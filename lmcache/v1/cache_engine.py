# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Union,
)
import asyncio
import gc
import multiprocessing
import threading
import time
import uuid
from contextlib import nullcontext

# Third Party
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.observability import LMCacheStatsLogger, LMCStatsMonitor
from lmcache.usage_context import InitializeUsageContext
from lmcache.utils import (
    CacheEngineKey,
    CacheRemoveEvent,
    CacheStoreEvent,
    _lmcache_nvtx_annotate,
    convert_tokens_to_list,
)
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.event_manager import EventManager, EventStatus, EventType
from lmcache.v1.gpu_connector import (
    GPUConnectorInterface,
    SGLangLayerwiseGPUConnector,
    VLLMBufferLayerwiseGPUConnector,
    VLLMPagedMemLayerwiseGPUConnector,
)
from lmcache.v1.memory_management import CuFileMemoryAllocator  # noqa: E501
from lmcache.v1.memory_management import (  # noqa: E501
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    MixedMemoryAllocator,
    PagedTensorMemoryAllocator,
    TensorMemoryObj,
)
from lmcache.v1.prefetch import PrefetchContext
from lmcache.v1.storage_backend.storage_manager import StorageManager
from lmcache.v1.system_detection import NUMADetector, NUMAMapping
from lmcache.v1.token_database import (
    ChunkedTokenDatabase,
    SegmentTokenDatabase,
    TokenDatabase,
)

logger = init_logger(__name__)

# Type aliases for processed chunks
# (cache_key, memory_obj, start_index, end_index)
ProcessedChunk = Tuple[CacheEngineKey, MemoryObj, int, int]
# (list of processed chunks, total kv size)
ProcessTokensInternalResult = Tuple[List[ProcessedChunk], int]


def _select_cxl_prefetch_keys(
    token_database: TokenDatabase,
    tokens: Iterable[int] | torch.Tensor,
    max_chunks: int,
    candidate_block_indices: Optional[Iterable[int]] = None,
) -> list[CacheEngineKey]:
    """Build bounded CXL-promotion keys, including sparse prefix positions.

    Prefix hashes are cumulative, so ``process_tokens`` must still walk every
    chunk before the furthest candidate.  The candidate index set prevents
    local/GPU blocks between CXL blocks from being submitted as promotions.
    ``None`` preserves the legacy contiguous-front behavior for older callers.
    """
    if max_chunks <= 0:
        return []

    requested_indices = (
        None
        if candidate_block_indices is None
        else {max(0, int(index)) for index in candidate_block_indices}
    )
    if requested_indices is not None and not requested_indices:
        return []

    target_count = (
        max_chunks
        if requested_indices is None
        else min(len(requested_indices), max_chunks)
    )
    keys: list[CacheEngineKey] = []
    for chunk_index, (_, _, key) in enumerate(
        token_database.process_tokens(tokens=tokens, mask=None)
    ):
        assert isinstance(key, CacheEngineKey)
        if requested_indices is not None and chunk_index not in requested_indices:
            continue
        keys.append(key)
        if len(keys) >= target_count:
            break
    return keys


class CacheEngineEndSignal:
    pass


@dataclass(frozen=True, slots=True)
class PromotionResult:
    requested: int
    promoted: tuple[CacheEngineKey, ...]
    already_present: tuple[CacheEngineKey, ...]
    source_missing: tuple[CacheEngineKey, ...]
    failed: tuple[CacheEngineKey, ...]
    bytes_promoted: int


@dataclass(frozen=True, slots=True)
class BackendOffloadResult:
    """One worker's bounded CPU-to-backend offload result."""

    success: bool
    committed_chunks: int
    already_present_chunks: int
    failed_chunks: int
    bytes_written: int
    error: Optional[str] = None


class _CxlPrefetchExecutor:
    """Worker-local data-plane executor for router-owned global prefetch.

    This class intentionally contains no predictor, session state, or pattern
    table.  Dynamo owns those decisions globally; LMCache only translates the
    full token sequence into rank-local keys and performs a bounded CXL to CPU
    copy.
    """

    _SOURCE = "CxlBackend"
    _TARGET = "LocalCPUBackend"

    @classmethod
    def maybe_create(
        cls, engine: "LMCacheEngine", config: LMCacheEngineConfig
    ) -> Optional["_CxlPrefetchExecutor"]:
        extra = config.extra_config or {}
        # Prediction and admission are router-owned.  LMCache is enabled only
        # when the explicit global flag is set; the former worker-local
        # association/cxl flags are intentionally no longer activation paths.
        enabled = bool(extra.get("cxl_global_prefetch_enabled", False))
        if not enabled or engine.storage_manager is None:
            return None
        backends = engine.storage_manager.storage_backends
        if cls._SOURCE not in backends or cls._TARGET not in backends:
            logger.warning(
                "Global CXL prefetch requires CxlBackend and LocalCPUBackend; disabled."
            )
            return None
        return cls(engine, extra)

    def __init__(self, engine: "LMCacheEngine", extra: dict) -> None:
        self.engine = engine
        self.max_chunks = max(
            1,
            int(
                extra.get(
                    "cxl_global_prefetch_max_chunks",
                    8,
                )
            ),
        )
        self.max_inflight = max(
            1,
            int(
                extra.get(
                    "cxl_global_prefetch_workers",
                    2,
                )
            ),
        )
        self.max_inflight_bytes = max(
            1,
            int(
                extra.get(
                    "cxl_global_prefetch_max_inflight_bytes",
                    128 * 1024**2,
                )
            ),
        )
        self.chunk_bytes = 1
        for dimension in engine.metadata.kv_shape:
            self.chunk_bytes *= int(dimension)
        self.chunk_bytes *= torch.empty(
            (), dtype=engine.metadata.kv_dtype
        ).element_size()
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_inflight,
            thread_name_prefix="lmcache-global-prefetch",
        )
        self._lock = threading.Lock()
        self._inflight_bytes = 0
        self._inflight_keys: set[CacheEngineKey] = set()
        self._request_futures: dict[str, set[Any]] = {}
        self._route_epochs: dict[str, int] = {}
        self._task_status: dict[str, dict[str, int | str | None]] = {}
        self._max_task_status = max(128, self.max_inflight * 256)
        self._closed = False
        self.scheduled = 0
        self.promoted = 0
        self.failed = 0
        logger.info(
            "LMCache global prefetch executor enabled (max_chunks=%d, workers=%d, "
            "max_inflight_bytes=%d)",
            self.max_chunks,
            self.max_inflight,
            self.max_inflight_bytes,
        )

    def prefetch_keys(
        self,
        keys: Iterable[CacheEngineKey],
        *,
        request_id: Optional[str] = None,
        route_epoch: int = 0,
        deadline_ns: Optional[int] = None,
        priority: int = 0,
        max_prefetch_bytes: Optional[int] = None,
        task_id: Optional[str] = None,
    ) -> int:
        unique = list(dict.fromkeys(keys))
        if not unique:
            return 0
        if deadline_ns is not None and time.time_ns() >= deadline_ns:
            return 0
        if request_id:
            with self._lock:
                previous = self._route_epochs.get(request_id, -1)
                if route_epoch < previous:
                    return 0
                self._route_epochs[request_id] = route_epoch
                while len(self._route_epochs) > self._max_task_status:
                    self._route_epochs.pop(next(iter(self._route_epochs)))
        byte_limit = min(
            self.max_inflight_bytes,
            max_prefetch_bytes if max_prefetch_bytes is not None else self.max_inflight_bytes,
        )
        accepted: list[CacheEngineKey] = []
        reserved = 0
        with self._lock:
            if self._closed:
                return 0
            for key in unique:
                if len(accepted) >= self.max_chunks:
                    break
                if key in self._inflight_keys:
                    continue
                if reserved + self.chunk_bytes > byte_limit:
                    break
                accepted.append(key)
                reserved += self.chunk_bytes
            if not accepted or self._inflight_bytes + reserved > self.max_inflight_bytes:
                return 0
            self._inflight_bytes += reserved
            self._inflight_keys.update(accepted)
            self.scheduled += len(accepted)
        event_id = f"global-prefetch-{request_id or uuid.uuid4().hex[:12]}"
        logical_task_id = task_id or event_id
        with self._lock:
            while len(self._task_status) >= self._max_task_status:
                self._task_status.pop(next(iter(self._task_status)))
            self._task_status[logical_task_id] = {
                "task_id": logical_task_id,
                "state": "SUBMITTED",
                "requested_chunks": len(accepted),
                "ready_chunks": 0,
                "bytes_copied": 0,
                "started_ns": time.time_ns(),
                "completed_ns": None,
            }
        try:
            future = self._executor.submit(
                self.engine.promote_keys_intra_node,
                accepted,
                self._SOURCE,
                self._TARGET,
                event_id,
                do_copy=True,
                max_chunks=self.max_chunks,
                allow_eviction=False,
            )
            if request_id:
                with self._lock:
                    self._request_futures.setdefault(request_id, set()).add(future)
            future.add_done_callback(
                lambda done: self._done(
                    done, accepted, reserved, request_id, logical_task_id
                )
            )
        except Exception:
            self._release(accepted, reserved, request_id)
            logger.exception("Failed to submit global CXL prefetch task")
            return 0
        return len(accepted)

    def _done(
        self,
        future: Any,
        keys: list[CacheEngineKey],
        reserved: int,
        request_id: Optional[str],
        task_id: str,
    ) -> None:
        try:
            result = future.result()
            with self._lock:
                self.promoted += len(result.promoted)
                self.failed += len(result.failed) + len(result.source_missing)
                status = self._task_status.get(task_id)
                if status is not None:
                    # A duplicate hint is a successful no-op when the target
                    # object reached LocalCPU between admission and execution.
                    # Count already-present chunks as ready; otherwise the
                    # router would observe FAILED, release its reservation,
                    # and retrigger the same physical work on every request.
                    ready_chunks = len(result.promoted) + len(result.already_present)
                    status.update(
                        state="READY" if ready_chunks == len(keys) else "FAILED",
                        ready_chunks=ready_chunks,
                        bytes_copied=int(result.bytes_promoted),
                        completed_ns=time.time_ns(),
                    )
        except Exception:
            with self._lock:
                self.failed += len(keys)
                status = self._task_status.get(task_id)
                if status is not None:
                    status.update(state="FAILED", completed_ns=time.time_ns())
            logger.exception("Global CXL prefetch task failed")
        finally:
            self._release(keys, reserved, request_id, future)

    def _release(
        self,
        keys: list[CacheEngineKey],
        reserved: int,
        request_id: Optional[str] = None,
        future: Any = None,
    ) -> None:
        with self._lock:
            self._inflight_bytes = max(0, self._inflight_bytes - reserved)
            self._inflight_keys.difference_update(keys)
            if request_id and future is not None:
                futures = self._request_futures.get(request_id)
                if futures is not None:
                    futures.discard(future)
                    if not futures:
                        self._request_futures.pop(request_id, None)

    def cancel(self, request_id: str, route_epoch: int) -> int:
        with self._lock:
            previous = self._route_epochs.get(request_id, -1)
            if route_epoch < previous:
                return 0
            self._route_epochs[request_id] = route_epoch
            futures = list(self._request_futures.get(request_id, ()))
        return sum(1 for future in futures if future.cancel())

    def close(self) -> None:
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)

    def status(self, task_id: str) -> Optional[dict[str, int | str | None]]:
        with self._lock:
            status = self._task_status.get(task_id)
            return dict(status) if status is not None else None



class LMCacheEngine:
    """The main class for the cache engine.

    When storing the KV caches into the cache engine, it takes GPU KV
    caches from the serving engine and convert them into MemoryObjs that
    resides in the CPU. The MemoryObjs are then being stored into the
    StorageBackends in an asynchronous manner.

    When retrieving the KV caches from the cache engine, it fetches the
    MemoryObjs from the StorageBackends and convert them into GPU KV caches
    by GPUConnectors specialized for the serving engine.

    It also supports prefetching the KV caches from the StorageBackends.
    It relies on the StorageBackends to manage the requests of prefetching
    and real retrieval and avoid the conflicts.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
        token_database: TokenDatabase,
        gpu_connector: Optional[GPUConnectorInterface],
        broadcast_fn: Callable[[torch.Tensor, int], None],
        broadcast_object_fn: Callable[[Any, int], Any],
    ):
        logger.info(f"Creating LMCacheEngine with config: {config}")
        self.config = config
        self.metadata = metadata
        self.token_database = token_database
        self.gpu_connector = gpu_connector
        self.broadcast_fn = broadcast_fn
        self.broadcast_object_fn = broadcast_object_fn
        # save_only_first_rank only works when use mla
        self.save_only_first_rank = (
            self.config.get_extra_config_value("save_only_first_rank", metadata.use_mla)
            and metadata.use_mla
        )

        if self.save_only_first_rank and self.gpu_connector is not None:
            self.broadcast_stream = (
                self.gpu_connector.load_stream
                if hasattr(self.gpu_connector, "load_stream")
                else torch.cuda.Stream()
            )

        self.enable_controller = config.enable_controller

        # NOTE: Unix systems use fork by default
        multiprocessing.set_start_method("spawn", force=True)

        # avoid circular import
        # First Party
        from lmcache.v1.cache_controller import LMCacheWorker

        self.lmcache_worker: Optional[LMCacheWorker] = None
        lmcache_worker_ids = config.get_lmcache_worker_ids(
            metadata.use_mla, metadata.world_size
        )
        # lmcache_worker_ids is empty means start on all workers
        if (
            self.enable_controller
            and self.metadata.role != "scheduler"
            and (not lmcache_worker_ids or metadata.worker_id in lmcache_worker_ids)
        ):
            self.lmcache_worker = LMCacheWorker(config, metadata, self)
        else:
            self.lmcache_worker = None
            logger.info(
                "LMCacheWorker is not initialized (related configs: "
                "enable_controller: %s, role: %s, worker_id: %s, worker_ids: %s).",
                self.enable_controller,
                self.metadata.role,
                self.metadata.worker_id,
                lmcache_worker_ids,
            )

        self.async_loading = config.enable_async_loading
        self.event_manager = EventManager()

        self.use_layerwise = config.use_layerwise

        # TODO: support save_only_first_rank when use layerwise
        # if use_layerwise is True, all ranks will initialize the storage_manager
        # if save_only_first_rank is False, all ranks will initialize
        # the storage_manager
        # if save_only_first_rank is True, only the first rank and
        # lookup server workers will initialize the storage_manager
        self.storage_manager = None
        lookup_server_worker_ids = self.config.get_lookup_server_worker_ids(
            metadata.use_mla, metadata.world_size
        )
        if (
            self.lmcache_worker is not None
            or self.use_layerwise
            or not self.save_only_first_rank
            or self.metadata.is_first_rank()
            or len(lookup_server_worker_ids) == 0
            or self.metadata.worker_id in lookup_server_worker_ids
        ):
            logger.info(
                f"Initialize storage manager on rank {self.metadata.worker_id}, "
                f"use layerwise: {self.use_layerwise},"
                f"save only first rank: {self.save_only_first_rank}"
            )
            self.storage_manager = StorageManager(
                config,
                metadata,
                # self.memory_allocator,
                event_manager=self.event_manager,
                lmcache_worker=self.lmcache_worker,
            )

        # KV events
        self.kv_events_enabled = False
        self.kv_events_enabled = config.enable_kv_events
        if self.kv_events_enabled:
            self.kv_events: List[CacheStoreEvent | CacheRemoveEvent] = []
            self._kv_event_tail_hash_by_req: Dict[str, int] = {}
            # block_hash -> (parent_block_hash, token_ids, block_size, lora_id)
            self._kv_store_meta_by_hash: Dict[int, Tuple[Optional[int], List[int], Optional[int], Optional[int]]] = {}
            if self.storage_manager is not None:
                self._register_backend_kv_event_sinks()
            logger.info("KV events are enabled.")
        else:
            logger.info("KV events are disabled.")

        # Dynamo owns global prediction and placement.  LMCache only keeps a
        # bounded, rank-local CXL->CPU data-plane executor.
        self._cxl_prefetch_executor = _CxlPrefetchExecutor.maybe_create(
            self, config
        )
        # Route-time hints are monotonic per request.  This prevents a stale
        # node reservation from starting new work after a reroute.
        self._prefetch_route_epochs: dict[str, int] = {}

        # HACK: remove this in the future
        # NOTE (Jiayi): This is currently used to support
        # dropping the kv cache from the buffer in PD backend
        # at decoder.
        self.remove_after_retrieve = config.enable_pd and config.pd_role == "receiver"

        self.num_layers = metadata.kv_shape[0]
        self.fmt = None
        if self.use_layerwise:
            if metadata.use_mla:
                self.fmt = MemoryFormat.KV_MLA_FMT
            elif config.enable_blending:
                self.fmt = MemoryFormat.KV_2TD
            else:
                self.fmt = MemoryFormat.KV_T2D
        if metadata.use_mla:
            self.fmt = MemoryFormat.KV_MLA_FMT

        # NOTE(ApostaC): we haven't support lookup-cache yet
        self.lookup_cache: dict[CacheEngineKey, Any] = {}

        # lookup_id -> {location -> [pinned keys]}
        self.lookup_pins: dict[str, dict[str, list]] = defaultdict(
            lambda: defaultdict(list)
        )
        # CPU-to-CXL offload is prepared rank-locally and published only after
        # the controller has observed a successful prepare on every TP rank.
        # Keep the newly written keys separate from lookup pin state: the CPU
        # source remains resident and the CXL objects are intentionally not
        # routable until the explicit commit message arrives.  Only newly
        # written keys are retained here so an abort cannot remove an object
        # that predated this operation and was reported as ALREADY_PRESENT.
        self._pending_offloads: dict[
            str, tuple[str, tuple[CacheEngineKey, ...]]
        ] = {}
        # After a rank publishes its events, keep the prepared keys until the
        # coordinator confirms that every TP rank committed.  If another rank
        # fails, the coordinator can still roll back already-published shards.
        self._committed_offloads: dict[
            str, tuple[str, tuple[CacheEngineKey, ...]]
        ] = {}
        self._pending_offloads_lock = threading.Lock()

        InitializeUsageContext(config.to_original_config(), metadata)
        self.stats_monitor = LMCStatsMonitor.GetOrCreate()

        self.post_inited = False

        # Whether to force store to wait if no CPU buffer is available
        self.force_store_wait = config.extra_config and config.extra_config.get(
            "force_store_wait", False
        )

        gc.collect()
        if not config.py_enable_gc:
            gc.disable()

    def post_init(self, **kwargs) -> None:
        if "async_lookup_server" in kwargs:
            self.async_lookup_server = kwargs["async_lookup_server"]
        if not self.post_inited:
            if self.storage_manager is not None:
                self.storage_manager.post_init(**kwargs)
            logger.info("Post-initializing LMCacheEngine")
            if self.gpu_connector is not None:
                self.gpu_connector.initialize_kvcaches_ptr(**kwargs)
            self.post_inited = True

    def _infer_parent_block_hash(
        self,
        start: int,
        tokens: Optional[Union[torch.Tensor, list[int]]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        request_configs: Optional[dict] = None,
    ) -> Optional[int]:
        """Infer parent block hash for the first stored chunk when start > 0."""
        if start <= 0:
            return None

        try:
            if tokens is not None:
                chunk_iter = self.token_database.process_tokens(
                    tokens=tokens,
                    mask=None,
                    make_key=False,
                    request_configs=request_configs,
                )
            elif hashes is not None and offsets is not None:
                chunk_iter = self.token_database.process_tokens(
                    hashes=hashes,
                    offsets=offsets,
                    make_key=False,
                    request_configs=request_configs,
                )
            else:
                return None

            for chunk_start, chunk_end, chunk_hash in chunk_iter:
                if chunk_end == start:
                    return int(chunk_hash)
                if chunk_start >= start:
                    break
        except Exception as e:
            logger.warning(
                "Failed to infer KV event parent hash (start=%s): %s",
                start,
                e,
            )
        return None

    def _log_kv_event_parent_check(
        self,
        request_id: str,
        is_first_event: bool,
        first_block_start: Optional[int],
        parent_block_hash: Optional[int],
    ) -> None:
        # For the first event of a request, parent can be non-None if storing
        # starts from a non-zero offset (e.g. prefix chunks were skipped).
        expects_parent_none = is_first_event and first_block_start == 0
        parent_is_none = parent_block_hash is None

        if expects_parent_none and not parent_is_none:
            logger.warning(
                "KV event parent-check failed: first event starting at token 0 "
                "should have parent_block_hash=None (req_id=%s, parent=%s)",
                request_id,
                parent_block_hash,
            )
            return

        if (not expects_parent_none) and parent_is_none:
            logger.warning(
                "KV event parent-check failed: expected non-None parent "
                "(req_id=%s, is_first_event=%s, first_block_start=%s)",
                request_id,
                is_first_event,
                first_block_start,
            )
            return

        logger.info(
            "KV event parent-check passed: req_id=%s, is_first_event=%s, "
            "first_block_start=%s, parent_block_hash=%s",
            request_id,
            is_first_event,
            first_block_start,
            parent_block_hash,
        )

    def freeze(self, enabled: bool) -> None:
        """
        Set the freeze mode for the cache engine.

        When freeze mode is enabled:
        - All store operations will be skipped (no new data stored)
        - Only local_cpu backend will be used for retrieval
        - No admit/evict messages will be generated
        This protects the local_cpu hot cache from changes.

        Args:
            enabled (bool): Whether to enable freeze mode
        """
        if self.storage_manager is not None:
            self.storage_manager.set_freeze(enabled)
            logger.info("freeze mode %s", "enabled" if enabled else "disabled")

    def is_frozen(self) -> bool:
        """
        Get the current freeze mode status.

        Returns:
            bool: True if freeze mode is enabled, False otherwise
        """
        if self.storage_manager is not None:
            return self.storage_manager.is_frozen()
        return False

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def store(
        self,
        tokens: Optional[Union[torch.Tensor, list[int]]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        """Store the tokens/hashes and mask into the cache engine.

        :param Optional[torch.Tensor] tokens: The tokens of the corresponding KV caches.

        :param Optional[List[int]] hashes: The hashes of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched,
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.
            Should include KV cache specific information (e.g., paged KV buffer
            and the page tables).

        :raises: ValueError if the number of Falses in the mask is not a
            multiple of the chunk size.
        """
        assert self.gpu_connector is not None, (
            "gpu_connector is required for store operation"
        )

        if self._is_passive():
            logger.debug(f"rank={self.metadata.worker_id} ignore store")
            return

        assert self.storage_manager is not None

        # Initialize num_to_store_tokens to avoid reference before assignment
        num_to_store_tokens = 0

        if mask is not None:
            num_to_store_tokens = torch.sum(mask).item()
        elif tokens is not None:
            num_to_store_tokens = len(tokens)
        elif hashes is not None:
            assert offsets is not None, (
                "Offsets should be set when hashes are provided during store"
            )
            num_to_store_tokens = sum(offsets)
            kwargs["slot_mapping"] = torch.tensor(
                kwargs["slot_mapping"], dtype=torch.long, device="cuda"
            )

        assert tokens is not None or hashes is not None, (
            "Either 'tokens' or 'hashes' must be provided."
        )

        # Check if freeze mode is enabled
        if self.is_frozen():
            logger.debug(
                "Freeze mode enabled, skipping store operation for %d tokens",
                num_to_store_tokens,
            )
            return

        monitor_req_id = self.stats_monitor.on_store_request(num_to_store_tokens)

        starts: List[int] = []
        ends: List[int] = []
        keys: List[CacheEngineKey] = []
        memory_objs: List[MemoryObj] = []

        offload_time = 0.0
        put_time = 0.0
        tot_kv_size = 0
        tot_token_num = 0
        t = time.perf_counter()

        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)
        request_id = kwargs.get("request_id")

        prev_key = 0
        event_block_hashes: List[int] = []
        event_token_ids: List[int] = []
        event_parent_block_hash: Optional[int] = None
        event_first_block_start: Optional[int] = None
        event_block_size = getattr(self.token_database, "chunk_size", self.config.chunk_size)
        for start, end, key in self.token_database.process_tokens(
            tokens,
            hashes,
            offsets,
            mask,
            request_configs=request_configs,
        ):
            assert isinstance(key, CacheEngineKey)
            # Allocate the memory object
            num_tokens = end - start
            kv_shape = self.gpu_connector.get_shape(num_tokens)
            kv_dtype = self.metadata.kv_dtype

            # TODO (Jiayi): should be batched in the future
            memory_obj = self.storage_manager.allocate(
                kv_shape,
                kv_dtype,
                busy_loop=self.force_store_wait,
                fmt=self.fmt,
            )
            if memory_obj is None:
                logger.warning(
                    "Local cpu memory under pressure so"
                    " choosing to store only "
                    f" {len(memory_objs)}"
                    " total chunks of KV cache."
                )
                break

            starts.append(start)
            ends.append(end)
            keys.append(key)
            memory_objs.append(memory_obj)
            tot_kv_size += memory_obj.get_size()
            tot_token_num += num_tokens

            # Aggregate KV event for this store() call
            if self.kv_events_enabled:
                if not event_block_hashes:
                    event_first_block_start = start
                    if start == 0:
                        event_parent_block_hash = None
                    elif request_id is not None:
                        event_parent_block_hash = self._kv_event_tail_hash_by_req.get(request_id)
                        if event_parent_block_hash is None:
                            event_parent_block_hash = self._infer_parent_block_hash(
                                start=start,
                                tokens=tokens,
                                hashes=hashes,
                                offsets=offsets,
                                request_configs=request_configs,
                            )
                    else:
                        event_parent_block_hash = self._infer_parent_block_hash(
                            start=start,
                            tokens=tokens,
                            hashes=hashes,
                            offsets=offsets,
                            request_configs=request_configs,
                        )
                        if event_parent_block_hash is None:
                            event_parent_block_hash = prev_key if prev_key != 0 else None
                event_block_hashes.append(key.chunk_hash)
                if tokens is not None:
                    event_token_ids.extend(
                        convert_tokens_to_list(
                            tokens,
                            start,
                            end,
                        )
                    )
                elif hashes is not None:
                    event_token_ids.extend(hashes[start:end])
                prev_key = key.chunk_hash

        if self.kv_events_enabled and event_block_hashes:
            stored_event = CacheStoreEvent(
                block_hashes=event_block_hashes,
                parent_block_hash=event_parent_block_hash,
                token_ids=event_token_ids,
                block_size=event_block_size,
                lora_id=None,
                medium="CPU",
            )
            self.kv_events.append(stored_event)
            self._cache_store_event_metadata(stored_event)
            logger.info(
                "Queued KV store event: req_id=%s, num_blocks=%d, medium=%s, "
                "block_size=%d, parent_block_hash=%s",
                request_id,
                len(stored_event.block_hashes),
                stored_event.medium or "unknown",
                stored_event.block_size,
                stored_event.parent_block_hash,
            )
            if request_id is not None:
                is_first_event = request_id not in self._kv_event_tail_hash_by_req
                self._log_kv_event_parent_check(
                    request_id=request_id,
                    is_first_event=is_first_event,
                    first_block_start=event_first_block_start,
                    parent_block_hash=stored_event.parent_block_hash,
                )
            if request_id is not None:
                self._kv_event_tail_hash_by_req[request_id] = event_block_hashes[-1]


        # memory_objs might be empty, directly return to avoid sending tokens
        if not memory_objs:
            return
        self.gpu_connector.batched_from_gpu(memory_objs, starts, ends, **kwargs)
        offload_time += time.perf_counter() - t

        t = time.perf_counter()

        transfer_spec = kwargs.get("transfer_spec", None)
        # TODO: we implicitly rely on batched_put to call ref_count_down
        # this management should be done in a cleaner way
        self.storage_manager.batched_put(keys, memory_objs, transfer_spec=transfer_spec)
        put_time += time.perf_counter() - t

        tot_time = offload_time + put_time

        logger.info(
            "Stored %d out of total %d tokens. size: %.4f gb, cost %.4f ms, "
            "throughput: %.4f GB/s; offload_time: %.4f ms, put_time: %.4f ms",
            tot_token_num,
            num_to_store_tokens,
            tot_kv_size / 1024**3,
            tot_time * 1000,
            tot_kv_size / tot_time / 1024**3,
            offload_time * 1000,
            put_time * 1000,
        )

        self.stats_monitor.on_store_finished(monitor_req_id, tot_token_num)

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def store_layer(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Generator[None, None, None]:
        """
        Store the KV cache in a layerwise manner.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.

        return: A generator that yields None. In the first iteration, the
            generator allocates the memory objects for all layers and moves
            the KV cache of the first layer from GPU to CPU. In the next
            iterations, it moves the KV cache of layer i from GPU to the memory
            objects (on CPU) and puts the memory objects of layer i-1 to the
            storage backends. In the last iteration, it puts the memory objects
            of the last layer to the storage backends.
        """
        assert self.storage_manager is not None
        assert self.gpu_connector is not None, (
            "gpu_connector is required for store_layer operation"
        )

        if mask is not None:
            num_to_store_tokens = torch.sum(mask).item()
        else:
            num_to_store_tokens = len(tokens)
        monitor_req_id = self.stats_monitor.on_store_request(num_to_store_tokens)

        # Check if freeze mode is enabled
        if self.is_frozen():
            logger.debug(
                "Freeze mode enabled, skipping store_layer for %d tokens",
                num_to_store_tokens,
            )
            # Still need to yield to avoid StopIteration
            for layer_id in range(self.num_layers):
                yield
            return

        starts = []
        ends = []
        keys = []
        memory_objs = []
        tot_token_num = 0
        kv_dtype = self.metadata.kv_dtype
        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)
        request_id = kwargs.get("request_id")

        prev_key = 0
        event_block_hashes: List[int] = []
        event_token_ids: List[int] = []
        event_parent_block_hash: Optional[int] = None
        event_first_block_start: Optional[int] = None
        event_block_size = getattr(self.token_database, "chunk_size", self.config.chunk_size)
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens, mask=mask, request_configs=request_configs
        ):
            assert isinstance(key, CacheEngineKey)

            keys_multi_layer = key.split_layers(self.num_layers)
            # Only check the first layer
            if self.storage_manager.contains(keys_multi_layer[0]):
                continue

            # Allocate the memory object
            num_tokens = end - start
            kv_shape_single_layer = self.gpu_connector.get_shape(num_tokens)

            memory_objs_multi_layer = self.storage_manager.batched_allocate(
                kv_shape_single_layer,
                kv_dtype,
                batch_size=self.num_layers,
                fmt=self.fmt,
                busy_loop=self.force_store_wait,
            )

            if memory_objs_multi_layer is None:
                logger.warning(
                    "Local cpu memory under pressure so"
                    " choosing to not store the KV cache."
                )
                break

            starts.append(start)
            ends.append(end)
            keys.append(keys_multi_layer)
            memory_objs.append(memory_objs_multi_layer)
            tot_token_num += num_tokens

            # Aggregate KV event for this store_layer() call
            if self.kv_events_enabled and tokens is not None:
                if not event_block_hashes:
                    event_first_block_start = start
                    if start == 0:
                        event_parent_block_hash = None
                    elif request_id is not None:
                        event_parent_block_hash = self._kv_event_tail_hash_by_req.get(request_id)
                        if event_parent_block_hash is None:
                            event_parent_block_hash = self._infer_parent_block_hash(
                                start=start,
                                tokens=tokens,
                                request_configs=request_configs,
                            )
                    else:
                        event_parent_block_hash = self._infer_parent_block_hash(
                            start=start,
                            tokens=tokens,
                            request_configs=request_configs,
                        )
                        if event_parent_block_hash is None:
                            event_parent_block_hash = prev_key if prev_key != 0 else None
                event_block_hashes.append(key.chunk_hash)
                event_token_ids.extend(
                    convert_tokens_to_list(
                        tokens,
                        start,
                        end,
                    )
                )
                prev_key = key.chunk_hash

        if self.kv_events_enabled and event_block_hashes:
            stored_event = CacheStoreEvent(
                block_hashes=event_block_hashes,
                parent_block_hash=event_parent_block_hash,
                token_ids=event_token_ids,
                block_size=event_block_size,
                lora_id=None,
                medium="CPU",
            )
            self.kv_events.append(stored_event)
            self._cache_store_event_metadata(stored_event)
            logger.info(
                "Queued KV store event: req_id=%s, num_blocks=%d, medium=%s, "
                "block_size=%d, parent_block_hash=%s",
                request_id,
                len(stored_event.block_hashes),
                stored_event.medium or "unknown",
                stored_event.block_size,
                stored_event.parent_block_hash,
            )
            if request_id is not None:
                is_first_event = request_id not in self._kv_event_tail_hash_by_req
                self._log_kv_event_parent_check(
                    request_id=request_id,
                    is_first_event=is_first_event,
                    first_block_start=event_first_block_start,
                    parent_block_hash=stored_event.parent_block_hash,
                )
            if request_id is not None:
                self._kv_event_tail_hash_by_req[request_id] = event_block_hashes[-1]


        if keys:
            # Transpose the keys and memory objects into layer major format
            memory_objs = [list(row) for row in zip(*memory_objs, strict=False)]
            keys = [list(row) for row in zip(*keys, strict=False)]

            assert isinstance(
                self.gpu_connector,
                (
                    VLLMPagedMemLayerwiseGPUConnector,
                    VLLMBufferLayerwiseGPUConnector,
                    SGLangLayerwiseGPUConnector,
                ),
            )

            mem_obj_generator = self.gpu_connector.batched_from_gpu(
                memory_objs, starts, ends, **kwargs
            )

            next(mem_obj_generator)

            for layer_id in range(self.num_layers):
                yield
                next(mem_obj_generator)
                self.storage_manager.batched_put(keys[layer_id], memory_objs[layer_id])
        else:
            # If no cache are found, we still need to yield to avoid
            # `StopIteration`
            for layer_id in range(self.num_layers):
                yield

        self.stats_monitor.on_store_finished(monitor_req_id, tot_token_num)
        logger.debug(f"Stored {tot_token_num} out of total {len(tokens)} tokens")
        yield

    def submit_cxl_prefetch(
        self,
        request_id: str,
        tokens: Iterable[int] | torch.Tensor,
        max_chunks: int,
        candidate_block_indices: Optional[Iterable[int]] = None,
    ) -> dict[str, int | bool | str | None]:
        """Translate logical tokens and submit a bounded reactive hint.

        Dynamo never constructs LMCache keys.  The local TokenDatabase keeps
        chunk size, hash configuration, model namespace, dtype, and TP-rank
        semantics on the LMCache side where they belong.
        """
        if self.storage_manager is None or max_chunks <= 0:
            return {
                "enabled": False,
                "scheduled": 0,
                "deduplicated": 0,
                "already_cpu": 0,
                "capacity_rejected": 0,
                "status": "unavailable",
            }

        keys = _select_cxl_prefetch_keys(
            self.token_database,
            tokens,
            max_chunks,
            candidate_block_indices,
        )

        return self.storage_manager.submit_cxl_prefetch(
            keys,
            max_chunks=max_chunks,
            request_id=request_id,
        )

    def get_cxl_prefetch_stats(self) -> dict[str, int]:
        """Return bounded reactive CXL prefetch admission/completion counters."""
        if self.storage_manager is None:
            return {}
        return self.storage_manager.get_cxl_prefetch_stats()

    def prefetch_segments(
        self,
        keys: Iterable[CacheEngineKey],
        *,
        trigger_key: Optional[CacheEngineKey] = None,
        request_id: Optional[str] = None,
        deadline_ns: Optional[int] = None,
        priority: int = 0,
        context: Optional[PrefetchContext] = None,
        max_prefetch_bytes: Optional[int] = None,
        route_epoch: int = 0,
        task_id: Optional[str] = None,
    ) -> int:
        """Best-effort prefetch of an arbitrary set of cache segments.

        Keys are intentionally independent (they need not form a prefix), so
        a Dynamo/router hint can be forwarded without going through the legacy
        prefix-only ``async_lookup_and_prefetch`` path.  The operation is
        default-off with the global prefetch configuration and returns
        the number of entries accepted by its bounded background queue.
        """
        executor = self._cxl_prefetch_executor
        if executor is None:
            return 0
        return executor.prefetch_keys(
            keys,
            request_id=request_id,
            deadline_ns=deadline_ns,
            priority=priority,
            max_prefetch_bytes=max_prefetch_bytes,
            route_epoch=route_epoch,
            task_id=task_id,
        )

    def prefetch_tokens(
        self,
        tokens: Iterable[int] | torch.Tensor,
        *,
        request_id: Optional[str] = None,
        route_epoch: int = 0,
        deadline_ns: Optional[int] = None,
        priority: int = 0,
        max_prefetch_bytes: Optional[int] = None,
        request_configs: Optional[dict] = None,
        task_id: Optional[str] = None,
        start_chunk: Optional[int] = None,
        end_chunk: Optional[int] = None,
        ttl_ms: int = 5000,
    ) -> int:
        """Route-time hint entry point using logical prompt tokens.

        Token-to-key conversion remains local to the worker, preserving TP
        rank, model namespace, dtype, tags, and the configured hash function.
        Older epochs are rejected before any CXL work is submitted.
        """
        if request_id:
            previous = self._prefetch_route_epochs.get(request_id, -1)
            if route_epoch < previous:
                return 0
            self._prefetch_route_epochs[request_id] = route_epoch
        infos = list(
            self.token_database.process_tokens(
                tokens=tokens,
                mask=None,
                request_configs=request_configs,
            )
        )
        begin = max(0, int(start_chunk or 0))
        finish = int(end_chunk) if end_chunk is not None else len(infos)
        finish = max(begin, min(finish, len(infos)))
        keys = [key for _, _, key in infos[begin:finish]]
        return self.prefetch_segments(
            keys,
            request_id=request_id,
            deadline_ns=deadline_ns,
            priority=priority,
            route_epoch=route_epoch,
            max_prefetch_bytes=max_prefetch_bytes,
            task_id=task_id,
            context=PrefetchContext(
                model_namespace=str(self.metadata.model_name),
                tp_rank=int(self.metadata.worker_id),
            ),
        )

    def prefetch_status(self, task_id: str) -> Optional[dict]:
        """Return a bounded snapshot for a router-issued prefetch task."""
        executor = self._cxl_prefetch_executor
        if executor is None:
            return None
        return executor.status(task_id)

    def cancel_prefetch_hint(self, request_id: str, route_epoch: int) -> int:
        """Cancel queued work for an obsolete route epoch."""
        previous = self._prefetch_route_epochs.get(request_id, -1)
        if route_epoch < previous:
            return 0
        self._prefetch_route_epochs[request_id] = route_epoch
        executor = self._cxl_prefetch_executor
        if executor is None:
            return 0
        return executor.cancel(request_id, route_epoch)

    def drain_prefetch_access_events(self):
        """The predictor moved to Dynamo; no worker-local events are emitted."""
        return []

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def retrieve(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Retrieve the KV caches from the cache engine. And put the retrieved
        KV cache to the serving engine via the GPU connector.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched,
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.
            Should include KV cache specific information (e.g., paged KV buffer
            and the page tables).

        :return: the boolean mask indicating which tokens are retrieved. The
            length of the mask should be the same as the tokens. On CPU.

        :raises: ValueError if the number of Falses in the mask is not a
            multiple of the chunk size.
        """
        assert self.gpu_connector is not None, (
            "gpu_connector is required for retrieve operation"
        )

        tot_kv_size = 0
        t = time.perf_counter()

        if mask is not None:
            num_required_tokens = torch.sum(mask).item()
        else:
            num_required_tokens = len(tokens)
        monitor_req_id = self.stats_monitor.on_retrieve_request(num_required_tokens)

        ret_mask = torch.zeros(len(tokens), dtype=torch.bool, device="cpu")

        reordered_chunks: List[ProcessedChunk] = []
        if not self._is_passive():
            if self.async_loading:
                reordered_chunks, tot_kv_size = self._async_process_tokens_internal(  # noqa: E501
                    tokens,
                    mask,
                    ret_mask,
                    **kwargs,
                )
            else:
                reordered_chunks, tot_kv_size = self._process_tokens_internal(
                    tokens,
                    mask,
                    ret_mask,
                    **kwargs,
                )
        if self.save_only_first_rank:
            with torch.cuda.stream(self.broadcast_stream):
                self._broadcast_or_receive_memory_objs(
                    reordered_chunks,
                    ret_mask,
                )

            # if self.gpu_connector has load_stream, self.broadcast_stream is equals
            # to self.gpu_connector.load_stream, the broadcast and to_gpu operation
            # will execute sequentially within the stream.
            # if self.gpu_connector does not have load_stream, self.broadcast_stream
            # is created by torch.cuda.Stream(), we need to synchronize broadcast
            # operation, and then process to_cpu operation.
            if not hasattr(self.gpu_connector, "load_stream"):
                self.broadcast_stream.synchronize()

        # NOTE(Jiayi): memory_obj doesn't have to be a pinned
        # cpu tensor for the sake of performance.
        # For example, disk->gpu is faster than disk->cpu->gpu.
        # RDMA is another example.
        if len(reordered_chunks) > 0:
            _, memory_objs, starts, ends = zip(*reordered_chunks, strict=False)
            self.gpu_connector.batched_to_gpu(
                list(memory_objs), list(starts), list(ends), **kwargs
            )

        # TODO(Jiayi): Remove the following for loop with batched operations
        # TODO(Jiayi): Need to refactor the `remove_after_retrieve` logic.
        for key, memory_obj, _, _ in reordered_chunks:
            if self.remove_after_retrieve and not self._is_passive():
                assert self.storage_manager is not None
                self.storage_manager.remove(key)
            memory_obj.ref_count_down()

        onload_time = time.perf_counter() - t

        retrieved_tokens = torch.sum(ret_mask)
        self.stats_monitor.on_retrieve_finished(monitor_req_id, retrieved_tokens)
        logger.info(
            "Retrieved %d out of %d required tokens (from %d total tokens)."
            " size: %.4f gb,"
            " cost %.4f ms, throughput: %.4f GB/s;",
            retrieved_tokens,
            num_required_tokens,
            len(tokens),
            tot_kv_size / 1024**3,
            onload_time * 1000,
            tot_kv_size / onload_time / 1024**3 if onload_time > 0 else 0,
        )

        return ret_mask

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def retrieve_layer(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Generator[Optional[torch.Tensor], None, None]:
        """
        Retrieve the KV cache in a layerwise manner.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.

        return: A generator that yields Optional[torch.Tensor]. The tensor will
            be the boolean mask indicating which tokens are retrieved and will
            only be returned in the last iteration. In the first iteration,
            the generator retrieve the memory objects of the first layer from
            the storage backends. In the next iterations, it moves the KV cache
            of layer i from the memory objects (on CPU) to GPU and retrieves
            the memory objects of layer i+1 from the storage backends. In the
            last iteration, it moves the memory objects of the last layer to
            the GPU.
        """
        assert self.storage_manager is not None
        assert self.gpu_connector is not None, (
            "gpu_connector is required for retrieve_layer operation"
        )

        if mask is not None:
            num_required_tokens = torch.sum(mask).item()
        else:
            num_required_tokens = len(tokens)
        monitor_req_id = self.stats_monitor.on_retrieve_request(num_required_tokens)

        ret_mask = torch.zeros(len(tokens), dtype=torch.bool, device="cpu")

        starts = []
        ends = []
        keys = []

        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)

        location = None
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens,
            mask=mask,
            request_configs=request_configs,
        ):
            assert isinstance(key, CacheEngineKey)

            keys_multi_layer = key.split_layers(self.num_layers)

            # NOTE: Only check the first layer
            if current_location := self.storage_manager.contains(keys_multi_layer[0]):
                if location is None:
                    location = current_location
                else:
                    # TODO(Jiayi): Support multi-location retrieval in the future
                    assert location == current_location, (
                        "All retrieved keys should be from the same location "
                        "when use layerwise retrieval."
                        "Please support multi-location retrieval in the future."
                    )
            else:
                break

            starts.append(start)
            ends.append(end)
            keys.append(keys_multi_layer)

            ret_mask[start:end] = True

        if keys:
            # Transpose the keys into layer major format
            keys_layer_major = [list(row) for row in zip(*keys, strict=False)]

            get_generator = self.storage_manager.layerwise_batched_get(
                keys_layer_major,
                location=location,
            )

            assert isinstance(
                self.gpu_connector,
                (
                    VLLMPagedMemLayerwiseGPUConnector,
                    VLLMBufferLayerwiseGPUConnector,
                    SGLangLayerwiseGPUConnector,
                ),
            )
            mem_obj_consumer = self.gpu_connector.batched_to_gpu(starts, ends, **kwargs)
            next(mem_obj_consumer)

            to_count_down = []
            for layer_id in range(self.num_layers):
                task = next(get_generator)

                assert task is not None

                if layer_id == 0:
                    # NOTE(Yuwei): For sglang integration we need to provide retrieved
                    # tokens number in the first layer loading since there is no lookup
                    yield torch.sum(ret_mask)
                else:
                    yield None

                mem_objs_layer = task.result()
                mem_obj_consumer.send(mem_objs_layer)
                to_count_down.extend(mem_objs_layer)

            for mem_obj in to_count_down:
                mem_obj.ref_count_down()
        else:
            # If no cache are found, we still need to yield to avoid
            # `StopIteration`
            for layer_id in range(self.num_layers):
                yield None

        yield None

        # synchronize the last layer
        next(mem_obj_consumer)

        retrieved_tokens = torch.sum(ret_mask)
        self.stats_monitor.on_retrieve_finished(monitor_req_id, retrieved_tokens)
        logger.info(
            f"Retrieved {retrieved_tokens} "
            f"out of {num_required_tokens} "
            f"out of total {len(tokens)} tokens"
        )

        yield ret_mask

    @_lmcache_nvtx_annotate
    def lookup(
        self,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        search_range: Optional[List[str]] = None,
        lookup_id: Optional[str] = None,
        pin: bool = False,
        request_configs: Optional[dict] = None,
        num_computed_tokens: int = 0,
    ) -> int:
        """
        Checks the existence of KV cache of the tokens from the cache engine.

        :param Optional[Union[torch.Tensor, List[int]]] tokens: the input tokens,
        with shape [seq_len]

        :param Optional[List[int]] hashes: the input hashes, with length [num_chunks]
        :param Optional[List[int]] offsets: the offsets of each chunk,
        with length [num_chunks]

        :param Optional[List[str]] search_range: The range of storage backends
        to search in. Should be a subset of
        ["LocalCPUBackend", "LocalDiskBackend"] for now.
        If None, search in all backends.

        :param Optional[str] lookup_id: The lookup ID to
            associate with the lookup. When pin is true, this argument is
            required to be not None.

        :param bool pin: If True, pin the KV cache in the storage.

        :param Optional[dict] request_configs: the configs of the request.

        :param int num_computed_tokens: Number of leading tokens those are already
            available in the caller.

        :return: An int indicating how many prefix tokens are cached.
        """
        assert self.storage_manager is not None

        if tokens is not None:
            lookup_request_id = self.stats_monitor.on_lookup_request(len(tokens))
        else:
            assert offsets is not None
            assert hashes is not None
            lookup_request_id = self.stats_monitor.on_lookup_request(sum(offsets))

        # Skip the number of tokens that are already computed (aligned upstream to
        # chunk size)
        aligned_computed_tokens = num_computed_tokens
        res = aligned_computed_tokens
        try:
            chunk_info_iterator = self.token_database.process_tokens(
                tokens=tokens,
                hashes=hashes,
                offsets=offsets,
                request_configs=request_configs,
            )

            # TODO: support batched_contains when layerwise is enabled
            if self.use_layerwise:
                for start, end, key in chunk_info_iterator:
                    if end <= aligned_computed_tokens:
                        continue
                    assert isinstance(key, CacheEngineKey)

                    # TODO(Jiayi): Optimize by checking only the existence of the key
                    # of one layer
                    key_all_layers = key.split_layers(self.num_layers)

                    hit_chunks, block_mapping = self.storage_manager.batched_contains(
                        key_all_layers,  # type: ignore
                        search_range,
                        pin,
                    )
                    # Only all layers are hit and hit in one location,
                    # we consider this key as a hit
                    if hit_chunks == self.num_layers and len(block_mapping) == 1:
                        if pin:
                            assert lookup_id is not None, (
                                "lookup_id is required when pin is True"
                            )
                            location = next(iter(block_mapping.keys()))
                            self.lookup_pins[lookup_id][location].extend(key_all_layers)
                        res = end
                        continue
                    return res
            else:
                chunk_info_list = []
                keys = []
                for chunk_info in chunk_info_iterator:
                    assert isinstance(chunk_info[2], CacheEngineKey)
                    start, end, _ = chunk_info
                    if end <= aligned_computed_tokens:
                        continue
                    chunk_info_list.append(chunk_info)
                    # chunk_info contains (start, end, key)
                    # chunk_info[2] is the key
                    keys.append(chunk_info[2])
                # If no tokens to lookup, return immediately
                if not keys:
                    return res
                # hit chunks by prefix matching
                hit_chunks, block_mapping = self.storage_manager.batched_contains(
                    keys, search_range, pin
                )
                if pin and block_mapping:
                    assert lookup_id is not None, (
                        "lookup_id is required when pin is True"
                    )
                    self.lookup_pins[lookup_id] = block_mapping
                for idx, (start, end, key) in enumerate(chunk_info_list):
                    if idx < hit_chunks:
                        res = end
                        continue
                    return res

            # all tokens where found, return the maximal end
            return res
        finally:
            # When num_computed_tokens is greater than a chunk, we skip
            # some tokens to reduce the number of lookup requests.
            # It is possible that res equals aligned_computed_tokens and no lookup is
            # performed.
            # In this case, using res as the number of hit tokens will overcount
            # the number of hit tokens.
            # TODO deprecate this metric and use retrieve metrics instead.
            self.stats_monitor.on_lookup_finished(lookup_request_id, res)
            # vllm lookup sets pin to True
            if pin:
                self.storage_manager.touch_cache()

    @_lmcache_nvtx_annotate
    def move(
        self,
        tokens: Union[torch.Tensor, List[int]],
        old_position: str,
        new_position: tuple[str, str],
        event_id: str,
        do_copy: bool = True,
    ) -> int:
        """
        Perform cross-node move of the KV cache.
        """
        assert self.storage_manager is not None

        memory_objs: list[Optional[MemoryObj]] = []
        num_tokens = self.lookup(
            tokens,
            search_range=[old_position],
            lookup_id=event_id,
            pin=True,
        )
        try:
            keys = self.lookup_pins.get(event_id, {}).get(old_position, [])
            if not num_tokens or not keys:
                logger.debug("Move is not performed as there are no tokens to move.")
                return 0

            memory_objs = self.storage_manager.batched_get(
                keys=keys,
                location=old_position,
            )
            assert memory_objs is not None, "Failed to get memory objects to move"
            if any(memory_obj is None for memory_obj in memory_objs):
                raise RuntimeError("Failed to get a complete pinned prefix to move")
            logger.debug(
                f"Trying to send {len(memory_objs)} memory objects to {new_position}"
            )

            typed_memory_objs = [memory_obj for memory_obj in memory_objs if memory_obj]
            # TODO: reduce loops
            token_dim = typed_memory_objs[0].meta.fmt.token_dim()
            offsets = [memory_obj.meta.shape[token_dim] for memory_obj in typed_memory_objs]

            transfer_spec = {
                "target_peer_init_url": new_position[0],
                "offsets": offsets,
            }

            p2p_backend = self.storage_manager.storage_backends["P2PBackend"]
            future = asyncio.run_coroutine_threadsafe(
                p2p_backend.async_batched_submit_put_task(
                    keys,
                    typed_memory_objs,
                    transfer_spec=transfer_spec,
                ),
                self.storage_manager.loop,
            )
            future.result()

            if not do_copy:
                self.storage_manager.batched_remove(keys, locations=[old_position])

            logger.debug(
                f"Moving {len(keys)} chunks from {old_position} to {new_position}"
            )
            return num_tokens
        finally:
            for memory_obj in memory_objs:
                if memory_obj is not None:
                    memory_obj.ref_count_down()
            self.lookup_unpin(event_id)

    def move_intra_node(
        self,
        tokens: Union[torch.Tensor, List[int]],
        old_position: str,
        new_position: tuple[str, str],
        event_id: str,
        do_copy: bool = True,
        max_chunks: Optional[int] = None,
    ) -> int:
        """
        Intra-node (same worker) promotion of a KV prefix into LocalCPUBackend.

        This implements the "prefetch" branch of a controller-driven move whose
        destination is the *same* worker (see cache_controller/worker.py). It
        pulls a contiguous prefix from a slower local tier (e.g. ``CxlBackend``
        or ``LocalDiskBackend``) up into ``LocalCPUBackend`` so a subsequent
        lookup hits the faster CPU tier instead of the source tier.

        Unlike :meth:`move`, this does NOT go through ``P2PBackend`` / the
        network: the copy is a local (e.g. CXL->CPU) write-back.

        :param tokens: The token ids of the prefix to promote.
        :param old_position: Source backend name, e.g. ``"CxlBackend"``.
        :param new_position: ``(worker_url, backend_name)``; ``backend_name``
            must be ``"LocalCPUBackend"`` for now.
        :param event_id: Unique id used to pin/unpin the source keys.
        :param do_copy: If True (default) keep the source copy; if False, remove
            the promoted keys from the source tier after the copy.
        :param max_chunks: Optional budget cap; promote at most this many leading
            chunks. Used by the predictive prefetcher to bound CPU-tier pressure.

        :return: Number of chunks newly promoted into LocalCPUBackend. (Reported
            back to the controller via ``MoveWorkerRetMsg.num_tokens`` as a unit
            count.)
        """
        assert self.storage_manager is not None
        assert new_position[1] == "LocalCPUBackend", (
            "move_intra_node currently only supports promoting to LocalCPUBackend."
        )

        # 1) Lookup + pin the contiguous prefix in the source tier so it cannot
        #    be evicted between the lookup and the copy.
        num_tokens = self.lookup(
            tokens,
            search_range=[old_position],
            lookup_id=event_id,
            pin=True,
        )
        try:
            if not num_tokens:
                logger.debug(
                    "move_intra_node: nothing to prefetch from %s.", old_position
                )
                return 0

            keys = self.lookup_pins.get(event_id, {}).get(old_position, [])
            if not keys:
                return 0

            # Budget cap: only promote the leading `max_chunks` chunks. The rest
            # stay pinned until the `finally` below unpins the whole event.
            if max_chunks is not None and max_chunks >= 0:
                keys = keys[:max_chunks]
                if not keys:
                    return 0

            # 2) Pull the prefix from the source tier. For CxlBackend this copies
            #    CXL -> a standalone pinned CPU staging buffer (see
            #    CxlBackend.load_bytes_from_cxl). We own the returned MemoryObjs
            #    and MUST ref_count_down() each of them.
            memory_objs = self.storage_manager.batched_get(
                keys=keys,
                location=old_position,
            )
            if not memory_objs:
                return 0

            local_cpu_backend = self.storage_manager.storage_backends[
                "LocalCPUBackend"
            ]

            promoted = 0  # chunks newly copied into the CPU tier
            contiguous = 0  # leading chunks confirmed present in the CPU tier
            stop = False  # once a gap/failure is seen, keep the region a prefix
            for key, mem_obj in zip(keys, memory_objs, strict=False):
                if mem_obj is None:
                    # Should not happen (keys were just pinned) but guard anyway.
                    stop = True
                    continue
                try:
                    if stop or mem_obj.tensor is None:
                        stop = True
                        continue
                    if local_cpu_backend.contains(key):
                        # Already in the CPU tier; still part of the prefix.
                        contiguous += 1
                        continue
                    # IMPORTANT: the source staging buffer is NOT owned by
                    # LocalCPUBackend's allocator (CxlBackend returns a standalone
                    # pinned tensor). We must copy into a LocalCPUBackend-allocated
                    # object before caching it, otherwise ownership/free is wrong.
                    cached_obj = local_cpu_backend.allocate(
                        mem_obj.get_shape(),
                        mem_obj.get_dtype(),
                        fmt=mem_obj.meta.fmt,
                        # A controller move used for speculation must not evict
                        # demand-resident CPU entries.
                        eviction=False,
                        busy_loop=False,
                    )
                    if cached_obj is None or cached_obj.tensor is None:
                        # CPU tier full and eviction could not free space.
                        stop = True
                        continue
                    cached_obj.tensor.copy_(mem_obj.tensor, non_blocking=False)
                    # submit_put_task takes its own ref (ref_count_up); release our
                    # local ref afterwards so hot_cache holds the only owning ref.
                    if hasattr(local_cpu_backend, "submit_prefetch_put_task"):
                        local_cpu_backend.submit_prefetch_put_task(key, cached_obj)
                    else:
                        local_cpu_backend.submit_put_task(key, cached_obj)
                    cached_obj.ref_count_down()
                    promoted += 1
                    contiguous += 1
                finally:
                    # Release the staging buffer we received from batched_get.
                    mem_obj.ref_count_down()

            # 3) Move (not copy) semantics: drop the promoted prefix from source.
            if not do_copy and contiguous:
                self.storage_manager.batched_remove(
                    keys[:contiguous], locations=[old_position]
                )

            logger.debug(
                "move_intra_node: promoted %d/%d chunks from %s to LocalCPUBackend.",
                promoted,
                len(keys),
                old_position,
            )
            return promoted
        finally:
            # Always release the source-tier pins, even on error.
            # NOTE: the cross-node move() path currently omits this and can leak
            # pins; move_intra_node deliberately does not repeat that bug.
            self.lookup_unpin(event_id)

    def promote_keys_intra_node(
        self,
        keys: List[CacheEngineKey],
        source: str,
        target: str,
        event_id: str,
        *,
        do_copy: bool = True,
        max_chunks: Optional[int] = None,
        allow_eviction: bool = True,
    ) -> PromotionResult:
        """Promote arbitrary cache keys between local storage tiers.

        CXL reads return standalone pinned staging objects. Each successful
        promotion therefore allocates a fresh target-owned object and copies
        into it. Source pins and all staging refs are balanced on every path.
        """
        assert self.storage_manager is not None
        if target != "LocalCPUBackend":
            raise ValueError("only LocalCPUBackend is a supported promotion target")
        backends = self.storage_manager.storage_backends
        if source not in backends or target not in backends:
            raise ValueError(f"promotion backend unavailable: {source} -> {target}")

        selected = list(dict.fromkeys(keys))
        if max_chunks is not None and max_chunks >= 0:
            selected = selected[:max_chunks]
        requested = len(selected)
        if not selected:
            return PromotionResult(0, (), (), (), (), 0)

        source_backend = backends[source]
        target_backend = backends[target]
        already_present: list[CacheEngineKey] = []
        source_missing: list[CacheEngineKey] = []
        failed: list[CacheEngineKey] = []
        promoted: list[CacheEngineKey] = []
        pinned: list[CacheEngineKey] = []
        bytes_promoted = 0

        try:
            # Filter before reading so association speculation does not consume
            # CXL bandwidth for objects that are already CPU-resident.
            for key in selected:
                if target_backend.contains(key):
                    already_present.append(key)
                    continue
                # Pin explicitly instead of contains(..., pin=True): backend
                # demand lookup bookkeeping (keys_in_request) must not retain
                # speculative association keys.
                if source_backend.contains(key) and source_backend.pin(key):
                    pinned.append(key)
                else:
                    source_missing.append(key)

            if not pinned:
                return PromotionResult(
                    requested,
                    (),
                    tuple(already_present),
                    tuple(source_missing),
                    (),
                    0,
                )

            memory_objs = self.storage_manager.batched_get(
                keys=pinned,
                location=source,
            )
            if memory_objs is None:
                failed.extend(pinned)
                memory_objs = []

            handled = 0
            for key, mem_obj in zip(pinned, memory_objs, strict=False):
                handled += 1
                if mem_obj is None:
                    failed.append(key)
                    continue
                try:
                    if mem_obj.tensor is None:
                        failed.append(key)
                        continue
                    if target_backend.contains(key):
                        already_present.append(key)
                        continue
                    cached_obj = target_backend.allocate(
                        mem_obj.get_shape(),
                        mem_obj.get_dtype(),
                        fmt=mem_obj.meta.fmt,
                        eviction=allow_eviction,
                        busy_loop=False,
                    )
                    if cached_obj is None or cached_obj.tensor is None:
                        failed.append(key)
                        continue
                    try:
                        cached_obj.tensor.copy_(
                            mem_obj.tensor, non_blocking=False
                        )
                        if not allow_eviction and hasattr(
                            target_backend, "submit_prefetch_put_task"
                        ):
                            target_backend.submit_prefetch_put_task(key, cached_obj)
                        else:
                            target_backend.submit_put_task(key, cached_obj)
                    finally:
                        cached_obj.ref_count_down()
                    promoted.append(key)
                    bytes_promoted += mem_obj.get_size()
                except Exception:
                    failed.append(key)
                    logger.exception(
                        "Promotion failed for key=%s event_id=%s",
                        getattr(key, "chunk_hash", key),
                        event_id,
                    )
                finally:
                    mem_obj.ref_count_down()
            if handled < len(pinned):
                failed.extend(pinned[handled:])

            if not do_copy and promoted:
                self.storage_manager.batched_remove(promoted, locations=[source])
            return PromotionResult(
                requested,
                tuple(promoted),
                tuple(already_present),
                tuple(source_missing),
                tuple(failed),
                bytes_promoted,
            )
        finally:
            for key in pinned:
                source_backend.unpin(key)

    def offload_to_backend(
        self,
        tokens: Union[torch.Tensor, List[int]],
        source_backend: str = "LocalCPUBackend",
        target_backend: str = "CxlBackend",
        copy: bool = True,
        max_chunks: int = 8,
        operation_id: str = "offload",
        publish_events: bool = True,
        max_bytes: int = 0,
    ) -> BackendOffloadResult:
        """Copy a bounded LocalCPU prefix into CXL and release source pins.

        With ``publish_events=False`` this is the prepare half of a
        controller-coordinated TP offload.  Successful keys are retained in a
        bounded operation table and become visible to Dynamo only when
        :meth:`publish_pending_offload` is called.  ``copy=False`` is rejected
        until a coordinated TP-wide CPU-removal barrier exists; automatic
        offload is copy-only.
        """
        assert self.storage_manager is not None
        if len(tokens) == 0:
            return BackendOffloadResult(False, 0, 0, 0, 0, "empty token prefix")
        if max_chunks <= 0:
            return BackendOffloadResult(False, 0, 0, 0, 0, "max_chunks must be positive")
        if not copy:
            return BackendOffloadResult(
                False,
                0,
                0,
                0,
                0,
                "CPU move is disabled until a TP-wide removal barrier is implemented",
            )
        if source_backend != "LocalCPUBackend" or target_backend != "CxlBackend":
            return BackendOffloadResult(
                False,
                0,
                0,
                0,
                0,
                "automatic offload currently supports only LocalCPUBackend -> CxlBackend",
            )
        backends = self.storage_manager.storage_backends
        if source_backend not in backends or target_backend not in backends:
            return BackendOffloadResult(
                False,
                0,
                0,
                0,
                0,
                f"backend unavailable: {source_backend} -> {target_backend}",
            )

        lookup_id = operation_id
        pinned_keys: list[CacheEngineKey] = []
        memory_objs: list[Optional[MemoryObj]] = []
        committed = 0
        already_present = 0
        failed = 0
        bytes_written = 0
        error: Optional[str] = None
        completed_keys: list[CacheEngineKey] = []
        prepared_keys: list[CacheEngineKey] = []

        try:
            matched_tokens = self.lookup(
                tokens,
                search_range=[source_backend],
                lookup_id=lookup_id,
                pin=True,
            )
            pinned_keys = list(
                self.lookup_pins.get(lookup_id, {}).get(source_backend, [])
            )[:max_chunks]
            if matched_tokens <= 0 or not pinned_keys:
                return BackendOffloadResult(
                    False, 0, 0, 0, 0, "source prefix not found"
                )

            memory_objs = self.storage_manager.batched_get(
                keys=pinned_keys, location=source_backend
            ) or []
            target = backends[target_backend]
            selected_keys = list(pinned_keys)
            if max_bytes > 0:
                selected_keys = []
                selected_bytes = 0
                for index, key in enumerate(pinned_keys):
                    memory_obj = memory_objs[index] if index < len(memory_objs) else None
                    if memory_obj is None or memory_obj.tensor is None:
                        selected_keys.append(key)
                        continue
                    try:
                        object_bytes = int(memory_obj.get_physical_size())
                    except Exception:
                        object_bytes = int(memory_obj.get_size())
                    if selected_keys and selected_bytes + object_bytes > max_bytes:
                        break
                    if not selected_keys and object_bytes > max_bytes:
                        break
                    selected_keys.append(key)
                    selected_bytes += object_bytes
                if not selected_keys:
                    for memory_obj in memory_objs:
                        if memory_obj is not None:
                            memory_obj.ref_count_down()
                    return BackendOffloadResult(
                        False,
                        0,
                        0,
                        0,
                        0,
                        f"max_bytes={max_bytes} is smaller than one CXL chunk",
                    )
            selected_key_set = set(selected_keys)
            background_slot = getattr(target, "background_offload_slot", None)
            for index, key in enumerate(pinned_keys):
                memory_obj = memory_objs[index] if index < len(memory_objs) else None
                if key not in selected_key_set:
                    if memory_obj is not None:
                        memory_obj.ref_count_down()
                    continue
                if memory_obj is None or memory_obj.tensor is None:
                    failed += 1
                    if memory_obj is not None:
                        memory_obj.ref_count_down()
                    continue
                try:
                    # Offload owns at most one CXL chunk at a time.  The
                    # governor never waits for serving work; losing this
                    # admission race defers the remainder to the next
                    # planner round and releases every outstanding CPU ref.
                    slot = (
                        background_slot()
                        if background_slot is not None
                        else nullcontext(True)
                    )
                    with slot as admitted:
                        if not admitted:
                            remaining = sum(
                                1
                                for remaining_key in pinned_keys[index:]
                                if remaining_key in selected_key_set
                            )
                            failed += remaining
                            error = (
                                "CXL offload deferred: serving-priority resource "
                                "is busy"
                            )
                            logger.debug(
                                "Deferring CPU-to-CXL offload operation=%s at "
                                "chunk=%s; remaining_chunks=%d",
                                operation_id,
                                getattr(key, "chunk_hash", key),
                                remaining,
                            )
                            for remaining_index in range(index + 1, len(pinned_keys)):
                                remaining_obj = (
                                    memory_objs[remaining_index]
                                    if remaining_index < len(memory_objs)
                                    else None
                                )
                                if remaining_obj is not None:
                                    remaining_obj.ref_count_down()
                            break
                        result = target.submit_put_task(
                            key, memory_obj, publish_events=publish_events
                        )
                    status = getattr(getattr(result, "status", None), "value", result)
                    if status in ("success", "stored"):
                        committed += 1
                        bytes_written += int(
                            getattr(result, "bytes_written", memory_obj.get_size()) or 0
                        )
                        completed_keys.append(key)
                        prepared_keys.append(key)
                    elif status in ("already_present",):
                        already_present += 1
                        completed_keys.append(key)
                        if not publish_events and getattr(
                            result, "needs_publish", False
                        ):
                            prepared_keys.append(key)
                    elif status in ("deferred",):
                        failed += 1
                        error = (
                            getattr(result, "detail", None)
                            or "CXL offload deferred: serving-priority resource is busy"
                        )
                    else:
                        failed += 1
                        error = getattr(result, "detail", None) or str(status)
                except Exception as exc:
                    failed += 1
                    error = str(exc)
                    logger.exception(
                        "CPU-to-CXL offload failed for key=%s operation=%s",
                        getattr(key, "chunk_hash", key),
                        operation_id,
                    )
                finally:
                    memory_obj.ref_count_down()

            success = failed == 0 and len(completed_keys) == len(selected_keys)
            if not publish_events and (success or prepared_keys):
                self._remember_pending_offload(
                    operation_id,
                    target_backend,
                    prepared_keys,
                )
                logger.info(
                    "CXL offload prepared: operation=%s chunks=%d bytes=%d; "
                    "waiting for CXL_READY commit",
                    operation_id,
                    len(completed_keys),
                    bytes_written,
                )
            return BackendOffloadResult(
                success,
                committed,
                already_present,
                failed,
                bytes_written,
                error,
            )
        except Exception as exc:
            error = str(exc)
            logger.exception("CPU-to-backend offload failed: operation=%s", operation_id)
            # An exception outside the per-key block can still occur after a
            # previous key has created a CXL object.  Preserve those keys in
            # the operation table so a controller-level abort can destroy
            # them instead of leaving invisible objects behind.
            if not publish_events and prepared_keys:
                self._remember_pending_offload(
                    operation_id,
                    target_backend,
                    prepared_keys,
                )
            return BackendOffloadResult(
                False,
                committed,
                already_present,
                failed + 1,
                bytes_written,
                error,
            )
        finally:
            # lookup_unpin releases every key pinned by lookup, including keys
            # beyond max_chunks that were intentionally not copied.
            self.lookup_unpin(lookup_id)

    def _remember_pending_offload(
        self,
        operation_id: str,
        target_backend: str,
        prepared_keys: Sequence[CacheEngineKey],
    ) -> None:
        """Record the new CXL keys that an explicit commit must publish."""
        new_keys = tuple(dict.fromkeys(prepared_keys))
        evicted: Optional[tuple[str, tuple[CacheEngineKey, ...]]] = None
        cleanup_replacement: Optional[
            tuple[str, tuple[CacheEngineKey, ...]]
        ] = None
        with self._pending_offloads_lock:
            if (
                operation_id not in self._pending_offloads
                and len(self._pending_offloads) >= 128
            ):
                evicted_operation = next(iter(self._pending_offloads))
                evicted = self._pending_offloads.pop(evicted_operation, None)
                if evicted is not None:
                    logger.warning(
                        "Aborting stale prepared offload operation=%s "
                        "to keep the pending table bounded",
                        evicted_operation,
                    )
            previous = self._pending_offloads.get(operation_id)
            if previous is None:
                pending = (target_backend, new_keys)
            elif previous[0] == target_backend:
                # Reusing an operation ID must not discard keys from the first
                # attempt.  A retry may observe those keys as ALREADY_PRESENT;
                # retaining the union keeps a later abort able to remove them.
                pending = (
                    target_backend,
                    tuple(dict.fromkeys((*previous[1], *new_keys))),
                )
            else:
                # This indicates a caller bug, but silently replacing the old
                # record would leak its objects.  Keep the old transaction and
                # clean the newly supplied keys immediately.
                logger.error(
                    "Cannot reuse offload operation=%s across backends %s -> %s",
                    operation_id,
                    previous[0],
                    target_backend,
                )
                pending = previous
                cleanup_replacement = (target_backend, new_keys)
            self._pending_offloads[operation_id] = pending

        if evicted is not None:
            self._cleanup_pending_offload(evicted)
        if cleanup_replacement is not None:
            self._cleanup_pending_offload(cleanup_replacement)

    def _cleanup_pending_offload(
        self,
        pending: tuple[str, tuple[CacheEngineKey, ...]],
    ) -> BackendOffloadResult:
        """Destroy the physical objects belonging to one prepared operation."""
        target_backend, keys = pending
        if not keys:
            return BackendOffloadResult(True, 0, 0, 0, 0, None)
        if self.storage_manager is None:
            return BackendOffloadResult(
                False, 0, 0, len(keys), 0, "storage manager unavailable"
            )
        target = self.storage_manager.storage_backends.get(target_backend)
        if target is None:
            return BackendOffloadResult(
                False,
                0,
                0,
                len(keys),
                0,
                f"backend unavailable: {target_backend}",
            )

        removed = 0
        errors: list[str] = []
        native_exists = getattr(target, "native_exists", None)
        for key in keys:
            try:
                removed_key = target.remove(key, force=True)
                if not removed_key:
                    # A backend may report False for an already-missing key.
                    # Treat a confirmed miss as clean,
                    # but retry once when a native object is still present;
                    # this matters for a CXL destroy that lost a lock race.
                    present = target.contains(key)
                    if not present and callable(native_exists):
                        present = bool(native_exists(key))
                    if present:
                        removed_key = target.remove(key, force=True)
                        if not removed_key:
                            present = target.contains(key)
                            if not present and callable(native_exists):
                                present = bool(native_exists(key))
                    if not present:
                        removed_key = True
                if removed_key:
                    removed += 1
                else:
                    errors.append(
                        f"{getattr(key, 'chunk_hash', key)}: object remains present"
                    )
            except Exception as exc:
                errors.append(f"{getattr(key, 'chunk_hash', key)}: {exc}")
                logger.exception(
                    "Failed to abort prepared offload key=%s",
                    getattr(key, "chunk_hash", key),
                )
        failed = len(keys) - removed
        return BackendOffloadResult(
            failed == 0,
            removed,
            0,
            failed,
            0,
            "; ".join(errors) if errors else None,
        )

    def abort_pending_offload(self, operation_id: str) -> BackendOffloadResult:
        """Abort a prepared or partially committed offload."""
        with self._pending_offloads_lock:
            pending = self._pending_offloads.pop(operation_id, None)
            committed = getattr(self, "_committed_offloads", {}).pop(
                operation_id, None
            )
            if pending is None:
                pending = committed
            elif committed is not None and pending[0] == committed[0]:
                # Normally an operation is either pending or committed.  If a
                # retried operation ID created both records, abort both sets
                # instead of dropping the older, already-published keys.
                pending = (
                    pending[0],
                    tuple(dict.fromkeys((*pending[1], *committed[1]))),
                )
        # Abort is idempotent: a prepare that produced no new keys, or a
        # duplicate cleanup message, is already in the desired final state.
        if pending is None:
            return BackendOffloadResult(True, 0, 0, 0, 0, None)
        result = self._cleanup_pending_offload(pending)
        if not result.success:
            # Keep failed cleanup retryable.  The controller may retry the
            # abort, and engine shutdown will make one more best-effort pass.
            with self._pending_offloads_lock:
                if (
                    operation_id not in self._pending_offloads
                    and operation_id
                    not in getattr(self, "_committed_offloads", {})
                ):
                    self._pending_offloads[operation_id] = pending
        return result

    def finalize_pending_offload(self, operation_id: str) -> BackendOffloadResult:
        """Release rollback bookkeeping after every TP rank committed."""
        with self._pending_offloads_lock:
            committed = getattr(self, "_committed_offloads", {}).pop(
                operation_id, None
            )
            pending = self._pending_offloads.get(operation_id)
        if committed is not None:
            return BackendOffloadResult(
                True, len(committed[1]), 0, 0, 0, None
            )
        if pending is not None:
            return BackendOffloadResult(
                False, 0, 0, len(pending[1]), 0, "offload has not committed"
            )
        # Finalization is bookkeeping-only and idempotent.  This also makes a
        # duplicate finalize safe after a worker restart or retry.
        return BackendOffloadResult(True, 0, 0, 0, 0, None)

    def abort_all_pending_offloads(self) -> None:
        """Best-effort cleanup for prepared objects during engine shutdown."""
        with self._pending_offloads_lock:
            operation_ids = list(self._pending_offloads)
        for operation_id in operation_ids:
            result = self.abort_pending_offload(operation_id)
            if not result.success:
                logger.error(
                    "Failed to abort prepared offload during shutdown: operation=%s "
                    "removed=%d failed=%d error=%s",
                    operation_id,
                    result.committed_chunks,
                    result.failed_chunks,
                    result.error,
                )

    def publish_pending_offload(self, operation_id: str) -> BackendOffloadResult:
        """Publish one successfully prepared offload as ``CXL_READY``.

        Keep the operation in the pending table until every key is published.
        This makes a failed commit retryable and lets the controller explicitly
        abort the still-unpublished physical objects.
        """
        with self._pending_offloads_lock:
            pending = self._pending_offloads.get(operation_id)
        if pending is None:
            return BackendOffloadResult(
                False, 0, 0, 0, 0, f"no prepared offload for operation {operation_id}"
            )
        target_backend, keys = pending
        if self.storage_manager is None:
            return BackendOffloadResult(False, 0, 0, 0, 0, "storage manager unavailable")
        target = self.storage_manager.storage_backends.get(target_backend)
        if target is None:
            return BackendOffloadResult(
                False, 0, 0, len(keys), 0, f"backend unavailable: {target_backend}"
            )
        try:
            published = target.publish_keys(keys)
            success = len(published) == len(keys)
            if not success:
                return BackendOffloadResult(
                    False,
                    len(published),
                    0,
                    len(keys) - len(published),
                    0,
                    "one or more prepared CXL keys disappeared before commit",
                )
            with self._pending_offloads_lock:
                # Keep the keys rollback-capable until the coordinator has
                # observed commit success on every TP rank.  This closes the
                # window where one rank publishes and another rank fails.
                if self._pending_offloads.get(operation_id) == pending:
                    self._pending_offloads.pop(operation_id, None)
                    committed_offloads = getattr(
                        self, "_committed_offloads", None
                    )
                    if committed_offloads is None:
                        committed_offloads = {}
                        self._committed_offloads = committed_offloads
                    if len(committed_offloads) >= 128:
                        evicted_operation = next(iter(committed_offloads))
                        committed_offloads.pop(evicted_operation, None)
                        logger.warning(
                            "Dropping stale committed-offload bookkeeping=%s",
                            evicted_operation,
                        )
                    committed_offloads[operation_id] = pending
            logger.info(
                "CXL_READY published: operation=%s chunks=%d",
                operation_id,
                len(published),
            )
            return BackendOffloadResult(True, len(published), 0, 0, 0, None)
        except Exception as exc:
            logger.exception("CXL_READY publication failed: operation=%s", operation_id)
            return BackendOffloadResult(False, 0, 0, len(keys), 0, str(exc))

    # TODO(Jiayi): Add layerwise support.
    @_lmcache_nvtx_annotate
    def async_lookup_and_prefetch(
        self,
        lookup_id: str,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        search_range: Optional[List[str]] = None,
        pin: bool = False,
        request_configs: Optional[dict] = None,
    ) -> None:
        """
        An async version of lookup + prefetch.

        There are three categories of backends:
        (1) sync lookup + sync retrieval (e.g., cpu)
        (2) sync lookup + async retrieval (e.g., disk)
        (3) async lookup + async retrieval (e.g., p2p)
        """
        assert self.storage_manager is not None

        keys: list[CacheEngineKey] = []
        cum_chunk_lengths = [0]

        # TODO(Jiayi): make token database able to return list.
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens,
            hashes=hashes,
            offsets=offsets,
            request_configs=request_configs,
        ):
            assert isinstance(key, CacheEngineKey)
            keys.append(key)
            cum_chunk_lengths.append(end)

        asyncio.run_coroutine_threadsafe(
            self.storage_manager.async_lookup_and_prefetch(
                lookup_id, keys, cum_chunk_lengths, search_range, pin
            ),
            self.storage_manager.loop,
        )

    def cleanup_memory_objs(self, lookup_id: str) -> None:
        """
        Cleanup memory objects allocated during prefetch for an aborted lookup.

        Called by the scheduler when it determines that an aborted lookup
        has finished its prefetch tasks.
        """
        try:
            # Get the completed future from event_manager
            if (
                self.event_manager.get_event_status(EventType.LOADING, lookup_id)
                != EventStatus.DONE
            ):
                logger.debug(
                    "No completed event found for lookup_id=%s to clean up.", lookup_id
                )
                return
            future = self.event_manager.pop_event(EventType.LOADING, lookup_id)

            # Get memory objects from the future result
            memory_objs = future.result()
            # Flatten nested lists (each backend returns a list of chunks)
            memory_objs_flat = [mm for m in memory_objs for mm in m]

            # Release each memory object
            for memory_obj in memory_objs_flat:
                try:
                    logger.debug("Releasing memory object for lookup_id=%s", lookup_id)
                    memory_obj.ref_count_down()
                except Exception as e:
                    logger.error(f"Error releasing memory object: {e}")
        except Exception as e:
            logger.error(
                f"Error during cleanup_memory_objs for lookup_id={lookup_id}: {e}"
            )

    # TODO(Jiayi): Need to handle the case where `tokens=None`.
    # In this case, we compress all tokens.
    # TODO(Jiayi): support other compression methods.
    @_lmcache_nvtx_annotate
    def compress(
        self,
        tokens: Union[torch.Tensor, List[int]],
        method: str,
        location: str,
        event_id: str,
    ) -> int:
        assert self.storage_manager is not None
        if method not in ["cachegen"]:
            logger.warning(f"Unsupported compression method: {method}.")
            return 0

        # First Party
        from lmcache.v1.storage_backend.naive_serde import CreateSerde

        serializer, _ = CreateSerde(method, self.metadata, self.config)

        num_tokens = self.lookup(
            tokens,
            search_range=[location],
            lookup_id=event_id,
            pin=True,
        )

        if not num_tokens:
            logger.debug("Move is not performed as there are no tokens to move.")
            return 0

        block_mapping = self.lookup_pins[event_id]
        assert len(block_mapping) == 1
        keys = block_mapping[location]

        memory_objs = self.storage_manager.batched_get(
            keys=keys,
            location=location,
        )
        assert memory_objs is not None, (
            "LMCacheEngine.compress: Failed to get memory objects to compress"
        )

        compressed_memory_objs = []
        for memory_obj in memory_objs:
            assert memory_obj is not None
            compressed_memory_obj = serializer.serialize(memory_obj)
            memory_obj.unpin()
            compressed_memory_objs.append(compressed_memory_obj)

        self.storage_manager.batched_remove(keys, locations=[location])

        self.storage_manager.batched_put(
            keys=keys,
            memory_objs=compressed_memory_objs,
            location=location,
        )

        return num_tokens

    @_lmcache_nvtx_annotate
    def decompress(
        self,
        tokens: Union[torch.Tensor, List[int]],
        method: str,
        location: str,
        event_id: str,
    ) -> int:
        assert self.storage_manager is not None
        if method not in ["cachegen"]:
            logger.warning(f"Unsupported decompression method: {method}.")
            return 0

        # First Party
        from lmcache.v1.storage_backend.naive_serde import CreateSerde

        _, deserializer = CreateSerde(method, self.metadata, self.config)

        num_tokens = self.lookup(
            tokens,
            search_range=[location],
            lookup_id=event_id,
            pin=True,
        )

        if not num_tokens:
            logger.debug("there are no tokens to decompress.")
            return 0

        block_mapping = self.lookup_pins[event_id]
        assert len(block_mapping) == 1
        keys = block_mapping[location]

        compressed_memory_objs = self.storage_manager.batched_get(
            keys=keys,
            location=location,
        )

        assert compressed_memory_objs is not None, (
            "LMCacheEngine.compress: Failed to get compressed "
            "memory objects to decompress"
        )

        memory_objs = []
        for compressed_memory_obj in compressed_memory_objs:
            assert compressed_memory_obj is not None
            memory_obj = deserializer.deserialize(compressed_memory_obj)
            compressed_memory_obj.unpin()
            memory_objs.append(memory_obj)

        self.storage_manager.batched_remove(keys, locations=[location])

        self.storage_manager.batched_put(
            keys=keys,
            memory_objs=memory_objs,
            location=location,
        )

        return num_tokens

    @_lmcache_nvtx_annotate
    def lookup_unpin(self, lookup_id: str) -> None:
        if lookup_id in self.lookup_pins:
            assert self.storage_manager is not None
            for location, keys in self.lookup_pins.pop(lookup_id).items():
                self.storage_manager.batched_unpin(keys, [location])

    @_lmcache_nvtx_annotate
    def clear(
        self,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        locations: Optional[List[str]] = None,
        request_configs: Optional[dict] = None,
    ) -> int:
        # TODO: need to clear by request_configs
        if self.save_only_first_rank:
            if self.metadata.is_first_rank():
                num_removed = self._clear(tokens, locations, request_configs)
                return num_removed
            else:
                return 0
        return self._clear(tokens, locations, request_configs)

    @_lmcache_nvtx_annotate
    def get_kv_events(self) -> Iterable[CacheStoreEvent | CacheRemoveEvent]:
        if self.kv_events_enabled and (events := self.kv_events):
            self.kv_events = []
            return events
        return []

    def _emit_prefetch_tier_event(
        self,
        keys: Iterable[CacheEngineKey],
        *,
        medium: str = "CPU",
    ) -> None:
        """Publish speculative tier transitions to the Dynamo bridge."""
        if not self.kv_events_enabled:
            return
        keys = list(keys)
        if not keys:
            return
        for key in keys:
            meta = self._kv_store_meta_by_hash.get(int(key.chunk_hash))
            parent, token_ids, block_size, lora_id = meta or (
                None,
                [],
                self.config.chunk_size,
                None,
            )
            self.kv_events.append(
                CacheStoreEvent(
                    block_hashes=[int(key.chunk_hash)],
                    parent_block_hash=parent,
                    token_ids=list(token_ids),
                    block_size=int(block_size or self.config.chunk_size),
                    lora_id=lora_id,
                    medium=medium,
                    origin="PREFETCH",
                    worker_id=int(self.metadata.worker_id),
                )
            )

    def _register_backend_kv_event_sinks(self) -> None:
        assert self.storage_manager is not None
        for backend in self.storage_manager.storage_backends.values():
            setter = getattr(backend, "set_kv_event_sink", None)
            if callable(setter):
                setter(self._on_backend_kv_event)

    def _compute_event_block_token_sizes(
        self, num_blocks: int, block_size: int, token_ids_len: int
    ) -> List[int]:
        if num_blocks == 0:
            return []
        if block_size == 0:
            sizes = [0 for _ in range(num_blocks)]
            sizes[0] = token_ids_len
            return sizes
        sizes: List[int] = [0 for _ in range(num_blocks)]
        remaining = token_ids_len
        for idx in range(num_blocks):
            if remaining <= 0:
                sizes[idx] = 0
                continue
            take = min(remaining, block_size)
            sizes[idx] = take
            remaining -= take
        return sizes

    def _cache_store_event_metadata(self, event: CacheStoreEvent) -> None:
        if not event.block_hashes:
            return
        block_sizes = self._compute_event_block_token_sizes(
            num_blocks=len(event.block_hashes),
            block_size=event.block_size,
            token_ids_len=len(event.token_ids),
        )
        token_offset = 0
        parent_hash = event.parent_block_hash
        for idx, block_hash in enumerate(event.block_hashes):
            take = block_sizes[idx] if idx < len(block_sizes) else 0
            end = token_offset + take
            block_tokens = event.token_ids[token_offset:end]
            self._kv_store_meta_by_hash[int(block_hash)] = (
                parent_hash,
                block_tokens,
                event.block_size,
                event.lora_id,
            )
            token_offset = end
            parent_hash = int(block_hash)

    def _on_backend_kv_event(self, event: CacheStoreEvent | CacheRemoveEvent) -> None:
        if not self.kv_events_enabled:
            return
        if isinstance(event, CacheStoreEvent) and (event.medium or "").upper() == "CXL":
            for block_hash in event.block_hashes:
                meta = self._kv_store_meta_by_hash.get(int(block_hash))
                if meta is None:
                    logger.warning(
                        "CXL_ENRICH_MISS: block_hash=%d had no CPU metadata; "
                        "falling back to empty token_ids (this will cause tokens_hash mismatch in indexer). "
                        "meta_map_size=%d",
                        int(block_hash),
                        len(self._kv_store_meta_by_hash),
                    )
                    parent = event.parent_block_hash
                    token_ids = event.token_ids
                    block_size = event.block_size
                    lora_id = event.lora_id
                else:
                    parent, token_ids, block_size, lora_id = meta
                cxl_event = CacheStoreEvent(
                    block_hashes=[int(block_hash)],
                    parent_block_hash=parent,
                    token_ids=list(token_ids),
                    block_size=int(block_size or event.block_size),
                    lora_id=lora_id,
                    medium="CXL",
                )
                self.kv_events.append(cxl_event)
                logger.info(
                    "Queued KV store event: req_id=%s, num_blocks=%d, medium=%s, "
                    "block_size=%d, parent_block_hash=%s",
                    None,
                    len(cxl_event.block_hashes),
                    cxl_event.medium or "unknown",
                    cxl_event.block_size,
                    cxl_event.parent_block_hash,
                )
            return
        self.kv_events.append(event)

    def _clear(
        self,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        locations: Optional[List[str]] = None,
        request_configs: Optional[dict] = None,
    ) -> int:
        assert self.storage_manager is not None
        assert isinstance(self.storage_manager, StorageManager)
        # Clear all caches if tokens is None
        if tokens is None or len(tokens) == 0:
            num_cleared = self.storage_manager.clear(locations)
            return num_cleared

        num_removed = 0
        # Only remove the caches for the given tokens
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens, request_configs=request_configs
        ):
            assert isinstance(key, CacheEngineKey)
            removed = self.storage_manager.remove(key, locations)
            num_removed += removed
        return num_removed

    @_lmcache_nvtx_annotate
    def health(
        self,
    ) -> int:
        """
        Check the health of the cache engine.
        return: 0 if healthy, otherwise the error code
        """
        assert self.storage_manager is not None
        return 0 if self.storage_manager.memcheck() else -1

    def close(self) -> None:
        """Close the cache engine and free all the resources"""
        logger.info("Closing LMCacheEngine...")

        if self._cxl_prefetch_executor is not None:
            try:
                self._cxl_prefetch_executor.close()
            except Exception:
                logger.exception("Failed to close global prefetch executor.")

        try:
            self.abort_all_pending_offloads()
        except Exception:
            logger.exception("Failed to abort prepared CPU-to-CXL offloads on shutdown.")

        if self.lmcache_worker is not None:
            try:
                logger.info("Closing lmcache_worker...")
                self.lmcache_worker.close()
                logger.info("lmcache_worker closed successfully")
            except Exception as e:
                logger.error(f"Error closing lmcache_worker: {e}")

        try:
            logger.info("Closing storage_manager...")
            if self.storage_manager is not None:
                self.storage_manager.close()
            logger.info("storage_manager closed successfully")
        except Exception as e:
            logger.error(f"Error closing storage_manager: {e}")

        logger.info("LMCacheEngine closed.")

    def _async_process_tokens_internal(
        self,
        tokens,
        mask,
        ret_mask,
        **kwargs,
    ) -> ProcessTokensInternalResult:
        """
        This function is used to get the memory objects from the event manager.

        Args:
            tokens: Input tokens to process
            mask: Mask indicating valid token positions
            ret_mask: Output mask updated with cache hit positions
            **kwargs: Additional keyword arguments
        """
        assert "req_id" in kwargs, "req_id is required for async loading"
        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)

        tot_kv_size = 0
        chunks: List[ProcessedChunk] = []
        future = self.event_manager.pop_event(EventType.LOADING, kwargs["req_id"])

        # As mentioned in async_lookup_and_prefetch(), the future.result()
        # is key data pair for each chunk in each tier. So extract the key
        # and memory object pairs to memory_obj_map
        try:
            keyed_memory_objs = future.result()
            memory_obj_map: dict[CacheEngineKey, MemoryObj] = {}
        except Exception as e:
            logger.error(f"Error popping event for request {kwargs['req_id']}: {e}")
            return [], 0

        for backend_results in keyed_memory_objs:
            for key, memory_obj in backend_results:
                memory_obj_map[key] = memory_obj

        # TODO(Jiayi): hashing inside `process_tokens` can be skipped.
        used_keys: set[CacheEngineKey] = set()
        chunk_infos = list(
            self.token_database.process_tokens(
                tokens=tokens,
                mask=mask,
                request_configs=request_configs,
            )
        )
        for start, end, key in chunk_infos:
            assert isinstance(key, CacheEngineKey)
            memory_obj = memory_obj_map.get(key)
            if memory_obj is None:
                # returned chunks are expected to be contiguous.
                # break at the first missing chunk.
                break
            chunks.append((key, memory_obj, start, end))
            tot_kv_size += memory_obj.get_size()
            ret_mask[start:end] = True
            used_keys.add(key)

        # NOTE: free the memory objects that are not hit.
        for key, mem_obj in memory_obj_map.items():
            if key not in used_keys:
                mem_obj.ref_count_down()

        return chunks, tot_kv_size

    def _process_tokens_internal(
        self,
        tokens,
        mask,
        ret_mask,
        **kwargs,
    ) -> ProcessTokensInternalResult:
        """Process tokens and populate the reordered lists.

        This function is used to process tokens and populate the reordered lists.

        Args:
            tokens: Input tokens to process
            mask: Mask indicating valid token positions
            ret_mask: Output mask updated with cache hit positions
            **kwargs: Additional keyword arguments
        """
        assert self.storage_manager is not None

        tot_kv_size = 0
        reordered_chunks: List[ProcessedChunk] = []
        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)

        # In some scenarios, lookup is called first, and then the original tokens
        # is sliced based on the lookup result. In these scenarios, the tokens
        # passed in must exist in LMCache, and we can set skip_contains_check to True.
        # When skip_contains_check is True and there is only one backend, the `contains`
        # call can be skipped.
        skip_contains_check = (
            kwargs["skip_contains_check"] if "skip_contains_check" in kwargs else False
        )
        chunk_infos = []
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens,
            mask=mask,
            request_configs=request_configs,
        ):
            assert isinstance(key, CacheEngineKey)
            chunk_infos.append((key, start, end))

        # block_mapping: location -> [(CacheEngineKey, start, end)]
        if (
            skip_contains_check
            and len(self.storage_manager.non_allocator_backends) == 1
        ):
            location = self.storage_manager.non_allocator_backends[0]
            block_mapping = {location: chunk_infos}
        else:
            block_mapping = self.storage_manager.get_block_mapping(chunk_infos)

        last_failed_block_start = None
        backend_hit_tokens = kwargs.get("backend_hit_tokens")
        hit_segments: list[tuple[str, int, int]] = []
        # Resolve CxlBackend once per call so per-chunk attribution stays cheap.
        cxl_backend_for_attr = (
            self.storage_manager.storage_backends.get("CxlBackend")
            if self.storage_manager is not None
            else None
        )
        # Build a per-call chain parent map so we can backfill
        # _kv_store_meta_by_hash for chunks that get auto-cached into
        # LocalCPUBackend via storage_manager.get_blocking (the CXL/Disk -> CPU
        # back-fill path). Without this, those chunks have no metadata, and
        # when they later get evicted and spilled back to CXL, the CXL
        # CacheStoreEvent gets empty token_ids -> wrong tokens_hash -> indexer
        # block_hash mismatch warnings.
        #
        # chunk_infos is in token order (from token_database.process_tokens)
        # so we can derive each chunk's chain-parent as the previous chunk's
        # chunk_hash. Only fill entries that don't already exist to avoid
        # clobbering store-time records.
        meta_record_enabled = (
            self.kv_events_enabled
            and tokens is not None
            and hasattr(self, "_kv_store_meta_by_hash")
        )
        if meta_record_enabled:
            chain_parent_by_key: dict[int, Optional[int]] = {}
            prev_chunk_hash: Optional[int] = None
            for ci_key, _, _ in chunk_infos:
                chain_parent_by_key[int(ci_key.chunk_hash)] = prev_chunk_hash
                prev_chunk_hash = int(ci_key.chunk_hash)

        for location, blocks in block_mapping.items():
            keys = [key for key, _, _ in blocks]
            memory_objs = self.storage_manager.batched_get(
                keys=keys,
                location=location,
                request_id=kwargs.get("req_id"),
            )
            assert memory_objs is not None, (
                "Failed to get memory objects from storage backend"
            )

            is_cxl_location = str(location) == "CxlBackend"
            for (key, start, end), memory_obj in zip(blocks, memory_objs, strict=False):
                if memory_obj is None:
                    logger.warning(
                        "The cache block is in the storage, but it can't be retrieved "
                        "(location=%s, key=%s, start=%s, end=%s)",
                        location,
                        getattr(key, "chunk_hash", key),
                        start,
                        end,
                    )
                    if (
                        last_failed_block_start is None
                        or last_failed_block_start < start
                    ):
                        last_failed_block_start = start
                    break
                reordered_chunks.append((key, memory_obj, start, end))
                # Attribute CXL hits between local- and shared-origin: a "shared"
                # hit is a key that was first discovered via SHM metadata without
                # a local put (i.e. another node created it). See CxlBackend.
                attr_location = str(location)
                if is_cxl_location and cxl_backend_for_attr is not None:
                    if cxl_backend_for_attr.is_shared_origin(key):
                        attr_location = "CxlBackend:shared"
                hit_segments.append((attr_location, int(start), int(end)))
                tot_kv_size += memory_obj.get_size()
                ret_mask[start:end] = True

                # Backfill _kv_store_meta_by_hash for this chunk if it wasn't
                # populated by a prior store(). The CXL/Disk -> CPU back-fill
                # path in storage_manager.get_blocking caches chunks into
                # LocalCPUBackend without going through cache_engine.store(),
                # so without this backfill, downstream CXL spill events would
                # be emitted with empty token_ids and cause hash mismatches in
                # the router indexer.
                if (
                    meta_record_enabled
                    and int(key.chunk_hash) not in self._kv_store_meta_by_hash
                ):
                    try:
                        chunk_token_ids = convert_tokens_to_list(tokens, start, end)
                    except Exception:
                        chunk_token_ids = []
                    if chunk_token_ids:
                        self._kv_store_meta_by_hash[int(key.chunk_hash)] = (
                            chain_parent_by_key.get(int(key.chunk_hash)),
                            chunk_token_ids,
                            int(end - start),
                            None,
                        )

        if last_failed_block_start is not None:
            ret_mask[last_failed_block_start:] = False

            reordered_chunks = [
                (key, memory_obj, start, end)
                for key, memory_obj, start, end in reordered_chunks
                if end < last_failed_block_start
            ]
            hit_segments = [
                (loc, start, end)
                for loc, start, end in hit_segments
                if end < last_failed_block_start
            ]
        if isinstance(backend_hit_tokens, dict):
            for location, start, end in hit_segments:
                backend_hit_tokens[location] = int(backend_hit_tokens.get(location, 0)) + int(
                    end - start
                )
        return reordered_chunks, tot_kv_size

    def _broadcast_or_receive_memory_objs(
        self,
        reordered_chunks,
        ret_mask,
    ):
        """
        Handles broadcasting or receiving memory objects in a distributed environment.

        This function implements the communication logic where:
        - The first rank (coordinator) broadcasts memory objects and metadata to others
        - Other ranks receive and reconstruct the memory objects

        Parameters:
        reordered_chunks: List of tuples containing [key, memory object, start, end]
        ret_mask: Boolean mask indicating which positions have been processed

        Side Effects:
        - On first rank:
          * Broadcasts chunk count and each chunk's combined metadata
          * Broadcasts tensor data
        - On other ranks:
          * Receives chunk data and populates reordered_chunks
          * Updates ret_mask to mark received positions as True
        """
        if self.metadata.is_first_rank():
            # Broadcast total chunk count
            chunk_count = len(reordered_chunks)
            self.broadcast_object_fn(chunk_count, self.metadata.first_rank)

            # Broadcast each chunk's data
            for key, memory_obj, start, end in reordered_chunks:
                # Combine (start, end) and metadata into single broadcast
                metadata_dict = memory_obj.metadata.to_dict()
                combined_metadata = (start, end, metadata_dict)
                self.broadcast_object_fn(combined_metadata, self.metadata.first_rank)

                # Broadcast tensor data
                tensor_to_broadcast = memory_obj.tensor.to(
                    f"cuda:{self.metadata.worker_id}"
                )
                self.broadcast_fn(tensor_to_broadcast, self.metadata.first_rank)
        else:
            # Receive total chunk count
            chunk_count = self.broadcast_object_fn(None, self.metadata.first_rank)
            if chunk_count is None:
                logger.warning(
                    f"rank={self.metadata.worker_id} received None chunk_count"
                )
                return

            # Fill reordered_chunks with received data
            for _ in range(chunk_count):
                # Receive combined metadata (start, end, metadata_dict)
                combined_metadata = self.broadcast_object_fn(
                    None, self.metadata.first_rank
                )
                if combined_metadata is None:
                    logger.warning(
                        f"rank={self.metadata.worker_id} "
                        "received None combined_metadata"
                    )
                    break
                start, end, metadata_dict = combined_metadata
                ret_mask[start:end] = True

                # Create tensor and receive data
                metadata = MemoryObjMetadata.from_dict(metadata_dict)
                local_rank = self.metadata.worker_id % torch.cuda.device_count()
                tensor = torch.empty(
                    metadata.shape,
                    dtype=metadata.dtype,
                    device=f"cuda:{local_rank}",
                )
                self.broadcast_fn(tensor, self.metadata.first_rank)

                # Create temporary memory object (key not needed for other ranks)
                memory_obj = TensorMemoryObj(
                    raw_data=tensor, metadata=metadata, parent_allocator=None
                )
                reordered_chunks.append((None, memory_obj, start, end))

    def _is_passive(self):
        """
        A 'passive' CacheEngine means that the node itself will not store/retrieve
        the data directly, but from the "active" worker (i.e., rank 0 in MLA)
        """
        return self.save_only_first_rank and not self.metadata.is_first_rank()


class LMCacheEngineBuilder:
    _instances: Dict[str, LMCacheEngine] = {}
    _cfgs: Dict[str, LMCacheEngineConfig] = {}
    _metadatas: Dict[str, LMCacheEngineMetadata] = {}
    _stat_loggers: Dict[str, LMCacheStatsLogger] = {}

    # TODO(Jiayi): Please remove this helper function in the future.
    # Currently, it's only used for testing.
    @staticmethod
    def _Create_memory_allocator(
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
        numa_mapping: Optional[NUMAMapping] = None,
    ) -> MemoryAllocatorInterface:
        # NOTE: should remove this function after fixing the unit tests:
        # raise RuntimeError("_Create_memory_allocator is deprecated!")
        extra_config = config.extra_config
        enable_nixl_storage = extra_config is not None and extra_config.get(
            "enable_nixl_storage"
        )

        if enable_nixl_storage:
            # TODO(Jiayi): weird to import from transfer utils.
            # First Party
            from lmcache.v1.transfer_channel.transfer_utils import (
                get_correct_device,
            )

            corrected_device = get_correct_device(
                config.nixl_buffer_device,
                metadata.worker_id,
            )

            buffer = torch.empty(
                config.nixl_buffer_size,
                dtype=torch.uint8,
                device=corrected_device,
            )

            if corrected_device == "cpu":
                torch.cuda.cudart().cudaHostRegister(
                    buffer.data_ptr(), config.nixl_buffer_size, 0
                )
            else:
                logger.info(f"Setting cuda device to {corrected_device} ")
                torch.cuda.set_device(corrected_device)

            return PagedTensorMemoryAllocator(
                buffer,
                [torch.Size(metadata.kv_shape)],
                [metadata.kv_dtype],
                MemoryFormat.KV_2LTD,
            )

        if config.weka_path is not None or config.gds_path is not None:
            assert config.cufile_buffer_size is not None
            return CuFileMemoryAllocator(config.cufile_buffer_size * 1024**2)

        max_local_cpu_size = config.max_local_cpu_size
        # save_only_first_rank only works when use mla
        save_only_first_rank = (
            config.get_extra_config_value("save_only_first_rank", metadata.use_mla)
            and metadata.use_mla
        )
        if save_only_first_rank and metadata.is_first_rank():
            # Only the first rank will save the cache,
            # so we need to set it lager than other ranks
            first_rank_max_local_cpu_size = (
                config.extra_config.get(
                    "first_rank_max_local_cpu_size", max_local_cpu_size
                )
                if config.extra_config
                else max_local_cpu_size
            )
            return MixedMemoryAllocator(
                int(first_rank_max_local_cpu_size * 1024**3),
                numa_mapping=numa_mapping,
            )
        return MixedMemoryAllocator(
            int(max_local_cpu_size * 1024**3),
            numa_mapping=numa_mapping,
        )

    @staticmethod
    def _Create_token_database(
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
    ) -> TokenDatabase:
        if config.enable_blending:
            return SegmentTokenDatabase(config, metadata)
        return ChunkedTokenDatabase(config, metadata)

    @classmethod
    def get_or_create(
        cls,
        instance_id: str,
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
        gpu_connector: Optional[GPUConnectorInterface],
        broadcast_fn: Callable[[torch.Tensor, int], None],
        broadcast_object_fn: Callable[[Any, int], Any],
    ) -> LMCacheEngine:
        """
        Builds a new LMCacheEngine instance if it doesn't already exist for the
        given ID.

        raises: ValueError if the instance already exists with a different
            configuration.
        """
        logger.info(f"Creating LMCacheEngine instance {instance_id}")
        if instance_id not in cls._instances:
            numa_mapping = NUMADetector.get_numa_mapping(config)
            logger.info(f"NUMA mapping for instance {instance_id}: {numa_mapping}")
            token_database = cls._Create_token_database(config, metadata)
            stat_logger = LMCacheStatsLogger(metadata, log_interval=10)

            engine = LMCacheEngine(
                config,
                metadata,
                token_database,
                gpu_connector,
                broadcast_fn,
                broadcast_object_fn,
            )

            cls._instances[instance_id] = engine
            cls._cfgs[instance_id] = config
            cls._metadatas[instance_id] = metadata
            cls._stat_loggers[instance_id] = stat_logger
            return engine
        else:
            if (
                cls._cfgs[instance_id] != config
                or cls._metadatas[instance_id] != metadata
            ):
                raise ValueError(
                    f"Instance {instance_id} already exists with a different "
                    f"configuration or metadata."
                )
            return cls._instances[instance_id]

    @classmethod
    def get(cls, instance_id: str) -> Optional[LMCacheEngine]:
        """Returns the LMCacheEngine instance associated with the instance ID,
        or None if not found."""
        return cls._instances.get(instance_id)

    @classmethod
    def destroy(cls, instance_id: str) -> None:
        """Close and delete the LMCacheEngine instance by the instance ID"""
        # TODO: unit test for this
        logger.info(f"Destroying LMCacheEngine instance: {instance_id}")

        if instance_id in cls._instances:
            stat_logger = cls._stat_loggers[instance_id]
            try:
                logger.info("Shutting down stats logger...")
                stat_logger.shutdown()
                logger.info("Stats logger shut down successfully")
            except Exception as e:
                logger.error(f"Error shutting down stats logger: {e}")

            engine = cls._instances[instance_id]
            try:
                logger.info("Closing cache engine...")
                engine.close()
                logger.info("Cache engine closed successfully")
            except Exception as e:
                logger.error(f"Error closing cache engine: {e}")

            try:
                logger.info("Cleaning up instance dictionaries...")
                cls._instances.pop(instance_id, None)
                cls._cfgs.pop(instance_id, None)
                cls._metadatas.pop(instance_id, None)
                cls._stat_loggers.pop(instance_id, None)
                logger.info("Instance dictionaries cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up instances: {e}")

            try:
                logger.info("Destroying stats monitor...")
                LMCStatsMonitor.DestroyInstance()
                logger.info("Stats monitor destroyed successfully")
            except Exception as e:
                logger.error(f"Error destroying stats monitor: {e}")

            logger.info(f"LMCacheEngine instance {instance_id} destroyed")
        else:
            logger.warning(f"Instance {instance_id} not found for destruction")
