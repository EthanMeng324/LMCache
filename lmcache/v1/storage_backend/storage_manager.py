# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from typing import (
    TYPE_CHECKING,
    Any,
    Coroutine,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    Callable,
    Awaitable,
    cast,
)
import asyncio
import functools
import os
import threading
import time

# Third Party
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.observability import PrometheusLogger
from lmcache.utils import (
    CacheEngineKey,
    _lmcache_nvtx_annotate,
    start_loop_in_thread_with_exceptions,
)
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.event_manager import EventManager, EventStatus, EventType
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.storage_backend import CreateStorageBackends, is_cuda_worker
from lmcache.v1.storage_backend.abstract_backend import (
    AllocatorBackendInterface,
    StorageBackendInterface,
)
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.prefetch import PrefetchState, PrefetchTask

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_controller.worker import LMCacheWorker
    from lmcache.v1.lookup_client.lmcache_async_lookup_client import (
        LMCacheAsyncLookupServer,
    )

logger = init_logger(__name__)


# Helper function to get the class name of the backend
def get_backend_cname(backend: StorageBackendInterface):
    return backend.__class__.__name__


# Helper function to allocate and copy memory objects between D and H
def allocate_and_copy_objects(
    allocator_backend: AllocatorBackendInterface,
    keys: Sequence[CacheEngineKey],
    src_memory_objs: list[MemoryObj],
    stream: torch.cuda.Stream,
) -> tuple[Sequence[CacheEngineKey], list[MemoryObj]]:
    """
    Allocate the memory objects and copy the data from src_memory_objs to
    the newly allocated memory objects

    Args:
        allocator_backend: the allocator backend to allocate the new memory
          objects
        keys: the cache engine keys corresponding to the memory objects
        src_memory_objs: the memory objects to copy from
        stream: the cuda stream to run the copy in

    Returns:
        - list of cache engine keys that corresponds to the memory objects
          that has been successfully allocated
        - list of the memory objects that has been successfully allocated
    """
    allocated_objects = []
    for key, src_memory_obj in zip(keys, src_memory_objs, strict=False):
        if allocator_backend.contains(key):
            continue
        memory_obj = allocator_backend.allocate(
            src_memory_obj.get_shape(),
            src_memory_obj.get_dtype(),
            fmt=src_memory_obj.meta.fmt,
            eviction=True,
            busy_loop=False,
        )

        if memory_obj is None or memory_obj.tensor is None:
            break

        with torch.cuda.stream(stream):
            memory_obj.tensor.copy_(src_memory_obj.tensor, non_blocking=True)
        allocated_objects.append(memory_obj)

    stream.synchronize()
    return keys[: len(allocated_objects)], allocated_objects


class WeightedSemaphore:
    def __init__(self, chunk_budget: int):
        # it is physically impossible to have more fragmentation than 50%
        # when all of the chunks are of the same size (save_unfull_chunk=False)
        # so we can safely allocate half of the chunk budget for concurrent requests
        self._concurrent_budget_cap = chunk_budget // 2
        self._chunk_budget_cap = chunk_budget
        self._current_chunks = self._concurrent_budget_cap
        self._cond = asyncio.Condition()

    async def acquire(self, n: int = 1) -> None:
        if n > self._chunk_budget_cap:
            raise ValueError(
                f"Trying to acquire {n} chunks, "
                f"Cannot acquire more than {self._chunk_budget_cap} chunks"
                "Please set the max local cpu size to a larger value"
            )

        async with self._cond:
            logger.debug(f"WeightedSemaphore: Attempting to acquire {n} chunks")
            if n <= self._concurrent_budget_cap:
                await self._cond.wait_for(lambda: self._current_chunks >= n)
                self._current_chunks -= n
            else:
                # Oversized case: require exclusive access
                await self._cond.wait_for(
                    lambda: self._current_chunks == self._concurrent_budget_cap
                )
                # Reserve everything
                self._current_chunks = 0
            logger.debug(
                f"WeightedSemaphore: Acquired {n} chunks, "
                f"remaining chunks: {self._current_chunks}"
            )

    async def release(self, n: int = 1) -> None:
        async with self._cond:
            if n <= self._concurrent_budget_cap:
                self._current_chunks += n
            else:
                self._current_chunks = self._concurrent_budget_cap
            self._cond.notify_all()


class AsyncMultiSerializer:
    """
    Prevent race conditions where multiple batched_get's cause the local CPU
    backend to allocate memory objects in parallel and get deadlocked.
    Make the assumption that the save_unfull_chunk is False so that we
    can assume that we can always use 50% of the given memory
    """

    def __init__(
        self,
        allocator_backend: AllocatorBackendInterface,
        loop: asyncio.AbstractEventLoop,
    ):
        self.chunk_budget = allocator_backend.calculate_chunk_budget()
        self._sem = WeightedSemaphore(self.chunk_budget)
        self.loop = loop

    async def run(
        self,
        coro_fn: Coroutine[Any, Any, Any],
        num_chunks: int,
    ) -> Any:
        await self._sem.acquire(num_chunks)
        try:
            return await coro_fn
        finally:
            await self._sem.release(num_chunks)


class AsyncSingleSerializer:
    """
    Prevent race conditions in a naive way by forcing each request that
    is passed through to be serialized
    """

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop
        # lazy init in run
        self.lock: Optional[asyncio.Lock] = None

    async def run(self, coro_fn: Coroutine[Any, Any, Any], *args, **kwargs) -> Any:
        # we need to lazily initialize the lock to
        # place it on the calling event loop
        if self.lock is None:
            self.lock = asyncio.Lock()
        async with self.lock:  # type: ignore
            return await coro_fn


AsyncSerializer = Union[AsyncSingleSerializer, AsyncMultiSerializer]


# TODO: extend this class to implement caching policies and eviction policies
class StorageManager:
    """
    The StorageManager is responsible for managing the storage backends.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
        event_manager: EventManager,
        lmcache_worker: Optional["LMCacheWorker"] = None,
    ):
        self.config = config
        self.metadata = metadata
        self.loop = asyncio.new_event_loop()

        self.thread = threading.Thread(
            target=start_loop_in_thread_with_exceptions,
            args=(self.loop,),
            name="storage-manger-event-loop",
        )
        self.thread.start()

        # For scheduler role, always use CPU device
        if is_cuda_worker(metadata):
            dst_device = "cuda"
        else:
            dst_device = "cpu"
        self.storage_backends: OrderedDict[str, StorageBackendInterface] = (
            CreateStorageBackends(
                config,
                metadata,
                self.loop,
                dst_device,
                lmcache_worker,
            )
        )

        # the backend used for actual storage
        self.non_allocator_backends = self.get_non_allocator_backends()

        self.enable_pd = config.enable_pd

        self.allocator_backend = None
        if metadata.role != "scheduler":
            self.allocator_backend = self._get_allocator_backend(config)
        if config.local_cpu:
            self.local_cpu_backend = self.storage_backends["LocalCPUBackend"]

        self.manager_lock = threading.Lock()
        # Node-local route/runtime prefetch registry.  The actual copy is
        # submitted by CacheEngine's bounded executor, while this registry
        # provides one inflight task per logical key and demand escalation for
        # callers that race with a hint.
        self.prefetch_inflight: dict[CacheEngineKey, PrefetchTask] = {}
        self.prefetch_completed: dict[CacheEngineKey, PrefetchTask] = {}
        self._prefetch_lock = threading.Lock()

        # Waiting-route CXL promotion is deliberately separate from the
        # existing predictor/association registry above.  The Future table is
        # the join point shared by speculative work and synchronous demand
        # reads, while the semaphore bounds both executor admission and CPU
        # pressure from speculative copies.
        extra_config = (
            config.extra_config if isinstance(config.extra_config, dict) else {}
        )
        self.cxl_prefetch_enabled = bool(
            extra_config.get("enable_cxl_prefetch", False)
        )
        self.cxl_prefetch_max_inflight = max(
            1, int(extra_config.get("cxl_prefetch_max_inflight", 2))
        )
        self.cxl_prefetch_max_chunks = max(
            1, int(extra_config.get("cxl_prefetch_max_chunks", 8))
        )
        self._cxl_prefetch_executor: Optional[ThreadPoolExecutor] = None
        self._cxl_prefetch_semaphore: Optional[threading.BoundedSemaphore] = None
        self._cxl_prefetch_futures: dict[CacheEngineKey, Future] = {}
        self._cxl_prefetch_lock = threading.Lock()
        self._cxl_prefetch_stats = {
            "submitted": 0,
            "completed": 0,
            "succeeded": 0,
            "failed": 0,
            "deduplicated": 0,
            "already_cpu": 0,
            "capacity_rejected": 0,
            "demand_joins": 0,
            "missing_cxl": 0,
            "metadata_missing": 0,
            "cxl_read_failed": 0,
            "cpu_capacity_rejected": 0,
            "cpu_admission_failed": 0,
            "backend_unavailable": 0,
            "unknown_failed": 0,
        }
        # Timeline-only, bounded per-key state.  The normal prefetch path does
        # not retain request/key history; this small registry exists solely so
        # a later demand lookup can be joined to the exact promotion that it
        # consumed during a funnel validation run.
        try:
            trace_capacity = int(
                os.environ.get("DYN_CXL_PREFETCH_TRACE_CAPACITY", "4096")
            )
        except (TypeError, ValueError):
            trace_capacity = 4096
        self._cxl_prefetch_trace_capacity = max(1, trace_capacity)
        self._cxl_prefetch_admissions: dict[
            CacheEngineKey, dict[str, str | int | bool | None]
        ] = {}
        self._cxl_prefetch_completed_trace: OrderedDict[
            CacheEngineKey, dict[str, str | int | bool | None]
        ] = OrderedDict()
        if self.cxl_prefetch_enabled:
            self._cxl_prefetch_executor = ThreadPoolExecutor(
                max_workers=self.cxl_prefetch_max_inflight,
                thread_name_prefix="lmcache-cxl-prefetch",
            )
            self._cxl_prefetch_semaphore = threading.BoundedSemaphore(
                self.cxl_prefetch_max_inflight
            )
            logger.info(
                "CXL lookahead promotion enabled (max_inflight=%d, max_chunks=%d)",
                self.cxl_prefetch_max_inflight,
                self.cxl_prefetch_max_chunks,
            )

        self.lmcache_worker = lmcache_worker
        self.instance_id = config.lmcache_instance_id
        self.worker_id = metadata.worker_id

        self.event_manager = event_manager

        self.async_lookup_server: Optional["LMCacheAsyncLookupServer"] = None
        self.async_serializer: Optional[AsyncSerializer] = None

        # The cuda stream for internal copies during put
        if is_cuda_worker(metadata):
            self.internal_copy_stream = torch.cuda.Stream()
        else:
            self.internal_copy_stream = None

        # freeze mode: only use local_cpu backend for retrieval
        self._freeze = False
        self._freeze_lock = threading.RLock()

        self._setup_metrics()
        
        # Initialize cache migration service if enabled
        self.cache_migration_service = None
        self._init_cache_migration_service()

        # Storage write policy (write-through by default).
        # write-through: store to all configured backends (current behavior).
        # write-back: store only to L1 (LocalCPUBackend), spill to lower tiers on eviction.
        self.storage_write_policy = "write_through"
        if self.config.extra_config is not None:
            self.storage_write_policy = self.config.extra_config.get(
                "storage_write_policy", self.storage_write_policy
            )
        self._init_write_back_policy()

    def register_prefetch_task(self, task: PrefetchTask) -> bool:
        """Register one task per key; return False for duplicate work."""
        with self._prefetch_lock:
            existing = self.prefetch_inflight.get(task.key)
            if existing is not None:
                if task.priority > existing.priority:
                    existing.priority = task.priority
                    existing.state = PrefetchState.ESCALATED_DEMAND
                return False
            task.state = PrefetchState.QUEUED
            self.prefetch_inflight[task.key] = task
            return True

    async def prefetch_keys(
        self,
        tasks: Sequence[PrefetchTask],
        *,
        submitter: Optional[Callable[[PrefetchTask], Awaitable[None]]] = None,
    ) -> list[PrefetchTask]:
        """Register bounded arbitrary-key prefetch tasks.

        ``submitter`` is optional because CacheEngine owns the CXL staging and
        promotion executor.  When supplied, each newly admitted task is
        handed to it asynchronously; demand lookups can still call
        :meth:`escalate_prefetch` while the submitter is running.
        """
        accepted = [task for task in tasks if self.register_prefetch_task(task)]
        if submitter is not None:
            for task in accepted:
                await submitter(task)
        return accepted

    def escalate_prefetch(self, key: CacheEngineKey) -> Optional[PrefetchTask]:
        """Upgrade a hint task when demand races with its CXL copy."""
        with self._prefetch_lock:
            task = self.prefetch_inflight.get(key)
            if task is not None:
                task.priority = max(task.priority, 2**31 - 1)
                task.state = PrefetchState.ESCALATED_DEMAND
            return task

    def complete_prefetch_task(
        self, key: CacheEngineKey, *, state: PrefetchState
    ) -> Optional[PrefetchTask]:
        with self._prefetch_lock:
            task = self.prefetch_inflight.pop(key, None)
            if task is not None:
                task.state = state
                task.completed_ns = time.time_ns()
                self.prefetch_completed[key] = task
            return task

    def cancel_prefetch_task(self, key: CacheEngineKey) -> bool:
        with self._prefetch_lock:
            task = self.prefetch_inflight.pop(key, None)
            if task is None:
                return False
            task.state = PrefetchState.CANCELLED
            self.prefetch_completed[key] = task
            return True

    def _get_prefetch_future(self, key: CacheEngineKey) -> Optional[Future]:
        """Return the active lookahead-promotion Future for ``key`` if any."""
        with self._cxl_prefetch_lock:
            return self._cxl_prefetch_futures.get(key)

    @staticmethod
    def _cxl_prefetch_timeline_enabled() -> bool:
        return os.environ.get("DYN_CXL_PREFETCH_TIMELINE_ENABLED", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def _log_cxl_prefetch_admission(
        self,
        *,
        request_id: Optional[str],
        key: CacheEngineKey,
        cxl_present: Optional[bool],
        local_cpu_present: Optional[bool],
        outcome: str,
    ) -> None:
        if not self._cxl_prefetch_timeline_enabled() or not request_id:
            return
        logger.info(
            "CXL prefetch funnel admission request_id=%s worker_id=%s key=%s "
            "cxl_present=%s local_cpu_present=%s outcome=%s admission_unix_ns=%d",
            request_id,
            getattr(self, "worker_id", "unknown"),
            getattr(key, "chunk_hash", key),
            cxl_present,
            local_cpu_present,
            outcome,
            time.time_ns(),
        )

    def _log_cxl_prefetch_demand(
        self,
        *,
        request_id: Optional[str],
        key: CacheEngineKey,
        memory_obj: Optional[MemoryObj],
        backend_name: str,
    ) -> None:
        if (
            not self._cxl_prefetch_timeline_enabled()
            or not request_id
            or memory_obj is None
            or backend_name != "LocalCPUBackend"
        ):
            return
        with self._cxl_prefetch_lock:
            trace = self._cxl_prefetch_completed_trace.get(key)
            if trace is not None:
                trace = dict(trace)
        if trace is None:
            return
        prefetch_request_id = trace.get("request_id")
        request_matches_prefetch = bool(
            prefetch_request_id == request_id
            or (
                isinstance(prefetch_request_id, str)
                and request_id.startswith(prefetch_request_id + "-")
            )
        )
        consumed = bool(
            request_matches_prefetch
            and trace.get("cxl_present") is True
            and trace.get("local_cpu_present") is False
        )
        logger.info(
            "CXL prefetch funnel demand request_id=%s worker_id=%s key=%s "
            "backend=%s prefetch_request_id=%s promotion_complete_unix_ns=%s "
            "consumed_prefetch=%s demand_unix_ns=%d",
            request_id,
            getattr(self, "worker_id", "unknown"),
            getattr(key, "chunk_hash", key),
            backend_name,
            prefetch_request_id,
            trace.get("promotion_complete_unix_ns"),
            consumed,
            time.time_ns(),
        )

    def _local_cpu_backend(self) -> Optional[LocalCPUBackend]:
        backend = self.storage_backends.get("LocalCPUBackend")
        # The production backend is a LocalCPUBackend.  Keep this lookup based
        # on the stable backend name rather than an exact class check so
        # instrumentation/test wrappers can still participate in the same
        # demand-join protocol.
        return cast(LocalCPUBackend, backend) if backend is not None else None

    def _run_cxl_prefetch_task(
        self,
        key: CacheEngineKey,
        cxl_backend: StorageBackendInterface,
        local_cpu_backend: LocalCPUBackend,
        request_id: Optional[str] = None,
    ) -> bool:
        timeline_enabled = self._cxl_prefetch_timeline_enabled()
        if timeline_enabled and request_id:
            logger.info(
                "CXL prefetch timeline promotion_start request_id=%s "
                "promotion_start_unix_ns=%d key=%s",
                request_id,
                time.time_ns(),
                getattr(key, "chunk_hash", key),
            )
        # A demand lookup may have populated CPU after admission but before
        # the worker got scheduled.  Avoid a redundant CXL read in that case.
        if local_cpu_backend.contains(key):
            return True
        submit = getattr(cxl_backend, "submit_prefetch_task", None)
        if not callable(submit):
            with self._cxl_prefetch_lock:
                self._cxl_prefetch_stats["backend_unavailable"] += 1
            return False
        try:
            succeeded = bool(submit(key))
            if not succeeded:
                reason_getter = getattr(
                    cxl_backend, "get_prefetch_failure_reason", None
                )
                reason = reason_getter() if callable(reason_getter) else None
                stat_name = (
                    reason
                    if reason in {
                        "missing_cxl",
                        "metadata_missing",
                        "cxl_read_failed",
                        "cpu_capacity_rejected",
                        "cpu_admission_failed",
                        "backend_unavailable",
                    }
                    else "unknown_failed"
                )
                with self._cxl_prefetch_lock:
                    self._cxl_prefetch_stats[stat_name] += 1
                logger.debug(
                    "CXL lookahead promotion skipped for key %s: %s",
                    getattr(key, "chunk_hash", key),
                    stat_name,
                )
            return succeeded
        except Exception:
            with self._cxl_prefetch_lock:
                self._cxl_prefetch_stats["unknown_failed"] += 1
            logger.debug(
                "CXL lookahead promotion task failed for key %s",
                getattr(key, "chunk_hash", key),
                exc_info=True,
            )
            return False

    def _cxl_prefetch_done(
        self,
        key: CacheEngineKey,
        future: Future,
        request_id: Optional[str] = None,
    ) -> None:
        succeeded = False
        try:
            succeeded = bool(future.result())
        except Exception:
            logger.debug(
                "CXL lookahead promotion future failed for key %s",
                getattr(key, "chunk_hash", key),
                exc_info=True,
            )

        complete_ns = time.time_ns()
        admission_trace: Optional[dict[str, str | int | bool | None]] = None
        with self._cxl_prefetch_lock:
            if self._cxl_prefetch_futures.get(key) is future:
                self._cxl_prefetch_futures.pop(key, None)
            admission_trace = self._cxl_prefetch_admissions.pop(key, None)
            self._cxl_prefetch_stats["completed"] += 1
            if succeeded:
                self._cxl_prefetch_stats["succeeded"] += 1
                if admission_trace is not None:
                    admission_trace["promotion_complete_unix_ns"] = complete_ns
                    self._cxl_prefetch_completed_trace[key] = admission_trace
                    self._cxl_prefetch_completed_trace.move_to_end(key)
                    while (
                        len(self._cxl_prefetch_completed_trace)
                        > self._cxl_prefetch_trace_capacity
                    ):
                        self._cxl_prefetch_completed_trace.popitem(last=False)
            else:
                self._cxl_prefetch_stats["failed"] += 1

        if request_id and self._cxl_prefetch_timeline_enabled():
            logger.info(
                "CXL prefetch timeline promotion_complete request_id=%s "
                "promotion_complete_unix_ns=%d success=%s key=%s "
                "cxl_present=%s local_cpu_present=%s",
                request_id,
                complete_ns,
                succeeded,
                getattr(key, "chunk_hash", key),
                None if admission_trace is None else admission_trace.get("cxl_present"),
                None
                if admission_trace is None
                else admission_trace.get("local_cpu_present"),
            )

        semaphore = self._cxl_prefetch_semaphore
        if semaphore is not None:
            try:
                semaphore.release()
            except ValueError:
                logger.warning("CXL lookahead promotion semaphore release overflow")

    def submit_cxl_prefetch(
        self,
        keys: Sequence[CacheEngineKey],
        max_chunks: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> dict[str, int | bool | str | None]:
        """Submit bounded CXL-to-CPU work without waiting for completion.

        The returned counters describe admission only.  A ``False`` result
        from a background task is intentionally not propagated as a request
        error; the demand path will fall back to the normal storage lookup.
        """
        result: dict[str, int | bool | str | None] = {
            "enabled": self.cxl_prefetch_enabled,
            "scheduled": 0,
            "deduplicated": 0,
            "already_cpu": 0,
            "capacity_rejected": 0,
            "status": "disabled" if not self.cxl_prefetch_enabled else "accepted",
        }
        if not self.cxl_prefetch_enabled:
            return result

        limit = self.cxl_prefetch_max_chunks if max_chunks is None else int(max_chunks)
        if limit <= 0:
            result["status"] = "empty"
            return result
        limit = min(limit, self.cxl_prefetch_max_chunks)

        cxl_backend = self.storage_backends.get("CxlBackend")
        local_cpu_backend = self._local_cpu_backend()
        executor = self._cxl_prefetch_executor
        semaphore = self._cxl_prefetch_semaphore
        if (
            cxl_backend is None
            or local_cpu_backend is None
            or executor is None
            or semaphore is None
        ):
            result["status"] = "unavailable"
            return result

        for key in dict.fromkeys(keys):
            if int(result["scheduled"]) >= limit:
                break

            timeline_enabled = self._cxl_prefetch_timeline_enabled()
            cxl_present: Optional[bool] = None
            if timeline_enabled:
                try:
                    cxl_present = bool(cxl_backend.contains(key))
                except Exception:
                    logger.debug(
                        "Unable to inspect CXL residency for funnel key %s",
                        getattr(key, "chunk_hash", key),
                        exc_info=True,
                    )

            local_cpu_present = bool(local_cpu_backend.contains(key))
            if local_cpu_present:
                self._log_cxl_prefetch_admission(
                    request_id=request_id,
                    key=key,
                    cxl_present=cxl_present,
                    local_cpu_present=True,
                    outcome="already_cpu",
                )
                result["already_cpu"] = int(result["already_cpu"]) + 1
                with self._cxl_prefetch_lock:
                    self._cxl_prefetch_stats["already_cpu"] += 1
                continue

            with self._cxl_prefetch_lock:
                if key in self._cxl_prefetch_futures:
                    self._log_cxl_prefetch_admission(
                        request_id=request_id,
                        key=key,
                        cxl_present=cxl_present,
                        local_cpu_present=False,
                        outcome="deduplicated",
                    )
                    result["deduplicated"] = int(result["deduplicated"]) + 1
                    self._cxl_prefetch_stats["deduplicated"] += 1
                    continue

            if not semaphore.acquire(blocking=False):
                self._log_cxl_prefetch_admission(
                    request_id=request_id,
                    key=key,
                    cxl_present=cxl_present,
                    local_cpu_present=False,
                    outcome="capacity_rejected",
                )
                result["capacity_rejected"] = int(result["capacity_rejected"]) + 1
                with self._cxl_prefetch_lock:
                    self._cxl_prefetch_stats["capacity_rejected"] += 1
                break

            # Recheck after acquiring the slot to close races with another
            # submitter or a demand hit.
            if local_cpu_backend.contains(key):
                semaphore.release()
                self._log_cxl_prefetch_admission(
                    request_id=request_id,
                    key=key,
                    cxl_present=cxl_present,
                    local_cpu_present=True,
                    outcome="already_cpu_race",
                )
                result["already_cpu"] = int(result["already_cpu"]) + 1
                with self._cxl_prefetch_lock:
                    self._cxl_prefetch_stats["already_cpu"] += 1
                continue

            future: Optional[Future] = None
            with self._cxl_prefetch_lock:
                if key in self._cxl_prefetch_futures:
                    semaphore.release()
                    self._log_cxl_prefetch_admission(
                        request_id=request_id,
                        key=key,
                        cxl_present=cxl_present,
                        local_cpu_present=False,
                        outcome="deduplicated_race",
                    )
                    result["deduplicated"] = int(result["deduplicated"]) + 1
                    self._cxl_prefetch_stats["deduplicated"] += 1
                    continue
                try:
                    future = executor.submit(
                        self._run_cxl_prefetch_task,
                        key,
                        cxl_backend,
                        local_cpu_backend,
                        request_id,
                    )
                except Exception:
                    semaphore.release()
                    self._log_cxl_prefetch_admission(
                        request_id=request_id,
                        key=key,
                        cxl_present=cxl_present,
                        local_cpu_present=False,
                        outcome="executor_rejected",
                    )
                    result["capacity_rejected"] = int(result["capacity_rejected"]) + 1
                    self._cxl_prefetch_stats["capacity_rejected"] += 1
                    logger.debug(
                        "Unable to submit CXL lookahead promotion task",
                        exc_info=True,
                    )
                    break
                self._cxl_prefetch_futures[key] = future
                self._cxl_prefetch_admissions[key] = {
                    "request_id": request_id,
                    "cxl_present": cxl_present,
                    "local_cpu_present": False,
                    "admission_unix_ns": time.time_ns(),
                }
                self._cxl_prefetch_stats["submitted"] += 1
                result["scheduled"] = int(result["scheduled"]) + 1
            self._log_cxl_prefetch_admission(
                request_id=request_id,
                key=key,
                cxl_present=cxl_present,
                local_cpu_present=False,
                outcome="submitted",
            )
            # A very fast fake/backend task can already be complete here.  Add
            # the callback outside the non-reentrant bookkeeping lock so the
            # immediate callback path cannot deadlock.
            if future is not None:
                future.add_done_callback(
                    lambda completed, prefetch_key=key, prefetch_request_id=request_id: self._cxl_prefetch_done(
                        prefetch_key, completed, prefetch_request_id
                    )
                )

        return result

    def _wait_for_cxl_prefetch(self, key: CacheEngineKey) -> bool:
        future = self._get_prefetch_future(key)
        if future is None:
            return False
        with self._cxl_prefetch_lock:
            self._cxl_prefetch_stats["demand_joins"] += 1
        try:
            return bool(future.result())
        except Exception:
            logger.debug(
                "Demand join observed failed CXL lookahead promotion for key %s",
                getattr(key, "chunk_hash", key),
                exc_info=True,
            )
            return False

    def _wait_for_cxl_prefetches(self, keys: Sequence[CacheEngineKey]) -> None:
        # Keep the Future references first: completion callbacks remove entries
        # from the table while a demand lookup is traversing the prefix.
        futures: list[Future] = []
        seen: set[int] = set()
        for key in keys:
            future = self._get_prefetch_future(key)
            if future is None or id(future) in seen:
                continue
            seen.add(id(future))
            futures.append(future)
        for future in futures:
            with self._cxl_prefetch_lock:
                self._cxl_prefetch_stats["demand_joins"] += 1
            try:
                future.result()
            except Exception:
                logger.debug(
                    "Demand join observed failed CXL lookahead promotion",
                    exc_info=True,
                )

    async def _await_cxl_prefetch_prefix(
        self,
        keys: Sequence[CacheEngineKey],
        search_range: Optional[list[str]],
    ) -> None:
        if (
            not self.cxl_prefetch_enabled
            or "LocalCPUBackend" not in self.storage_backends
            or (search_range is not None and "LocalCPUBackend" not in search_range)
        ):
            return
        local_cpu_backend = self._local_cpu_backend()
        if local_cpu_backend is None:
            return
        for key in keys:
            if local_cpu_backend.contains(key):
                continue
            future = self._get_prefetch_future(key)
            if future is None:
                break
            with self._cxl_prefetch_lock:
                self._cxl_prefetch_stats["demand_joins"] += 1
            try:
                await asyncio.wrap_future(future)
            except Exception:
                logger.debug(
                "Async demand join observed failed CXL lookahead promotion",
                    exc_info=True,
                )
            if not local_cpu_backend.contains(key):
                # Preserve the existing fallback behavior after a failed or
                # incomplete speculative task.
                break

    def get_cxl_prefetch_stats(self) -> dict[str, int]:
        with self._cxl_prefetch_lock:
            return {key: int(value) for key, value in self._cxl_prefetch_stats.items()}

    def _init_write_back_policy(self) -> None:
        """Initialize write-back eviction spilling if configured."""
        if self.storage_write_policy != "write_back":
            return
        if not getattr(self.config, "local_cpu", False):
            logger.warning(
                "storage_write_policy=write_back requested but local_cpu is disabled; "
                "falling back to write_through"
            )
            self.storage_write_policy = "write_through"
            return
        if "LocalCPUBackend" not in self.storage_backends:
            logger.warning(
                "storage_write_policy=write_back requested but LocalCPUBackend not found; "
                "falling back to write_through"
            )
            self.storage_write_policy = "write_through"
            return

        # Determine spill targets: either explicit list from extra_config, or
        # all non-L1 backends excluding transient transfer backends.
        spill_target_names: Optional[list[str]] = None
        if self.config.extra_config is not None:
            spill_target_names = self.config.extra_config.get("write_back_targets")

        excluded = {"LocalCPUBackend", "PDBackend", "P2PBackend"}
        spill_targets: list[tuple[str, StorageBackendInterface]] = []
        missing_targets: list[str] = []
        if spill_target_names is not None:
            existing = set(self.storage_backends.keys())
            missing_targets = [name for name in spill_target_names if name not in existing]
            if missing_targets:
                logger.warning(
                    "storage_write_policy=write_back requested spill targets not found: %s. "
                    "Available backends: %s",
                    missing_targets,
                    sorted(existing),
                )
        for name, backend in self.storage_backends.items():
            if name in excluded:
                continue
            if spill_target_names is not None and name not in spill_target_names:
                continue
            spill_targets.append((name, backend))

        if not spill_targets:
            logger.warning(
                "storage_write_policy=write_back enabled but no spill targets configured; "
                "falling back to write_through"
            )
            self.storage_write_policy = "write_through"
            return

        from lmcache.v1.storage_backend.write_back_listener import (
            WriteBackEvictionListener,
        )

        listener = WriteBackEvictionListener(spill_targets)
        local_cpu_backend = self.storage_backends["LocalCPUBackend"]
        local_cpu_backend.add_listener(listener)
        logger.info(
            "Enabled write-back policy: L1=LocalCPUBackend, spill_targets=%s",
            [name for name, _ in spill_targets],
        )

    def _setup_metrics(self):
        prometheus_logger = PrometheusLogger.GetInstanceOrNone()
        if prometheus_logger is None:
            logger.warning(
                "PrometheusLogger is not initialized, "
                "event metrics will not be collected"
            )
            return

        metric_map = {
            "storage_events_ongoing_count": EventStatus.ONGOING,
            "storage_events_done_count": EventStatus.DONE,
            "storage_events_not_found_count": EventStatus.NOT_FOUND,
        }

        for metric_name, status in metric_map.items():
            metric = getattr(prometheus_logger, metric_name)
            metric.set_function(
                lambda s=status: self.event_manager.get_events_count_by_status(
                    EventType.LOADING, s
                )
            )

    def post_init(self, **kwargs) -> None:
        if "async_lookup_server" in kwargs:
            self.async_lookup_server = kwargs.pop("async_lookup_server")
        # PDBackend has't supported calculate_chunk_budget
        if not self.enable_pd and self.config.enable_async_loading:
            assert self.allocator_backend is not None
            self.async_serializer = AsyncSingleSerializer(self.loop)

    def _init_cache_migration_service(self) -> None:
        """Initialize cache migration service if configured."""
        extra_config = self.config.extra_config
        if extra_config is None:
            return
            
        enable_migration = extra_config.get("enable_cache_migration", False)
        if not enable_migration:
            return
            
        # Get source and target backends
        source_backend_name = extra_config.get("cache_migration_source_backend", "LocalCPUBackend")
        target_backend_name = extra_config.get("cache_migration_target_backend", "LocalDiskBackend")
        
        source_backend = self.storage_backends.get(source_backend_name)
        target_backend = self.storage_backends.get(target_backend_name)
        
        if source_backend is None or target_backend is None:
            logger.warning(
                f"Cache migration enabled but backends not found: "
                f"source={source_backend_name}, target={target_backend_name}. "
                f"Available backends: {list(self.storage_backends.keys())}"
            )
            return
            
        # Import here to avoid circular dependency
        from lmcache.v1.storage_backend.cache_migration_service import (
            CacheMigrationService,
        )
        
        top_n = extra_config.get("cache_migration_top_n", 10)
        migration_interval = extra_config.get("cache_migration_interval", 60.0)
        copy_mode = extra_config.get("cache_migration_copy_mode", True)
        
        self.cache_migration_service = CacheMigrationService(
            source_backend=source_backend,
            target_backend=target_backend,
            top_n=top_n,
            migration_interval=migration_interval,
            copy_mode=copy_mode,
        )
        
        logger.info(
            f"Cache migration service initialized: "
            f"source={source_backend_name}, target={target_backend_name}, "
            f"top_n={top_n}, interval={migration_interval}s, copy_mode={copy_mode}"
        )

    def _get_allocator_backend(
        self, config: LMCacheEngineConfig
    ) -> AllocatorBackendInterface:
        if self.enable_pd:
            allocator_backend = self.storage_backends["PDBackend"]
        else:
            allocator_backend = self.storage_backends["LocalCPUBackend"]
        assert isinstance(allocator_backend, AllocatorBackendInterface)
        return allocator_backend

    @_lmcache_nvtx_annotate
    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction=True,
        busy_loop=True,
    ) -> Optional[MemoryObj]:
        """
        Allocate memory object with memory allocator.
        Use LRU evictor if eviction is enabled.
        """
        # TODO (Jiayi): We might need to pre-allocate and management
        # disk in a similar way as CPU.
        assert self.allocator_backend is not None
        return self.allocator_backend.allocate(
            shapes, dtypes, fmt, eviction=eviction, busy_loop=busy_loop
        )

    @_lmcache_nvtx_annotate
    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction=True,
        busy_loop=True,
    ) -> Optional[list[MemoryObj]]:
        """
        Batched allocate memory object with memory allocator.
        Use LRU evictor if eviction is enabled.
        """
        # TODO (Jiayi): We might need to pre-allocate and management
        # disk in a similar way as CPU.
        if self.allocator_backend is None:
            raise RuntimeError("Allocator backend not available for scheduler role")
        return self.allocator_backend.batched_allocate(
            shapes, dtypes, batch_size, fmt, eviction=eviction, busy_loop=busy_loop
        )

    def put(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        location: Optional[str] = None,
    ) -> None:
        """
        Non-blocking function to put the memory object into the storages.
        Do not store if the same object is being stored (handled here by
        storage manager) or has been stored (handled by storage backend).
        """
        raise RuntimeError(
            "StorageManager.put is deprecated and should not be called anymore"
        )

    def batched_put(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec=None,
        location: Optional[str] = None,
    ) -> None:
        """
        Non-blocking function to batched put the memory objects into the
        storage backends.
        Do not store if the same object is being stored (handled here by
        storage manager) or has been stored (handled by storage backend).
        """
        # The dictionary from backend cname to objects and keys
        obj_dict: dict[
            str,
            tuple[Sequence[CacheEngineKey], list[MemoryObj]],
        ] = {}
        if self.allocator_backend is None:
            # For scheduler role, no allocator backend available
            raise RuntimeError("Batched put not available for scheduler role")
        obj_dict[get_backend_cname(self.allocator_backend)] = (
            keys,
            memory_objs,
        )

        # write-back policy: if caller didn't specify a location, write only to
        # the L1 hot cache. Lower tiers will be filled by eviction spilling.
        # NOTE: If a transfer_spec is provided (e.g. disagg / p2p), we also
        # include P2PBackend even in write-back mode.
        write_back_mode = self.storage_write_policy == "write_back" and location is None

        for backend_name, backend in self.storage_backends.items():
            if location and backend_name != location:
                continue
            if write_back_mode and backend_name != "LocalCPUBackend":
                # Preserve transfer backends when explicitly requested by the caller
                # via transfer_spec (best-effort heuristic).
                if transfer_spec is not None and backend_name == "P2PBackend":
                    pass
                else:
                    continue

            allocator_backend = backend.get_allocator_backend()
            cname = get_backend_cname(allocator_backend)
            if cname not in obj_dict:
                new_keys, new_objs = allocate_and_copy_objects(
                    allocator_backend, keys, memory_objs, self.internal_copy_stream
                )
                obj_dict[cname] = (new_keys, new_objs)

            # NOTE: the handling of exists_in_put_tasks
            # is done in the backend
            ks, objs = obj_dict[cname]
            try:
                backend.batched_submit_put_task(ks, objs, transfer_spec=transfer_spec)
            except Exception:
                # Cache storage should be best-effort: never crash the engine due to a
                # single backend failing to write.
                logger.exception(
                    "batched_put: backend %s failed to store %d objects; skipping",
                    backend_name,
                    len(ks),
                )

        for cname, (ks, objs) in obj_dict.items():
            for memory_obj in objs:
                memory_obj.ref_count_down()

    def get(
        self,
        key: CacheEngineKey,
        location: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Optional[MemoryObj]:
        """
        Blocking function to get the memory object from the storages.
        """

        # Local CPU is the demand-facing L1.  If a route-time task is already
        # copying this key, join it before allowing the normal CXL fallback to
        # start another read.
        if location in (None, "LocalCPUBackend"):
            local_cpu_backend = self._local_cpu_backend()
            if local_cpu_backend is not None:
                memory_obj = local_cpu_backend.get_blocking(key)
                if memory_obj is not None:
                    local_cpu_backend.mark_demand_access(key)
                    self._log_cxl_prefetch_demand(
                        request_id=request_id,
                        key=key,
                        memory_obj=memory_obj,
                        backend_name="LocalCPUBackend",
                    )
                    return memory_obj
                self._wait_for_cxl_prefetch(key)
                memory_obj = local_cpu_backend.get_blocking(key)
                if memory_obj is not None:
                    local_cpu_backend.mark_demand_access(key)
                    self._log_cxl_prefetch_demand(
                        request_id=request_id,
                        key=key,
                        memory_obj=memory_obj,
                        backend_name="LocalCPUBackend",
                    )
                    return memory_obj

        # Search all backends for blocking get
        for backend_name, backend in self.get_active_storage_backends(location):
            # TODO(Jiayi): need to make sure all memory_objs returned
            # are allocated by the allocator backend.
            memory_obj = backend.get_blocking(key)
            if memory_obj:
                if backend_name == "LocalCPUBackend" and isinstance(
                    backend, LocalCPUBackend
                ):
                    backend.mark_demand_access(key)
                if (
                    backend_name not in ["LocalCPUBackend", "PDBackend"]
                    and "LocalCPUBackend" in self.storage_backends
                ):
                    local_cpu_backend = self.storage_backends["LocalCPUBackend"]
                    assert isinstance(local_cpu_backend, LocalCPUBackend)
                    # IMPORTANT:
                    # Do NOT insert a foreign-allocator MemoryObj directly into LocalCPUBackend.
                    # LocalCPUBackend assumes its cached objects are freed by its own allocator.
                    # Instead, best-effort copy into a LocalCPUBackend-allocated object and cache that.
                    try:
                        if memory_obj.tensor is not None:
                            cached_obj = local_cpu_backend.allocate(
                                memory_obj.get_shape(),
                                memory_obj.get_dtype(),
                                fmt=memory_obj.meta.fmt,
                                eviction=True,
                                busy_loop=False,
                            )
                            if cached_obj is not None and cached_obj.tensor is not None:
                                cached_obj.tensor.copy_(memory_obj.tensor, non_blocking=True)
                                local_cpu_backend.submit_put_task(key, cached_obj)
                                # Release our local ref; backend keeps its own.
                                cached_obj.ref_count_down()
                    except Exception:
                        logger.exception(
                            "get: failed to cache key %s into LocalCPUBackend; skipping",
                            getattr(key, "chunk_hash", key),
                        )
                return memory_obj

        return None

    def get_non_blocking(
        self,
        key: CacheEngineKey,
        location: Optional[str] = None,
    ) -> Optional[Future]:
        """
        Non-blocking function to get the memory object from the storages.
        """
        # TODO (Jiayi): incorporate prefetching here

        # Search all backends for non-blocking get
        for backend_name, backend in self.get_active_storage_backends(location):
            # NOTE(Jiayi): bypass the allocator for now
            task = backend.get_non_blocking(key)
            if task:
                # TODO (Jiayi): add write-back logic here
                return task
        return None

    def batched_get(
        self,
        keys: List[CacheEngineKey],
        location: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Optional[List[Optional[MemoryObj]]]:
        """
        Blocking function to get the memory objects from the storages.
        """
        if location in (None, "LocalCPUBackend"):
            self._wait_for_cxl_prefetches(keys)

        # TODO (ApostaC): remove the nested optional here
        for backend_name, storage_backend in self.get_active_storage_backends(location):
            memory_objs = storage_backend.batched_get_blocking(keys)
            if memory_objs:
                if backend_name == "LocalCPUBackend" and isinstance(
                    storage_backend, LocalCPUBackend
                ):
                    for key, memory_obj in zip(keys, memory_objs, strict=False):
                        if memory_obj is not None:
                            storage_backend.mark_demand_access(key)
                            self._log_cxl_prefetch_demand(
                                request_id=request_id,
                                key=key,
                                memory_obj=memory_obj,
                                backend_name=backend_name,
                            )
                return memory_objs
        return None

    def layerwise_batched_get(
        self,
        keys: List[List[CacheEngineKey]],
        location: Optional[str] = None,
    ) -> Generator[Future, None, None]:
        """
        Non-blocking function to get the memory objects into the storages
        in a layerwise manner.
        Do not store if the same object is being stored (handled here by
        storage manager) or has been stored (handled by storage backend).

        :param List[List[CacheEngineKey]] keys: The keys to get. The first
            dimension corresponds to the number of layers, and the second
            dimension corresponds to the number of chunks.

        :return: A generator that yields a future for each layer.
        """
        if location is None:
            location = "LocalCPUBackend"
        for keys_multi_chunk in keys:
            # Retrieve all chunks for one layer
            backend = self.storage_backends[location]
            # TODO(Jiayi): need to make async loading and layerwise compatible
            coro = backend.batched_get_non_blocking("fake_lookup_id", keys_multi_chunk)
            task = asyncio.run_coroutine_threadsafe(coro, self.loop)
            yield task

    def prefetch_single_done_callback(
        self,
        future: asyncio.Future,
        keys: list[CacheEngineKey],
        backend_name: str,
    ) -> None:
        """
        Callback function when a single prefetch task
        (i.e., prefetching from a single backend) is done.
        """
        # TODO(Jiayi): support write-back policy here
        pass

    def prefetch_all_done_callback(
        self,
        task: asyncio.Future,
        lookup_id: str,
        cum_chunk_lengths_total: list[int],
        tier_expected_chunks: list[int],
    ) -> None:
        """
        Callback function when all prefetch tasks
        (i.e., prefetching from all backends for the entire request) are done.
        """
        assert self.async_lookup_server is not None
        self.event_manager.update_event_status(
            EventType.LOADING, lookup_id, status=EventStatus.DONE
        )
        res = task.result()

        # Calculate total retrieved chunks across all tiers based on actual results
        # from batched_get_non_blocking, not the batched_async_contains results.
        # This handles the case where chunks may be evicted between contains check
        # and actual retrieval.
        #
        # Example: chunk_size=256, 7 chunks total (1792 tokens) across 3 tiers
        #   cum_chunk_lengths_total = [0, 256, 512, 768, 1024, 1280, 1536, 1792]
        #   tier_expected_chunks = [3, 2, 2]  # Tier 0: 3, Tier 1: 2, Tier 2: 2
        #
        #   Chunks:
        #   [0 1 2 3 4 5 6]
        #   |-----|          <--- stored in Tier0, tier_expected_chunks[0]==3
        #         |---|      <--- stored in Tier1, tier_expected_chunks[1]==2
        #             |---|  <--- stored in Tier2, tier_expected_chunks[2]==2
        #
        # Case 1: All chunks retrieved successfully
        #   [0 1 2 3 4 5 6]
        #   |-----|          <--- Tier0: retrieved 3 chunks (obj0, obj1, obj2)
        #         |---|      <--- Tier1: retrieved 2 chunks (obj3, obj4)
        #             |---|  <--- Tier2: retrieved 2 chunks (obj5, obj6)
        #   res = [[obj0, obj1, obj2], [obj3, obj4], [obj5, obj6]]
        #   total_retrieved_chunks = 7
        #   retrieved_length = cum_chunk_lengths_total[7] = 1792
        #
        # Case 2: Tier 1 only got 1 chunk (eviction), Tier 2 got all 2 chunks
        #   [0 1 2 3 4 5 6]
        #   |-----|          <--- Tier0: retrieved 3 chunks (obj0, obj1, obj2)
        #         |-|X|      <--- Tier1: retrieved 1 chunk (obj3), missing obj4
        #             |---|  <--- Tier2: retrieved 2 chunks (obj5, obj6) - IGNORED
        #   res = [[obj0, obj1, obj2], [obj3], [obj5, obj6]]
        #   total_retrieved_chunks = 4 (stop at tier 1, tier 2 chunks ignored)
        #   retrieved_length = cum_chunk_lengths_total[4] = 1024
        #   Note: Even though tier 2 successfully retrieved 2 chunks, they are
        #   not counted because tier 1 has a gap, breaking prefix continuity.
        #
        # Case 3: Tier 0 only got 2 chunks (eviction), other tiers got all
        #   [0 1 2 3 4 5 6]
        #   |---|X|          <--- Tier0: retrieved 2 chunks (obj0, obj1), missing obj2
        #         |---|      <--- Tier1: retrieved 2 chunks (obj3, obj4) - IGNORED
        #             |---|  <--- Tier2: retrieved 2 chunks (obj5, obj6) - IGNORED
        #   res = [[obj0, obj1], [obj3, obj4], [obj5, obj6]]
        #   total_retrieved_chunks = 2 (stop at tier 0, all subsequent ignored)
        #   retrieved_length = cum_chunk_lengths_total[2] = 512
        total_retrieved_chunks = 0
        for tier_idx, tier_result in enumerate(res):
            actual_chunks = len(tier_result)
            expected_chunks = tier_expected_chunks[tier_idx]
            total_retrieved_chunks += actual_chunks

            # If a tier retrieved fewer chunks than expected, we stop counting
            # because subsequent chunks are not contiguous
            if actual_chunks < expected_chunks:
                # Release all chunks in subsequent tiers since they won't be used
                for subsequent_tier in res[tier_idx + 1 :]:
                    for mem_obj in subsequent_tier:
                        mem_obj.ref_count_down()
                break

        retrieved_length = cum_chunk_lengths_total[total_retrieved_chunks]
        logger.info(
            f"Responding to scheduler for lookup id {lookup_id}"
            f" with retrieved length {retrieved_length}"
        )
        self.async_lookup_server.send_response_to_scheduler(lookup_id, retrieved_length)

    async def async_lookup_and_prefetch(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        cum_chunk_lengths: list[int],
        search_range: Optional[list[str]] = None,
        pin: bool = False,
    ) -> None:
        """
        Perform asynchronous lookup and prefetching across all storage backends.

        :param str lookup_id: The unique id (e.g., request id) for the request.
        :param list[CacheEngineKey] keys: The keys to lookup and prefetch.
        :param list[int] cum_chunk_lengths: The cumulative token lengths of the chunks.
            This is a list where cum_chunk_lengths[i] represents the total number of
            tokens from chunk 0 to chunk i-1 (inclusive).
            Example: If chunk_size=256 and we have 3 chunks:
                - chunk 0: 256 tokens (tokens 0-255)
                - chunk 1: 256 tokens (tokens 256-511)
                - chunk 2: 128 tokens (tokens 512-639)
            Then cum_chunk_lengths = [0, 256, 512, 640]
            Note: len(cum_chunk_lengths) = len(keys) + 1
        :param Optional[list[str]] search_range: The range of storage backends
        to search in. Common values include ["LocalCPUBackend", "CxlBackend",
        "LocalDiskBackend"]. If None, search in all active backends.
        :param bool pin: Whether to pin the keys.
        """

        # NOTE(Jiayi): Currently, the retrieval pattern is always
        # prefix-based. That is, we retrieve 0-t1 tokens from backend 1
        # and retrieve t1-t2 tokens from backend 2, etc. The assumption
        # here is that the suffix chunks are more likely to be evicted
        # than the prefix chunks.
        # TODO(Jiayi): We need to change/optimize this for non-prefix
        # based retrieval patterns or cases where middle chunks are missing.

        # NOTE(Jiayi): We can tolerate the last tier to have fewer loaded
        # chunks than its lookup result indicated. This is especially helpful
        # for P2PBackend.

        # Let the currently executing route-time prefix complete before the
        # async lookup asks CXL for the same objects.  ``wrap_future`` keeps the
        # StorageManager event loop non-blocking while retaining fail-open
        # fallback semantics below.
        await self._await_cxl_prefetch_prefix(keys, search_range)

        num_total_chunks = len(keys)
        num_total_hit_chunks = 0
        # cum_chunk_lengths_total: A copy of the original cumulative chunk lengths
        # for all chunks. This is preserved to calculate the final token count
        # based on the actual retrieved chunks.
        # Example: If chunk_size=256 and we have 3 chunks with total 640 tokens:
        #     cum_chunk_lengths_total = [0, 256, 512, 640]
        # If we retrieve 2 chunks, the retrieved token count is:
        #     cum_chunk_lengths_total[2] = 512 tokens
        cum_chunk_lengths_total = cum_chunk_lengths[:]
        loading_tasks = []
        tier_expected_chunks = []
        # we also keep track of the keys for each tier and each chunk
        loading_task_keys: list[list[CacheEngineKey]] = []
        for backend_name, backend in self.get_active_storage_backends(
            search_range=search_range
        ):
            num_hit_chunks = await backend.batched_async_contains(lookup_id, keys, pin)

            if num_hit_chunks == 0:
                continue

            num_total_hit_chunks += num_hit_chunks
            tier_expected_chunks.append(num_hit_chunks)

            backend_keys = keys[:num_hit_chunks]
            loading_task_keys.append(backend_keys)

            assert self.async_serializer is not None, (
                "Async serializer must be initialized via post_init before using "
                "async_lookup_and_prefetch."
            )
            # num_hit_chunks is only used for the multi serializer
            get_coro = self.async_serializer.run(
                backend.batched_get_non_blocking(
                    lookup_id,
                    backend_keys,
                    {"cum_chunk_lengths": cum_chunk_lengths[: num_hit_chunks + 1]},
                ),
                num_hit_chunks,
            )
            loading_task = asyncio.create_task(get_coro)
            loading_task.add_done_callback(
                functools.partial(
                    self.prefetch_single_done_callback,
                    keys=keys,
                    backend_name=backend_name,
                )
            )

            loading_tasks.append(loading_task)

            cum_chunk_lengths = cum_chunk_lengths[num_hit_chunks:]

            if num_total_hit_chunks == num_total_chunks:
                break
            keys = keys[num_hit_chunks:]

        # If no chunks were hit across all backends, respond immediately and return.
        if num_total_hit_chunks == 0:
            if self.async_lookup_server is not None:
                self.async_lookup_server.send_response_to_scheduler(lookup_id, 0)
            return

        # gather_with_keys() here make a pair of (key, memory_obj) for each chunk
        # in each tier. The all_done result's layout is like following and
        # will be processed in _async_process_tokens_internal()
        # Tier 0:
        #  Tuple(loading_task_keys[0][0] : MemoryObj0)
        #  Tuple(loading_task_keys[0][1] : MemoryObj1)
        # Tier 1:
        #  Tuple(loading_task_keys[1][0] : MemoryObj2)
        #  Tuple(loading_task_keys[1][1] : MemoryObj3)
        async def gather_with_keys() -> list[list[tuple[CacheEngineKey, MemoryObj]]]:
            loading_results = await asyncio.gather(*loading_tasks)
            return [
                list(zip(keys, results, strict=False))
                for keys, results in zip(
                    loading_task_keys, loading_results, strict=False
                )
            ]

        all_done = asyncio.create_task(gather_with_keys())
        # Register the event before adding the callback to avoid race conditions
        self.event_manager.add_event(
            EventType.LOADING,
            lookup_id,
            all_done,
        )

        all_done.add_done_callback(
            lambda future: self.prefetch_all_done_callback(
                future,
                lookup_id,
                cum_chunk_lengths_total,
                tier_expected_chunks,
            )
        )

    def set_freeze(self, enabled: bool) -> None:
        """
        Set freeze mode.

        When enabled, only local_cpu backend will be used for retrieval.
        """
        with self._freeze_lock:
            self._freeze = enabled
        logger.info("StorageManager freeze mode set to %s", enabled)

    def is_frozen(self) -> bool:
        """
        Get freeze mode status.

        Returns:
            bool: True if freeze mode is enabled, False otherwise
        """
        with self._freeze_lock:
            return self._freeze

    def contains(
        self,
        key: CacheEngineKey,
        search_range: Optional[List[str]] = None,
        pin: bool = False,
    ) -> Optional[str]:
        """
        Check whether the key exists in the storage backend.

        :param CacheEngineKey key: The key to check.

        :param Optional[List[str]] search_range: The range of storage backends
        to search in. Should be a subset of ["LocalCPUBackend",
        "LocalDiskBackend"] for now.
        If None, search in all backends.

        :param bool pin: Whether to pin the key.

        return: True if the key exists in the specified storage backends.
        """

        for backend_name, backend in self.get_active_storage_backends(
            search_range=search_range
        ):
            # NOTE(Jiayi): We do not pin for PDBackend
            pin_in_backend = pin if backend_name != "PDBackend" else False

            if backend.contains(key, pin_in_backend):
                return backend_name

        return None

    def batched_contains(
        self,
        keys: List[CacheEngineKey],
        search_range: Optional[List[str]] = None,
        pin: bool = False,
    ) -> tuple[int, dict]:
        """
        Check whether the key exists in the storage backend.

        :param List[CacheEngineKey] keys: The keys to check.

        :param Optional[List[str]] search_range: The range of storage backends
        to search in. Should be a subset of ["LocalCPUBackend",
        "LocalDiskBackend"] for now.
        If None, search in all backends.

        :param bool pin: Whether to pin the key.

        return: Return hit chunks and block mapping by prefix match.
        """
        total_keys = len(keys)
        total_hit_chunks = 0
        block_mapping = {}
        for backend_name, backend in self.get_active_storage_backends(
            search_range=search_range
        ):
            # NOTE(Jiayi): We do not pin for PDBackend
            pin_in_backend = pin if backend_name != "PDBackend" else False

            hit_chunks = backend.batched_contains(keys, pin_in_backend)
            if hit_chunks == 0:
                continue
            block_mapping[backend_name] = keys[:hit_chunks]
            total_hit_chunks += hit_chunks
            if total_hit_chunks == total_keys:
                break
            keys = keys[hit_chunks:]

        return total_hit_chunks, block_mapping

    def get_block_mapping(
        self, chunk_infos: List[Tuple[CacheEngineKey, int, int]]
    ) -> Dict[str, List[Tuple[CacheEngineKey, int, int]]]:
        """
        Get block mapping for the given chunk infos, works by prefix match.

        :param List[Tuple[CacheEngineKey, int, int]] chunk_infos:
        List of chunk infos, each tuple contains (key, begin, end)

        :return: Dict[str, List[Tuple[CacheEngineKey, int, int]]]:
        Block mapping for the given chunk infos, each key is the backend name,
        each value is a list of chunk infos in the backend.
        """
        keys = [chunk_info[0] for chunk_info in chunk_infos]
        total_keys = len(keys)
        block_mapping = {}
        total_hit_chunks = 0
        for backend_name, backend in self.get_active_storage_backends():
            hit_chunks = backend.batched_contains(keys)
            if hit_chunks == 0:
                continue
            block_mapping[backend_name] = chunk_infos[
                total_hit_chunks : total_hit_chunks + hit_chunks
            ]
            total_hit_chunks += hit_chunks
            if total_hit_chunks == total_keys:
                break
            keys = keys[hit_chunks:]
        return block_mapping

    def touch_cache(self):
        for backend_name, backend in self.storage_backends.items():
            if backend_name == "LocalCPUBackend" or backend_name == "LocalDiskBackend":
                backend.touch_cache()

    def remove(
        self,
        key: CacheEngineKey,
        locations: Optional[List[str]] = None,
    ) -> int:
        """
        Remove the key and the corresponding cache in the specified
        locations.

        :param CacheEngineKey key: The key to remove.

        :param Optional[List[str]] locations: The range of storage backends
        to perform `remove` in.
        Should be a subset of ["LocalCPUBackend", "LocalDiskBackend"] for now.
        If None, perform `remove` in all backends.

        return: Total number of removed caches in the specified
        storage backends.
        """

        num_removed = 0
        for backend_name, backend in self.storage_backends.items():
            # TODO(Jiayi): need to handle remove in non-cpu backends
            if locations is None or backend_name in locations:
                num_removed += backend.remove(key)

        return num_removed

    def batched_remove(
        self,
        keys: List[CacheEngineKey],
        locations: Optional[List[str]] = None,
    ) -> int:
        """
        Batched remove the keys and the corresponding cache in the specified
        locations.

        :param List[CacheEngineKey] keys: The keys to remove.

        :param Optional[List[str]] locations: The range of storage backends
        to perform `remove` in.
        Should be a subset of ["LocalCPUBackend", "LocalDiskBackend"] for now.
        If None, perform `remove` in all backends.

        return: Total number of removed caches in the specified
        storage backends.
        """
        num_removed = 0
        for backend_name, backend in self.storage_backends.items():
            if locations is None or backend_name in locations:
                num_removed += backend.batched_remove(keys)

        return num_removed

    def batched_unpin(
        self,
        keys: List[CacheEngineKey],
        locations: Optional[List[str]] = None,
    ) -> None:
        """
        Unpin the keys in the specified locations.

        :param List[CacheEngineKey] keys: The keys to unpin.

        :param Optional[List[str]] locations: The range of storage backends
        to perform `unpin` in.
        Should be a subset of ["LocalCPUBackend", "LocalDiskBackend"] for now.
        If None, perform `unpin` in all backends.
        """
        for backend_name, backend in self.storage_backends.items():
            if locations is None or backend_name in locations:
                for key in keys:
                    backend.unpin(key)

    def clear(
        self,
        locations: Optional[List[str]] = None,
    ) -> int:
        """
        Clear all caches in the specified locations.

        :param Optional[List[str]] locations: The range of storage backends
        to perform `clear` in.
        Should be a subset of ["LocalCPUBackend", "LocalDiskBackend"] for now.
        If None, perform `clear` in all backends.

        return: Total number of cleared tokens in the specified
        storage backends.
        """

        num_cleared_tokens = 0
        for backend_name, backend in self.storage_backends.items():
            # TODO(Jiayi): need to handle remove in non-cpu backends
            if locations is None or backend_name in locations:
                if hasattr(backend, "clear"):
                    num_cleared_tokens += backend.clear()
                else:
                    logger.warning(
                        f"Storage backend {backend_name} does not support "
                        "clear operation. Skipping."
                    )

        return num_cleared_tokens

    def memcheck(self) -> bool:
        """
        Check the integrity of the underlying storage backend's
        memory allocators

        Returns:
            True if everything is good otherwise False
        """
        for backend in self.storage_backends.values():
            if not isinstance(backend, AllocatorBackendInterface):
                continue
            if not backend.get_memory_allocator().memcheck():
                return False
        return True

    def get_active_storage_backends(
        self,
        location: Optional[str] = None,
        search_range: Optional[List[str]] = None,
    ) -> Generator[Tuple[str, StorageBackendInterface], None, None]:
        """
        Get the active storage backends based on freeze mode and filters.

        :param Optional[str] location: If specified, only yield backends
            matching this exact name.
        :param Optional[List[str]] search_range: If specified, only yield
            backends whose names are in this list.

        :return: Generator of (backend_name, backend) tuples.
        """
        for backend_name, backend in self.storage_backends.items():
            # In freeze mode, only use local_cpu backend
            with self._freeze_lock:
                if self._freeze and backend_name != "LocalCPUBackend":
                    continue
            if location and backend_name != location:
                continue
            if search_range and backend_name not in search_range:
                continue
            yield backend_name, backend

    def get_non_allocator_backends(self) -> List[str]:
        """
        Get the names of the actual storage backends. Some backends,
        such as LocalCPUBackend and PDBackend, in some cases, only
        serve as a backend for allocation.
        """
        storage_names = []
        for backend_name, backend in self.storage_backends.items():
            if "LocalCPUBackend" == backend_name and not self.config.local_cpu:
                # if local_cpu is False, means LocalCPUBackend is only a allocator
                continue
            if "PDBackend" == backend_name and backend.pd_config.role == "sender":  # type: ignore
                # if pd_config.role is sender, means PDBackend is only a allocator
                continue
            storage_names.append(backend_name)
        return storage_names

    def close(self):
        logger.info("Closing StorageManager...")

        with self._prefetch_lock:
            for task in self.prefetch_inflight.values():
                task.state = PrefetchState.CANCELLED
            self.prefetch_inflight.clear()

        # Stop speculative workers before closing CXL/CPU backends.  Waiting
        # for admitted tasks preserves object ownership and lets their Future
        # callbacks release the bounded semaphore cleanly.
        if self._cxl_prefetch_executor is not None:
            try:
                self._cxl_prefetch_executor.shutdown(
                    wait=True,
                    cancel_futures=True,
                )
            except Exception:
                logger.exception("Failed to close CXL lookahead promotion executor")
            finally:
                self._cxl_prefetch_executor = None

        # Stop migration service if enabled
        if self.cache_migration_service is not None:
            self.cache_migration_service.stop()

        # Close all backends
        for name, backend in self.storage_backends.items():
            try:
                logger.info(f"Closing storage backend: {name}")
                backend.close()
                logger.info(f"Storage backend {name} closed successfully")
            except Exception as e:
                logger.error(f"Error closing backend {name}: {e}")

        # Stop event loop
        try:
            if self.loop.is_running():
                logger.info("Stopping event loop...")
                self.loop.call_soon_threadsafe(self.loop.stop)
                logger.info("Event loop stop signaled")
        except Exception as e:
            logger.error(f"Error stopping event loop: {e}")

        # Wait for thread with timeout
        if self.thread.is_alive():
            logger.info("Waiting for storage manager thread to finish...")
            self.thread.join(timeout=10.0)

            if self.thread.is_alive():
                logger.warning(
                    "Storage manager thread did not terminate within 10s timeout. "
                    "Proceeding with shutdown anyway."
                )
            else:
                logger.info("Storage manager thread terminated successfully")
        else:
            logger.info("Storage manager thread already stopped")

        logger.info("Storage manager closed.")
