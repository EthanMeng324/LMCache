# SPDX-License-Identifier: Apache-2.0
"""
CXL Backend using cxl_shm.c C library wrapper.
This backend maintains the Python StorageBackendInterface but uses
the C implementation from cxl_shm.c for actual memory management.
"""
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Iterator, List, Optional, Sequence
import asyncio
import ctypes
import itertools
import os
import queue
import re
import subprocess
import threading
import time

# Third Party
import ctypes
import numpy as np
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor
from lmcache.utils import (
    CacheEngineKey,
    CacheRemoveEvent,
    CacheStoreEvent,
    DiskCacheMetadata,
    _lmcache_nvtx_annotate,
)
from lmcache.v1.cache_controller.message import KVAdmitMsg, KVEvictMsg
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat, MemoryObj, MemoryObjMetadata, TensorMemoryObj
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.v1.storage_backend.cache_policy import get_cache_policy
from lmcache.v1.storage_backend.cxl_resource_governor import (
    CxlResourceGovernor,
    CxlResourceSnapshot,
    CxlSharedResourceLock,
)
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.cxl_shm_binding import CxlShmWrapper, CxlShmHnd

if TYPE_CHECKING:
    # First Party
    from lmcache.config import LMCacheEngineMetadata
    from lmcache.v1.cache_controller.worker import LMCacheWorker

logger = init_logger(__name__)


def _with_cxl_priority(method: Callable[..., Any]) -> Callable[..., Any]:
    """Run a CXL backend operation under the local serving-priority gate."""

    @wraps(method)
    def wrapped(self: "CxlBackend", *args: Any, **kwargs: Any) -> Any:
        with self._cxl_access():
            return method(self, *args, **kwargs)

    return wrapped


def _with_background_cxl_priority(method: Callable[..., Any]) -> Callable[..., Any]:
    """Make a speculative CXL read best-effort and non-blocking."""

    @wraps(method)
    def wrapped(self: "CxlBackend", *args: Any, **kwargs: Any) -> Any:
        with self.background_offload_slot() as admitted:
            if not admitted:
                self._prefetch_status.reason = "cxl_resource_busy"
                return False
            return method(self, *args, **kwargs)

    return wrapped

# Default DAX device path (from cxl_shm.c)
DEFAULT_DAX_DEVICE = "/dev/dax1.0"
# Fallback used only when extra_config.max_cxl_size is explicitly none/null.
DEFAULT_CXL_DAX_SIZE_BYTES = 1 << 36  # 64GB

# Constants from cxl_shm.h
CXL_SHM_ONAME_LEN = 20  # Maximum object name length


class CxlPutStatus(str, Enum):
    """Outcome of one synchronous CXL write attempt.

    Historically ``submit_put_task`` returned ``None`` for both success and
    failure.  That is sufficient for best-effort write-back, but not for a
    CPU->CXL offload controller which must never remove its CPU source unless
    the CXL copy is known to exist.
    """

    SUCCESS = "success"
    ALREADY_PRESENT = "already_present"
    IN_PROGRESS = "in_progress"
    DEFERRED = "deferred"
    CAPACITY_REJECTED = "capacity_rejected"
    IO_FAILED = "io_failed"


@dataclass(frozen=True, slots=True)
class CxlPutResult:
    status: CxlPutStatus
    bytes_written: int = 0
    detail: Optional[str] = None
    # True means an existing local object is usable as this operation's
    # prepare result, but still needs publish_keys() at commit time.
    needs_publish: bool = False


def _resolve_cxl_device_size_bytes(extra_config: Optional[dict[str, Any]]) -> tuple[int, bool]:
    """
    Resolve CXL device size from YAML extra_config.max_cxl_size.

    Returns:
        (size_bytes, used_default)
    """
    raw_value = None if extra_config is None else extra_config.get("max_cxl_size", None)

    # Only none/null maps to default value.
    if raw_value is None:
        return (DEFAULT_CXL_DAX_SIZE_BYTES, True)

    if isinstance(raw_value, str):
        text = raw_value.strip().lower()
        if text in ("none", "null", "~"):
            return (DEFAULT_CXL_DAX_SIZE_BYTES, True)
        if text == "":
            raise ValueError("extra_config.max_cxl_size must be a number in GB or 'none'.")
        raw_value = text

    try:
        size_gb = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid extra_config.max_cxl_size={raw_value!r}. "
            "Expected a positive number (GB) or 'none'."
        ) from exc

    if size_gb <= 0:
        raise ValueError(
            f"Invalid extra_config.max_cxl_size={raw_value!r}. "
            "Expected a positive number (GB) or 'none'."
        )

    return (int(size_gb * 1024**3), False)


class CxlWorker:
    """
    Deprecated: CxlBackend is now implemented as a synchronous backend (like LocalCPUBackend).
    This stub is kept only to avoid breaking imports in older code paths.
    """

    def __init__(self) -> None:
        raise RuntimeError("CxlWorker is no longer used; CxlBackend is synchronous.")


class CxlBackend(StorageBackendInterface):
    """
    CXL Backend using cxl_shm.c C library.
    Wraps the C implementation while maintaining Python StorageBackendInterface.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        dst_device: str = "cuda",
        lmcache_worker: Optional["LMCacheWorker"] = None,
        metadata: Optional["LMCacheEngineMetadata"] = None,
    ):
        self.cache_policy = get_cache_policy(config.cache_policy)
        self.dict = self.cache_policy.init_mutable_mapping()

        self.dst_device = dst_device
        self.local_cpu_backend = local_cpu_backend
        self.cxl_lock = threading.Lock()
        # CXL is shared by the serving path and the best-effort CPU->CXL
        # offload path.  Keep a separate local arbiter so an async offload
        # thread cannot turn into an unprioritized CXL writer.
        self.cxl_resource_governor = CxlResourceGovernor()
        self._cxl_resource_context = threading.local()
        # Per-executor-thread outcome for the bool-compatible prefetch API.
        self._prefetch_status = threading.local()
        self.chunk_size = int(config.chunk_size)

        # Track actual bytes consumed in CXL per key (includes header + padding).
        self._stored_sizes: dict[CacheEngineKey, int] = {}

        # Get configuration for C library.
        num_procs = config.extra_config.get("cxl_num_procs", 1) if config.extra_config else 1
        rank = config.extra_config.get("cxl_rank", 0) if config.extra_config else 0
        
        # Get DAX device path from config and set environment variable for C library
        dax_device = None
        if config.extra_config is not None:
            dax_device = config.extra_config.get("cxl_dax_device")
        if dax_device is None:
            dax_device = os.getenv("LMCACHE_CXL_DAX_DEVICE", DEFAULT_DAX_DEVICE)
        
        # Set environment variable for C library to use
        os.environ["LMCACHE_CXL_DAX_DEVICE"] = dax_device
        logger.info(f"Using CXL DAX device: {dax_device}")

        # Single source for CXL size:
        # - parse extra_config.max_cxl_size from YAML
        # - pass it to C layer via env before cxl_shm_init()
        cxl_size_bytes, used_default_size = _resolve_cxl_device_size_bytes(config.extra_config)
        os.environ["LMCACHE_CXL_DAX_DEVICE_SIZE"] = str(cxl_size_bytes)
        if used_default_size:
            logger.info(
                "extra_config.max_cxl_size is none/null, using default CXL size: %d bytes",
                cxl_size_bytes,
            )
        else:
            logger.info(
                "Configured CXL size from extra_config.max_cxl_size: %d bytes",
                cxl_size_bytes,
            )

        resource_lock_enabled = True
        resource_lock_dir = None
        if config.extra_config is not None:
            raw_lock_enabled = config.extra_config.get(
                "cxl_resource_lock_enabled", True
            )
            if isinstance(raw_lock_enabled, str):
                resource_lock_enabled = raw_lock_enabled.strip().lower() not in {
                    "0",
                    "false",
                    "off",
                    "no",
                }
            else:
                resource_lock_enabled = bool(raw_lock_enabled)
            resource_lock_dir = config.extra_config.get("cxl_resource_lock_dir")
        self.cxl_shared_resource_lock = CxlSharedResourceLock(
            f"{dax_device}|{cxl_size_bytes}|{num_procs}",
            lock_dir=resource_lock_dir,
            enabled=resource_lock_enabled,
        )
        if resource_lock_enabled and not self.cxl_shared_resource_lock.enabled:
            logger.warning(
                "CXL process-shared payload lock is unavailable; using the "
                "process-local serving-priority governor only"
            )

        # Initialize C library wrapper
        self.cxl_shm = CxlShmWrapper(num_procs=num_procs, rank=rank)
        result = self.cxl_shm.init()
        if result != 0:
            raise RuntimeError(f"Failed to initialize CXL shared memory: {result}")

        # Optional: auto-reset on init to recover from unclean shutdowns.
        #
        # Why this exists:
        # - If a previous process was interrupted, cxl_shm may leave metadata
        #   (in_use slots / initialized flag) behind. A new init() will then see
        #   those slots as occupied and can immediately fail with
        #   "No free slots available for shared objects", even at i=0.
        #
        # Modes:
        # - "none" (default): preserve existing CXL contents.
        # - "full": call finalize() then init() again (slow; zeros entire mapped region).
        # - "metadata": call reset_metadata() (fast; clears metadata + cursor only).
        #
        # NOTE: reset_metadata is for debugging/recovery; it makes existing objects
        # unreachable by name and will overwrite old data as new objects are created.
        reset_mode = None
        if config.extra_config is not None:
            reset_mode = config.extra_config.get("cxl_reset_on_init", None)
        if reset_mode is not None and reset_mode not in ("none", "full", "metadata"):
            logger.warning(
                "Unknown cxl_reset_on_init=%s; expected one of none/full/metadata. Ignoring.",
                reset_mode,
            )
            reset_mode = None

        if reset_mode == "full":
            if rank != 0:
                logger.warning(
                    "cxl_reset_on_init=full requested on non-zero rank=%d; skipping",
                    rank,
                )
            else:
                logger.warning(
                    "cxl_reset_on_init=full: calling cxl_shm_finalize() then re-init "
                    "(this will clear ALL existing CXL data and may be slow)"
                )
                self.cxl_shm.finalize()
                re_rc = self.cxl_shm.init()
                if re_rc != 0:
                    raise RuntimeError(
                        f"Failed to re-initialize CXL shared memory after full reset: {re_rc}"
                    )
        elif reset_mode == "metadata":
            if rank != 0:
                logger.warning(
                    "cxl_reset_on_init=metadata requested on non-zero rank=%d; skipping",
                    rank,
                )
            else:
                logger.warning(
                    "cxl_reset_on_init=metadata: resetting only CXL metadata/cursor "
                    "(existing objects become unreachable)"
                )
                md_rc = self.cxl_shm.reset_metadata()
                if md_rc != 0:
                    raise RuntimeError(f"Failed to reset CXL metadata: {md_rc}")

        # Store handles for each key
        self.key_handles: dict[CacheEngineKey, CxlShmHnd] = {}
        # Map from CacheEngineKey to CXL key (for long keys that use hash)
        self.key_to_cxl_key: dict[CacheEngineKey, str] = {}
        # Process-local duplicate suppression for synchronous writes.  The C
        # library still serializes create/destroy across processes; this set
        # prevents two local offload jobs from both reserving capacity before
        # either one has published Python metadata.
        self._put_inflight: set[CacheEngineKey] = set()
        # Track event publication independently from physical residency.  A
        # successful offload publishes immediately, while a retry can repair
        # a sink failure without allocating a duplicate native object.
        self._published_keys: set[CacheEngineKey] = set()
        # Native objects are fully written before insertion, but an explicit
        # CPU-to-CXL prepare must not be reused by a second offload before its
        # commit barrier. This local marker also lets an
        # interrupted event publication be adopted by a later prepare.
        self._prepared_keys: set[CacheEngineKey] = set()

        self.loop = loop
        self.use_local_cpu = config.local_cpu

        # Keep Python-side accounting aligned with C mapping size.
        self.max_cache_size = cxl_size_bytes
        self.current_cache_size = 0.0

        # to help maintain suffix -> prefix order in the dict
        self.keys_in_request: List[CacheEngineKey] = []

        self.lmcache_worker = lmcache_worker
        self.instance_id = config.lmcache_instance_id
        self.stats_monitor = LMCStatsMonitor.GetOrCreate()
        self.usage = 0
        self._kv_event_sink: Optional[
            Callable[[CacheStoreEvent | CacheRemoveEvent], None]
        ] = None

        # Template to infer metadata for objects created by other nodes.
        # Assumption: within a deployment, KV format/dtype/num_layers/hidden_dim
        # are consistent for this model; token count may vary (e.g., unfull chunks).
        self._kv_template_fmt: Optional[MemoryFormat] = None
        self._kv_template_dtype: Optional[torch.dtype] = None
        self._kv_template_num_layers: Optional[int] = None
        self._kv_template_hidden_dim: Optional[int] = None
        if metadata is not None:
            self.set_kv_template_from_engine_metadata(metadata)

        # Lightweight logging for "how big is one CXL object".
        # Default: log the first few writes, then every N writes.
        self._put_counter = 0
        self._put_log_first_n = 1
        self._put_log_every = 100
        if config.extra_config is not None:
            self._put_log_first_n = int(
                config.extra_config.get("cxl_log_put_first_n", self._put_log_first_n)
            )
            self._put_log_every = int(
                config.extra_config.get("cxl_log_put_every", self._put_log_every)
            )

        logger.info(f"Initialized CXL backend using cxl_shm.c (num_procs={num_procs}, rank={rank})")

    # Public hook: allow other components (e.g., StorageManager / write-back listener)
    # to provide the KV "schema" early, without requiring a prior CXL write.
    def set_kv_template_from_memory_obj(self, memory_obj: MemoryObj) -> None:
        self._maybe_update_kv_template(memory_obj)

    def set_kv_template_from_engine_metadata(
        self, metadata: "LMCacheEngineMetadata"
    ) -> None:
        """Seed shared-object reconstruction from authoritative engine metadata.

        A process-local CXL index starts empty. When a fresh replica discovers an
        object written by another replica, native metadata provides its byte sizes
        but not the tensor dtype/shape. Standard vLLM metadata is sufficient to
        reconstruct the KV_2LTD schema without requiring this replica to perform a
        local put first.
        """
        if metadata.use_mla:
            logger.warning(
                "Skipping CXL schema bootstrap from engine metadata for MLA; "
                "shared-object reconstruction currently supports KV_2LTD only."
            )
            return

        kv_shape = tuple(metadata.kv_shape)
        if len(kv_shape) != 5 or int(kv_shape[1]) != 2:
            logger.warning(
                "Skipping CXL schema bootstrap for unexpected KV shape: %s",
                kv_shape,
            )
            return

        num_layers = int(kv_shape[0])
        hidden_dim = int(kv_shape[3]) * int(kv_shape[4])
        dtype = metadata.kv_dtype
        if num_layers <= 0 or hidden_dim <= 0 or dtype is None:
            logger.warning(
                "Skipping CXL schema bootstrap for invalid engine metadata: "
                "layers=%d hidden_dim=%d dtype=%s",
                num_layers,
                hidden_dim,
                dtype,
            )
            return

        self._kv_template_fmt = MemoryFormat.KV_2LTD
        self._kv_template_dtype = dtype
        self._kv_template_num_layers = num_layers
        self._kv_template_hidden_dim = hidden_dim
        logger.info(
            "Initialized CXL shared KV schema from engine metadata: "
            "fmt=%s dtype=%s layers=%d hidden_dim=%d",
            self._kv_template_fmt,
            self._kv_template_dtype,
            self._kv_template_num_layers,
            self._kv_template_hidden_dim,
        )

    def _logical_size_bytes(self, dtype: Optional[torch.dtype], shape: torch.Size) -> int:
        if dtype is None:
            return 0
        return int(torch.tensor([], dtype=dtype).element_size()) * int(np.prod(tuple(shape)))

    def _alloc_size_bytes(self, dtype: Optional[torch.dtype], shape: torch.Size, fmt: MemoryFormat) -> int:
        """Fixed-size allocation in CXL (pad tokens to chunk_size) for KV_2LTD."""
        if dtype is None:
            return self._logical_size_bytes(dtype, shape)
        if fmt == MemoryFormat.KV_2LTD and len(shape) == 4:
            # shape: (2, num_layers, num_tokens, hidden_dim)
            num_layers = int(shape[1])
            hidden_dim = int(shape[3])
            elem = int(torch.tensor([], dtype=dtype).element_size())
            return int(2 * num_layers * self.chunk_size * hidden_dim * elem)
        return self._logical_size_bytes(dtype, shape)

    def _maybe_update_kv_template(self, memory_obj: MemoryObj) -> None:
        try:
            fmt = memory_obj.metadata.fmt
            dtype = memory_obj.metadata.dtype
            shape = tuple(memory_obj.metadata.shape)
        except Exception:
            return
        if dtype is None:
            return
        # We only infer template for KV_2LTD (most common vLLM path).
        if fmt == MemoryFormat.KV_2LTD and len(shape) == 4:
            # shape: (2, num_layers, num_tokens, hidden_dim)
            self._kv_template_fmt = fmt
            self._kv_template_dtype = dtype
            self._kv_template_num_layers = int(shape[1])
            self._kv_template_hidden_dim = int(shape[3])

    def _infer_kv_2ltd_shape_from_size(self, obj_size: int) -> Optional[torch.Size]:
        """Infer KV_2LTD shape (2, L, T, H) from raw byte size using template."""
        if (
            self._kv_template_dtype is None
            or self._kv_template_num_layers is None
            or self._kv_template_hidden_dim is None
        ):
            return None
        L = int(self._kv_template_num_layers)
        H = int(self._kv_template_hidden_dim)
        elem = int(torch.tensor([], dtype=self._kv_template_dtype).element_size())
        denom = 2 * L * H * elem
        if denom <= 0 or obj_size % denom != 0:
            return None
        T = int(obj_size // denom)
        if T <= 0:
            return None
        return torch.Size([2, L, T, H])

    def __str__(self):
        return "CxlBackend"

    @contextmanager
    def _cxl_access(self) -> Iterator[None]:
        """Enter CXL access with serving priority.

        ``background_offload_slot`` marks the current executor thread after it
        has acquired the non-blocking background permit.  Nested backend calls
        (for example ``get_blocking`` -> ``contains``) then reuse that permit
        instead of acquiring it twice.
        """

        if getattr(self._cxl_resource_context, "background", False):
            yield
            return
        with self.cxl_resource_governor.serving():
            with self.cxl_shared_resource_lock.serving():
                yield

    @contextmanager
    def background_offload_slot(self) -> Iterator[bool]:
        """Try to reserve CXL for one bounded background offload chunk.

        The context manager is intentionally non-blocking.  Callers should
        release the slot after one chunk and retry later if ``False`` is
        yielded.  This is the only supported entry point for background CXL
        writes; normal ``submit_put_task`` calls remain serving-priority.
        """

        with self.cxl_resource_governor.try_background() as local_admitted:
            if not local_admitted:
                yield False
                return

            with self.cxl_shared_resource_lock.try_background() as admitted:
                if not admitted:
                    yield False
                    return

                previous = getattr(self._cxl_resource_context, "background", False)
                self._cxl_resource_context.background = True
                try:
                    yield True
                finally:
                    self._cxl_resource_context.background = previous

    def cxl_resource_snapshot(self) -> CxlResourceSnapshot:
        """Return local serving/background arbitration counters."""

        return self.cxl_resource_governor.snapshot()

    def set_kv_event_sink(
        self, sink: Callable[[CacheStoreEvent | CacheRemoveEvent], None]
    ) -> None:
        self._kv_event_sink = sink

    def _get_cxl_key(self, key: CacheEngineKey) -> str:
        """
        Get CXL key string, handling length limit.
        If key is too long, use hash prefix.
        Caches the mapping to ensure consistency.
        """
        # Check cache first
        if key in self.key_to_cxl_key:
            return self.key_to_cxl_key[key]
        
        key_str = key.to_string()
        if len(key_str) > CXL_SHM_ONAME_LEN:
            # Use hash if too long (keep first char and hash)
            import hashlib
            key_hash = hashlib.sha256(key_str.encode()).hexdigest()
            # Use format: "H" + hash (total 17 chars, fits in 20)
            cxl_key = f"H{key_hash[:16]}"
            logger.debug(f"Key '{key_str[:20]}...' too long ({len(key_str)}), using hash: {cxl_key}")
        else:
            cxl_key = key_str
        
        # Cache the mapping
        self.key_to_cxl_key[key] = cxl_key
        return cxl_key

    @_with_cxl_priority
    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        # Multi-node access: objects may be created by other nodes, so `self.dict`
        # may not have the key yet. Use open_obj as the source of truth.
        with self.cxl_lock:
            cxl_key = self._get_cxl_key(key)
            result, hnd = self.cxl_shm.open_obj(cxl_key)
            if result != 0 or hnd is None or not hnd.obj_contents or not hnd.mapped_addr:
                # If we had a stale entry, drop it.
                self.dict.pop(key, None)
                return False
            try:
                # Both sizes must be read here: cxl_shm_close() NULLs hnd->obj,
                # so any read after the finally below would silently yield 0.
                actual_size = int(hnd.obj_contents.size)
                logical_size = int(getattr(hnd.obj_contents, "actual_size", 0))
                if actual_size <= 0 or logical_size <= 0:
                    self._published_keys.discard(key)
                    return False
                # debug-only logging removed (timing-sensitive)
            finally:
                # Do not retain handles from contains(); avoid accumulating mappings.
                try:
                    self.cxl_shm.close(hnd)
                except Exception:
                    pass

            meta = self.dict.get(key)
            if meta is None:
                # First time seeing this key (likely created by another node).
                if logical_size <= 0 or logical_size > actual_size:
                    return False

                fmt = self._kv_template_fmt or MemoryFormat.KV_2LTD
                dtype = self._kv_template_dtype
                if dtype is None:
                    return False
                if fmt != MemoryFormat.KV_2LTD:
                    return False
                inferred_shape = self._infer_kv_2ltd_shape_from_size(logical_size)
                if inferred_shape is None:
                    return False
                path = f"cxl_obj_{cxl_key}"
                self.dict[key] = DiskCacheMetadata(
                    path=path,
                    size=int(logical_size),
                    shape=inferred_shape,
                    dtype=dtype,
                    cached_positions=None,
                    fmt=fmt,
                    pin_count=0,
                    shared=True,
                )
                meta = self.dict[key]
                # Stored size is fixed (header + padded payload).
                self._stored_sizes[key] = int(actual_size)
            else:
                # If we already track it, accept as long as actual_size is >= header+logical.
                # (actual_size is fixed padded size; meta.size is logical payload size)
                if int(meta.size) > 0 and int(actual_size) < int(meta.size):
                    self.dict.pop(key, None)
                    self._stored_sizes.pop(key, None)
                    return False
                self._stored_sizes[key] = int(actual_size)

            if pin:
                meta.pin()
                self.keys_in_request.append(key)
            return True

    def is_shared_origin(self, key: CacheEngineKey) -> bool:
        # Returns True iff the entry was first discovered via CXL SHM metadata
        # without a local put — i.e. another node created it. Used by the cache
        # engine to attribute CXL hits between local-origin and shared-origin.
        with self.cxl_lock:
            meta = self.dict.get(key)
            return bool(meta is not None and getattr(meta, "shared", False))

    def touch_cache(self):
        # flip the order of the keys in the request
        with self.cxl_lock:
            for key in reversed(self.keys_in_request):
                self.cache_policy.update_on_hit(key, self.dict)
            self.keys_in_request = []

    def _destroy_native_handle(
        self,
        hnd: CxlShmHnd,
        *,
        key: Optional[CacheEngineKey] = None,
    ) -> bool:
        """Destroy one native object and close its process-local handle."""
        destroyed = False
        try:
            result = self.cxl_shm.destroy(hnd)
            destroyed = result in (None, 0)
            if not destroyed:
                logger.warning(
                    "Failed to destroy CXL object after rollback: key=%s result=%s",
                    getattr(key, "chunk_hash", key),
                    result,
                )
        except Exception:
            logger.exception(
                "Exception destroying CXL object after rollback: key=%s",
                getattr(key, "chunk_hash", key),
            )
        finally:
            try:
                self.cxl_shm.close(hnd)
            except Exception:
                logger.debug(
                    "Exception closing CXL rollback handle: key=%s",
                    getattr(key, "chunk_hash", key),
                    exc_info=True,
                )
        return destroyed

    def _destroy_owned_native_handle(self, key: CacheEngineKey) -> bool:
        """Destroy the native object owned by a completed local write."""
        with self.cxl_lock:
            hnd = self.key_handles.pop(key, None)
        if hnd is None:
            return False
        return self._destroy_native_handle(hnd, key=key)

    def _mark_native_ready(self, key: CacheEngineKey, logical_size: int) -> None:
        """Make one fully copied native object visible to CXL readers."""
        temporary = False
        with self.cxl_lock:
            hnd = self.key_handles.get(key)
        if hnd is None:
            result, hnd = self.cxl_shm.open_obj(self._get_cxl_key(key))
            if result != 0 or hnd is None or not hnd.obj_contents:
                raise RuntimeError(
                    f"CXL object disappeared before publish: {getattr(key, 'chunk_hash', key)}"
                )
            temporary = True
        try:
            native_name = bytes(hnd.obj_contents.name).split(b"\0", 1)[0].decode(
                "utf-8", errors="ignore"
            )
            if not hnd.obj_contents.in_use or native_name != self._get_cxl_key(key):
                raise RuntimeError(
                    f"CXL object handle is stale before publish: "
                    f"expected={self._get_cxl_key(key)} actual={native_name}"
                )
            native_size = int(hnd.obj_contents.size)
            if logical_size <= 0 or logical_size > native_size:
                raise RuntimeError(
                    f"Invalid CXL logical size for {getattr(key, 'chunk_hash', key)}: "
                    f"logical={logical_size} native={native_size}"
                )
            current_size = int(getattr(hnd.obj_contents, "actual_size", 0))
            if current_size == logical_size:
                return
            if current_size > 0:
                raise RuntimeError(
                    f"CXL object has conflicting logical size for "
                    f"{getattr(key, 'chunk_hash', key)}: "
                    f"current={current_size} requested={logical_size}"
                )
            result = self.cxl_shm.set_actual_size(hnd, logical_size)
            if result not in (None, 0):
                raise RuntimeError(
                    f"Failed to publish CXL object for {getattr(key, 'chunk_hash', key)}: "
                    f"{result}"
                )
        finally:
            if temporary:
                try:
                    self.cxl_shm.close(hnd)
                except Exception:
                    pass

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        # Synchronous backend: no async put tasks.
        return False

    def pin(
        self,
        key: CacheEngineKey,
    ) -> bool:
        with self.cxl_lock:
            if key in self.dict:
                self.dict[key].pin()
                return True
            else:
                return False

    def unpin(
        self,
        key: CacheEngineKey,
    ) -> bool:
        with self.cxl_lock:
            if key in self.dict:
                self.dict[key].unpin()
                return True
            else:
                return False

    @_with_cxl_priority
    def remove(
        self,
        key: CacheEngineKey,
        force: bool = True,
    ) -> bool:
        if force:
            self.cxl_lock.acquire()

        if not (meta := self.dict.pop(key, None)):
            self._prepared_keys.discard(key)
            self._published_keys.discard(key)
            cxl_key = self._get_cxl_key(key)
            hnd = self.key_handles.pop(key, None)
            if hnd is None:
                result, tmp_hnd = self.cxl_shm.open_obj(cxl_key)
                if result == 0 and tmp_hnd is not None:
                    hnd = tmp_hnd
            destroyed = (
                self._destroy_native_handle(hnd, key=key)
                if hnd is not None
                else False
            )
            if force:
                self.cxl_lock.release()
            return destroyed

        # Prepared CPU->CXL objects are deliberately absent from the global
        # admission stream.  Removing one during a transaction abort must not
        # emit a remove event for an object that was never admitted.  A cold
        # replica reconstructed from shared metadata is still considered
        # externally visible and must retain the normal eviction events.
        was_published = key in self._published_keys or bool(
            getattr(meta, "shared", False)
        )
        # meta.size is the logical payload size (bytes). Actual bytes consumed in CXL
        # may include header + padding; track via _stored_sizes.
        stored_size = int(self._stored_sizes.pop(key, int(meta.size)))
        was_prepared = key in self._prepared_keys
        self._prepared_keys.discard(key)
        cxl_key = self._get_cxl_key(key)
        
        # Destroy object in C library.
        #
        # IMPORTANT: key_handles is not a source of truth. If we only destroy when
        # a cached handle exists, we can end up with "dict removed but CXL object
        # still present", which later shows up as open_obj succeeding while dict
        # misses. Always attempt to destroy the CXL object.
        hnd = self.key_handles.pop(key, None)
        if hnd is None:
            result, tmp_hnd = self.cxl_shm.open_obj(cxl_key)
            if result == 0 and tmp_hnd is not None:
                hnd = tmp_hnd

        destroyed = True
        if hnd is not None:
            try:
                destroy_result = self.cxl_shm.destroy(hnd)
                destroyed = destroy_result in (None, 0)
                if not destroyed:
                    logger.error(
                        "Failed to destroy CXL object during remove: key=%s result=%s",
                        getattr(key, "chunk_hash", key),
                        destroy_result,
                    )
            except Exception:
                destroyed = False
                logger.exception(
                    "Exception destroying CXL object during remove: key=%s",
                    getattr(key, "chunk_hash", key),
                )
            finally:
                # close best-effort to avoid leaking mappings
                try:
                    self.cxl_shm.close(hnd)
                except Exception:
                    pass

        if not destroyed:
            # Do not report a successful removal, or drop Python accounting, if
            # the native object is still potentially reachable.  Keeping the
            # metadata here lets a later retry reopen and destroy the object;
            # otherwise an abort would turn a failed native delete into an
            # invisible CXL allocation.
            self.dict[key] = meta
            self._stored_sizes[key] = stored_size
            if was_published:
                self._published_keys.add(key)
            if was_prepared:
                self._prepared_keys.add(key)
            if force:
                self.cxl_lock.release()
            return False

        self._published_keys.discard(key)

        self.usage -= stored_size
        self.stats_monitor.update_local_storage_usage(self.usage)
        self.current_cache_size -= stored_size

        if force:
            self.cache_policy.update_on_force_evict(key)
            self.cxl_lock.release()

        # push kv evict msg
        if was_published and self.lmcache_worker is not None:
            self.lmcache_worker.put_msg(
                KVEvictMsg(self.instance_id, key.worker_id, key.chunk_hash, str(self))
            )
        if was_published and self._kv_event_sink is not None:
            self._kv_event_sink(
                CacheRemoveEvent(block_hashes=[key.chunk_hash], medium="CXL")
            )

        return True

    @_with_cxl_priority
    def native_exists(self, key: CacheEngineKey) -> bool:
        """Check native CXL residency without requiring Python metadata."""
        with self.cxl_lock:
            result, hnd = self.cxl_shm.open_obj(self._get_cxl_key(key))
            if result != 0 or hnd is None or not hnd.obj_contents:
                return False
            try:
                return bool(hnd.obj_contents.in_use)
            finally:
                try:
                    self.cxl_shm.close(hnd)
                except Exception:
                    pass

    def insert_key(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        *,
        publish_events: bool = True,
    ) -> None:
        # Learn template for cross-node reconstruction.
        self._maybe_update_kv_template(memory_obj)
        shape = memory_obj.metadata.shape
        dtype = memory_obj.metadata.dtype
        fmt = memory_obj.metadata.fmt
        logical_size = self._logical_size_bytes(dtype, shape)
        if logical_size <= 0:
            logical_size = int(memory_obj.get_physical_size())

        has_stored = False
        with self.cxl_lock:
            # Need to do reinsert to update cache recency
            if key in self.dict:
                self.dict.pop(key)
                has_stored = True

            # Use key string as path identifier
            cxl_key = self._get_cxl_key(key)
            path = f"cxl_obj_{cxl_key}"
            # Use keyword args to avoid positional mistakes.
            self.dict[key] = DiskCacheMetadata(
                path=path,
                size=int(logical_size),
                shape=shape,
                dtype=dtype,
                cached_positions=None,
                fmt=fmt,
                pin_count=0,
                shared=False,
            )
            # stored size (padding) is tracked separately for capacity accounting
            self._stored_sizes[key] = int(self._alloc_size_bytes(dtype, shape, fmt))
            if publish_events:
                self._prepared_keys.discard(key)
            else:
                self._prepared_keys.add(key)

        if publish_events and not has_stored:
            self.publish_keys([key])

    def publish_keys(self, keys: Sequence[CacheEngineKey]) -> tuple[CacheEngineKey, ...]:
        """Publish already-written CXL keys after a TP coordinator barrier."""
        published: list[CacheEngineKey] = []
        for key in dict.fromkeys(keys):
            with self.cxl_lock:
                if key not in self.dict:
                    continue
                if key in self._published_keys:
                    published.append(key)
                    continue
                logical_size = int(self.dict[key].size)
            try:
                # The native metadata remains actual_size=0 through prepare.
                # Only the commit path makes the object discoverable remotely.
                self._mark_native_ready(key, logical_size)
                with self.cxl_lock:
                    self._published_keys.add(key)
                if self.lmcache_worker is not None:
                    self.lmcache_worker.put_msg(
                        KVAdmitMsg(
                            self.instance_id,
                            key.worker_id,
                            key.chunk_hash,
                            str(self),
                        )
                    )
                if self._kv_event_sink is not None:
                    # cache_engine enriches this with parent/tokens using the
                    # still-resident CPU source metadata.
                    self._kv_event_sink(
                        CacheStoreEvent(
                            block_hashes=[key.chunk_hash],
                            parent_block_hash=None,
                            token_ids=[],
                            block_size=int(self.chunk_size),
                            lora_id=None,
                            medium="CXL",
                        )
                    )
                with self.cxl_lock:
                    self._prepared_keys.discard(key)
                published.append(key)
            except Exception:
                with self.cxl_lock:
                    self._published_keys.discard(key)
                raise
        return tuple(published)

    @_with_cxl_priority
    def submit_put_task(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        *,
        publish_events: bool = True,
    ) -> CxlPutResult:
        assert memory_obj.tensor is not None
        existing = False
        already_published = False
        existing_is_prepared = False
        existing_is_shared = False
        with self.cxl_lock:
            if key in self.dict:
                existing = True
                already_published = key in self._published_keys
                existing_is_prepared = key in self._prepared_keys
                existing_is_shared = bool(
                    getattr(self.dict[key], "shared", False)
                )
            if key in self._put_inflight:
                return CxlPutResult(CxlPutStatus.IN_PROGRESS)
            if not existing:
                self._put_inflight.add(key)
        if existing:
            if existing_is_prepared:
                return CxlPutResult(
                    CxlPutStatus.IN_PROGRESS,
                    detail="CXL object is prepared by another offload operation",
                )
            if already_published:
                return CxlPutResult(CxlPutStatus.ALREADY_PRESENT)
            if not publish_events:
                return CxlPutResult(
                    CxlPutStatus.ALREADY_PRESENT,
                    needs_publish=not existing_is_shared,
                )
            if existing_is_shared:
                return CxlPutResult(CxlPutStatus.ALREADY_PRESENT)
            # A previous writer may have completed the native write but failed
            # while notifying the event sink. Retry the publication before
            # acknowledging an idempotent write; the router must not be told
            # that a CXL copy is ready solely because the object exists.
            try:
                self.publish_keys([key])
                return CxlPutResult(CxlPutStatus.ALREADY_PRESENT)
            except Exception as exc:
                logger.exception(
                    "CXL event publication failed for existing key=%s", key
                )
                return CxlPutResult(CxlPutStatus.IO_FAILED, detail=str(exc))

        # Logical size written by caller (no allocator tail bytes).
        shape = memory_obj.metadata.shape
        dtype = memory_obj.metadata.dtype
        fmt = memory_obj.metadata.fmt
        payload_size = self._logical_size_bytes(dtype, shape)
        if payload_size <= 0:
            payload_size = int(memory_obj.get_physical_size())
        alloc_size = self._alloc_size_bytes(dtype, shape, fmt)
        if alloc_size <= 0:
            alloc_size = payload_size
        required_size = int(alloc_size)

        # Log object size / token count for capacity planning.
        # This is the *actual* per-object size passed to cxl_shm_create().
        self._put_counter += 1
        do_log = (self._put_counter <= self._put_log_first_n) or (
            self._put_log_every > 0 and self._put_counter % self._put_log_every == 0
        )
        if do_log:
            try:
                num_tokens = memory_obj.get_num_tokens()
            except Exception:
                num_tokens = -1
            try:
                fmt = memory_obj.metadata.fmt
                dtype = memory_obj.metadata.dtype
                shape = tuple(memory_obj.metadata.shape)
            except Exception:
                fmt = None
                dtype = None
                shape = None
            logger.info(
                "cxl_put[%d]: key=%s cxl_key=%s tokens=%s phy_bytes=%d (%.2f MiB) "
                "fmt=%s dtype=%s shape=%s",
                self._put_counter,
                getattr(key, "chunk_hash", key),
                self._get_cxl_key(key),
                num_tokens,
                int(required_size),
                float(required_size) / 1024.0 / 1024.0,
                fmt,
                dtype,
                shape,
            )
        reserved = False
        write_succeeded = False
        usage_added = False
        try:
            with self.cxl_lock:
                while self.current_cache_size + required_size > self.max_cache_size:
                    evict_keys = self.cache_policy.get_evict_candidates(
                        self.dict, num_candidates=1
                    )
                    if not evict_keys:
                        logger.warning("No eviction candidates found. CXL space under pressure.")
                        return CxlPutResult(CxlPutStatus.CAPACITY_REJECTED)

                    # remove(..., force=False) owns the single accounting
                    # decrement.  Do not pre-decrement here: doing both made
                    # current_cache_size drift below actual CXL usage.
                    self.batched_remove(evict_keys, force=False)
                self.current_cache_size += required_size
                reserved = True

            self.cache_policy.update_on_put(key)
            buffer = memory_obj.byte_array
            self.write_cxl_mmap(
                key, buffer, payload_size=payload_size, alloc_size=alloc_size
            )
            write_succeeded = True

            self.usage += required_size
            usage_added = True
            self.stats_monitor.update_local_storage_usage(self.usage)
            self.insert_key(key, memory_obj, publish_events=publish_events)
            return CxlPutResult(CxlPutStatus.SUCCESS, bytes_written=required_size)
        except Exception as exc:
            logger.exception("CXL put failed for key=%s", getattr(key, "chunk_hash", key))
            cleaned = False

            # Once write_cxl_mmap() returned, this call owns the native object.
            # If insert_key() or event publication fails, remove the Python
            # metadata and destroy that object instead of letting contains()
            # turn a failed write into ALREADY_PRESENT.
            if write_succeeded:
                try:
                    with self.cxl_lock:
                        inserted = key in self.dict
                    if inserted:
                        cleaned = self.remove(key, force=True)
                    if not cleaned:
                        cleaned = self._destroy_owned_native_handle(key)
                except Exception:
                    logger.exception(
                        "Failed to roll back CXL object after put failure for key=%s",
                        getattr(key, "chunk_hash", key),
                    )
                    try:
                        cleaned = self._destroy_owned_native_handle(key) or cleaned
                    except Exception:
                        logger.exception(
                            "Failed to destroy owned CXL handle for key=%s",
                            getattr(key, "chunk_hash", key),
                        )

            if cleaned:
                # remove() owns both accounting decrements when metadata was
                # inserted.  Do not subtract the reservation a second time.
                usage_added = False
                reserved = False
            else:
                if usage_added:
                    self.usage = max(0, self.usage - required_size)
                    self.stats_monitor.update_local_storage_usage(self.usage)
                if reserved:
                    with self.cxl_lock:
                        self.current_cache_size = max(
                            0.0, self.current_cache_size - required_size
                        )

            # A concurrent process can win the native create race after this
            # process performed its local precheck.  Only that pre-create
            # failure path is allowed to use contains() for deduplication.
            if not write_succeeded:
                try:
                    if self.contains(key):
                        return CxlPutResult(CxlPutStatus.ALREADY_PRESENT)
                except Exception:
                    logger.debug(
                        "Unable to inspect concurrent CXL writer for key=%s",
                        getattr(key, "chunk_hash", key),
                        exc_info=True,
                    )
            return CxlPutResult(CxlPutStatus.IO_FAILED, detail=str(exc))
        finally:
            with self.cxl_lock:
                self._put_inflight.discard(key)

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec=None,
    ) -> None:
        for key, memory_obj in zip(keys, memory_objs, strict=False):
            self.submit_put_task(key, memory_obj)

    @_with_background_cxl_priority
    def submit_prefetch_task(
        self,
        key: CacheEngineKey,
        *,
        task_id: Optional[str] = None,
        ttl_ms: Optional[int] = None,
    ) -> bool:
        # Synchronous prefetch: best-effort read from CXL and cache into LocalCPUBackend.
        #
        # IMPORTANT: do not busy-loop on LocalCPU allocation here; under memory pressure
        # it can stall the engine. If LocalCPU can't allocate, we just skip prefetch.
        # Keep the reason on the worker thread so StorageManager can expose a
        # useful completion counter without changing this long-standing bool API.
        status = self._prefetch_status
        status.reason = None

        def fail(reason: str) -> bool:
            status.reason = reason
            return False

        if not self.use_local_cpu:
            return fail("backend_unavailable")
        if self.local_cpu_backend is None:
            return fail("backend_unavailable")
        # A demand path may have filled the target between admission and
        # execution.  Avoid rereading the shared object for an already-hot key.
        if self.local_cpu_backend.contains(key):
            return True
        # ``dict`` is process-local.  A shared CXL mapping can contain an
        # object created by another worker, so always let ``contains`` hydrate
        # metadata before taking the read path.  This also makes the direct
        # backend API safe (the engine promotion path already calls contains).
        if not self.contains(key, pin=False):
            return fail("missing_cxl")
        with self.cxl_lock:
            self.cache_policy.update_on_hit(key, self.dict)
            cxl_meta = self.dict.get(key)
            if cxl_meta is None:
                return fail("metadata_missing")
            dtype = cxl_meta.dtype
            shape = cxl_meta.shape
            fmt = cxl_meta.fmt

        if dtype is None or shape is None:
            return fail("metadata_missing")

        tmp_obj = self.load_bytes_from_cxl(key, dtype=dtype, shape=shape, fmt=fmt)
        if tmp_obj is None or tmp_obj.tensor is None:
            if tmp_obj is not None:
                tmp_obj.ref_count_down()
            return fail("cxl_read_failed")

        cached_obj = self.local_cpu_backend.allocate(
            shape,
            dtype,
            fmt,
            # Speculation is never allowed to evict demand-resident CPU KV.
            eviction=False,
            busy_loop=False,
        )
        if cached_obj is None or cached_obj.tensor is None:
            if cached_obj is not None:
                cached_obj.ref_count_down()
            tmp_obj.ref_count_down()
            return fail("cpu_capacity_rejected")

        try:
            cached_obj.tensor.copy_(tmp_obj.tensor, non_blocking=True)
            expire_ns = None
            if ttl_ms is not None and int(ttl_ms) > 0:
                expire_ns = time.time_ns() + int(ttl_ms) * 1_000_000
            if hasattr(self.local_cpu_backend, "submit_prefetch_put_task"):
                self.local_cpu_backend.submit_prefetch_put_task(
                    key, cached_obj, expire_ns=expire_ns
                )
            else:
                self.local_cpu_backend.submit_put_task(key, cached_obj)
        except Exception:
            # Both objects are owned by this function until the backend admits
            # cached_obj.  Always balance those references, including a failed
            # tensor copy or backend admission.
            status.reason = "cpu_admission_failed"
            logger.debug(
                "Reactive CXL prefetch admission failed for key %s",
                getattr(key, "chunk_hash", key),
                exc_info=True,
            )
            return False
        finally:
            # Release local refs; a successful backend admission keeps its own
            # reference to cached_obj.
            cached_obj.ref_count_down()
            tmp_obj.ref_count_down()
        return True

    def get_prefetch_failure_reason(self) -> Optional[str]:
        """Return the last failure reason on the calling worker thread."""
        status = getattr(self, "_prefetch_status", None)
        return None if status is None else getattr(status, "reason", None)

    @_with_cxl_priority
    def get_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        """Blocking get function."""
        # contains() acquires cxl_lock and, for a cold replica, reconstructs the
        # process-local metadata from the native shared object. Calling it while
        # already holding this non-reentrant lock deadlocks on the cold path.
        if not self.contains(key, pin=False):
            return None

        with self.cxl_lock:
            cxl_meta = self.dict.get(key)
            if cxl_meta is None:
                return None
            self.cache_policy.update_on_hit(key, self.dict)

            dtype = cxl_meta.dtype
            shape = cxl_meta.shape
            fmt = cxl_meta.fmt
            assert dtype is not None
            assert shape is not None

            return self.load_bytes_from_cxl(
                key, dtype=dtype, shape=shape, fmt=fmt
            )

    def _batched_contains_prefix_blocking(
        self,
        keys: list[CacheEngineKey],
        pin: bool,
    ) -> int:
        """Blocking helper for async prefix-contains.

        Returns the number of *contiguous prefix* keys that exist in CXL.
        """
        num_hit_chunks = 0
        for key in keys:
            # NOTE: prefix-based counting (stop at first miss), consistent with
            # LocalCPUBackend/LocalDiskBackend async loading semantics.
            if not self.contains(key, pin=pin):
                return num_hit_chunks
            num_hit_chunks += 1
        return num_hit_chunks

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        """Async prefix-contains for CXL.

        We wrap the blocking `contains()` (which calls into the C library) in a
        worker thread to avoid blocking the storage manager event loop.
        """
        # lookup_id is currently unused for CXL.
        _ = lookup_id
        # asyncio.to_thread() returns an awaitable executed in the default pool.
        return await asyncio.to_thread(
            self._batched_contains_prefix_blocking, list(keys), pin
        )

    def _batched_get_prefix_blocking(
        self,
        keys: list[CacheEngineKey],
    ) -> list[MemoryObj]:
        """Blocking helper for async batched_get_non_blocking.

        Best-effort: if a key disappears after lookup/pin, stop early and return
        the successfully loaded prefix. This matches the engine's contiguous-hit
        expectation (break at first missing chunk).
        """
        mem_objs: list[MemoryObj] = []
        for key in keys:
            mem_obj = self.get_blocking(key)
            if mem_obj is None:
                break
            mem_objs.append(mem_obj)
        return mem_objs

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> list[MemoryObj]:
        """Async CXL prefetch.

        Loads KV from CXL into pinned CPU staging buffers (see `load_bytes_from_cxl`)
        using a background thread so this call can run concurrently with other
        backends under `async_lookup_and_prefetch()`.
        """
        _ = lookup_id
        _ = transfer_spec
        return await asyncio.to_thread(self._batched_get_prefix_blocking, list(keys))

    def get_non_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[Future]:
        """Non-blocking get function."""
        raise NotImplementedError(
            "Non-blocking get is not implemented for CxlBackend."
        )

    def get_allocator_backend(self):
        return self.local_cpu_backend

    def load_bytes_from_cxl(
        self,
        key: CacheEngineKey,
        dtype: torch.dtype,
        shape: torch.Size,
        fmt: MemoryFormat,
    ) -> Optional[MemoryObj]:
        """Load bytes from CXL into a pinned CPU MemoryObj (staging buffer).

        IMPORTANT: vLLM's GPU connector expects host tensors to be pinned/registered.
        A tensor created via `torch.frombuffer()` on a DAX-mapped address is not
        guaranteed to be pinned/registered, and can crash with:
          "Host tensor not registered/pinned (or bad ptr)".

        Therefore, we always copy CXL -> pinned CPU tensor here.

        NOTE:
        We intentionally do NOT allocate this staging buffer from LocalCPUBackend's
        internal allocator pool. Under heavy memory pressure, `LocalCPUBackend.allocate`
        can busy-loop on eviction (retrieve path), stalling the engine. Using a
        standalone pinned tensor avoids coupling CXL reads to the L1 cache capacity.

        ----
        Historical note / future optimization:
        We previously tried a "zero-copy" path by constructing a CPU tensor that
        directly views the DAX-mapped CXL region (via `torch.frombuffer()` on a
        ctypes array created from the mapped address), and then wrapping it in a
        `MemoryObj` to pass downstream.

        That approach is kept below as commented reference, but it is NOT safe
        with the current vLLM integration because the downstream CUDA transfer
        path uses custom kernels that assume the host pointer is registered
        (pinned) for DMA. DAX-backed mappings are not automatically registered,
        so the kernel fails at runtime.

        TODO(future):
        - Implement a true GPU-direct read path (GPU DMA reading from CXL/DAX),
          e.g. via GPUDirect RDMA / DMA-BUF / a driver-assisted path, so we can
          avoid the extra CXL->CPU copy and the CPU pinned-memory requirement.
        - If we keep a CPU-staging path, consider explicit host registration
          (cudaHostRegister) for the mapped region only if it is supported and
          safe for DAX (alignment, lifetime, unregistration, and security).

        Reference "zero-copy" sketch (DO NOT USE as-is):

            # cxl_key = self._get_cxl_key(key)
            # result, hnd = self.cxl_shm.open_obj(cxl_key)
            # if result != 0 or hnd is None:
            #     return None
            # if not hnd.obj_contents or not hnd.mapped_addr:
            #     return None
            # size = hnd.obj_contents.size
            # mapped_addr = hnd.mapped_addr
            #
            # addr_int = ctypes.cast(mapped_addr, ctypes.c_void_p).value
            # if addr_int is None:
            #     return None
            # self.cxl_shm.flush(ctypes.c_void_p(addr_int), size)
            #
            # array_type = ctypes.c_uint8 * size
            # cxl_array = array_type.from_address(addr_int)
            # cxl_tensor = torch.frombuffer(cxl_array, dtype=torch.uint8)
            # num_elements = size // dtype.itemsize
            # cxl_tensor = cxl_tensor[: num_elements * dtype.itemsize].view(dtype).view(shape)
            #
            # metadata = MemoryObjMetadata(
            #     shape=shape,
            #     dtype=dtype,
            #     address=0,
            #     phy_size=size,
            #     ref_count=1,
            #     pin_count=0,
            #     fmt=fmt,
            # )
            # memory_obj = TensorMemoryObj(raw_data=cxl_tensor, metadata=metadata, parent_allocator=None)
            # self.key_handles[key] = hnd
            # return memory_obj
        """
        cxl_key = self._get_cxl_key(key)
        result, hnd = self.cxl_shm.open_obj(cxl_key)
        if result != 0 or hnd is None or not hnd.obj_contents or not hnd.mapped_addr:
            # Debug-friendly miss reason (kept at DEBUG to avoid spam).
            logger.debug(
                "CXL load: open_obj failed: key=%s cxl_key=%s result=%s hnd=%s has_obj=%s has_addr=%s",
                getattr(key, "chunk_hash", key),
                cxl_key,
                result,
                "None" if hnd is None else "OK",
                bool(getattr(hnd, "obj_contents", None)),
                bool(getattr(hnd, "mapped_addr", None)),
            )
            return None

        try:
            obj_size = int(hnd.obj_contents.size)
            if obj_size <= 0:
                logger.debug(
                    "CXL load: obj_size<=0: key=%s cxl_key=%s obj_size=%d",
                    getattr(key, "chunk_hash", key),
                    cxl_key,
                    obj_size,
                )
                return None

            # Compute expected logical payload size from dtype/shape (for debug).
            logical_size = int(torch.tensor([], dtype=dtype).element_size()) * int(
                np.prod(tuple(shape))
            )

            # New actual_size path: take logical payload size from C metadata when present.
            meta_actual = int(getattr(hnd.obj_contents, "actual_size", 0))
            if meta_actual <= 0:
                # actual_size=0 is the native prepare marker. Never read an
                # object whose payload has not crossed the CXL commit point.
                return None
            logical_size = meta_actual

            # debug-only logging removed (timing-sensitive)

            if obj_size < logical_size:
                logger.warning(
                    "CXL object size too small: key=%s cxl_key=%s obj_size=%d logical=%d",
                    getattr(key, "chunk_hash", key),
                    cxl_key,
                    obj_size,
                    logical_size,
                )
                return None

            # Allocate pinned CPU memory on GPU-capable workers.  A CUDA build
            # of torch raises when pin_memory=True on a CPU-only host, so use a
            # pageable staging buffer there (not a production GPU data path).
            # Pinned allocation pressure on GPU workers remains a cache miss.
            try:
                raw_data = torch.empty(
                    logical_size,
                    dtype=torch.uint8,
                    device="cpu",
                    pin_memory=torch.cuda.is_available(),
                )
            except Exception:
                # This is a common intermittent failure mode under host-pinned pressure.
                # Keep at WARNING (with stack) so we can confirm root-cause quickly.
                logger.exception(
                    "CXL load: failed to allocate pinned staging buffer: key=%s cxl_key=%s size=%d obj_size=%d meta_actual=%d shape=%s dtype=%s",
                    getattr(key, "chunk_hash", key),
                    cxl_key,
                    logical_size,
                    obj_size,
                    meta_actual,
                    tuple(shape),
                    str(dtype),
                )
                return None

            # Flush before read to avoid stale cache lines.
            self.cxl_shm.flush(ctypes.c_void_p(hnd.mapped_addr), logical_size)

            # Copy CXL -> pinned CPU buffer directly (no torch.frombuffer).
            try:
                ctypes.memmove(
                    ctypes.c_void_p(raw_data.data_ptr()),
                    ctypes.c_void_p(int(hnd.mapped_addr)),
                    logical_size,
                )
            except Exception:
                logger.exception(
                    "CXL load: memmove failed: key=%s cxl_key=%s size=%d obj_size=%d meta_actual=%d",
                    getattr(key, "chunk_hash", key),
                    cxl_key,
                    logical_size,
                    obj_size,
                    meta_actual,
                )
                return None

            metadata = MemoryObjMetadata(
                shape=shape,
                dtype=dtype,
                address=0,
                phy_size=logical_size,
                ref_count=1,
                pin_count=0,
                fmt=fmt,
            )
            return TensorMemoryObj(raw_data=raw_data, metadata=metadata, parent_allocator=None)
        finally:
            # Best-effort close to avoid accumulating mappings/handles.
            try:
                self.cxl_shm.close(hnd)
            except Exception:
                pass

    def write_cxl_mmap(
        self,
        key: CacheEngineKey,
        buffer: bytearray,
        payload_size: int,
        alloc_size: int,
    ):
        """Write buffer to CXL memory using C library."""
        start_time = time.time()
        size = int(alloc_size)
        cxl_key = self._get_cxl_key(key)
        
        hnd = None
        stored_handle = False
        try:
            # Create object in C library.  The object is owned by this method
            # until the complete payload has been copied and flushed.
            result, hnd = self.cxl_shm.create(
                cxl_key, size, actual_size=0
            )
            if result != 0 or hnd is None:
                raise RuntimeError(
                    f"Failed to create CXL object for key {cxl_key}: {result}"
                )

            if not hnd.mapped_addr:
                raise RuntimeError(f"CXL object has no mapped address for key {cxl_key}")

            # Write data to mapped address. The native object remains at
            # actual_size=0 until publish_keys() marks the commit; it is
            # destroyed on every exception after create(), including a failed
            # flush.
            mapped_addr = hnd.mapped_addr
            # Layout: [payload][zero padding]
            #
            # Prefer zero-copy from a writable buffer (memoryview/bytearray) to avoid
            # an extra Python-side bytes() materialization.
            try:
                mv = buffer if isinstance(buffer, memoryview) else memoryview(buffer)
                src_arr = (ctypes.c_uint8 * payload_size).from_buffer(mv)  # zero-copy
                src_ptr = ctypes.cast(src_arr, ctypes.c_void_p)
                ctypes.memmove(ctypes.c_void_p(int(mapped_addr)), src_ptr, payload_size)
                # zero padding tail (if any)
                tail = int(size) - int(payload_size)
                if tail > 0:
                    ctypes.memset(
                        ctypes.c_void_p(int(mapped_addr) + int(payload_size)),
                        0,
                        tail,
                    )
            except (TypeError, BufferError):
                # Fallback: materialize bytes (extra copy) if buffer isn't writable/compatible.
                buffer_bytes = bytes(buffer)
                ctypes.memmove(
                    ctypes.c_void_p(int(mapped_addr)),
                    buffer_bytes,
                    len(buffer_bytes),
                )
                tail = int(size) - int(len(buffer_bytes))
                if tail > 0:
                    ctypes.memset(
                        ctypes.c_void_p(int(mapped_addr) + int(len(buffer_bytes))),
                        0,
                        tail,
                    )

            # Flush after write to guarantee persistence / visibility.
            flush_result = self.cxl_shm.flush(ctypes.c_void_p(mapped_addr), size)
            if flush_result not in (None, 0):
                raise RuntimeError(
                    f"Failed to flush CXL object for key {cxl_key}: {flush_result}"
                )

            # Store handle only after the complete payload write succeeds.
            with self.cxl_lock:
                self.key_handles[key] = hnd
                stored_handle = True
        except Exception:
            if stored_handle:
                with self.cxl_lock:
                    owned_hnd = self.key_handles.pop(key, None)
                if owned_hnd is not None:
                    hnd = owned_hnd
            if hnd is not None:
                self._destroy_native_handle(hnd, key=key)
            raise

        cxl_write_time = time.time() - start_time
        logger.debug(
            f"CXL write size: {size} bytes, "
            f"Bandwidth: {size / cxl_write_time / 1e6:.2f} MB/s"
        )

    def read_cxl_mmap(self, key: CacheEngineKey, buffer: bytearray):
        """Read buffer from CXL memory using C library."""
        start_time = time.time()
        size = len(buffer)
        cxl_key = self._get_cxl_key(key)
        
        # Open object in C library
        result, hnd = self.cxl_shm.open_obj(cxl_key)
        if result != 0 or hnd is None:
            raise ValueError(f"Key {cxl_key} not found in CXL")
        
        if not hnd.mapped_addr:
            raise ValueError(f"Invalid handle for key {cxl_key}")
        
        mapped_addr = hnd.mapped_addr
        obj_size = hnd.obj_contents.size if hnd.obj_contents else size
        
        if obj_size != size:
            raise ValueError(f"Size mismatch: expected {size}, got {obj_size}")
        
        # Flush before read to avoid stale cache lines.
        self.cxl_shm.flush(ctypes.c_void_p(mapped_addr), size)

        # Copy from mapped memory to the provided buffer (zero-copy destination).
        mv = buffer if isinstance(buffer, memoryview) else memoryview(buffer)
        dst_arr = (ctypes.c_uint8 * size).from_buffer(mv)
        ctypes.memmove(dst_arr, ctypes.c_void_p(mapped_addr), size)

        cxl_read_time = time.time() - start_time
        logger.debug(
            f"CXL read size: {size} bytes, "
            f"Bandwidth: {size / cxl_read_time / 1e6:.2f} MB/s"
        )

    def close(self) -> None:
        # Close all handles
        with self.cxl_lock:
            for key, hnd in list(self.key_handles.items()):
                try:
                    self.cxl_shm.close(hnd)
                except Exception as e:
                    logger.warning(f"Error closing handle for key {key}: {e}")
            self.key_handles.clear()
            self._stored_sizes.clear()
            self._prepared_keys.clear()

        # Finalize C library
        if hasattr(self, "cxl_shm"):
            try:
                self.cxl_shm.finalize()
            except Exception as e:
                logger.warning(f"Error finalizing CXL shared memory: {e}")

        # Synchronous backend: no worker to close.
