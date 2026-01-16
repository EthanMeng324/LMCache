# SPDX-License-Identifier: Apache-2.0
"""
CXL Backend using cxl_shm.c C library wrapper.
This backend maintains the Python StorageBackendInterface but uses
the C implementation from cxl_shm.c for actual memory management.
"""
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence
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
from lmcache.utils import CacheEngineKey, DiskCacheMetadata, _lmcache_nvtx_annotate
from lmcache.v1.cache_controller.message import KVAdmitMsg, KVEvictMsg
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat, MemoryObj, MemoryObjMetadata, TensorMemoryObj
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.v1.storage_backend.cache_policy import get_cache_policy
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.cxl_shm_binding import CxlShmWrapper, CxlShmHnd

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_controller.worker import LMCacheWorker

logger = init_logger(__name__)

# Default DAX device path (from cxl_shm.c)
DEFAULT_DAX_DEVICE = "/dev/dax1.0"

# Constants from cxl_shm.h
CXL_SHM_ONAME_LEN = 20  # Maximum object name length


class CxlWorker:
    def __init__(self) -> None:
        self.pq: queue.PriorityQueue[tuple[int, int, str, Callable, dict[str, Any]]] = (
            queue.PriorityQueue()
        )

        num_workers = 1
        self.executor = ThreadPoolExecutor(max_workers=num_workers)

        self.put_lock = threading.Lock()
        self.prefetch_lock = threading.Lock()
        self.put_tasks: List[CacheEngineKey] = []

        self.prefetch_tasks: dict[CacheEngineKey, Optional[Future]] = {}

        self.counter = itertools.count()
        self._shutdown = False

        self.thread = threading.Thread(target=self.process_task, daemon=True)
        self.thread.start()

    def submit_task(
        self,
        task_type: str,
        task: Callable,
        **kwargs,
    ):
        if task_type == "prefetch":
            priority = 0
            self.insert_prefetch_task(kwargs["key"], None)
        elif task_type == "delete":
            priority = 1
        elif task_type == "put":
            priority = 2
            self.insert_put_task(kwargs["key"])
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        self.pq.put((priority, next(self.counter), task_type, task, kwargs))

    def process_task(self):
        while not self._shutdown:
            try:
                _, _, task_type, task, kwargs = self.pq.get(block=True, timeout=0.1)
            except queue.Empty:
                continue

            if task_type == "exit":
                break

            future = self.executor.submit(task, **kwargs)
            if task_type == "prefetch":
                self.insert_prefetch_task(kwargs["key"], future)

            self.pq.task_done()

    def remove_put_task(self, key: CacheEngineKey):
        with self.put_lock:
            if key in self.put_tasks:
                self.put_tasks.remove(key)
            else:
                logger.warning(f"Key {key} not found in put tasks.")

    def insert_put_task(self, key: CacheEngineKey):
        with self.put_lock:
            self.put_tasks.append(key)

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        with self.put_lock:
            return key in self.put_tasks

    def remove_prefetch_task(self, key: CacheEngineKey):
        with self.prefetch_lock:
            if key in self.prefetch_tasks:
                self.prefetch_tasks.pop(key)
            else:
                logger.warning(f"Key {key} not found in prefetch tasks.")

    def insert_prefetch_task(
        self,
        key: CacheEngineKey,
        future_or_none: Optional[Future] = None,
    ):
        with self.prefetch_lock:
            self.prefetch_tasks[key] = future_or_none

    def exists_in_prefetch_tasks(self, key: CacheEngineKey) -> bool:
        with self.prefetch_lock:
            return key in self.prefetch_tasks

    def wait_prefetch_task(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Wait for the prefetch task to complete and return the MemoryObj."""
        while True:
            self.prefetch_lock.acquire()
            if key not in self.prefetch_tasks:
                self.prefetch_lock.release()
                return None

            logger.debug(f"Waiting for prefetch task for key {key} to complete.")
            future = self.prefetch_tasks[key]
            if future is None:
                self.prefetch_lock.release()
                time.sleep(0.01)
                continue

            self.prefetch_lock.release()

            memory_obj = future.result()
            return memory_obj

    def close(self):
        self._shutdown = True
        # Put a sentinel to wake up the thread
        try:
            self.pq.put_nowait((999, 0, "exit", lambda: None, {}))
        except queue.Full:
            pass
        # Wait for thread to finish (with timeout)
        self.thread.join(timeout=1.0)
        self.executor.shutdown(wait=True)


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
    ):
        self.cache_policy = get_cache_policy(config.cache_policy)
        self.dict = self.cache_policy.init_mutable_mapping()

        self.dst_device = dst_device
        self.local_cpu_backend = local_cpu_backend
        self.cxl_lock = threading.Lock()

        # Get configuration for C library
        # Note: cxl_shm.c uses hardcoded values, but we can override via environment
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

        # Initialize C library wrapper
        self.cxl_shm = CxlShmWrapper(num_procs=num_procs, rank=rank)
        result = self.cxl_shm.init()
        if result != 0:
            raise RuntimeError(f"Failed to initialize CXL shared memory: {result}")

        # Store handles for each key
        self.key_handles: dict[CacheEngineKey, CxlShmHnd] = {}
        # Map from CacheEngineKey to CXL key (for long keys that use hash)
        self.key_to_cxl_key: dict[CacheEngineKey, str] = {}

        self.loop = loop
        self.use_local_cpu = config.local_cpu

        self.cxl_worker = CxlWorker()

        # Get max CXL cache size from config
        max_cxl_size_gb = 0.0
        if config.extra_config is not None:
            max_cxl_size_gb = config.extra_config.get("max_cxl_size", 0.0)
        max_cxl_size_bytes = int(max_cxl_size_gb * 1024**3) if max_cxl_size_gb > 0 else 0
        
        # cxl_shm.c uses hardcoded 32GB (1UL << 35)
        CXL_SHM_DAX_SIZE = 1 << 35  # 32GB
        if max_cxl_size_bytes > 0:
            self.max_cache_size = max_cxl_size_bytes
        else:
            self.max_cache_size = CXL_SHM_DAX_SIZE
        self.current_cache_size = 0.0

        # to help maintain suffix -> prefix order in the dict
        self.keys_in_request: List[CacheEngineKey] = []

        self.lmcache_worker = lmcache_worker
        self.instance_id = config.lmcache_instance_id
        self.stats_monitor = LMCStatsMonitor.GetOrCreate()
        self.usage = 0

        logger.info(f"Initialized CXL backend using cxl_shm.c (num_procs={num_procs}, rank={rank})")

    def __str__(self):
        return "CxlBackend"

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

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        with self.cxl_lock:
            if key not in self.dict:
                # Try to open object in C library
                cxl_key = self._get_cxl_key(key)
                result, hnd = self.cxl_shm.open_obj(cxl_key)
                if result != 0 or hnd is None:
                    return False
                
                # Object exists, add to dict
                if hnd.obj_contents:
                    size = hnd.obj_contents.size
                    # We don't have shape/dtype in C structure, use defaults
                    # In practice, you might want to store this separately
                    shape = None
                    dtype = None
                    fmt = None
                    path = f"cxl_obj_{cxl_key}"
                    self.dict[key] = DiskCacheMetadata(path, size, shape, dtype, fmt, False)
                    self.key_handles[key] = hnd
            
            if pin:
                self.dict[key].pin()
                self.keys_in_request.append(key)
            return True

    def touch_cache(self):
        # flip the order of the keys in the request
        with self.cxl_lock:
            for key in reversed(self.keys_in_request):
                self.cache_policy.update_on_hit(key, self.dict)
            self.keys_in_request = []

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        return self.cxl_worker.exists_in_put_tasks(key)

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

    def remove(
        self,
        key: CacheEngineKey,
        force: bool = True,
    ) -> bool:
        if force:
            self.cxl_lock.acquire()

        if not (meta := self.dict.pop(key, None)):
            if force:
                self.cxl_lock.release()
            return False

        size = meta.size
        cxl_key = self._get_cxl_key(key)
        
        # Destroy object in C library
        if key in self.key_handles:
            hnd = self.key_handles.pop(key)
            self.cxl_shm.destroy(hnd)
            self.cxl_shm.close(hnd)

        self.usage -= size
        self.stats_monitor.update_local_storage_usage(self.usage)
        self.current_cache_size -= size

        if force:
            self.cache_policy.update_on_force_evict(key)
            self.cxl_lock.release()

        # push kv evict msg
        if self.lmcache_worker is not None:
            self.lmcache_worker.put_msg(
                KVEvictMsg(self.instance_id, key.worker_id, key.chunk_hash, str(self))
            )

        return True

    def insert_key(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        size = memory_obj.get_physical_size()
        shape = memory_obj.metadata.shape
        dtype = memory_obj.metadata.dtype
        fmt = memory_obj.metadata.fmt

        has_stored = False
        with self.cxl_lock:
            # Need to do reinsert to update cache recency
            if key in self.dict:
                self.dict.pop(key)
                has_stored = True

            # Use key string as path identifier
            cxl_key = self._get_cxl_key(key)
            path = f"cxl_obj_{cxl_key}"
            self.dict[key] = DiskCacheMetadata(path, size, shape, dtype, fmt, False)

        # push kv admit msg
        if self.lmcache_worker is not None and not has_stored:
            self.lmcache_worker.put_msg(
                KVAdmitMsg(self.instance_id, key.worker_id, key.chunk_hash, str(self))
            )

    def submit_put_task(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
    ):
        assert memory_obj.tensor is not None

        # skip repeated save
        if self.exists_in_put_tasks(key):
            logger.debug(f"Put task for {key} is already in progress.")
            return None

        required_size = memory_obj.get_physical_size()
        with self.cxl_lock:
            # Check space (simplified - C library handles actual allocation)
            while self.current_cache_size + required_size > self.max_cache_size:
                evict_keys = self.cache_policy.get_evict_candidates(
                    self.dict, num_candidates=1
                )
                if not evict_keys:
                    logger.warning(
                        "No eviction candidates found. CXL space under pressure."
                    )
                    return None

                for evict_key in evict_keys:
                    self.current_cache_size -= self.dict[evict_key].size

                self.batched_remove(evict_keys, force=False)
            self.current_cache_size += required_size

        self.cache_policy.update_on_put(key)
        memory_obj.ref_count_up()

        self.cxl_worker.submit_task(
            "put",
            self.async_save_bytes_to_cxl,
            key=key,
            memory_obj=memory_obj,
        )

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec=None,
    ) -> None:
        for key, memory_obj in zip(keys, memory_objs, strict=False):
            self.submit_put_task(key, memory_obj)

    def submit_prefetch_task(
        self,
        key: CacheEngineKey,
    ) -> bool:
        assert self.use_local_cpu, "prefetch and local_cpu must be enabled together"

        logger.debug("Submitting prefetch task")

        self.cxl_lock.acquire()
        if key not in self.dict:
            self.cxl_lock.release()
            return False

        self.cache_policy.update_on_hit(key, self.dict)

        if self.cxl_worker.exists_in_prefetch_tasks(key):
            logger.debug(f"Prefetch task for {key} is already in progress.")
            self.cxl_lock.release()
            return False

        cxl_meta = self.dict[key]
        dtype = cxl_meta.dtype
        shape = cxl_meta.shape
        fmt = cxl_meta.fmt

        assert dtype is not None
        assert shape is not None

        memory_obj = self.local_cpu_backend.allocate(shape, dtype, fmt)
        if memory_obj is None:
            self.cxl_lock.release()
            logger.debug("Memory allocation failed during async CXL load.")
            return False

        self.dict[key].pin()

        self.cache_policy.update_on_hit(key, self.dict)

        self.cxl_lock.release()
        logger.debug(f"Prefetching {key} from CXL.")

        self.cxl_worker.submit_task(
            "prefetch",
            self.async_load_bytes_from_cxl,
            key=key,
            memory_obj=memory_obj,
        )

        return True

    def get_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        """Blocking get function."""
        self.cxl_lock.acquire()
        if key not in self.dict:
            if not self.contains(key, pin=False):
                self.cxl_lock.release()
                return None

        self.cache_policy.update_on_hit(key, self.dict)

        self.cxl_lock.release()

        if memory_obj := self.cxl_worker.wait_prefetch_task(key):
            if self.local_cpu_backend.contains(key, pin=True):
                return memory_obj

        self.cxl_lock.acquire()
        self.cache_policy.update_on_hit(key, self.dict)

        cxl_meta = self.dict[key]
        dtype = cxl_meta.dtype
        shape = cxl_meta.shape
        fmt = cxl_meta.fmt
        assert dtype is not None
        assert shape is not None

        memory_obj = self.load_bytes_from_cxl(
            key, dtype=dtype, shape=shape, fmt=fmt
        )
        self.cxl_lock.release()

        return memory_obj

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

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def async_save_bytes_to_cxl(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
    ) -> None:
        """Convert KV to bytes and async store bytes to CXL memory via C library."""
        kv_chunk = memory_obj.tensor
        assert kv_chunk is not None
        buffer = memory_obj.byte_array

        size = len(buffer)
        self.usage += size
        self.stats_monitor.update_local_storage_usage(self.usage)

        # Write to CXL via C library
        self.write_cxl_mmap(key, buffer, memory_obj)

        self.insert_key(key, memory_obj)

        memory_obj.ref_count_down()

        self.cxl_worker.remove_put_task(key)

    def async_load_bytes_from_cxl(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
    ):
        """Async load bytearray from CXL memory via C library."""

        logger.debug("Executing `async_load_bytes` from CXL.")
        buffer = memory_obj.byte_array
        self.read_cxl_mmap(key, buffer)

        self.cxl_lock.acquire()
        self.dict[key].unpin()
        self.cxl_lock.release()

        self.local_cpu_backend.submit_put_task(key, memory_obj)

        self.cxl_worker.remove_prefetch_task(key)

        return memory_obj

    def load_bytes_from_cxl(
        self,
        key: CacheEngineKey,
        dtype: torch.dtype,
        shape: torch.Size,
        fmt: MemoryFormat,
    ) -> Optional[MemoryObj]:
        """Load bytearray from CXL memory via C library."""
        cxl_key = self._get_cxl_key(key)
        
        # Open object in C library
        result, hnd = self.cxl_shm.open_obj(cxl_key)
        if result != 0 or hnd is None:
            return None
        
        if not hnd.obj_contents or not hnd.mapped_addr:
            return None
        
        size = hnd.obj_contents.size
        mapped_addr = hnd.mapped_addr
        
        # Create tensor directly from CXL memory
        # mapped_addr is a c_void_p, convert to int for from_address
        addr_int = ctypes.cast(mapped_addr, ctypes.c_void_p).value
        if addr_int is None:
            return None
        
        array_type = ctypes.c_uint8 * size
        cxl_array = array_type.from_address(addr_int)
        cxl_tensor = torch.frombuffer(cxl_array, dtype=torch.uint8)
        
        # Reshape and view as the correct dtype
        num_elements = size // dtype.itemsize
        cxl_tensor = cxl_tensor[:num_elements * dtype.itemsize].view(dtype).view(shape)
        
        metadata = MemoryObjMetadata(
            shape=shape,
            dtype=dtype,
            address=0,
            phy_size=size,
            ref_count=1,
            pin_count=0,
            fmt=fmt,
        )
        
        memory_obj = TensorMemoryObj(
            raw_data=cxl_tensor,
            metadata=metadata,
            parent_allocator=None,
        )
        
        # Store handle
        self.key_handles[key] = hnd
        
        return memory_obj

    def write_cxl_mmap(self, key: CacheEngineKey, buffer: bytearray, memory_obj: MemoryObj):
        """Write buffer to CXL memory using C library."""
        start_time = time.time()
        size = len(buffer)
        cxl_key = self._get_cxl_key(key)
        
        # Create object in C library
        result, hnd = self.cxl_shm.create(cxl_key, size)
        if result != 0 or hnd is None:
            raise RuntimeError(f"Failed to create CXL object for key {cxl_key}: {result}")
        
        # Write data to mapped address
        if hnd.mapped_addr:
            mapped_addr = hnd.mapped_addr
            # Copy buffer to mapped memory using ctypes
            if isinstance(buffer, memoryview):
                buffer_bytes = bytes(buffer)
            else:
                buffer_bytes = bytes(buffer)
            
            # Use ctypes to write to memory
            ctypes.memmove(ctypes.c_void_p(mapped_addr), buffer_bytes, size)
            
            # Flush to ensure persistence
            self.cxl_shm.flush(ctypes.c_void_p(mapped_addr), size)
        
        # Store handle
        with self.cxl_lock:
            self.key_handles[key] = hnd

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
        
        # Copy from mapped memory to buffer
        if isinstance(buffer, memoryview):
            buffer_bytes = bytes(buffer)
        else:
            buffer_bytes = bytes(buffer)
        
        # Use ctypes to read from memory
        ctypes.memmove(buffer_bytes, ctypes.c_void_p(mapped_addr), size)
        
        # Copy back to buffer if it was a bytearray
        if isinstance(buffer, bytearray):
            buffer[:] = buffer_bytes

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

        # Finalize C library
        if hasattr(self, "cxl_shm"):
            try:
                self.cxl_shm.finalize()
            except Exception as e:
                logger.warning(f"Error finalizing CXL shared memory: {e}")

        self.cxl_worker.close()

