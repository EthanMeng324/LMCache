# SPDX-License-Identifier: Apache-2.0
"""
Latency benchmark for the Python-level CxlBackend (end-to-end).

This is intentionally separate from unit/functionality tests.

Typical usage (device permission is usually root-only):

  sudo -E ./venv/bin/python tests/v1/storage_backend/bench_cxl_backend_latency.py \
    --dax-device /dev/dax1.0 --mode write_sync --flush --size-bytes $((4*1024*1024))

Notes:
- This benchmark uses *CxlBackend APIs* (submit_put_task / get_blocking),
  so it includes the worker thread/executor scheduling overhead and backend bookkeeping.
- Use --flush to include flush+fence costs (write-after-flush, read-before-flush).
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import os
import statistics
import sys
import time
from unittest.mock import Mock

import torch
import threading

from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat, MemoryObj, MemoryObjMetadata, TensorMemoryObj
from lmcache.v1.storage_backend.cxl_backend import CxlBackend
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

DEFAULT_ITERS = 200
DEFAULT_WARMUP = 20
RUN_ID = (os.getpid() << 32) ^ (time.time_ns() & 0xFFFFFFFFFFFFFFFF)


def _percentile(sorted_vals: list[int], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    if p <= 0:
        return float(sorted_vals[0])
    if p >= 100:
        return float(sorted_vals[-1])
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return float(sorted_vals[f])
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return float(d0 + d1)


def _print_latency(label: str, samples_ns: list[int]) -> None:
    samples_ns.sort()
    mean_us = statistics.fmean(samples_ns) / 1e3
    p50_us = _percentile(samples_ns, 50.0) / 1e3
    p90_us = _percentile(samples_ns, 90.0) / 1e3
    p95_us = _percentile(samples_ns, 95.0) / 1e3
    p99_us = _percentile(samples_ns, 99.0) / 1e3
    max_us = samples_ns[-1] / 1e3

    print(f"\n{label} latency (microseconds):")
    print(f"- mean: {mean_us:.2f}")
    print(f"- p50 : {p50_us:.2f}")
    print(f"- p90 : {p90_us:.2f}")
    print(f"- p95 : {p95_us:.2f}")
    print(f"- p99 : {p99_us:.2f}")
    print(f"- max : {max_us:.2f}")


def _make_mock_config(*, num_procs: int, rank: int, max_cxl_size_gb: float, dax_device: str | None):
    config = Mock(spec=LMCacheEngineConfig)
    config.cache_policy = "LRU"
    # CxlBackend expects a real integer for chunk_size.
    # When using Mock(spec=...), unset attributes become Mock objects, which
    # breaks int(config.chunk_size).
    config.chunk_size = 256
    config.local_cpu = False
    config.lmcache_instance_id = "bench_instance"
    extra = {
        "cxl_num_procs": num_procs,
        "cxl_rank": rank,
        "max_cxl_size": max_cxl_size_gb,
    }
    if dax_device is not None:
        extra["cxl_dax_device"] = dax_device
    config.extra_config = extra
    return config


def _make_dummy_local_cpu_backend() -> LocalCPUBackend:
    # CxlBackend requires it in ctor even when local_cpu=False.
    backend = Mock(spec=LocalCPUBackend)
    backend.allocate = Mock(return_value=None)
    backend.contains = Mock(return_value=False)
    backend.submit_put_task = Mock()
    return backend


def _make_payload(size_bytes: int) -> TensorMemoryObj:
    # Use a uint8 tensor so size_bytes maps 1:1 to numel, simpler to reason about.
    t = torch.empty((size_bytes,), dtype=torch.uint8)
    # Touch a little so pages exist.
    t[: min(size_bytes, 256)] = torch.arange(min(size_bytes, 256), dtype=torch.uint8)

    meta = MemoryObjMetadata(
        shape=t.shape,
        dtype=t.dtype,
        address=0,
        phy_size=size_bytes,
        ref_count=1,
        pin_count=0,
        fmt=MemoryFormat.BINARY_BUFFER,
    )
    return TensorMemoryObj(raw_data=t, metadata=meta, parent_allocator=None)


def _make_key(i: int) -> CacheEngineKey:
    # Make each key unique to avoid "object already exists" from cxl_shm_create.
    return CacheEngineKey(
        fmt="bench",
        model_name="cxl",
        world_size=1,
        worker_id=0,
        chunk_hash=(RUN_ID + i) & 0xFFFFFFFFFFFFFFFF,
        dtype=torch.uint8,
    )


def _wait_put_done(backend: CxlBackend, key: CacheEngineKey, timeout_s: float = 30.0) -> None:
    t0 = time.time()
    while backend.exists_in_put_tasks(key):
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"Timed out waiting for put task to finish for key={key}")
        # NOTE: sleep granularity can be ~10ms on some systems, which will dominate "latency"
        # measurements if used in a tight polling loop. The benchmark prefers event-based waiting.
        time.sleep(0.0005)


class _PutProfiler:
    """
    Instrument CxlBackend put execution to measure:
    - queue_delay: record_submit() -> submit_put_task entry (near-zero for sync backend)
    - exec_time  : submit_put_task entry -> submit_put_task exit

    Also provides an Event per key so the benchmark can wait precisely without polling sleeps.
    """

    def __init__(self, backend: CxlBackend):
        self.backend = backend
        self._orig_submit = backend.submit_put_task
        self._orig_write = backend.write_cxl_mmap
        self._orig_insert = backend.insert_key
        self._lock = threading.Lock()
        self._submit_ns: dict[CacheEngineKey, int] = {}
        self._entry_ns: dict[CacheEngineKey, int] = {}
        self._exit_ns: dict[CacheEngineKey, int] = {}
        self._events: dict[CacheEngineKey, threading.Event] = {}
        # Stage timings inside worker (all ns, per key)
        self._ba_ns: dict[CacheEngineKey, int] = {}  # byte_array extraction
        self._write_total_ns: dict[CacheEngineKey, int] = {}
        self._getkey_ns: dict[CacheEngineKey, int] = {}
        self._create_ns: dict[CacheEngineKey, int] = {}
        self._bytes_copy_ns: dict[CacheEngineKey, int] = {}
        self._memmove_ns: dict[CacheEngineKey, int] = {}
        self._flush_ns: dict[CacheEngineKey, int] = {}
        self._storehnd_ns: dict[CacheEngineKey, int] = {}
        self._insert_ns: dict[CacheEngineKey, int] = {}

        def _wrapped_write_cxl_mmap(key: CacheEngineKey, buffer, payload_size: int, alloc_size: int):
            # Re-implement the current write path with timings.
            t0 = time.perf_counter_ns()
            size = int(alloc_size)
            cxl_key = backend._get_cxl_key(key)
            t1 = time.perf_counter_ns()

            t2 = time.perf_counter_ns()
            result, hnd = backend.cxl_shm.create(cxl_key, size, actual_size=int(payload_size))
            t3 = time.perf_counter_ns()
            if result != 0 or hnd is None:
                raise RuntimeError(f"Failed to create CXL object for key {cxl_key}: {result}")

            mapped_addr = hnd.mapped_addr
            if mapped_addr:
                # Prefer zero-copy from a writable buffer; fall back to bytes() if needed.
                t4 = time.perf_counter_ns()
                used_copy = False
                try:
                    mv = buffer if isinstance(buffer, memoryview) else memoryview(buffer)
                    src_arr = (ctypes.c_uint8 * int(payload_size)).from_buffer(mv)
                    src_ptr = ctypes.cast(src_arr, ctypes.c_void_p)
                except (TypeError, BufferError):
                    used_copy = True
                    buffer_bytes = bytes(buffer)
                    src_ptr = buffer_bytes
                t5 = time.perf_counter_ns()

                t6 = time.perf_counter_ns()
                ctypes.memmove(ctypes.c_void_p(int(mapped_addr)), src_ptr, int(payload_size))
                tail = int(size) - int(payload_size)
                if tail > 0:
                    ctypes.memset(
                        ctypes.c_void_p(int(mapped_addr) + int(payload_size)),
                        0,
                        tail,
                    )
                t7 = time.perf_counter_ns()

                # Flush after write (matches current CxlBackend behavior)
                f0 = time.perf_counter_ns()
                backend.cxl_shm.flush(ctypes.c_void_p(mapped_addr), int(size))
                f1 = time.perf_counter_ns()
            else:
                t4 = t5 = t6 = t7 = f0 = f1 = time.perf_counter_ns()
                used_copy = False

            t8 = time.perf_counter_ns()
            with backend.cxl_lock:
                backend.key_handles[key] = hnd
            t9 = time.perf_counter_ns()

            t10 = time.perf_counter_ns()

            with self._lock:
                self._write_total_ns[key] = t10 - t0
                self._getkey_ns[key] = t1 - t0
                self._create_ns[key] = t3 - t2
                self._bytes_copy_ns[key] = (t5 - t4) if used_copy else 0
                self._memmove_ns[key] = t7 - t6
                self._flush_ns[key] = f1 - f0
                self._storehnd_ns[key] = t9 - t8

        backend.write_cxl_mmap = _wrapped_write_cxl_mmap  # type: ignore[assignment]

        def _wrapped_insert_key(key: CacheEngineKey, memory_obj: MemoryObj):
            t0 = time.perf_counter_ns()
            try:
                return self._orig_insert(key, memory_obj)
            finally:
                t1 = time.perf_counter_ns()
                with self._lock:
                    self._insert_ns[key] = t1 - t0

        backend.insert_key = _wrapped_insert_key  # type: ignore[assignment]

        def _wrapped_submit_put_task(key: CacheEngineKey, memory_obj: MemoryObj):
            entry = time.perf_counter_ns()
            with self._lock:
                self._entry_ns[key] = entry
                ev = self._events.get(key)
            try:
                return self._orig_submit(key, memory_obj)
            finally:
                exit_ns = time.perf_counter_ns()
                with self._lock:
                    self._exit_ns[key] = exit_ns
                    if ev is not None:
                        ev.set()

        backend.submit_put_task = _wrapped_submit_put_task  # type: ignore[assignment]

    def record_submit(self, key: CacheEngineKey) -> threading.Event:
        now = time.perf_counter_ns()
        ev = threading.Event()
        with self._lock:
            self._submit_ns[key] = now
            self._events[key] = ev
        return ev

    def wait(self, key: CacheEngineKey, ev: threading.Event, timeout_s: float = 30.0) -> None:
        if not ev.wait(timeout=timeout_s):
            raise TimeoutError(f"Timed out waiting for put completion for key={key}")

    def pop_stats_us(self, key: CacheEngineKey) -> tuple[float, float]:
        with self._lock:
            s = self._submit_ns.pop(key, None)
            e = self._entry_ns.pop(key, None)
            x = self._exit_ns.pop(key, None)
            self._events.pop(key, None)
        if s is None or e is None or x is None:
            return float("nan"), float("nan")
        return (e - s) / 1e3, (x - e) / 1e3

    def pop_stage_us(self, key: CacheEngineKey) -> dict[str, float]:
        with self._lock:
            ba = self._ba_ns.pop(key, None)
            wt = self._write_total_ns.pop(key, None)
            gk = self._getkey_ns.pop(key, None)
            cr = self._create_ns.pop(key, None)
            bc = self._bytes_copy_ns.pop(key, None)
            mm = self._memmove_ns.pop(key, None)
            fl = self._flush_ns.pop(key, None)
            sh = self._storehnd_ns.pop(key, None)
            ins = self._insert_ns.pop(key, None)
        def _us(v):
            return float("nan") if v is None else v / 1e3
        return {
            "byte_array_us": _us(ba),
            "write_total_us": _us(wt),
            "get_cxl_key_us": _us(gk),
            "cxl_create_us": _us(cr),
            "bytes_copy_us": _us(bc),
            "memmove_us": _us(mm),
            "flush_us": _us(fl),
            "store_handle_us": _us(sh),
            "insert_key_us": _us(ins),
        }


def bench_write(
    backend: CxlBackend,
    payload: TensorMemoryObj,
    *,
    iters: int,
    warmup: int,
    cleanup: bool,
) -> None:
    keys: list[CacheEngineKey] = []
    profiler = _PutProfiler(backend)

    for i in range(warmup):
        key = _make_key(10_000_000 + i)
        keys.append(key)
        ev = profiler.record_submit(key)
        backend.submit_put_task(key, payload)
        profiler.wait(key, ev)

    samples_ns: list[int] = []
    qdelay_us: list[float] = []
    exec_us: list[float] = []
    stage_samples: dict[str, list[float]] = {}
    for i in range(iters):
        key = _make_key(i)
        keys.append(key)
        ev = profiler.record_submit(key)
        t0 = time.perf_counter_ns()
        backend.submit_put_task(key, payload)
        profiler.wait(key, ev)
        t1 = time.perf_counter_ns()
        samples_ns.append(t1 - t0)
        qd, ex = profiler.pop_stats_us(key)
        qdelay_us.append(qd)
        exec_us.append(ex)
        stages = profiler.pop_stage_us(key)
        for k, v in stages.items():
            stage_samples.setdefault(k, []).append(v)

    _print_latency(f"BACKEND PUT (submit_put_task, size={payload.metadata.phy_size} bytes)", samples_ns)
    if qdelay_us and exec_us:
        qdelay_us_sorted = sorted(qdelay_us)
        exec_us_sorted = sorted(exec_us)
        print("\nBreakdown (microseconds):")
        print(f"- queue_delay p50: {_percentile([int(v * 1e3) for v in qdelay_us_sorted], 50.0)/1e3:.2f}  (submit -> worker entry)")
        print(f"- queue_delay p99: {_percentile([int(v * 1e3) for v in qdelay_us_sorted], 99.0)/1e3:.2f}")
        print(f"- exec_time   p50: {_percentile([int(v * 1e3) for v in exec_us_sorted], 50.0)/1e3:.2f}  (worker entry -> exit)")
        print(f"- exec_time   p99: {_percentile([int(v * 1e3) for v in exec_us_sorted], 99.0)/1e3:.2f}")

        # Per-stage stats (p50/p99) inside worker
        def _p(vals: list[float], pct: float) -> float:
            ints = [int(v * 1e3) for v in vals if v == v]  # filter NaN
            ints.sort()
            return _percentile(ints, pct) / 1e3 if ints else float("nan")

        print("\nWorker-stage breakdown (microseconds, p50 / p99):")
        for name in [
            "byte_array_us",
            "write_total_us",
            "get_cxl_key_us",
            "cxl_create_us",
            "bytes_copy_us",
            "memmove_us",
            "flush_us",
            "store_handle_us",
            "insert_key_us",
        ]:
            vals = stage_samples.get(name, [])
            print(f"- {name}: {_p(vals, 50.0):.2f} / {_p(vals, 99.0):.2f}")

    if cleanup:
        # Cleanup is outside timing; it can be expensive (destroy does memset+flush in C).
        removed = 0
        for k in keys:
            try:
                if backend.remove(k, force=True):
                    removed += 1
            except Exception:
                pass
        print(f"\nCleanup: removed {removed}/{len(keys)} objects")


def bench_write_sync(
    backend: CxlBackend,
    payload: TensorMemoryObj,
    *,
    iters: int,
    warmup: int,
    cleanup: bool,
) -> None:
    """
    Synchronous write benchmark:
    - execute the put path inline (no CxlWorker queue/executor)
    - still uses backend.write_cxl_mmap + insert_key + (optional) flush

    This is NOT how LMCacheEngine.store is designed to run (store expects non-blocking puts),
    but it's useful to compare overheads introduced by the async worker chain.
    """
    keys: list[CacheEngineKey] = []

    # Reuse the same per-stage instrumentation by wrapping write_cxl_mmap.
    profiler = _PutProfiler(backend)

    def _put_sync(key: CacheEngineKey):
        # Mirror submit_put_task bookkeeping as much as possible.
        buffer = payload.byte_array
        payload_size = int(payload.get_physical_size())
        alloc_size = int(payload_size)
        required_size = int(alloc_size)
        with backend.cxl_lock:
            backend.current_cache_size += required_size
        backend.cache_policy.update_on_put(key)

        payload.ref_count_up()
        backend.write_cxl_mmap(key, buffer, payload_size=payload_size, alloc_size=alloc_size)
        backend.usage += len(buffer)
        backend.stats_monitor.update_local_storage_usage(backend.usage)
        backend.insert_key(key, payload)
        payload.ref_count_down()

    # Warmup (not recorded)
    for i in range(warmup):
        key = _make_key(30_000_000 + i)
        keys.append(key)
        _put_sync(key)

    # Timed
    samples_ns: list[int] = []
    stage_samples: dict[str, list[float]] = {}
    for i in range(iters):
        key = _make_key(40_000_000 + i)
        keys.append(key)
        t0 = time.perf_counter_ns()
        _put_sync(key)
        t1 = time.perf_counter_ns()
        samples_ns.append(t1 - t0)

        stages = profiler.pop_stage_us(key)
        for k, v in stages.items():
            stage_samples.setdefault(k, []).append(v)

    _print_latency(f"BACKEND PUT SYNC (write_cxl_mmap, size={payload.metadata.phy_size} bytes)", samples_ns)

    # Per-stage stats (p50/p99)
    def _p(vals: list[float], pct: float) -> float:
        ints = [int(v * 1e3) for v in vals if v == v]  # filter NaN
        ints.sort()
        return _percentile(ints, pct) / 1e3 if ints else float("nan")

    print("\nSync worker-stage breakdown (microseconds, p50 / p99):")
    for name in [
        "write_total_us",
        "get_cxl_key_us",
        "cxl_create_us",
        "bytes_copy_us",
        "memmove_us",
        "flush_us",
        "store_handle_us",
    ]:
        vals = stage_samples.get(name, [])
        print(f"- {name}: {_p(vals, 50.0):.2f} / {_p(vals, 99.0):.2f}")

    if cleanup:
        removed = 0
        for k in keys:
            try:
                if backend.remove(k, force=True):
                    removed += 1
            except Exception:
                pass
        print(f"\nCleanup: removed {removed}/{len(keys)} objects")

def bench_read(
    backend: CxlBackend,
    payload: TensorMemoryObj,
    *,
    cleanup: bool,
    iters: int,
    warmup: int,
) -> None:
    class _GetProfiler:
        """
        Instrument CxlBackend read path (get_blocking -> load_bytes_from_cxl) to produce
        a stage breakdown similar to PUT.

        Note:
        - C-side cxl_shm_open_obj() flushes the mapped region internally; that cost is
          included in open_obj_us.
        - Python-side backend.cxl_shm.flush() can be enabled/disabled via --flush in main().
        """

        def __init__(self, backend: CxlBackend):
            self.backend = backend
            self._orig_load = backend.load_bytes_from_cxl
            self._lock = threading.Lock()
            self._stage_us: dict[str, list[float]] = {}

            def _push(name: str, ns: int):
                with self._lock:
                    self._stage_us.setdefault(name, []).append(ns / 1e3)

            def _wrapped_load_bytes_from_cxl(key, dtype, shape, fmt):
                t0 = time.perf_counter_ns()
                cxl_key = backend._get_cxl_key(key)
                t1 = time.perf_counter_ns()

                o0 = time.perf_counter_ns()
                result, hnd = backend.cxl_shm.open_obj(cxl_key)
                o1 = time.perf_counter_ns()
                if result != 0 or hnd is None:
                    return None
                if not hnd.obj_contents or not hnd.mapped_addr:
                    return None

                size = hnd.obj_contents.size
                mapped_addr = hnd.mapped_addr

                addr_int = ctypes.cast(mapped_addr, ctypes.c_void_p).value
                if addr_int is None:
                    return None

                f0 = time.perf_counter_ns()
                backend.cxl_shm.flush(ctypes.c_void_p(addr_int), size)
                f1 = time.perf_counter_ns()

                b0 = time.perf_counter_ns()
                array_type = ctypes.c_uint8 * size
                cxl_array = array_type.from_address(addr_int)
                cxl_tensor = torch.frombuffer(cxl_array, dtype=torch.uint8)

                num_elements = size // dtype.itemsize
                cxl_tensor = cxl_tensor[: num_elements * dtype.itemsize].view(dtype).view(shape)
                b1 = time.perf_counter_ns()

                m0 = time.perf_counter_ns()
                metadata = MemoryObjMetadata(
                    shape=shape,
                    dtype=dtype,
                    address=0,
                    phy_size=size,
                    ref_count=1,
                    pin_count=0,
                    fmt=fmt,
                )
                memory_obj = TensorMemoryObj(raw_data=cxl_tensor, metadata=metadata, parent_allocator=None)
                # NOTE: get_blocking() holds backend.cxl_lock while calling load_bytes_from_cxl().
                # Do NOT reacquire it here, otherwise we deadlock (threading.Lock is not re-entrant).
                backend.key_handles[key] = hnd
                m1 = time.perf_counter_ns()

                t2 = time.perf_counter_ns()

                _push("get_cxl_key_us", t1 - t0)
                _push("open_obj_us", o1 - o0)
                _push("flush_us", f1 - f0)
                _push("frombuffer_view_us", b1 - b0)
                _push("build_obj_us", m1 - m0)
                _push("load_total_us", t2 - t0)
                return memory_obj

            backend.load_bytes_from_cxl = _wrapped_load_bytes_from_cxl  # type: ignore[assignment]

        def stage_values(self, name: str) -> list[float]:
            with self._lock:
                return list(self._stage_us.get(name, []))

    profiler = _GetProfiler(backend)

    # Prepare a pool of keys to read.
    keys: list[CacheEngineKey] = []
    num_keys = max(1, min(64, iters))
    print(f"Preparing {num_keys} keys for READ benchmark...", flush=True)
    for i in range(num_keys):
        k = _make_key(20_000_000 + i)
        keys.append(k)
        backend.submit_put_task(k, payload)
        _wait_put_done(backend, k)
        if (i + 1) % 8 == 0 or (i + 1) == num_keys:
            print(f"- prepared {i+1}/{num_keys}", flush=True)

    # Warmup reads
    print(f"Warmup READs: {warmup} iterations...", flush=True)
    for i in range(warmup):
        _ = backend.get_blocking(keys[i % num_keys])
        if (i + 1) % 10 == 0 or (i + 1) == warmup:
            print(f"- warmup {i+1}/{warmup}", flush=True)

    samples_ns: list[int] = []
    print(f"Timed READs: {iters} iterations...", flush=True)
    for i in range(iters):
        k = keys[i % num_keys]
        t0 = time.perf_counter_ns()
        _ = backend.get_blocking(k)
        t1 = time.perf_counter_ns()
        samples_ns.append(t1 - t0)
        if (i + 1) % 25 == 0 or (i + 1) == iters:
            print(f"- timed {i+1}/{iters}", flush=True)

    _print_latency("BACKEND GET (get_blocking)", samples_ns)

    # Breakdown (p50/p99) for read mode
    def _p(vals: list[float], pct: float) -> float:
        ints = [int(v * 1e3) for v in vals if v == v]  # filter NaN
        ints.sort()
        return _percentile(ints, pct) / 1e3 if ints else float("nan")

    print("\nRead-stage breakdown (microseconds, p50 / p99):")
    for name in [
        "get_cxl_key_us",
        "open_obj_us",
        "flush_us",
        "frombuffer_view_us",
        "build_obj_us",
        "load_total_us",
    ]:
        vals = profiler.stage_values(name)
        print(f"- {name}: {_p(vals, 50.0):.2f} / {_p(vals, 99.0):.2f}")

    if cleanup:
        removed = 0
        for k in keys:
            try:
                if backend.remove(k, force=True):
                    removed += 1
            except Exception:
                pass
        print(f"\nCleanup: removed {removed}/{len(keys)} objects")


def main() -> int:
    ap = argparse.ArgumentParser(description="CxlBackend latency benchmark (end-to-end).")
    ap.add_argument("--dax-device", type=str, default=None, help="DAX device path (sets cxl_dax_device).")
    ap.add_argument("--size-bytes", type=int, default=4 * 1024 * 1024)
    ap.add_argument("--mode", choices=["write_sync", "read"], required=True)
    ap.add_argument("--flush", action="store_true", help="Enable flush+fence (write-after, read-before).")
    ap.add_argument("--cleanup", action="store_true", help="Destroy created objects after bench (outside timing).")
    args = ap.parse_args()

    if args.size_bytes <= 0:
        print("Invalid args: size-bytes>0 required.")
        return 2

    if args.dax_device is not None:
        os.environ["LMCACHE_CXL_DAX_DEVICE"] = args.dax_device

    print("Running CxlBackend latency benchmark (end-to-end)...")
    print(f"- dax_device: {os.getenv('LMCACHE_CXL_DAX_DEVICE', '<unset>')}")
    print(f"- mode: {args.mode}, flush: {args.flush}")
    print(f"- iters: {DEFAULT_ITERS}, warmup: {DEFAULT_WARMUP}, size_bytes: {args.size_bytes}")
    print("=" * 60)

    loop = asyncio.new_event_loop()
    try:
        cfg = _make_mock_config(num_procs=1, rank=0, max_cxl_size_gb=32.0, dax_device=args.dax_device)
        dummy_local_cpu = _make_dummy_local_cpu_backend()
        backend = CxlBackend(config=cfg, loop=loop, local_cpu_backend=dummy_local_cpu, dst_device="cuda")
        payload = _make_payload(args.size_bytes)

        # Control flush behavior at Python level for benchmarking.
        # Note: cxl_shm_open_obj() also flushes internally in C; this can't be disabled here.
        if not args.flush:
            backend.cxl_shm.flush = lambda *_args, **_kwargs: 0  # type: ignore[assignment]

        if args.mode == "write_sync":
            bench_write_sync(backend, payload, iters=DEFAULT_ITERS, warmup=DEFAULT_WARMUP, cleanup=args.cleanup)
        else:
            bench_read(backend, payload, iters=DEFAULT_ITERS, warmup=DEFAULT_WARMUP, cleanup=args.cleanup)

        backend.close()
        return 0
    finally:
        loop.close()


if __name__ == "__main__":
    sys.exit(main())

