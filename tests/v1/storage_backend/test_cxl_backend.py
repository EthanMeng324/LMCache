# SPDX-License-Identifier: Apache-2.0
"""
Tests + latency benchmark for the CXL backend.

This file is intentionally runnable with plain python (no pytest):

1) Real functional test (requires libcxl_shm.so + DAX device access):
   sudo -E ./venv/bin/python tests/v1/storage_backend/test_cxl_backend.py --real --dax-device /dev/dax1.0
   # optional: export LMCACHE_CXL_DAX_DEVICE=/dev/dax1.0 and omit --dax-device

Notes:
- If `torch` is not installed, unit tests will be SKIP'ed (because `lmcache.utils` imports torch).
- Latency benchmarking is split out into `tests/v1/storage_backend/bench_cxl_backend_latency.py`.
"""

import argparse
import asyncio
import ctypes
import os
import mmap
import statistics
import sys
import time
from unittest.mock import Mock, patch

# Keep consistent with cxl_shm.h
CXL_SHM_ONAME_LEN = 20

def _percentile(sorted_vals, p: float) -> float:
    """Compute percentile (0-100) with linear interpolation."""
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


def run_unit_tests() -> int:
    """Run the lightweight unit tests (mocked C library) without pytest."""
    try:
        import torch  # noqa: F401

        from lmcache.utils import CacheEngineKey, DiskCacheMetadata
        from lmcache.v1.config import LMCacheEngineConfig
        from lmcache.v1.memory_management import MemoryFormat
        from lmcache.v1.storage_backend.cxl_backend import CxlBackend
        from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
    except Exception as e:
        print("SKIP: unit tests require torch + lmcache imports, but they're not available here.")
        print(f"      reason: {e}")
        return 0

    print("Running CXL Backend unit tests (no pytest)...")
    print("=" * 60)

    passed = 0
    failed = 0

    def _run(name: str, fn):
        nonlocal passed, failed
        try:
            fn()
            print(f"   ✓ {name} passed")
            passed += 1
        except Exception as e:
            print(f"   ✗ {name} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    def create_mock_config():
        """Create a mock LMCacheEngineConfig."""
        config = Mock(spec=LMCacheEngineConfig)
        config.cache_policy = "LRU"
        config.local_cpu = True
        config.chunk_size = 256
        config.lmcache_instance_id = "test_instance"
        config.extra_config = {
            "cxl_num_procs": 1,
            "cxl_rank": 0,
            "max_cxl_size": 32.0,
        }
        return config

    def create_mock_local_cpu_backend():
        """Create a mock LocalCPUBackend."""
        backend = Mock(spec=LocalCPUBackend)
        backend.allocate = Mock(return_value=None)
        backend.contains = Mock(return_value=False)
        backend.submit_put_task = Mock()
        return backend

    class TestCxlBackendKeyHandling:
        """Test key handling and length limits."""

        def test_get_cxl_key_short(self):
            mock_config = create_mock_config()
            mock_local_cpu_backend = create_mock_local_cpu_backend()

            with patch("lmcache.v1.storage_backend.cxl_backend.CxlShmWrapper") as mock_cxl_shm:
                mock_wrapper = Mock()
                mock_wrapper.init.return_value = 0
                mock_cxl_shm.return_value = mock_wrapper

                loop = asyncio.new_event_loop()
                backend = CxlBackend(
                    config=mock_config,
                    loop=loop,
                    local_cpu_backend=mock_local_cpu_backend,
                )

                import torch

                key = CacheEngineKey(
                    fmt="vllm",
                    model_name="test",
                    world_size=1,
                    worker_id=0,
                    chunk_hash=hash("short"),
                    dtype=torch.bfloat16,
                )

                try:
                    cxl_key = backend._get_cxl_key(key)
                    assert len(cxl_key) <= CXL_SHM_ONAME_LEN
                    assert cxl_key == key.to_string() or cxl_key.startswith("H")
                finally:
                    backend.close()
                    loop.close()

        def test_get_cxl_key_long(self):
            mock_config = create_mock_config()
            mock_local_cpu_backend = create_mock_local_cpu_backend()

            with patch("lmcache.v1.storage_backend.cxl_backend.CxlShmWrapper") as mock_cxl_shm:
                mock_wrapper = Mock()
                mock_wrapper.init.return_value = 0
                mock_cxl_shm.return_value = mock_wrapper

                loop = asyncio.new_event_loop()
                backend = CxlBackend(
                    config=mock_config,
                    loop=loop,
                    local_cpu_backend=mock_local_cpu_backend,
                )

                import torch

                key = CacheEngineKey(
                    fmt="vllm",
                    model_name="very_long_model_name_that_exceeds_limit",
                    world_size=1,
                    worker_id=0,
                    chunk_hash=hash("very_long_chunk_hash_that_exceeds_limit"),
                    dtype=torch.bfloat16,
                )

                try:
                    key_str = key.to_string()
                    if len(key_str) > CXL_SHM_ONAME_LEN:
                        cxl_key = backend._get_cxl_key(key)
                        assert len(cxl_key) <= CXL_SHM_ONAME_LEN
                        assert cxl_key.startswith("H")
                        assert backend._get_cxl_key(key) == cxl_key
                finally:
                    backend.close()
                    loop.close()

        def test_get_cxl_key_caching(self):
            mock_config = create_mock_config()
            mock_local_cpu_backend = create_mock_local_cpu_backend()

            with patch("lmcache.v1.storage_backend.cxl_backend.CxlShmWrapper") as mock_cxl_shm:
                mock_wrapper = Mock()
                mock_wrapper.init.return_value = 0
                mock_cxl_shm.return_value = mock_wrapper

                loop = asyncio.new_event_loop()
                backend = CxlBackend(
                    config=mock_config,
                    loop=loop,
                    local_cpu_backend=mock_local_cpu_backend,
                )

                import torch

                key = CacheEngineKey(
                    fmt="vllm",
                    model_name="test",
                    world_size=1,
                    worker_id=0,
                    chunk_hash=hash("hash"),
                    dtype=torch.bfloat16,
                )

                try:
                    cxl_key1 = backend._get_cxl_key(key)
                    cxl_key2 = backend._get_cxl_key(key)
                    assert cxl_key1 == cxl_key2
                    assert key in backend.key_to_cxl_key
                finally:
                    backend.close()
                    loop.close()

    class TestCxlBackendBasicOperations:
        """Test basic backend operations."""

        def test_initialization(self):
            mock_config = create_mock_config()
            mock_local_cpu_backend = create_mock_local_cpu_backend()

            with patch("lmcache.v1.storage_backend.cxl_backend.CxlShmWrapper") as mock_cxl_shm:
                mock_wrapper = Mock()
                mock_wrapper.init.return_value = 0
                mock_cxl_shm.return_value = mock_wrapper

                loop = asyncio.new_event_loop()
                backend = CxlBackend(
                    config=mock_config,
                    loop=loop,
                    local_cpu_backend=mock_local_cpu_backend,
                )

                try:
                    assert backend.cxl_shm == mock_wrapper
                    assert len(backend.key_handles) == 0
                    assert len(backend.key_to_cxl_key) == 0
                finally:
                    backend.close()
                    loop.close()

        def test_initialization_failure(self):
            mock_config = create_mock_config()
            mock_local_cpu_backend = create_mock_local_cpu_backend()

            with patch("lmcache.v1.storage_backend.cxl_backend.CxlShmWrapper") as mock_cxl_shm:
                mock_wrapper = Mock()
                mock_wrapper.init.return_value = -1
                mock_cxl_shm.return_value = mock_wrapper

                loop = asyncio.new_event_loop()
                try:
                    try:
                        CxlBackend(
                            config=mock_config,
                            loop=loop,
                            local_cpu_backend=mock_local_cpu_backend,
                        )
                        assert False, "Should have raised RuntimeError"
                    except RuntimeError as e:
                        assert "Failed to initialize CXL shared memory" in str(e)
                finally:
                    loop.close()

        def test_pin_unpin(self):
            mock_config = create_mock_config()
            mock_local_cpu_backend = create_mock_local_cpu_backend()

            with patch("lmcache.v1.storage_backend.cxl_backend.CxlShmWrapper") as mock_cxl_shm:
                mock_wrapper = Mock()
                mock_wrapper.init.return_value = 0
                mock_cxl_shm.return_value = mock_wrapper

                loop = asyncio.new_event_loop()
                backend = CxlBackend(
                    config=mock_config,
                    loop=loop,
                    local_cpu_backend=mock_local_cpu_backend,
                )

                import torch

                sample_key = CacheEngineKey(
                    fmt="vllm",
                    model_name="test_model",
                    world_size=1,
                    worker_id=0,
                    chunk_hash=hash("test_hash"),
                    dtype=torch.bfloat16,
                )

                try:
                    assert backend.pin(sample_key) is False

                    backend.dict[sample_key] = DiskCacheMetadata(
                        path="test_path",
                        size=100,
                        shape=torch.Size([2, 4]),
                        dtype=torch.float32,
                        fmt=MemoryFormat.KV_T2D,
                    )

                    assert backend.pin(sample_key) is True
                    assert backend.dict[sample_key].is_pinned is True

                    assert backend.unpin(sample_key) is True
                    assert backend.dict[sample_key].is_pinned is False
                finally:
                    backend.close()
                    loop.close()

    print("\n1. Testing key handling...")
    test1 = TestCxlBackendKeyHandling()
    _run("test_get_cxl_key_short", test1.test_get_cxl_key_short)
    _run("test_get_cxl_key_long", test1.test_get_cxl_key_long)
    _run("test_get_cxl_key_caching", test1.test_get_cxl_key_caching)

    print("\n2. Testing basic operations...")
    test2 = TestCxlBackendBasicOperations()
    _run("test_initialization", test2.test_initialization)
    _run("test_initialization_failure", test2.test_initialization_failure)
    _run("test_pin_unpin", test2.test_pin_unpin)

    print("\n" + "=" * 60)
    print(f"Unit tests completed: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


def run_real_functional_test(*, dax_device: str | None, size_bytes: int) -> int:
    """
    Real functional test using CxlBackend APIs:
    - submit_put_task() writes known bytes into CXL/DAX
    - get_blocking() reads back and validates content
    - contains/remove basic behavior

    Requires:
    - torch installed (for CacheEngineKey dtype + MemoryObj)
    - libcxl_shm.so loadable
    - permission to open the DAX device (often root-only)
    """
    try:
        import torch

        from lmcache.utils import CacheEngineKey
        from lmcache.v1.config import LMCacheEngineConfig
        from lmcache.v1.memory_management import MemoryFormat, MemoryObjMetadata, TensorMemoryObj
        from lmcache.v1.storage_backend.cxl_backend import CxlBackend
        from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
    except Exception as e:
        print("SKIP: real test requires torch + lmcache imports.")
        print(f"      reason: {e}")
        return 0

    if size_bytes <= 0:
        print("ERROR: --size-bytes must be > 0")
        return 2

    if dax_device is not None:
        os.environ["LMCACHE_CXL_DAX_DEVICE"] = dax_device

    # Minimal config; CxlBackend reads DAX device from extra_config/env.
    config = Mock(spec=LMCacheEngineConfig)
    config.cache_policy = "LRU"
    config.local_cpu = False
    config.lmcache_instance_id = "test_instance"
    config.extra_config = {
        "cxl_num_procs": 1,
        "cxl_rank": 0,
        "max_cxl_size": 32.0,
    }
    if dax_device is not None:
        config.extra_config["cxl_dax_device"] = dax_device

    dummy_local_cpu = Mock(spec=LocalCPUBackend)
    dummy_local_cpu.allocate = Mock(return_value=None)
    dummy_local_cpu.contains = Mock(return_value=False)
    dummy_local_cpu.submit_put_task = Mock()

    # Prepare deterministic payload
    src = torch.empty((size_bytes,), dtype=torch.uint8)
    src[: min(size_bytes, 256)] = torch.arange(min(size_bytes, 256), dtype=torch.uint8)
    if size_bytes > 256:
        # simple pattern for the rest
        src[256:] = 7

    meta = MemoryObjMetadata(
        shape=src.shape,
        dtype=src.dtype,
        address=0,
        phy_size=size_bytes,
        ref_count=1,
        pin_count=0,
        fmt=MemoryFormat.BINARY_BUFFER,
    )
    mem_obj = TensorMemoryObj(raw_data=src, metadata=meta, parent_allocator=None)

    # Unique key per run to avoid "object already exists"
    run_id = (os.getpid() << 32) ^ (time.time_ns() & 0xFFFFFFFFFFFFFFFF)
    key = CacheEngineKey(
        fmt="bench",
        model_name="cxl",
        world_size=1,
        worker_id=0,
        chunk_hash=run_id & 0xFFFFFFFFFFFFFFFF,
        dtype=torch.uint8,
    )

    loop = asyncio.new_event_loop()
    try:
        backend = CxlBackend(config=config, loop=loop, local_cpu_backend=dummy_local_cpu, dst_device="cpu")
    except Exception as e:
        print("SKIP: failed to initialize CxlBackend (likely DAX permission / libcxl_shm.so issue).")
        print(f"      reason: {e}")
        return 0

    try:
        print("Running REAL CxlBackend functional test...")
        print(f"- dax_device: {os.getenv('LMCACHE_CXL_DAX_DEVICE', '<unset>')}")
        print(f"- size_bytes: {size_bytes}")
        print(f"- key: {key.to_string()}")

        # PUT
        backend.submit_put_task(key, mem_obj)

        # contains should be true after insert_key
        assert backend.contains(key) is True, "contains() should be True after submit_put_task"

        # GET and validate bytes
        got = backend.get_blocking(key)
        assert got is not None, "get_blocking() returned None after put"
        assert got.tensor is not None
        got_u8 = got.tensor.view(torch.uint8).clone()  # clone to detach from mmap
        assert got_u8.numel() == size_bytes
        if not torch.equal(got_u8, src):
            # Find first mismatch for debugging
            diff = (got_u8 != src).nonzero()
            first = int(diff[0].item()) if diff.numel() else -1
            raise AssertionError(f"Data mismatch at byte index {first}")

        # REMOVE and verify
        assert backend.remove(key, force=True) is True, "remove() should succeed"
        assert backend.contains(key) is False, "contains() should be False after remove"

        print("PASS: submit_put_task/get_blocking roundtrip + contains/remove")
        return 0
    except AssertionError as e:
        print(f"FAIL: {e}")
        return 1
    except Exception as e:
        print(f"ERROR: unexpected exception: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        try:
            backend.close()
        except Exception:
            pass
        loop.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="CXL backend tests + latency benchmark (no pytest).")
    parser.add_argument("--unit", action="store_true", help="Run mocked unit tests.")
    parser.add_argument("--real", action="store_true", help="Run real functional test against DAX device.")
    parser.add_argument("--dax-device", type=str, default=None, help="DAX device path (e.g. /dev/dax1.0).")
    parser.add_argument("--size-bytes", type=int, default=4 * 1024 * 1024, help="Payload size for real test.")
    args = parser.parse_args()

    if not args.unit and not args.real:
        # Default: safest mode.
        args.unit = True

    rc = 0
    if args.unit:
        rc = run_unit_tests()
        if rc != 0:
            return rc

    if args.real:
        return run_real_functional_test(dax_device=args.dax_device, size_bytes=args.size_bytes)

    return rc


if __name__ == "__main__":
    sys.exit(main())
