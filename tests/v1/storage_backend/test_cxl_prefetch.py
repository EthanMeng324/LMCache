# SPDX-License-Identifier: Apache-2.0
"""
Functional tests for the intra-node CXL<->CPU copy paths.

This exercises ``LMCacheEngine.move_intra_node`` (the implementation of the
controller-driven "prefetch" branch in cache_controller/worker.py) against a
REAL ``CxlBackend`` + REAL ``LocalCPUBackend``. No GPU and no real CXL hardware
are required: the CXL device is simulated with a file on ``/dev/shm`` (tmpfs =
DRAM), which needs no root.

Runnable with plain python (no pytest), matching test_cxl_backend.py style:

    # default: uses /dev/shm as the CXL device, ~1GB
    ./venv/bin/python tests/v1/storage_backend/test_cxl_prefetch.py

    # point at a real DAX device instead
    LMCACHE_CXL_DAX_DEVICE=/dev/dax1.0 \
        ./venv/bin/python tests/v1/storage_backend/test_cxl_prefetch.py --size-gb 32

It SKIPs (rc=0) if torch / lmcache / libcxl_shm.so are unavailable, or if the
CXL device cannot be initialized (e.g. tmpfs too small).

What is covered:
- T1  basic promotion CXL->CPU (content correctness, copy keeps source)
- T2  source pins are released after the move  (guards the #1 risk: pin leak)
- T3  idempotency: promoting an already-cached prefix is a no-op
- T4  partial prefix: stops at first missing chunk
- T5  move mode (do_copy=False) removes the promoted prefix from CXL
- T6  CPU->CXL offload copy preserves CPU, publishes a CXL event, and returns
      a rank-local result
- T7  CPU->CXL offload releases pins and reports a missing source prefix
"""

import argparse
import asyncio
import os
import sys
import threading
import time
from unittest.mock import Mock


# Global unique-key counter so repeated runs / sub-tests never collide in CXL.
_RUN_SALT = (os.getpid() << 20) ^ (time.time_ns() & 0xFFFFF)
_KEY_IDX = 0

# Small KV_2LTD-ish shape (2, num_layers, num_tokens, hidden) to keep the
# padded CXL allocation small.
_SHAPE = (2, 4, 8, 64)


def _skip(reason: str) -> int:
    print(f"SKIP: {reason}")
    return 0


def run_real_functional_test(*, dax_device, size_gb: float) -> int:
    try:
        import torch

        from lmcache.utils import CacheEngineKey
        from lmcache.v1.cache_engine import LMCacheEngine
        from lmcache.v1.config import LMCacheEngineConfig
        from lmcache.v1.memory_management import AdHocMemoryAllocator, MemoryFormat
        from lmcache.v1.storage_backend.cxl_backend import CxlBackend
        from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
    except Exception as e:  # torch or lmcache missing
        return _skip(f"real test requires torch + lmcache imports. reason: {e}")

    # ------------------------------------------------------------------ setup
    size_bytes = int(size_gb * 1024**3)
    if size_bytes <= 0:
        print("ERROR: --size-gb must be > 0")
        return 2

    # Choose CXL device: env override, else a tmpfs file (DRAM, no root).
    dax = dax_device or os.environ.get("LMCACHE_CXL_DAX_DEVICE")
    created_tmpfs = False
    if dax is None:
        dax = "/dev/shm/lmcache_cxl_prefetch_test.img"
        try:
            # Sparse file; C init memsets the mapped window, so keep size modest.
            with open(dax, "wb") as f:
                f.truncate(size_bytes)
            created_tmpfs = True
        except OSError as e:
            return _skip(f"cannot create tmpfs CXL file at {dax}: {e}")

    os.environ["LMCACHE_CXL_DAX_DEVICE"] = dax
    os.environ["LMCACHE_CXL_DAX_DEVICE_SIZE"] = str(size_bytes)

    # --- real LocalCPUBackend (pass allocator directly to skip metadata path) ---
    cpu_config = Mock(spec=LMCacheEngineConfig)
    cpu_config.cache_policy = "LRU"
    cpu_config.local_cpu = True
    cpu_config.lmcache_instance_id = "test_instance"
    cpu_config.use_layerwise = False
    cpu_config.enable_blending = False

    cpu_alloc = AdHocMemoryAllocator(device="cpu")
    local_cpu = LocalCPUBackend(
        config=cpu_config,
        memory_allocator=cpu_alloc,
        dst_device="cpu",
        lmcache_worker=None,
    )

    # --- real CxlBackend backed by the tmpfs/DAX device ---
    cxl_config = Mock(spec=LMCacheEngineConfig)
    cxl_config.cache_policy = "LRU"
    cxl_config.local_cpu = True
    cxl_config.chunk_size = 256
    cxl_config.lmcache_instance_id = "test_instance"
    cxl_config.extra_config = {
        "cxl_num_procs": 1,
        "cxl_rank": 0,
        "max_cxl_size": float(size_gb),
        "cxl_dax_device": dax,
        # Start from a clean device on every run.
        "cxl_reset_on_init": "full",
    }

    loop = asyncio.new_event_loop()
    try:
        cxl = CxlBackend(
            config=cxl_config,
            loop=loop,
            local_cpu_backend=local_cpu,
            dst_device="cpu",
        )
    except Exception as e:
        loop.close()
        if created_tmpfs:
            try:
                os.remove(dax)
            except OSError:
                pass
        return _skip(f"failed to init CxlBackend (DAX/libcxl_shm.so issue): {e}")

    # ------------------------------------------------------ payload / harness
    payload_alloc = AdHocMemoryAllocator(device="cpu")

    def make_key(dtype):
        global _KEY_IDX
        _KEY_IDX += 1
        return CacheEngineKey(
            fmt="prefetch",
            model_name="cxl",
            world_size=1,
            worker_id=0,
            chunk_hash=(_RUN_SALT ^ _KEY_IDX) & 0xFFFFFFFFFFFFFFFF,
            dtype=dtype,
        )

    def put_chunk_in_cxl():
        """Create a deterministic KV chunk, store it in CXL only.

        Returns (key, cloned_original_tensor).
        """
        dtype = torch.bfloat16
        key = make_key(dtype)
        obj = payload_alloc.allocate(_SHAPE, dtype, fmt=MemoryFormat.KV_2LTD)
        # Distinct, bf16-exact pattern per chunk.
        obj.tensor.fill_(float((_KEY_IDX % 13) + 1))
        original = obj.tensor.clone()
        cxl.submit_put_task(key, obj)
        return key, original

    def put_chunk_in_cpu():
        """Create a deterministic KV chunk in LocalCPUBackend only."""
        dtype = torch.bfloat16
        key = make_key(dtype)
        obj = payload_alloc.allocate(_SHAPE, dtype, fmt=MemoryFormat.KV_2LTD)
        obj.tensor.fill_(float((_KEY_IDX % 13) + 1))
        original = obj.tensor.clone()
        local_cpu.submit_put_task(key, obj)
        # submit_put_task took the backend ownership ref.
        obj.ref_count_down()
        return key, original

    # Fake StorageManager exposing only what move_intra_node touches.
    class FakeStorageManager:
        def __init__(self):
            self.storage_backends = {
                "CxlBackend": cxl,
                "LocalCPUBackend": local_cpu,
            }

        def batched_get(self, keys, location=None):
            # Mirrors StorageManager.batched_get for a single location.
            backend = self.storage_backends[location]
            return backend.batched_get_blocking(list(keys))

        def batched_remove(self, keys, locations=None):
            n = 0
            for k in keys:
                if cxl.remove(k, force=True):
                    n += 1
            return n

        def batched_unpin(self, keys, locations=None):
            backend = self.storage_backends[locations[0]]
            for k in keys:
                backend.unpin(k)

    # Build a bare LMCacheEngine and inject only the fields move_intra_node uses.
    engine = LMCacheEngine.__new__(LMCacheEngine)
    engine.storage_manager = FakeStorageManager()
    engine.lookup_pins = {}
    engine._pending_offloads = {}
    engine._committed_offloads = {}
    engine._pending_offloads_lock = threading.Lock()

    def fake_lookup(tokens, search_range=None, lookup_id=None, pin=False, **kw):
        """Stub for CacheEngine.lookup: treats ``tokens`` as a key list, pins a
        contiguous prefix in the source backend and records it in lookup_pins."""
        loc = search_range[0]
        src = engine.storage_manager.storage_backends[loc]
        pinned = []
        for k in tokens:
            if src.contains(k):
                if pin:
                    src.pin(k)
                pinned.append(k)
            else:
                break  # prefix semantics: stop at first miss
        engine.lookup_pins[lookup_id] = {loc: pinned}
        return len(pinned)

    engine.lookup = fake_lookup  # shadow the real (token-database-backed) lookup

    NEW_POS = ("*:0", "LocalCPUBackend")

    # ------------------------------------------------------------- run tests
    passed = 0
    failed = 0

    def _run(name, fn):
        nonlocal passed, failed
        try:
            fn()
            print(f"   ✓ {name} passed")
            passed += 1
        except Exception as e:
            print(f"   ✗ {name} FAILED: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    def t1_basic_promotion():
        import torch

        keys, originals = [], []
        for _ in range(3):
            k, t = put_chunk_in_cxl()
            keys.append(k)
            originals.append(t)
        for k in keys:
            assert not local_cpu.contains(k), "key should not be in CPU before prefetch"

        promoted = engine.move_intra_node(
            tokens=keys, old_position="CxlBackend",
            new_position=NEW_POS, event_id="evt-t1", do_copy=True,
        )
        assert promoted == len(keys), f"promoted {promoted} != {len(keys)}"
        for k, t in zip(keys, originals):
            assert local_cpu.contains(k), "key missing from CPU after prefetch"
            cached_t = local_cpu.hot_cache[k].tensor
            assert torch.equal(cached_t, t), "promoted tensor content mismatch"
            assert cxl.contains(k), "copy mode must keep the source copy"

    def t2_pins_released():
        # do_copy=True: after the move, no source key may remain pinned.
        keys = [put_chunk_in_cxl()[0] for _ in range(2)]
        engine.move_intra_node(
            tokens=keys, old_position="CxlBackend",
            new_position=NEW_POS, event_id="evt-t2", do_copy=True,
        )
        for k in keys:
            assert k in cxl.dict, "source metadata unexpectedly gone"
            assert cxl.dict[k].is_pinned is False, (
                "CXL pin leaked after prefetch (lookup_unpin did not run)"
            )
        # lookup_pins entry must have been popped by lookup_unpin.
        assert "evt-t2" not in engine.lookup_pins, "lookup_pins entry leaked"

    def t3_idempotent():
        keys = [put_chunk_in_cxl()[0] for _ in range(2)]
        p1 = engine.move_intra_node(
            tokens=keys, old_position="CxlBackend",
            new_position=NEW_POS, event_id="evt-t3a", do_copy=True,
        )
        assert p1 == len(keys)
        # Second promotion of the same, already-cached prefix: nothing new.
        p2 = engine.move_intra_node(
            tokens=keys, old_position="CxlBackend",
            new_position=NEW_POS, event_id="evt-t3b", do_copy=True,
        )
        assert p2 == 0, f"expected 0 new promotions, got {p2}"
        for k in keys:
            assert local_cpu.contains(k)
            assert cxl.dict[k].is_pinned is False

    def t4_partial_prefix():
        present = [put_chunk_in_cxl()[0] for _ in range(2)]
        missing = [make_key(__import__("torch").bfloat16) for _ in range(2)]
        all_keys = present + missing  # only the prefix `present` exists in CXL
        promoted = engine.move_intra_node(
            tokens=all_keys, old_position="CxlBackend",
            new_position=NEW_POS, event_id="evt-t4", do_copy=True,
        )
        assert promoted == len(present), f"promoted {promoted} != {len(present)}"
        for k in present:
            assert local_cpu.contains(k)
        for k in missing:
            assert not local_cpu.contains(k)

    def t5_move_mode_removes_source():
        k, _ = put_chunk_in_cxl()
        assert cxl.contains(k)
        promoted = engine.move_intra_node(
            tokens=[k], old_position="CxlBackend",
            new_position=NEW_POS, event_id="evt-t5", do_copy=False,
        )
        assert promoted == 1
        assert local_cpu.contains(k), "key must be in CPU after move"
        assert not cxl.contains(k), "move mode must remove the key from CXL"

    def t6_cpu_to_cxl_copy():
        import torch

        keys, originals = zip(*(put_chunk_in_cpu() for _ in range(2)))
        events = []
        cxl.set_kv_event_sink(events.append)
        result = engine.offload_to_backend(
            tokens=list(keys),
            operation_id="offload-t6",
            max_chunks=8,
        )
        assert result.success
        assert result.committed_chunks == len(keys)
        assert result.failed_chunks == 0
        assert result.bytes_written > 0
        assert events, "successful CXL writes must publish CacheStoreEvent"
        for key, original in zip(keys, originals):
            assert local_cpu.contains(key), "copy mode must retain CPU source"
            assert not local_cpu.hot_cache[key].is_pinned
            assert cxl.contains(key), "committed CXL object is not reopenable"
            staged = cxl.get_blocking(key)
            try:
                assert staged is not None and torch.equal(staged.tensor, original)
            finally:
                if staged is not None:
                    staged.ref_count_down()

        again = engine.offload_to_backend(
            list(keys), operation_id="offload-t6b", max_chunks=8
        )
        assert again.success
        assert again.already_present_chunks == len(keys)
        assert again.committed_chunks == 0

    def t7_cpu_to_cxl_budget_and_miss():
        missing = make_key(__import__("torch").bfloat16)
        result = engine.offload_to_backend(
            [missing], operation_id="offload-t7", max_chunks=8
        )
        assert not result.success
        assert result.failed_chunks == 0
        assert result.error == "source prefix not found"

    def t8_cpu_to_cxl_delayed_publish():
        keys, _ = zip(*(put_chunk_in_cpu() for _ in range(2)))
        events = []
        cxl.set_kv_event_sink(events.append)
        prepared = engine.offload_to_backend(
            tokens=list(keys),
            operation_id="offload-t8",
            max_chunks=8,
            publish_events=False,
        )
        assert prepared.success
        assert not events, "prepare must not publish a CXL-ready event"

        ready = engine.publish_pending_offload("offload-t8")
        assert ready.success
        assert ready.committed_chunks == len(keys)
        assert len(events) == len(keys)
        finalized = engine.finalize_pending_offload("offload-t8")
        assert finalized.success

    def t9_abort_after_one_rank_published():
        """A commit failure must roll back a rank that already published."""
        key, _ = put_chunk_in_cpu()
        prepared = engine.offload_to_backend(
            tokens=[key],
            operation_id="offload-t9",
            max_chunks=8,
            publish_events=False,
        )
        assert prepared.success
        ready = engine.publish_pending_offload("offload-t9")
        assert ready.success
        assert cxl.contains(key)

        aborted = engine.abort_pending_offload("offload-t9")
        assert aborted.success
        assert not cxl.contains(key), "commit abort must remove published CXL data"
        assert local_cpu.contains(key), "commit abort must retain the CPU source"

    def t10_abort_partial_cpu_to_cxl_prepare():
        """A partial prepare must destroy only the keys written by this op."""
        from lmcache.v1.storage_backend.cxl_backend import CxlPutResult, CxlPutStatus

        keys, _ = zip(*(put_chunk_in_cpu() for _ in range(2)))
        original_submit = cxl.submit_put_task
        calls = 0

        def fail_second(key, memory_obj, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                return CxlPutResult(CxlPutStatus.IO_FAILED, detail="injected failure")
            return original_submit(key, memory_obj, **kwargs)

        cxl.submit_put_task = fail_second
        try:
            prepared = engine.offload_to_backend(
                tokens=list(keys),
                operation_id="offload-t10",
                max_chunks=8,
                publish_events=False,
            )
        finally:
            cxl.submit_put_task = original_submit

        assert not prepared.success
        native_result, native_handle = cxl.cxl_shm.open_obj(
            cxl._get_cxl_key(keys[0])
        )
        try:
            assert native_result == 0, "successful prepare should allocate a CXL slot"
            assert not cxl.contains(
                keys[0]
            ), "prepared CXL data must stay invisible before commit"
        finally:
            if native_handle is not None:
                cxl.cxl_shm.close(native_handle)
        aborted = engine.abort_pending_offload("offload-t10")
        assert aborted.success
        assert aborted.committed_chunks == 1
        assert not cxl.contains(keys[0]), "abort must remove prepared CXL data"
        assert local_cpu.contains(keys[0]), "abort must retain the CPU source"

    def t11_native_write_failure_is_destroyed():
        """A failure after native create must not leave an in-use CXL slot."""
        key, _ = put_chunk_in_cpu()
        before = cxl.cxl_shm.debug_count_in_use()
        original_flush = cxl.cxl_shm.flush

        def fail_flush(*args, **kwargs):
            raise RuntimeError("injected flush failure")

        cxl.cxl_shm.flush = fail_flush
        try:
            failed = engine.offload_to_backend(
                tokens=[key],
                operation_id="offload-t11",
                max_chunks=8,
            )
        finally:
            cxl.cxl_shm.flush = original_flush

        assert not failed.success
        assert not cxl.contains(key), "failed native write must be destroyed"
        after = cxl.cxl_shm.debug_count_in_use()
        if before is not None and after is not None:
            assert after[0] == before[0]
            assert after[2] == before[2]
        assert local_cpu.contains(key), "failed write must retain the CPU source"

    def t12_reactive_backend_promotion():
        """Exercise the exact synchronous primitive used by route-time hints."""
        key, original = put_chunk_in_cxl()
        assert not local_cpu.contains(key)
        assert cxl.submit_prefetch_task(key) is True
        assert local_cpu.contains(key)
        assert torch.equal(local_cpu.hot_cache[key].tensor, original)
        assert cxl.contains(key), "reactive prefetch must keep the CXL source"

    print("Running CXL prefetch (move_intra_node) functional tests...")
    print("=" * 60)
    print(f"- cxl device : {dax}")
    print(f"- size       : {size_gb} GB")
    try:
        _run("t1_basic_promotion", t1_basic_promotion)
        _run("t2_pins_released", t2_pins_released)
        _run("t3_idempotent", t3_idempotent)
        _run("t4_partial_prefix", t4_partial_prefix)
        _run("t5_move_mode_removes_source", t5_move_mode_removes_source)
        _run("t6_cpu_to_cxl_copy", t6_cpu_to_cxl_copy)
        _run("t7_cpu_to_cxl_budget_and_miss", t7_cpu_to_cxl_budget_and_miss)
        _run("t8_cpu_to_cxl_delayed_publish", t8_cpu_to_cxl_delayed_publish)
        _run("t9_abort_after_one_rank_published", t9_abort_after_one_rank_published)
        _run("t10_abort_partial_cpu_to_cxl_prepare", t10_abort_partial_cpu_to_cxl_prepare)
        _run("t11_native_write_failure_is_destroyed", t11_native_write_failure_is_destroyed)
        _run("t12_reactive_backend_promotion", t12_reactive_backend_promotion)
    finally:
        try:
            cxl.close()
        except Exception:
            pass
        loop.close()
        if created_tmpfs:
            try:
                os.remove(dax)
            except OSError:
                pass

    print("\n" + "=" * 60)
    print(f"CXL prefetch tests completed: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="CXL intra-node prefetch functional tests (no pytest)."
    )
    parser.add_argument("--dax-device", type=str, default=None,
                        help="DAX device path (default: a /dev/shm tmpfs file).")
    parser.add_argument("--size-gb", type=float, default=1.0,
                        help="CXL device size in GB (default 1.0).")
    args = parser.parse_args()
    return run_real_functional_test(dax_device=args.dax_device, size_gb=args.size_gb)


if __name__ == "__main__":
    sys.exit(main())
