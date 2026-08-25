# SPDX-License-Identifier: Apache-2.0
"""
Regression test for cross-process reuse through one shared CXL mapping.

The writer and reader run as separate processes against one /dev/shm fake CXL
device (no root, no GPU, no Dynamo). The reader never performs a local put before
looking up the writer's object. Engine metadata must seed the KV tensor schema so
the cold reader can reconstruct Python metadata from native object sizes.

The test also verifies that a different TP rank does not alias the writer's
object. ``worker_id`` is the TP rank and intentionally remains part of the key;
cross-replica reuse is valid between equal ranks, not between different shards.

The writer and reader are separate processes sharing one device file, because a
fresh process is exactly what another worker or a restarted worker looks like:
its process-local CXL index is empty and must be rebuilt from shared metadata.

    python tests/v1/storage_backend/test_cxl_reuse_repro.py

The script exits non-zero if cold reuse, post-bootstrap reuse, or TP-rank
isolation does not match the required behavior.
"""

import argparse
import asyncio
import ctypes
import os
import subprocess
import sys
from unittest.mock import Mock

# Both processes must build the byte-identical key, so nothing here may vary.
CHUNK_HASH = 0x0BADC0DE12345678
SHAPE = (2, 4, 8, 64)
DEFAULT_DEV = "/dev/shm/lmcache_cxl_reuse_repro.img"


def _skip(reason: str) -> int:
    print(f"SKIP: {reason}")
    return 0


def build_backend(dax: str, size_gb: float, reset_mode: str):
    """Construct a real CxlBackend against `dax`. reset_mode: 'full' | 'none'."""
    import torch  # noqa: F401

    from lmcache.config import LMCacheEngineMetadata
    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.v1.memory_management import AdHocMemoryAllocator
    from lmcache.v1.storage_backend.cxl_backend import CxlBackend
    from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

    size_bytes = int(size_gb * 1024**3)
    os.environ["LMCACHE_CXL_DAX_DEVICE"] = dax
    os.environ["LMCACHE_CXL_DAX_DEVICE_SIZE"] = str(size_bytes)

    cpu_config = Mock(spec=LMCacheEngineConfig)
    cpu_config.cache_policy = "LRU"
    cpu_config.local_cpu = True
    cpu_config.lmcache_instance_id = "repro"
    cpu_config.use_layerwise = False
    cpu_config.enable_blending = False
    local_cpu = LocalCPUBackend(
        config=cpu_config,
        memory_allocator=AdHocMemoryAllocator(device="cpu"),
        dst_device="cpu",
        lmcache_worker=None,
    )

    cxl_config = Mock(spec=LMCacheEngineConfig)
    cxl_config.cache_policy = "LRU"
    cxl_config.local_cpu = True
    cxl_config.chunk_size = 256
    cxl_config.lmcache_instance_id = "repro"
    cxl_config.extra_config = {
        "cxl_num_procs": 1,
        "cxl_rank": 0,
        "max_cxl_size": float(size_gb),
        "cxl_dax_device": dax,
        "cxl_reset_on_init": reset_mode,
    }

    loop = asyncio.new_event_loop()
    metadata = LMCacheEngineMetadata(
        model_name="cxl",
        world_size=1,
        worker_id=0,
        fmt="repro",
        kv_dtype=torch.bfloat16,
        kv_shape=(4, 2, 256, 1, 64),
    )
    cxl = CxlBackend(
        config=cxl_config,
        loop=loop,
        local_cpu_backend=local_cpu,
        dst_device="cpu",
        metadata=metadata,
    )
    return cxl, AdHocMemoryAllocator(device="cpu"), loop


def make_key(worker_id: int, chunk_hash: int):
    import torch

    from lmcache.utils import CacheEngineKey

    return CacheEngineKey(
        fmt="repro",
        model_name="cxl",
        world_size=1,
        worker_id=worker_id,
        chunk_hash=chunk_hash,
        dtype=torch.bfloat16,
    )


def put(cxl, alloc, key):
    import torch

    from lmcache.v1.memory_management import MemoryFormat

    obj = alloc.allocate(SHAPE, torch.bfloat16, fmt=MemoryFormat.KV_2LTD)
    obj.tensor.fill_(1.0)
    cxl.submit_put_task(key, obj)


# --------------------------------------------------------------- phase: writer
def phase_writer(dax: str, size_gb: float) -> int:
    cxl, alloc, _ = build_backend(dax, size_gb, reset_mode="full")
    key = make_key(worker_id=0, chunk_hash=CHUNK_HASH)
    put(cxl, alloc, key)
    put(cxl, alloc, make_key(worker_id=0, chunk_hash=CHUNK_HASH ^ 0xAAAA))
    ok = cxl.contains(key)
    print(f"[writer] cxl_key      = {cxl._get_cxl_key(key)}")
    print(f"[writer] shm stats    = {cxl.cxl_shm.debug_count_in_use()}")
    print(f"[writer] put key (worker_id=0) -> contains={ok}")
    if not ok:
        print("[writer] FATAL: the writer cannot even see its own key.")
        return 2
    print("[writer] key is now resident in the CXL device; exiting.")
    return 0


# --------------------------------------------------------------- phase: reader
def phase_reader(dax: str, size_gb: float) -> int:
    """A fresh process whose CXL Python index is empty."""
    import torch

    cxl, alloc, _ = build_backend(dax, size_gb, reset_mode="none")
    key = make_key(worker_id=0, chunk_hash=CHUNK_HASH)

    # Is the writer's object still on the device at all, and do we address it
    # with the same name the writer used?
    ck = cxl._get_cxl_key(key)
    stats = cxl.cxl_shm.debug_count_in_use()
    rc, hnd = cxl.cxl_shm.open_obj(ck)
    raw_data_ok = False
    if rc == 0 and hnd is not None and hnd.obj_contents and hnd.mapped_addr:
        logical_size = int(hnd.obj_contents.actual_size)
        raw = ctypes.string_at(int(hnd.mapped_addr), logical_size)
        values = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16)
        raw_data_ok = bool(torch.all(values == 1.0))
        cxl.cxl_shm.close(hnd)
    print(f"[reader] cxl_key      = {ck}   (must equal the writer's)")
    print(f"[reader] shm stats    = {stats}")
    print(f"[reader] raw open_obj({ck}) -> rc={rc}   (0 = object IS on device)")
    print(f"[reader] raw payload matches writer                    -> {raw_data_ok}")

    # R2: a fresh worker must see an object another worker wrote without first
    #     performing a local put.
    r2 = cxl.contains(key)
    print(f"[reader] R2 cold contains(foreign key), no local put yet -> {r2}")

    # R3: direct get on another foreign key must neither deadlock nor require an
    #     earlier contains call. It must reconstruct and return the exact KV.
    cold_get_key = make_key(worker_id=0, chunk_hash=CHUNK_HASH ^ 0xAAAA)
    cold_obj = cxl.get_blocking(cold_get_key)
    r3 = bool(
        cold_obj is not None
        and cold_obj.tensor is not None
        and torch.all(cold_obj.tensor == 1.0)
    )
    if cold_obj is not None:
        cold_obj.ref_count_down()
    print(f"[reader] R3 direct cold get returns writer payload       -> {r3}")

    # R4: an unrelated local put must not change the already-working result.
    put(cxl, alloc, make_key(worker_id=0, chunk_hash=CHUNK_HASH ^ 0xFFFF))
    r4 = cxl.contains(key)
    print(f"[reader] R4 same key, after one unrelated local put      -> {r4}")

    # R5: same prefix (same chunk_hash), different worker_id. Tests whether the
    #     CXL pool is partitioned per worker.
    r5 = cxl.contains(make_key(worker_id=1, chunk_hash=CHUNK_HASH))
    print(f"[reader] R5 same chunk_hash, worker_id=1                 -> {r5}")

    print("\n================ VERDICT ================")
    if rc == 0 and raw_data_ok and r2 and r3 and r4:
        print("PASS: a cold process can adopt and read foreign CXL objects.")
    else:
        print("FAIL: cold shared-object reuse is not working.")

    if not r5:
        print("PASS: a different TP rank remains isolated from the writer's shard.")
    else:
        print("FAIL: different TP ranks unexpectedly alias one CXL object.")
    print("=========================================")
    return 0 if rc == 0 and raw_data_ok and r2 and r3 and r4 and not r5 else 3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["writer", "reader"], default=None)
    ap.add_argument("--device", default=os.environ.get("LMCACHE_CXL_DAX_DEVICE"))
    ap.add_argument("--size-gb", type=float, default=1.0)
    args = ap.parse_args()

    dax = args.device or DEFAULT_DEV

    # A child phase: run it and return.
    if args.phase:
        try:
            return (phase_writer if args.phase == "writer" else phase_reader)(
                dax, args.size_gb
            )
        except Exception as e:
            print(f"ERROR: phase {args.phase} could not run: {e}")
            return 4

    # Driver: create the device, then run writer and reader as separate processes.
    created = False
    if not os.path.exists(dax):
        try:
            with open(dax, "wb") as f:
                f.truncate(int(args.size_gb * 1024**3))
            created = True
        except OSError as e:
            return _skip(f"cannot create fake CXL device at {dax}: {e}")

    try:
        for phase in ("writer", "reader"):
            print(f"\n----- {phase} (fresh process) -----")
            try:
                rc = subprocess.call(
                    [
                        sys.executable,
                        __file__,
                        "--phase",
                        phase,
                        "--device",
                        dax,
                        "--size-gb",
                        str(args.size_gb),
                    ],
                    timeout=60,
                )
            except subprocess.TimeoutExpired:
                print(f"phase {phase} timed out (possible cold-read deadlock)")
                return 5
            if rc != 0:
                print(f"phase {phase} exited rc={rc}")
                return rc
    finally:
        if created:
            try:
                os.remove(dax)
            except OSError:
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
