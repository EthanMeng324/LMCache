# SPDX-License-Identifier: Apache-2.0
"""
End-to-end verification of intra-node CXL->CPU prefetch (Phase 1).

Unlike tests/v1/storage_backend/test_cxl_prefetch.py (which stubs `lookup`),
this drives a REAL `LMCacheEngine` so `move_intra_node` runs through the real
`lookup` + token_database + StorageManager. The only simulated piece is the CXL
device, which is a `/dev/shm` file (DRAM, no root).

Flow (the exact gap the unit test could not cover):
  1. store a token sequence  -> lands in LocalCPUBackend AND CxlBackend
  2. clear the LocalCPUBackend tier -> prefix now lives ONLY in CXL
  3. assert lookup(CPU)==0 and lookup(CXL)==full
  4. call the REAL engine.move_intra_node(tokens, CXL -> LocalCPUBackend)
  5. assert lookup(CPU) is full again, and retrieve() returns the correct KV

Requires CUDA (the vLLM paged GPU connector has no CPU implementation) and a
loadable libcxl_shm.so. SKIPs otherwise.

Run:
  pytest -q tests/v1/test_cxl_prefetch_e2e.py -s
"""

# Standard
import os
import random
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import mock_up_broadcast_fn, mock_up_broadcast_object_fn
from lmcache.v1.cache_engine import LMCacheEngineBuilder
from lmcache.v1.config import LMCacheEngineConfig

# Local
from .utils import (
    check_paged_kv_cache_equal,
    create_gpu_connector,
    dumb_metadata,
    generate_kv_cache_paged_list_tensors,
    generate_tokens,
    recover_engine_states,
)

_CXL_DEVICE = "/dev/shm/lmcache_cxl_e2e.img"
_CXL_SIZE_GB = 2.0


def _wait_until(pred, timeout=5.0, interval=0.01):
    start = time.time()
    while time.time() - start < timeout:
        if pred():
            return True
        time.sleep(interval)
    return False


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="vLLM paged GPU connector requires CUDA.",
)
def test_cxl_intra_node_prefetch_e2e(autorelease_v1):
    # Skip early if the CXL C library cannot be loaded.
    try:
        import lmcache.v1.storage_backend.cxl_shm_binding  # noqa: F401
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"libcxl_shm.so unavailable: {e}")

    size_bytes = int(_CXL_SIZE_GB * 1024**3)
    try:
        with open(_CXL_DEVICE, "wb") as f:
            f.truncate(size_bytes)
    except OSError as e:  # pragma: no cover
        pytest.skip(f"cannot create tmpfs CXL device {_CXL_DEVICE}: {e}")

    os.environ["LMCACHE_CXL_DAX_DEVICE"] = _CXL_DEVICE
    os.environ["LMCACHE_CXL_DAX_DEVICE_SIZE"] = str(size_bytes)

    device = "cuda"
    fmt = "vllm"
    chunk_size = 256
    num_tokens = 1024
    num_blocks = 1000
    block_size = 16
    dtype = torch.bfloat16
    kv_shape = (32, 2, chunk_size, 8, 128)

    connector = create_gpu_connector(1024, 32)
    tokens = generate_tokens(num_tokens, device)
    kv_cache = generate_kv_cache_paged_list_tensors(num_blocks, device, block_size, dtype)
    retrieved_cache = generate_kv_cache_paged_list_tensors(
        num_blocks, device, block_size, dtype
    )
    slot_mapping = torch.tensor(
        random.sample(range(0, num_blocks * block_size), num_tokens), device=device
    )

    cfg = LMCacheEngineConfig.from_defaults(
        chunk_size=chunk_size,
        local_cpu=True,
        max_local_cpu_size=2.0,
        save_unfull_chunk=False,
        extra_config={
            "cxl_dax_device": _CXL_DEVICE,
            "max_cxl_size": _CXL_SIZE_GB,
            "cxl_reset_on_init": "full",
        },
    )

    engine = autorelease_v1(
        LMCacheEngineBuilder.get_or_create(
            "cxl_prefetch_e2e",
            cfg,
            dumb_metadata(fmt, kv_shape),
            connector,
            mock_up_broadcast_fn,
            mock_up_broadcast_object_fn,
        )
    )

    expected = (num_tokens // chunk_size) * chunk_size

    # ---- 1. store (fans out to LocalCPUBackend + CxlBackend) -----------------
    engine.store(tokens=tokens, kvcaches=kv_cache, slot_mapping=slot_mapping)
    recover_engine_states(engine)
    assert _wait_until(lambda: engine.lookup(tokens) >= expected, timeout=5.0), (
        "store did not become visible in time"
    )

    # The prefix must be present in CXL specifically.
    assert engine.lookup(tokens, search_range=["CxlBackend"]) >= expected, (
        "store did not land in CxlBackend"
    )

    # ---- 2. drop the CPU tier: prefix now lives only in CXL ------------------
    engine.clear(locations=["LocalCPUBackend"])
    assert engine.lookup(tokens, search_range=["LocalCPUBackend"]) == 0, (
        "LocalCPUBackend not cleared"
    )
    assert engine.lookup(tokens, search_range=["CxlBackend"]) >= expected, (
        "CXL copy should survive clearing the CPU tier"
    )

    # ---- 3. REAL move_intra_node: promote CXL -> LocalCPUBackend --------------
    promoted = engine.move_intra_node(
        tokens=tokens,
        old_position="CxlBackend",
        new_position=("local", "LocalCPUBackend"),
        event_id="e2e-evt",
        do_copy=True,
    )
    assert promoted > 0, "move_intra_node promoted nothing"

    # ---- 4. the CPU tier now serves the prefix again -------------------------
    cpu_hit = engine.lookup(tokens, search_range=["LocalCPUBackend"])
    assert cpu_hit >= expected, (
        f"CPU tier should serve prefix after prefetch, got {cpu_hit} < {expected}"
    )

    # ---- 5. content correctness through the normal retrieve path -------------
    ret_mask = engine.retrieve(
        tokens, kvcaches=retrieved_cache, slot_mapping=slot_mapping
    )
    recover_engine_states(engine)
    assert torch.sum(ret_mask) == expected
    check_paged_kv_cache_equal(
        retrieved_cache, kv_cache, slot_mapping[:expected]
    )


def teardown_module(module):
    try:
        os.remove(_CXL_DEVICE)
    except OSError:
        pass
    LMCacheEngineBuilder.destroy("cxl_prefetch_e2e")
