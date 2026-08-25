# SPDX-License-Identifier: Apache-2.0
"""
Verification step 1 (automated half): does LMCache emit tier-tagged KV events?

This is the source end of the GPU->CPU->CXL tier-aware routing pipeline that the
Dynamo `kv-strata` router consumes. If this passes, LMCache is publishing the
`medium="CPU"` / `medium="CXL"` events (with non-empty token_ids) that the router
needs; the remaining verification (router ingests them, routing is sticky) needs
a live Dynamo cluster and is covered by verify_cxl_strata.sh's checklist.

Real `LMCacheEngine` with `enable_kv_events=True`, real CxlBackend on /dev/shm.
Requires CUDA. SKIPs otherwise.

Run:
  PYTHONHASHSEED=0 pytest -q tests/v1/test_cxl_strata_pipeline.py -s
"""

# Standard
import os
import random
from collections import Counter

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import (
    CacheStoreEvent,
    mock_up_broadcast_fn,
    mock_up_broadcast_object_fn,
)
from lmcache.v1.cache_engine import LMCacheEngineBuilder
from lmcache.v1.config import LMCacheEngineConfig

# Local
from .utils import (
    create_gpu_connector,
    dumb_metadata,
    generate_kv_cache_paged_list_tensors,
    generate_tokens,
    recover_engine_states,
)

_CXL_DEVICE = "/dev/shm/lmcache_cxl_strata.img"
_CXL_SIZE_GB = 1.0
_CHUNK = 256
_NUM_TOKENS = 1024
_EXPECTED = (_NUM_TOKENS // _CHUNK) * _CHUNK


def _require_env():
    if not torch.cuda.is_available():
        pytest.skip("vLLM paged GPU connector requires CUDA.")
    try:
        import lmcache.v1.storage_backend.cxl_shm_binding  # noqa: F401
    except Exception as e:
        pytest.skip(f"libcxl_shm.so unavailable: {e}")


def test_lmcache_emits_tier_tagged_kv_events():
    _require_env()

    if os.environ.get("PYTHONHASHSEED") is None:
        # Not fatal single-process, but REQUIRED for multi-node + shared CXL.
        print(
            "\nWARNING: PYTHONHASHSEED is not set. Fine for this single-process "
            "check, but you MUST set it (e.g. PYTHONHASHSEED=0) on every node "
            "before running multi-node with a shared CXL pool."
        )

    size_bytes = int(_CXL_SIZE_GB * 1024**3)
    with open(_CXL_DEVICE, "wb") as f:
        f.truncate(size_bytes)
    os.environ["LMCACHE_CXL_DAX_DEVICE"] = _CXL_DEVICE
    os.environ["LMCACHE_CXL_DAX_DEVICE_SIZE"] = str(size_bytes)

    device = "cuda"
    num_blocks, block_size, dtype = 1000, 16, torch.bfloat16
    kv_shape = (32, 2, _CHUNK, 8, 128)

    connector = create_gpu_connector(1024, 32)
    tokens = generate_tokens(_NUM_TOKENS, device)
    kv_cache = generate_kv_cache_paged_list_tensors(num_blocks, device, block_size, dtype)
    retrieved_cache = generate_kv_cache_paged_list_tensors(
        num_blocks, device, block_size, dtype
    )
    slot_mapping = torch.tensor(
        random.sample(range(0, num_blocks * block_size), _NUM_TOKENS), device=device
    )

    cfg = LMCacheEngineConfig.from_defaults(
        chunk_size=_CHUNK,
        local_cpu=True,
        max_local_cpu_size=2.0,
        save_unfull_chunk=False,
        enable_kv_events=True,  # <-- the switch under test
        extra_config={
            "cxl_dax_device": _CXL_DEVICE,
            "max_cxl_size": _CXL_SIZE_GB,
            "cxl_reset_on_init": "full",
        },
    )
    engine = LMCacheEngineBuilder.get_or_create(
        "cxl_strata_pipeline",
        cfg,
        dumb_metadata("vllm", kv_shape),
        connector,
        mock_up_broadcast_fn,
        mock_up_broadcast_object_fn,
    )

    try:
        # Store fans out to LocalCPUBackend (engine emits medium="CPU") and
        # CxlBackend (backend sink -> engine enriches to medium="CXL").
        engine.store(tokens=tokens, kvcaches=kv_cache, slot_mapping=slot_mapping)
        recover_engine_states(engine)

        # Ensure the store (incl. the CXL put + its sink event) has landed.
        import time

        deadline = time.time() + 5.0
        while (
            engine.lookup(tokens, search_range=["CxlBackend"]) < _EXPECTED
            and time.time() < deadline
        ):
            time.sleep(0.02)

        # A retrieve also exercises the CXL->CPU meta backfill path.
        engine.retrieve(
            tokens, kvcaches=retrieved_cache, slot_mapping=slot_mapping
        )
        recover_engine_states(engine)

        events = list(engine.get_kv_events())
        store_events = [e for e in events if isinstance(e, CacheStoreEvent)]

        mediums = Counter((e.medium or "GPU").upper() for e in store_events)
        print(f"\nKV store events by medium: {dict(mediums)}")

        assert store_events, "no KV store events emitted (is enable_kv_events on?)"
        assert mediums.get("CPU", 0) > 0, (
            "no medium=CPU store events; router would not see the CPU tier"
        )
        assert mediums.get("CXL", 0) > 0, (
            "no medium=CXL store events; router would not see the CXL tier "
            "(store did not reach CxlBackend, or the sink is not wired)"
        )

        # CXL events must carry token_ids, otherwise the indexer computes a wrong
        # tokens_hash and the block_hash will not line up (the CXL_ENRICH_MISS
        # warning path). This is the multi-node hash-consistency guard.
        cxl_events = [e for e in store_events if (e.medium or "").upper() == "CXL"]
        empty = [e for e in cxl_events if not e.token_ids]
        assert not empty, (
            f"{len(empty)}/{len(cxl_events)} CXL store events had EMPTY token_ids "
            "(CXL_ENRICH_MISS) -> would cause indexer block_hash mismatch. "
            "Check that store metadata backfill runs before the CXL sink fires."
        )

        print(
            f"OK: emitted {mediums.get('CPU', 0)} CPU + {len(cxl_events)} CXL "
            "tier-tagged store events, all CXL events carry token_ids."
        )
    finally:
        LMCacheEngineBuilder.destroy("cxl_strata_pipeline")
        try:
            os.remove(_CXL_DEVICE)
        except OSError:
            pass
