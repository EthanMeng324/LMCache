# SPDX-License-Identifier: Apache-2.0
"""
Phase 2: predictive CXL->CPU prefetch (session-continuation heuristic).

Strategy under test: A2 + B1.
  A2 (trigger)  only requests whose retrieve reused a prefix >= min_hit_tokens
                count as a "continuation".
  B1 (timing)   after serving such a request, asynchronously promote the
                CXL-resident part of its prefix into the CPU tier, targeting the
                NEXT turn.

Real `LMCacheEngine`, real CxlBackend (simulated with a /dev/shm file, no root),
real retrieve path. Requires CUDA (the vLLM paged connector is GPU-only).

Run:
  pytest -q tests/v1/test_cxl_prefetch_p2.py -s

Cases:
  P1/P6  baseline (prefetch off): retrieve does NOT write CXL hits back to CPU
         -> proves the problem is real AND that the off-switch is a no-op.
  P2     continuation triggers the promotion; CPU tier serves the prefix after.
  P3     A2 gate: hit below min_hit_tokens -> no promotion.
  P4     budget: max_chunks caps how much is promoted.
  P5     dedup/idempotency: repeated retrieves do not pile up in-flight moves.
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

_CXL_DEVICE = "/dev/shm/lmcache_cxl_p2.img"
_CXL_SIZE_GB = 1.0

_CHUNK = 256
_NUM_TOKENS = 1024
_EXPECTED = (_NUM_TOKENS // _CHUNK) * _CHUNK  # 1024


def _wait_until(pred, timeout=10.0, interval=0.02) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        if pred():
            return True
        time.sleep(interval)
    return False


def _require_env():
    if not torch.cuda.is_available():
        pytest.skip("vLLM paged GPU connector requires CUDA.")
    try:
        import lmcache.v1.storage_backend.cxl_shm_binding  # noqa: F401
    except Exception as e:
        pytest.skip(f"libcxl_shm.so unavailable: {e}")


class _Fixture:
    """A real engine with LocalCPUBackend + CxlBackend, plus GPU KV buffers."""

    def __init__(self, instance_id: str, prefetch_cfg: dict):
        size_bytes = int(_CXL_SIZE_GB * 1024**3)
        with open(_CXL_DEVICE, "wb") as f:
            f.truncate(size_bytes)
        os.environ["LMCACHE_CXL_DAX_DEVICE"] = _CXL_DEVICE
        os.environ["LMCACHE_CXL_DAX_DEVICE_SIZE"] = str(size_bytes)

        device = "cuda"
        num_blocks, block_size, dtype = 1000, 16, torch.bfloat16
        kv_shape = (32, 2, _CHUNK, 8, 128)

        self.instance_id = instance_id
        self.connector = create_gpu_connector(1024, 32)
        self.tokens = generate_tokens(_NUM_TOKENS, device)
        self.kv_cache = generate_kv_cache_paged_list_tensors(
            num_blocks, device, block_size, dtype
        )
        self.retrieved_cache = generate_kv_cache_paged_list_tensors(
            num_blocks, device, block_size, dtype
        )
        self.slot_mapping = torch.tensor(
            random.sample(range(0, num_blocks * block_size), _NUM_TOKENS),
            device=device,
        )

        extra = {
            "cxl_dax_device": _CXL_DEVICE,
            "max_cxl_size": _CXL_SIZE_GB,
            "cxl_reset_on_init": "full",
        }
        extra.update(prefetch_cfg)

        cfg = LMCacheEngineConfig.from_defaults(
            chunk_size=_CHUNK,
            local_cpu=True,
            max_local_cpu_size=2.0,
            save_unfull_chunk=False,
            extra_config=extra,
        )
        self.engine = LMCacheEngineBuilder.get_or_create(
            instance_id,
            cfg,
            dumb_metadata("vllm", kv_shape),
            self.connector,
            mock_up_broadcast_fn,
            mock_up_broadcast_object_fn,
        )

    # -- helpers -----------------------------------------------------------
    def store_and_spill_to_cxl(self):
        """Store the sequence, then drop the CPU tier so it lives only in CXL."""
        e = self.engine
        e.store(
            tokens=self.tokens, kvcaches=self.kv_cache, slot_mapping=self.slot_mapping
        )
        recover_engine_states(e)
        assert _wait_until(lambda: e.lookup(self.tokens) >= _EXPECTED), (
            "store did not become visible"
        )
        assert e.lookup(self.tokens, search_range=["CxlBackend"]) >= _EXPECTED, (
            "store did not land in CxlBackend"
        )
        e.clear(locations=["LocalCPUBackend"])
        assert self.cpu_hit() == 0, "CPU tier not cleared"
        assert e.lookup(self.tokens, search_range=["CxlBackend"]) >= _EXPECTED

    def retrieve(self, tokens=None, session_id=None, request_id=None):
        tokens = self.tokens if tokens is None else tokens
        request_configs = None
        if session_id is not None:
            request_configs = {"lmcache.session_id": session_id}
        mask = self.engine.retrieve(
            tokens,
            kvcaches=self.retrieved_cache,
            slot_mapping=self.slot_mapping[: len(tokens)],
            request_configs=request_configs,
            req_id=request_id,
        )
        recover_engine_states(self.engine)
        return mask

    def cpu_hit(self) -> int:
        return self.engine.lookup(self.tokens, search_range=["LocalCPUBackend"])

    def close(self):
        LMCacheEngineBuilder.destroy(self.instance_id)
        try:
            os.remove(_CXL_DEVICE)
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# P1 / P6: baseline. Prefetch OFF -> retrieve must NOT write CXL hits into CPU.
# This is both the "the problem is real" proof and the off-switch regression.
# --------------------------------------------------------------------------- #
def test_baseline_retrieve_does_not_write_back():
    _require_env()
    fx = _Fixture("cxl_p2_baseline", {"cxl_prefetch_enabled": False})
    try:
        assert fx.engine._cxl_prefetcher is None, "prefetcher must be disabled"
        fx.store_and_spill_to_cxl()

        mask = fx.retrieve()
        assert torch.sum(mask) == _EXPECTED, "retrieve should serve the prefix from CXL"

        # Give any (hypothetical) background work a chance, then assert nothing
        # was promoted: today's retrieve path leaves CXL hits in CXL.
        time.sleep(0.5)
        assert fx.cpu_hit() == 0, (
            "baseline: retrieve must not back-fill the CPU tier "
            f"(got {fx.cpu_hit()} tokens in CPU)"
        )
    finally:
        fx.close()


# --------------------------------------------------------------------------- #
# P2 + P5: a continuation triggers the promotion; repeated retrieves are safe.
# --------------------------------------------------------------------------- #
def test_continuation_triggers_prefetch():
    _require_env()
    fx = _Fixture(
        "cxl_p2_trigger",
        {
            "cxl_prefetch_enabled": True,
            "cxl_prefetch_min_hit_tokens": _CHUNK,  # 1 chunk
            "cxl_prefetch_max_chunks": 32,
        },
    )
    try:
        assert fx.engine._cxl_prefetcher is not None, "prefetcher must be enabled"
        fx.store_and_spill_to_cxl()

        # Turn N: serves from CXL (a continuation: hit == full prefix).
        mask = fx.retrieve()
        assert torch.sum(mask) == _EXPECTED

        # B1: the background promotion pulls the prefix back into the CPU tier.
        assert _wait_until(lambda: fx.cpu_hit() >= _EXPECTED, timeout=10.0), (
            f"prefetch did not promote the prefix to CPU (cpu_hit={fx.cpu_hit()})"
        )

        # Turn N+1 now hits the CPU tier, and the content is still correct.
        mask2 = fx.retrieve()
        assert torch.sum(mask2) == _EXPECTED
        check_paged_kv_cache_equal(
            fx.retrieved_cache, fx.kv_cache, fx.slot_mapping[:_EXPECTED]
        )

        # P5: no in-flight promotions leaked after everything settles.
        assert _wait_until(
            lambda: len(fx.engine._cxl_prefetcher._inflight) == 0, timeout=5.0
        ), "in-flight prefetch set leaked"
    finally:
        fx.close()


def test_global_temporal_prefetch_reconstructs_prefix_key():
    """A temporal hint must carry the full prefix and select one chunk.

    LMCache's ChunkedTokenDatabase uses a parent-dependent prefix hash.  This
    regression test exercises the same ``prefetch_tokens(..., start_chunk,
    end_chunk)`` contract used by Dynamo after the router sends a full prefix
    descriptor.
    """
    _require_env()
    fx = _Fixture(
        "cxl_global_temporal_prefix",
        {
            "cxl_global_prefetch_enabled": True,
            "cxl_global_prefetch_workers": 1,
            "cxl_global_prefetch_max_chunks": 1,
            "cxl_global_prefetch_max_inflight_bytes": 128 * 1024**2,
        },
    )
    try:
        fx.store_and_spill_to_cxl()
        infos = list(fx.engine.token_database.process_tokens(tokens=fx.tokens))
        target_key = infos[1][2]
        assert not fx.engine.storage_manager.storage_backends["LocalCPUBackend"].contains(
            target_key
        )

        task_id = "global-temporal-prefix-regression"
        scheduled = fx.engine.prefetch_tokens(
            fx.tokens,
            request_id="global-temporal-prefix-request",
            route_epoch=1,
            task_id=task_id,
            start_chunk=1,
            end_chunk=2,
        )
        assert scheduled == 1, f"expected one scheduled chunk, got {scheduled}"
        assert _wait_until(
            lambda: (fx.engine.prefetch_status(task_id) or {}).get("state")
            == "READY",
            timeout=10.0,
        ), f"global temporal prefetch did not become READY: {fx.engine.prefetch_status(task_id)}"
        status = fx.engine.prefetch_status(task_id)
        assert status is not None
        assert status["ready_chunks"] == 1
        assert status["bytes_copied"] > 0, (
            "first temporal prefetch must perform a physical CXL->CPU copy"
        )
        assert fx.engine.storage_manager.storage_backends["LocalCPUBackend"].contains(
            target_key
        )

        # A repeated hint is an idempotent success, not a physical failure.
        # This is the state Dynamo sees when another request predicts the same
        # chunk while the first promotion has already reached LocalCPU.
        duplicate_task_id = "global-temporal-prefix-duplicate"
        duplicate_scheduled = fx.engine.prefetch_tokens(
            fx.tokens,
            request_id="global-temporal-prefix-duplicate-request",
            route_epoch=1,
            task_id=duplicate_task_id,
            start_chunk=1,
            end_chunk=2,
        )
        assert duplicate_scheduled == 1
        assert _wait_until(
            lambda: (fx.engine.prefetch_status(duplicate_task_id) or {}).get("state")
            == "READY",
            timeout=10.0,
        ), (
            "already-present global temporal prefetch was reported as failure: "
            f"{fx.engine.prefetch_status(duplicate_task_id)}"
        )
        duplicate_status = fx.engine.prefetch_status(duplicate_task_id)
        assert duplicate_status is not None
        assert duplicate_status["ready_chunks"] == 1
        assert duplicate_status["bytes_copied"] == 0
    finally:
        fx.close()


# --------------------------------------------------------------------------- #
# P3: the A2 gate. A hit below min_hit_tokens is not treated as a continuation.
# --------------------------------------------------------------------------- #
def test_below_threshold_does_not_prefetch():
    _require_env()
    fx = _Fixture(
        "cxl_p2_threshold",
        {
            "cxl_prefetch_enabled": True,
            # Impossibly high: even a full-prefix hit is "not a continuation".
            "cxl_prefetch_min_hit_tokens": 10**9,
        },
    )
    try:
        fx.store_and_spill_to_cxl()
        mask = fx.retrieve()
        assert torch.sum(mask) == _EXPECTED

        time.sleep(0.5)
        assert fx.cpu_hit() == 0, (
            "A2 gate failed: promoted despite hit < min_hit_tokens"
        )
    finally:
        fx.close()


# --------------------------------------------------------------------------- #
# P4: the budget cap bounds how much a single event promotes.
# --------------------------------------------------------------------------- #
def test_budget_cap_limits_promotion():
    _require_env()
    fx = _Fixture(
        "cxl_p2_budget",
        {
            "cxl_prefetch_enabled": True,
            "cxl_prefetch_min_hit_tokens": _CHUNK,
            "cxl_prefetch_max_chunks": 1,  # promote at most one chunk
        },
    )
    try:
        fx.store_and_spill_to_cxl()
        mask = fx.retrieve()
        assert torch.sum(mask) == _EXPECTED

        # Exactly one chunk should land in CPU, and it must stay at one.
        assert _wait_until(lambda: fx.cpu_hit() >= _CHUNK, timeout=10.0), (
            "budget-capped prefetch promoted nothing"
        )
        time.sleep(0.5)
        assert fx.cpu_hit() == _CHUNK, (
            f"budget cap violated: expected {_CHUNK} tokens in CPU, got {fx.cpu_hit()}"
        )
    finally:
        fx.close()


# --------------------------------------------------------------------------- #
# Session association: a first-turn A-only hit followed by an A+B hit learns
# A -> B; a later session's A-only turn pulls B from CXL into LocalCPU.
# --------------------------------------------------------------------------- #
def test_association_prefetch_promotes_correlated_suffix():
    _require_env()
    fx = _Fixture(
        "cxl_association_prefetch",
        {
            "cxl_association_prefetch_enabled": True,
            "cxl_association_min_support": 0.5,
            "cxl_association_min_confidence": 0.5,
            "cxl_association_min_lift": 1.0,
            "cxl_association_max_prefetch_chunks": 1,
            # One test KV chunk is 32 MiB; reserve exactly one promotion.
            "cxl_association_max_prefetch_bytes": 32 * 1024 * 1024,
            "cxl_association_max_inflight_bytes": 32 * 1024 * 1024,
        },
    )
    try:
        prefetcher = fx.engine._cxl_association_prefetcher
        assert prefetcher is not None
        fx.store_and_spill_to_cxl()

        # An ordered session transition establishes A -> B. The first request
        # observes only A; the second reaches the suffix while all chunks are
        # still served from CXL.
        assert torch.sum(
            fx.retrieve(
                fx.tokens[:_CHUNK], session_id="train", request_id="train-1"
            )
        ) == _CHUNK
        fx.engine.clear(locations=["LocalCPUBackend"])
        assert torch.sum(
            fx.retrieve(fx.tokens, session_id="train", request_id="train-2")
        ) == _EXPECTED
        fx.engine.clear(locations=["LocalCPUBackend"])

        keys = [
            key
            for _, _, key in fx.engine.token_database.process_tokens(tokens=fx.tokens)
        ]
        assert len(keys) >= 2
        suffix = keys[1]

        # A new session demands only A. The temporal A -> B relationship
        # predicts the suffix, which is in CXL and not in this demand request.
        mask = fx.retrieve(
            fx.tokens[:_CHUNK], session_id="trigger", request_id="trigger-1"
        )
        assert torch.sum(mask) == _CHUNK
        cpu_backend = fx.engine.storage_manager.storage_backends["LocalCPUBackend"]
        assert _wait_until(lambda: cpu_backend.contains(suffix), timeout=10.0)
        assert prefetcher.stats()["promoted"] >= 1
    finally:
        fx.close()
