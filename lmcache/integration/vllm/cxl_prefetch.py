# SPDX-License-Identifier: Apache-2.0
"""Worker-side callable for Dynamo's waiting-route CXL prefetch hint."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.logging import init_logger
from lmcache.v1.cache_engine import LMCacheEngineBuilder

logger = init_logger(__name__)


def _worker_dp_rank(worker: Any) -> int:
    parallel_config = getattr(worker, "parallel_config", None)
    rank = getattr(parallel_config, "data_parallel_rank", None)
    if rank is None:
        vllm_config = getattr(worker, "vllm_config", None)
        parallel_config = getattr(vllm_config, "parallel_config", None)
        rank = getattr(parallel_config, "data_parallel_rank", None)
    return int(rank or 0)


def cxl_prefetch(
    worker: Any,
    request_id: str,
    tokens: Sequence[int],
    dp_rank: int,
    max_chunks: int,
    candidate_block_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Submit a bounded prefetch on the selected DP worker.

    This function is serialized by vLLM's ``collective_rpc`` and executed on
    every TP worker.  The rank check prevents a request targeted at one DP
    instance from warming another instance's CPU tier.  It intentionally
    catches every exception: a speculative control message must fail open.
    """
    try:
        actual_dp_rank = _worker_dp_rank(worker)
        target_dp_rank = int(dp_rank)
        if actual_dp_rank != target_dp_rank:
            return {
                "status": "ignored",
                "request_id": request_id,
                "scheduled": 0,
                "dp_rank": actual_dp_rank,
            }

        engine = LMCacheEngineBuilder.get(ENGINE_NAME)
        if engine is None:
            return {
                "status": "disabled",
                "request_id": request_id,
                "scheduled": 0,
            }

        scheduled = engine.submit_cxl_prefetch(
            request_id=request_id,
            tokens=list(tokens),
            max_chunks=max(0, int(max_chunks)),
            candidate_block_indices=(
                None
                if candidate_block_indices is None
                else [int(index) for index in candidate_block_indices]
            ),
        )
        prefetch_stats: dict[str, int] = {}
        get_stats = getattr(engine, "get_cxl_prefetch_stats", None)
        if callable(get_stats):
            try:
                stats = get_stats()
                if isinstance(stats, dict):
                    prefetch_stats = {
                        str(name): int(value)
                        for name, value in stats.items()
                        if isinstance(value, int)
                    }
            except Exception as exc:
                # Statistics are diagnostic only and must not turn a
                # best-effort prefetch admission into a failed RPC.
                logger.debug(
                    "Unable to read CXL lookahead promotion stats for request %s: %s",
                    request_id,
                    exc,
                )
        if isinstance(scheduled, dict):
            result = {"request_id": request_id, **scheduled}
        else:
            result = {
                "status": "accepted",
                "request_id": request_id,
                "scheduled": int(scheduled),
            }
        if prefetch_stats:
            result["prefetch_stats"] = prefetch_stats
        return result
    except Exception as exc:
        logger.debug(
            "CXL lookahead promotion failed open for request %s: %s",
            request_id,
            exc,
        )
        return {
            "status": "failed",
            "request_id": request_id,
            "scheduled": 0,
            "message": str(exc),
        }


class CxlPrefetchWorkerExtension:
    """vLLM worker extension exposing the prefetch RPC by name.

    EngineCore's safe message serializer cannot carry an arbitrary Python
    callable unless insecure pickle serialization is enabled.  Registering
    this small extension lets the request side use the string RPC name while
    keeping the default serializer enabled.
    """

    def cxl_prefetch(
        self,
        request_id: str,
        tokens: Sequence[int],
        dp_rank: int,
        max_chunks: int,
        candidate_block_indices: Sequence[int] | None = None,
    ) -> dict[str, Any]:
        return cxl_prefetch(
            self,
            request_id,
            tokens,
            dp_rank,
            max_chunks,
            candidate_block_indices,
        )


__all__ = ["CxlPrefetchWorkerExtension", "cxl_prefetch"]
