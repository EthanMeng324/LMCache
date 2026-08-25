# SPDX-License-Identifier: Apache-2.0
"""Shared types for bounded, tier-aware KV prefetching.

The types in this module deliberately contain logical identifiers only.  A
``SegmentID`` never embeds a process-local pointer, CXL offset, or request
identifier, so it is safe to put on a control-plane message.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import hashlib
import time
from typing import Any, Optional

from lmcache.utils import CacheEngineKey


class PrefetchState(StrEnum):
    CXL_READY = "CXL_READY"
    QUEUED = "PREFETCH_QUEUED"
    INFLIGHT = "PREFETCH_INFLIGHT"
    ESCALATED_DEMAND = "ESCALATED_DEMAND"
    CPU_READY = "CPU_PREFETCH_READY"
    DEMAND_PROTECTED = "CPU_DEMAND_PROTECTED"
    UNUSED = "UNUSED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


class AccessType(StrEnum):
    DEMAND = "demand"
    PREFETCH_HIT = "prefetch_hit"
    PREFETCH_LATE = "prefetch_late"


@dataclass(frozen=True, slots=True)
class PrefetchContext:
    model_namespace: str = ""
    phase: str = "prefill"
    request_class: int = 0
    attention_mode: int = 0
    tp_rank: int = 0
    layer_group: int = 0


@dataclass(frozen=True, slots=True)
class SegmentNamespace:
    model_hash: int
    model_revision: int = 0
    tensor_parallel_rank: int = 0
    pipeline_parallel_rank: int = 0
    kv_layout_version: int = 0
    kv_dtype: str = ""
    chunk_size: int = 0
    layer_group: int = 0

    def canonical(self) -> str:
        return ":".join(
            (
                str(self.model_hash),
                str(self.model_revision),
                str(self.tensor_parallel_rank),
                str(self.pipeline_parallel_rank),
                str(self.kv_layout_version),
                self.kv_dtype,
                str(self.chunk_size),
                str(self.layer_group),
            )
        )


@dataclass(frozen=True, slots=True)
class SegmentID:
    namespace: SegmentNamespace
    token_chunk_hash: int

    def canonical(self) -> str:
        return f"{self.namespace.canonical()}:{self.token_chunk_hash:032x}"

    def __str__(self) -> str:
        return self.canonical()


def segment_id_from_key(
    key: CacheEngineKey,
    *,
    model_revision: int = 0,
    pipeline_parallel_rank: int = 0,
    kv_layout_version: int = 0,
    layer_group: int = 0,
) -> SegmentID:
    """Build a stable logical ID from an LMCache key.

    ``worker_id`` is retained as the tensor-parallel rank because different TP
    shards contain different KV data.  The key's model/world-size/dtype fields
    are part of the namespace, while the chunk hash is the logical payload ID.
    """
    model_name = f"{key.fmt}:{key.model_name}:{key.world_size}"
    model_hash = int.from_bytes(
        hashlib.blake2b(model_name.encode("utf-8"), digest_size=8).digest(),
        "little",
    )
    return SegmentID(
        namespace=SegmentNamespace(
            model_hash=model_hash,
            model_revision=model_revision,
            tensor_parallel_rank=int(key.worker_id),
            pipeline_parallel_rank=pipeline_parallel_rank,
            kv_layout_version=kv_layout_version,
            kv_dtype=str(key.dtype),
            chunk_size=0,
            layer_group=layer_group,
        ),
        token_chunk_hash=int(key.chunk_hash),
    )


@dataclass(slots=True)
class KVAccessEvent:
    key: CacheEngineKey
    timestamp_ns: int
    req_id: str
    node_id: str
    model_id: str
    phase: str
    request_class: int
    source_tier: str
    access_type: AccessType
    context: PrefetchContext = field(default_factory=PrefetchContext)


@dataclass(slots=True)
class PrefetchTask:
    key: CacheEngineKey
    trigger_key: Optional[CacheEngineKey]
    request_id: Optional[str]
    context: PrefetchContext
    source: str
    target: str
    priority: int
    deadline_ns: int
    expected_use_ns: int
    score: float
    size_bytes: int
    generation: int = 0
    state: PrefetchState = PrefetchState.QUEUED
    created_ns: int = field(default_factory=time.time_ns)
    started_ns: Optional[int] = None
    completed_ns: Optional[int] = None


@dataclass(frozen=True, slots=True)
class PrefetchHandle:
    key: CacheEngineKey
    generation: int
    request_id: Optional[str]


@dataclass(frozen=True, slots=True)
class SegmentBlockMapping:
    segment_id: SegmentID
    router_block_hashes: tuple[int, ...]


__all__ = [
    "AccessType",
    "KVAccessEvent",
    "PrefetchContext",
    "PrefetchHandle",
    "PrefetchState",
    "PrefetchTask",
    "SegmentBlockMapping",
    "SegmentID",
    "SegmentNamespace",
    "segment_id_from_key",
]
