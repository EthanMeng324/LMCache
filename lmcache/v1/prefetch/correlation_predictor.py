# SPDX-License-Identifier: Apache-2.0
"""Bounded first/second-order local correlation predictor."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
import math
import threading
import time
from typing import Generic, Iterable, TypeVar

from lmcache.v1.prefetch.types import PrefetchContext

KeyT = TypeVar("KeyT")


@dataclass(slots=True)
class EdgeStats:
    count: float = 0.0
    useful_on_time: float = 0.0
    useful_late: float = 0.0
    unused: float = 0.0
    pollution: float = 0.0
    gap_ema_ns: float = 0.0
    last_seen_ns: int = 0


@dataclass(frozen=True, slots=True)
class CorrelationPrediction(Generic[KeyT]):
    key: KeyT
    score: float
    support: float
    confidence: float
    order: int
    gap_ema_ns: float


class CorrelationPredictor(Generic[KeyT]):
    """Learn directional streams with bounded top-K edges.

    The predictor stores only a small number of targets per context.  Counts
    decay lazily, and feedback is kept separate from training so speculative
    reads cannot create positive evidence.
    """

    def __init__(
        self,
        *,
        top_k: int = 4,
        min_support: float = 16.0,
        decay_half_life_ns: int = 60_000_000_000,
        second_order_enabled: bool = True,
        max_contexts: int = 100_000,
    ) -> None:
        if top_k <= 0 or min_support <= 0 or decay_half_life_ns <= 0:
            raise ValueError("invalid predictor bounds")
        self.top_k = top_k
        self.min_support = float(min_support)
        self.decay_half_life_ns = float(decay_half_life_ns)
        self.second_order_enabled = second_order_enabled
        self.max_contexts = max_contexts
        self._first: OrderedDict[tuple[PrefetchContext, KeyT], dict[KeyT, EdgeStats]] = OrderedDict()
        self._second: OrderedDict[tuple[PrefetchContext, KeyT, KeyT], dict[KeyT, EdgeStats]] = OrderedDict()
        self._lock = threading.Lock()

    def _decay(self, stats: EdgeStats, now_ns: int) -> float:
        if not stats.last_seen_ns:
            return stats.count
        age = max(0, now_ns - stats.last_seen_ns)
        return stats.count * math.exp(-math.log(2.0) * age / self.decay_half_life_ns)

    def _edge(self, table, state, target, now_ns):
        edges = table.setdefault(state, {})
        edge = edges.get(target)
        if edge is None:
            edge = edges[target] = EdgeStats(last_seen_ns=now_ns)
        else:
            edge.count = self._decay(edge, now_ns)
        edge.count += 1.0
        edge.last_seen_ns = now_ns
        table.move_to_end(state)
        while len(table) > self.max_contexts:
            table.popitem(last=False)
        if len(edges) > self.top_k * 2:
            weakest = sorted(edges, key=lambda k: self._decay(edges[k], now_ns))[: -self.top_k]
            for key in weakest:
                edges.pop(key, None)

    def observe(
        self,
        keys: Iterable[KeyT],
        *,
        context: PrefetchContext = PrefetchContext(),
        previous_key: KeyT | None = None,
        timestamp_ns: int | None = None,
    ) -> None:
        ordered = list(dict.fromkeys(keys))
        if not ordered:
            return
        now = timestamp_ns or time.time_ns()
        with self._lock:
            prev = previous_key
            for key in ordered:
                if prev is not None:
                    self._edge(self._first, (context, prev), key, now)
                    if self.second_order_enabled and previous_key is not None:
                        self._edge(self._second, (context, previous_key, prev), key, now)
                previous_key, prev = prev, key

    def predict(
        self,
        current_key: KeyT,
        *,
        previous_key: KeyT | None = None,
        context: PrefetchContext = PrefetchContext(),
        limit: int = 1,
        now_ns: int | None = None,
    ) -> list[CorrelationPrediction[KeyT]]:
        if limit <= 0:
            return []
        now = now_ns or time.time_ns()
        with self._lock:
            candidates: dict[KeyT, CorrelationPrediction[KeyT]] = {}
            tables = []
            if self.second_order_enabled and previous_key is not None:
                tables.append((2, self._second.get((context, previous_key, current_key))))
            tables.append((1, self._first.get((context, current_key))))
            for order, edges in tables:
                if not edges:
                    continue
                total = sum(self._decay(edge, now) for edge in edges.values())
                if total < self.min_support:
                    continue
                for key, edge in edges.items():
                    if key == current_key:
                        continue
                    support = self._decay(edge, now)
                    confidence = support / total if total else 0.0
                    score = confidence * min(1.0, support / self.min_support)
                    item = CorrelationPrediction(
                        key=key,
                        score=score,
                        support=support,
                        confidence=confidence,
                        order=order,
                        gap_ema_ns=edge.gap_ema_ns,
                    )
                    old = candidates.get(key)
                    if old is None or item.score > old.score:
                        candidates[key] = item
            return sorted(candidates.values(), key=lambda item: item.score, reverse=True)[:limit]

    def feedback(
        self,
        key: KeyT,
        *,
        source_key: KeyT,
        context: PrefetchContext = PrefetchContext(),
        on_time: bool,
        pollution: bool = False,
        timestamp_ns: int | None = None,
    ) -> None:
        now = timestamp_ns or time.time_ns()
        with self._lock:
            edges = self._first.get((context, source_key))
            if not edges or key not in edges:
                return
            edge = edges[key]
            edge.count = self._decay(edge, now)
            if pollution:
                edge.pollution += 1.0
            elif on_time:
                edge.useful_on_time += 1.0
            else:
                edge.useful_late += 1.0


__all__ = ["CorrelationPrediction", "CorrelationPredictor", "EdgeStats"]
