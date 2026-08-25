# SPDX-License-Identifier: Apache-2.0
"""Optional, off-hot-path pattern delta exporter."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Callable, Optional

from lmcache.v1.prefetch.types import PrefetchContext


@dataclass(frozen=True, slots=True)
class PatternDelta:
    context: PrefetchContext
    previous: str
    source: str
    target: str
    support_delta: float
    useful_delta: float
    gap_ema_ns: float
    epoch: int
    node_id: str


class GlobalPatternClient:
    """Buffers bounded heavy-edge deltas and exports them periodically.

    The exporter is disabled unless a callback is supplied.  This keeps the
    predictor hot path node-local while allowing deployments to connect NATS,
    Dynamo, or a small HTTP aggregator later.
    """

    def __init__(
        self,
        *,
        node_id: str,
        max_deltas: int = 10_000,
        export_interval_seconds: float = 2.0,
        sink: Optional[Callable[[list[PatternDelta]], None]] = None,
    ) -> None:
        self.node_id = node_id
        self.max_deltas = max_deltas
        self.export_interval_seconds = export_interval_seconds
        self._sink = sink
        self._deltas: list[PatternDelta] = []
        self._epoch = 0
        self._global_edges: dict[tuple[PrefetchContext, str, str], tuple[float, float]] = {}
        self._last_export = time.monotonic()
        self._lock = threading.Lock()

    def append(
        self,
        *,
        context: PrefetchContext,
        previous: str,
        source: str,
        target: str,
        support_delta: float = 1.0,
        useful_delta: float = 0.0,
        gap_ema_ns: float = 0.0,
    ) -> None:
        with self._lock:
            if len(self._deltas) >= self.max_deltas:
                self._deltas.pop(0)
            self._deltas.append(
                PatternDelta(
                    context,
                    previous,
                    source,
                    target,
                    support_delta,
                    useful_delta,
                    gap_ema_ns,
                    self._epoch,
                    self.node_id,
                )
            )
            due = time.monotonic() - self._last_export >= self.export_interval_seconds
        if due:
            self.flush()

    def flush(self) -> list[PatternDelta]:
        with self._lock:
            result = self._deltas
            self._deltas = []
            self._epoch += 1
            self._last_export = time.monotonic()
        if result and self._sink is not None:
            self._sink(result)
        return result

    def merge(self, deltas: list[PatternDelta], *, weight: float = 1.0) -> None:
        """Merge an aggregator response for cold-start/cross-node hints."""
        weight = max(0.0, min(1.0, float(weight)))
        with self._lock:
            for delta in deltas:
                state = (delta.context, delta.previous, delta.source)
                support, useful = self._global_edges.get(state, (0.0, 0.0))
                self._global_edges[state] = (
                    support + max(0.0, delta.support_delta) * weight,
                    useful + max(0.0, delta.useful_delta) * weight,
                )

    def global_score(
        self, context: PrefetchContext, previous: str, source: str
    ) -> tuple[float, float]:
        with self._lock:
            return self._global_edges.get((context, previous, source), (0.0, 0.0))


__all__ = ["GlobalPatternClient", "PatternDelta"]
