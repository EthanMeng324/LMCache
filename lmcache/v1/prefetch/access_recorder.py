# SPDX-License-Identifier: Apache-2.0
"""Demand-only access recording for the prefetch predictor."""

from __future__ import annotations

from collections import defaultdict, deque
import threading
import time
from typing import Callable, Iterable, Optional

from lmcache.utils import CacheEngineKey
from lmcache.v1.prefetch.types import (
    AccessType,
    KVAccessEvent,
    PrefetchContext,
)


class AccessRecorder:
    """Bounded request history and event sink.

    Lookup hits and speculative copies must not train the model.  Callers use
    :meth:`record_demand` only after a segment is consumed by the request; the
    recorder coalesces consecutive duplicate accesses and drops aborted
    request histories.
    """

    def __init__(
        self,
        *,
        node_id: str = "",
        max_events: int = 100_000,
        max_history_per_request: int = 64,
        sink: Optional[Callable[[KVAccessEvent], None]] = None,
    ) -> None:
        if max_events <= 0 or max_history_per_request <= 0:
            raise ValueError("access recorder bounds must be positive")
        self.node_id = node_id
        self.max_events = max_events
        self.max_history_per_request = max_history_per_request
        self._events: deque[KVAccessEvent] = deque(maxlen=max_events)
        self._history: dict[str, deque[CacheEngineKey]] = defaultdict(
            lambda: deque(maxlen=max_history_per_request)
        )
        self._lock = threading.Lock()
        self._sink = sink
        self.dropped = 0

    def record_demand(
        self,
        key: CacheEngineKey,
        *,
        req_id: str,
        model_id: str = "",
        phase: str = "prefill",
        request_class: int = 0,
        source_tier: str = "UNKNOWN",
        context: Optional[PrefetchContext] = None,
        timestamp_ns: Optional[int] = None,
    ) -> Optional[KVAccessEvent]:
        if not req_id:
            return None
        event = KVAccessEvent(
            key=key,
            timestamp_ns=timestamp_ns or time.time_ns(),
            req_id=req_id,
            node_id=self.node_id,
            model_id=model_id,
            phase=phase,
            request_class=request_class,
            source_tier=source_tier,
            access_type=AccessType.DEMAND,
            context=context or PrefetchContext(model_namespace=model_id),
        )
        with self._lock:
            history = self._history[req_id]
            if history and history[-1] == key:
                return None
            history.append(key)
            self._events.append(event)
        if self._sink is not None:
            self._sink(event)
        return event

    def request_history(self, req_id: str) -> tuple[CacheEngineKey, ...]:
        with self._lock:
            return tuple(self._history.get(req_id, ()))

    def abort_request(self, req_id: str) -> None:
        with self._lock:
            self._history.pop(req_id, None)

    def finish_request(self, req_id: str) -> tuple[CacheEngineKey, ...]:
        with self._lock:
            return tuple(self._history.pop(req_id, ()))

    def drain(self) -> list[KVAccessEvent]:
        with self._lock:
            events = list(self._events)
            self._events.clear()
            return events

    def events(self) -> tuple[KVAccessEvent, ...]:
        with self._lock:
            return tuple(self._events)


__all__ = ["AccessRecorder"]
