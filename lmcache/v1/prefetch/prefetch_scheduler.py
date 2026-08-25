# SPDX-License-Identifier: Apache-2.0
"""Bounded, priority-ordered prefetch task scheduler.

The engine's promotion executor is intentionally kept close to cache-engine
lifecycles.  This small scheduler provides the queue contract described in
``docs/prefetch.md`` for integrations that want a standalone producer/consumer
loop (and is also useful in tests).
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import threading
import time
from typing import Callable, Optional

from lmcache.v1.prefetch.types import PrefetchState, PrefetchTask


@dataclass(frozen=True, slots=True)
class SchedulerStats:
    queued: int
    inflight: int
    completed: int
    cancelled: int
    duplicate_suppressed: int


class PrefetchScheduler:
    """A bounded heap with per-key in-flight de-duplication."""

    def __init__(self, *, max_tasks: int = 16) -> None:
        if max_tasks <= 0:
            raise ValueError("max_tasks must be positive")
        self.max_tasks = max_tasks
        self._heap: list[tuple[int, int, PrefetchTask]] = []
        self._inflight: dict[object, PrefetchTask] = {}
        self._sequence = 0
        self._completed = 0
        self._cancelled = 0
        self._duplicates = 0
        self._lock = threading.Lock()

    def submit(self, task: PrefetchTask) -> bool:
        with self._lock:
            if task.key in self._inflight:
                self._duplicates += 1
                return False
            if len(self._inflight) >= self.max_tasks:
                return False
            task.state = PrefetchState.QUEUED
            self._sequence += 1
            heapq.heappush(self._heap, (-task.priority, self._sequence, task))
            self._inflight[task.key] = task
            return True

    def cancel(self, key: object) -> bool:
        with self._lock:
            task = self._inflight.pop(key, None)
            if task is None:
                return False
            task.state = PrefetchState.CANCELLED
            self._cancelled += 1
            return True

    def run_once(self, submitter: Callable[[PrefetchTask], bool]) -> bool:
        """Run one non-expired task; submitter returns completion status."""
        with self._lock:
            while self._heap:
                _, _, task = heapq.heappop(self._heap)
                if self._inflight.get(task.key) is task:
                    break
            else:
                return False
            if task.deadline_ns and time.time_ns() >= task.deadline_ns:
                self._inflight.pop(task.key, None)
                task.state = PrefetchState.CANCELLED
                self._cancelled += 1
                return True
            task.state = PrefetchState.INFLIGHT
            task.started_ns = time.time_ns()
        ok = False
        try:
            ok = bool(submitter(task))
        finally:
            with self._lock:
                self._inflight.pop(task.key, None)
                task.completed_ns = time.time_ns()
                task.state = PrefetchState.CPU_READY if ok else PrefetchState.FAILED
                if ok:
                    self._completed += 1
        return True

    def stats(self) -> SchedulerStats:
        with self._lock:
            return SchedulerStats(
                queued=sum(1 for task in self._inflight.values() if task.state == PrefetchState.QUEUED),
                inflight=sum(1 for task in self._inflight.values() if task.state == PrefetchState.INFLIGHT),
                completed=self._completed,
                cancelled=self._cancelled,
                duplicate_suppressed=self._duplicates,
            )


__all__ = ["PrefetchScheduler", "SchedulerStats"]
