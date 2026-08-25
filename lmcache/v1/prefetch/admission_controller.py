# SPDX-License-Identifier: Apache-2.0
"""Bandwidth/CPU admission for speculative segment copies."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Callable, Optional

from lmcache.v1.prefetch.types import PrefetchTask


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    admitted: bool
    score: float
    reason: str


class TokenBucket:
    """A monotonic byte token bucket used for CXL bandwidth budgeting."""

    def __init__(self, capacity: int, refill_bytes_per_second: float) -> None:
        if capacity <= 0 or refill_bytes_per_second <= 0:
            raise ValueError("token bucket parameters must be positive")
        self.capacity = float(capacity)
        self.tokens = float(capacity)
        self.refill = float(refill_bytes_per_second)
        self.updated = time.monotonic()
        self._lock = threading.Lock()

    def try_consume(self, amount: int) -> bool:
        if amount <= 0:
            return True
        now = time.monotonic()
        with self._lock:
            self.tokens = min(
                self.capacity,
                self.tokens + (now - self.updated) * self.refill,
            )
            self.updated = now
            if self.tokens < amount:
                return False
            self.tokens -= amount
            return True


class PrefetchAdmissionController:
    """Evaluate score, readiness and bounded resource budgets.

    The controller is intentionally independent of a storage backend.  The
    scheduler supplies callbacks for CXL readiness and probationary CPU space,
    allowing the same policy to be unit-tested without CXL hardware.
    """

    def __init__(
        self,
        *,
        score_threshold: float = 0.25,
        min_support: float = 16.0,
        cpu_capacity_bytes: int = 1,
        prefetch_cpu_capacity_ratio: float = 0.10,
        bandwidth_ratio: float = 0.15,
        bandwidth_bytes_per_second: float = 1.0,
        cxl_ready: Optional[Callable[[object], bool]] = None,
        cpu_space: Optional[Callable[[int], bool]] = None,
    ) -> None:
        if not 0 < prefetch_cpu_capacity_ratio <= 1:
            raise ValueError("prefetch_cpu_capacity_ratio must be in (0, 1]")
        if not 0 < bandwidth_ratio <= 1:
            raise ValueError("bandwidth_ratio must be in (0, 1]")
        self.score_threshold = float(score_threshold)
        self.min_support = float(min_support)
        self.prefetch_capacity = max(
            1, int(cpu_capacity_bytes * prefetch_cpu_capacity_ratio)
        )
        self.used_prefetch_bytes = 0
        self._lock = threading.Lock()
        self._cxl_ready = cxl_ready or (lambda _key: True)
        self._cpu_space = cpu_space or (lambda size: size <= self.prefetch_capacity)
        self.bucket = TokenBucket(
            max(1, int(bandwidth_bytes_per_second * bandwidth_ratio)),
            max(1.0, bandwidth_bytes_per_second * bandwidth_ratio),
        )

    def evaluate(self, task: PrefetchTask) -> AdmissionDecision:
        if task.deadline_ns and time.time_ns() >= task.deadline_ns:
            return AdmissionDecision(False, task.score, "deadline_expired")
        if task.score < self.score_threshold:
            return AdmissionDecision(False, task.score, "score_below_threshold")
        if not self._cxl_ready(task.key):
            return AdmissionDecision(False, task.score, "cxl_not_ready")
        with self._lock:
            if self.used_prefetch_bytes + task.size_bytes > self.prefetch_capacity:
                return AdmissionDecision(False, task.score, "cpu_probation_full")
        if not self._cpu_space(task.size_bytes):
            return AdmissionDecision(False, task.score, "cpu_space_unavailable")
        if not self.bucket.try_consume(task.size_bytes):
            return AdmissionDecision(False, task.score, "bandwidth_budget")
        with self._lock:
            self.used_prefetch_bytes += task.size_bytes
        return AdmissionDecision(True, task.score, "admitted")

    def release(self, size_bytes: int) -> None:
        with self._lock:
            self.used_prefetch_bytes = max(0, self.used_prefetch_bytes - size_bytes)


__all__ = ["AdmissionDecision", "PrefetchAdmissionController", "TokenBucket"]
