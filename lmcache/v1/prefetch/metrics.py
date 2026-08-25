# SPDX-License-Identifier: Apache-2.0
"""Small lock-protected counters for prefetch observability."""

from __future__ import annotations

from collections import Counter
import threading
from typing import Mapping


class PrefetchMetrics:
    def __init__(self) -> None:
        self._values: Counter[str] = Counter()
        self._lock = threading.Lock()

    def increment(self, name: str, value: int = 1) -> None:
        with self._lock:
            self._values[name] += value

    def snapshot(self) -> Mapping[str, int]:
        with self._lock:
            return dict(self._values)


__all__ = ["PrefetchMetrics"]
