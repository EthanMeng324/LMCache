# SPDX-License-Identifier: Apache-2.0
"""Bounded online co-access predictor for KV-cache keys."""

from collections import OrderedDict
from dataclasses import dataclass
import math
import threading
import time
from typing import Generic, Iterable, Mapping, TypeVar

KeyT = TypeVar("KeyT")
_LN2 = math.log(2.0)


@dataclass(frozen=True, slots=True)
class AssociationPrediction(Generic[KeyT]):
    key: KeyT
    source: KeyT
    confidence: float
    lift: float
    support: float
    score: float


@dataclass(slots=True)
class _Counter:
    value: float
    updated_at: float


@dataclass(slots=True)
class _SourceState(Generic[KeyT]):
    support: _Counter
    targets: OrderedDict[KeyT, _Counter]


class KVAssociationPredictor(Generic[KeyT]):
    """Learn directional associations from bounded demand co-access windows.

    Counts use lazy exponential decay. ``observe`` must only receive real demand
    hits; callers must not feed speculative prefetch reads back into the model.
    """

    def __init__(
        self,
        *,
        min_support: float = 20.0,
        min_confidence: float = 0.70,
        min_lift: float = 1.20,
        half_life_seconds: float = 3600.0,
        observation_window: int = 16,
        max_observation_keys: int = 64,
        max_sources: int = 100_000,
        max_targets_per_source: int = 8,
    ) -> None:
        if min_support <= 0:
            raise ValueError("min_support must be positive")
        if not 0 < min_confidence <= 1:
            raise ValueError("min_confidence must be in (0, 1]")
        if min_lift <= 0 or half_life_seconds <= 0:
            raise ValueError("min_lift and half_life_seconds must be positive")
        if min(observation_window, max_observation_keys, max_sources) <= 0:
            raise ValueError("predictor bounds must be positive")
        if max_targets_per_source <= 0:
            raise ValueError("max_targets_per_source must be positive")

        self.min_support = float(min_support)
        self.min_confidence = float(min_confidence)
        self.min_lift = float(min_lift)
        self.half_life_seconds = float(half_life_seconds)
        self.observation_window = int(observation_window)
        self.max_observation_keys = int(max_observation_keys)
        self.max_sources = int(max_sources)
        self.max_targets_per_source = int(max_targets_per_source)

        self._total = _Counter(0.0, 0.0)
        self._key_support: OrderedDict[KeyT, _Counter] = OrderedDict()
        self._sources: OrderedDict[KeyT, _SourceState[KeyT]] = OrderedDict()
        self._lock = threading.Lock()

    def _decayed(self, counter: _Counter, now: float) -> float:
        if counter.updated_at == 0.0 or now <= counter.updated_at:
            return counter.value
        return counter.value * math.exp(
            -_LN2 * (now - counter.updated_at) / self.half_life_seconds
        )

    def _increment(self, counter: _Counter, now: float) -> None:
        counter.value = self._decayed(counter, now) + 1.0
        counter.updated_at = now

    @staticmethod
    def _deduplicate(keys: Iterable[KeyT], limit: int) -> list[KeyT]:
        result: list[KeyT] = []
        seen: set[KeyT] = set()
        for key in keys:
            if key in seen:
                continue
            seen.add(key)
            result.append(key)
            if len(result) >= limit:
                break
        return result

    def observe(self, keys: Iterable[KeyT], *, timestamp: float | None = None) -> None:
        ordered = self._deduplicate(keys, self.max_observation_keys)
        if not ordered:
            return
        now = time.monotonic() if timestamp is None else float(timestamp)
        with self._lock:
            self._increment(self._total, now)
            for key in ordered:
                counter = self._key_support.get(key)
                if counter is None:
                    counter = _Counter(0.0, now)
                    self._key_support[key] = counter
                self._increment(counter, now)
                self._key_support.move_to_end(key)

                state = self._sources.get(key)
                if state is None:
                    state = _SourceState(_Counter(0.0, now), OrderedDict())
                    self._sources[key] = state
                self._increment(state.support, now)
                self._sources.move_to_end(key)

            # Only associate nearby ordered chunks, avoiding O(n^2) observations.
            for source_idx, source in enumerate(ordered):
                state = self._sources[source]
                begin = max(0, source_idx - self.observation_window)
                end = min(len(ordered), source_idx + self.observation_window + 1)
                for target_idx in range(begin, end):
                    if target_idx == source_idx:
                        continue
                    target = ordered[target_idx]
                    pair = state.targets.get(target)
                    if pair is None:
                        pair = _Counter(0.0, now)
                        state.targets[target] = pair
                    self._increment(pair, now)
                    state.targets.move_to_end(target)
                self._trim_targets(state, now)

            while len(self._sources) > self.max_sources:
                evicted, _ = self._sources.popitem(last=False)
                self._key_support.pop(evicted, None)
            while len(self._key_support) > self.max_sources:
                self._key_support.popitem(last=False)

    def _trim_targets(self, state: _SourceState[KeyT], now: float) -> None:
        overflow = len(state.targets) - self.max_targets_per_source
        if overflow <= 0:
            return
        weakest = sorted(
            state.targets,
            key=lambda key: self._decayed(state.targets[key], now),
        )[:overflow]
        for key in weakest:
            state.targets.pop(key, None)

    def predict(
        self,
        source_keys: Iterable[KeyT],
        *,
        limit: int = 8,
        timestamp: float | None = None,
    ) -> list[AssociationPrediction[KeyT]]:
        if limit <= 0:
            return []
        sources = self._deduplicate(source_keys, self.max_observation_keys)
        source_set = set(sources)
        now = time.monotonic() if timestamp is None else float(timestamp)
        best: dict[KeyT, AssociationPrediction[KeyT]] = {}
        with self._lock:
            total = self._decayed(self._total, now)
            if total <= 0:
                return []
            for source in sources:
                state = self._sources.get(source)
                if state is None:
                    continue
                source_support = self._decayed(state.support, now)
                if source_support < self.min_support:
                    continue
                for target, pair_counter in state.targets.items():
                    if target in source_set:
                        continue
                    pair_support = self._decayed(pair_counter, now)
                    confidence = pair_support / source_support
                    target_counter = self._key_support.get(target)
                    if target_counter is None:
                        continue
                    target_probability = self._decayed(target_counter, now) / total
                    if target_probability <= 0:
                        continue
                    lift = confidence / target_probability
                    if confidence < self.min_confidence or lift < self.min_lift:
                        continue
                    score = pair_support * confidence * lift
                    prediction = AssociationPrediction(
                        key=target,
                        source=source,
                        confidence=confidence,
                        lift=lift,
                        support=pair_support,
                        score=score,
                    )
                    previous = best.get(target)
                    if previous is None or prediction.score > previous.score:
                        best[target] = prediction
        return sorted(best.values(), key=lambda item: item.score, reverse=True)[:limit]

    def sizes(self) -> tuple[int, int]:
        """Return source and pair counts for tests and diagnostics."""
        with self._lock:
            return len(self._sources), sum(
                len(state.targets) for state in self._sources.values()
            )


@dataclass(slots=True)
class _SessionObservation(Generic[KeyT]):
    request_id: str | None
    requested_keys: tuple[KeyT, ...]
    hit_keys: tuple[KeyT, ...]
    hit_tiers: dict[KeyT, str]
    updated_at: float


class KVSessionAssociationPredictor(Generic[KeyT]):
    """Bounded next-request association predictor using session hit/miss state.

    Each observation carries the complete set of chunks requested by a turn and
    the subset that was actually retrieved from cache. A session transition
    learns only ``previous_turn.hit -> current_turn.hit`` associations; a miss
    never becomes positive evidence. The source support is incremented for
    every transition from a source hit, so later transitions where a target is
    absent or misses lower its conditional confidence naturally.
    """

    def __init__(
        self,
        *,
        min_support: float = 20.0,
        min_confidence: float = 0.70,
        min_lift: float = 1.20,
        half_life_seconds: float = 3600.0,
        observation_window: int = 16,
        max_observation_keys: int = 64,
        max_sources: int = 100_000,
        max_targets_per_source: int = 8,
        max_sessions: int = 10_000,
        session_ttl_seconds: float = 3600.0,
    ) -> None:
        if min_support <= 0:
            raise ValueError("min_support must be positive")
        if not 0 < min_confidence <= 1:
            raise ValueError("min_confidence must be in (0, 1]")
        if min_lift <= 0 or half_life_seconds <= 0:
            raise ValueError("min_lift and half_life_seconds must be positive")
        if min(observation_window, max_observation_keys, max_sources) <= 0:
            raise ValueError("predictor bounds must be positive")
        if min(max_targets_per_source, max_sessions, session_ttl_seconds) <= 0:
            raise ValueError("session and target bounds must be positive")

        self.min_support = float(min_support)
        self.min_confidence = float(min_confidence)
        self.min_lift = float(min_lift)
        self.half_life_seconds = float(half_life_seconds)
        self.observation_window = int(observation_window)
        self.max_observation_keys = int(max_observation_keys)
        self.max_sources = int(max_sources)
        self.max_targets_per_source = int(max_targets_per_source)
        self.max_sessions = int(max_sessions)
        self.session_ttl_seconds = float(session_ttl_seconds)

        self._total = _Counter(0.0, 0.0)
        self._key_support: OrderedDict[KeyT, _Counter] = OrderedDict()
        self._cxl_support: OrderedDict[KeyT, _Counter] = OrderedDict()
        self._sources: OrderedDict[KeyT, _SourceState[KeyT]] = OrderedDict()
        self._sessions: OrderedDict[str, _SessionObservation[KeyT]] = OrderedDict()
        self._lock = threading.Lock()

    def _decayed(self, counter: _Counter, now: float) -> float:
        if counter.updated_at == 0.0 or now <= counter.updated_at:
            return counter.value
        return counter.value * math.exp(
            -_LN2 * (now - counter.updated_at) / self.half_life_seconds
        )

    def _increment(self, counter: _Counter, now: float) -> None:
        counter.value = self._decayed(counter, now) + 1.0
        counter.updated_at = now

    @staticmethod
    def _deduplicate(keys: Iterable[KeyT], limit: int) -> list[KeyT]:
        return KVAssociationPredictor._deduplicate(keys, limit)

    def _trim_targets(self, state: _SourceState[KeyT], now: float) -> None:
        overflow = len(state.targets) - self.max_targets_per_source
        if overflow <= 0:
            return
        weakest = sorted(
            state.targets,
            key=lambda key: self._decayed(state.targets[key], now),
        )[:overflow]
        for key in weakest:
            state.targets.pop(key, None)

    def _trim_sources(self) -> None:
        while len(self._sources) > self.max_sources:
            evicted, _ = self._sources.popitem(last=False)
            self._key_support.pop(evicted, None)
            self._cxl_support.pop(evicted, None)
        while len(self._key_support) > self.max_sources:
            evicted, _ = self._key_support.popitem(last=False)
            self._cxl_support.pop(evicted, None)
        while len(self._cxl_support) > self.max_sources:
            self._cxl_support.popitem(last=False)

    def _trim_sessions(self, now: float) -> None:
        while self._sessions:
            session_id, observation = next(iter(self._sessions.items()))
            if (
                len(self._sessions) <= self.max_sessions
                and now - observation.updated_at <= self.session_ttl_seconds
            ):
                break
            self._sessions.pop(session_id)

    def observe_request(
        self,
        session_id: str | None,
        request_id: str | None,
        requested_keys: Iterable[KeyT],
        hit_keys: Iterable[KeyT],
        *,
        hit_tiers: Mapping[KeyT, str] | None = None,
        timestamp: float | None = None,
    ) -> bool:
        """Record one ordered request; return false when no session is supplied.

        ``requested_keys - hit_keys`` is the request's cache-miss set. Misses
        are retained in the session observation and deliberately contribute no
        positive target evidence.
        """

        if not session_id:
            return False
        requested = self._deduplicate(requested_keys, self.max_observation_keys)
        if not requested:
            return False
        requested_set = set(requested)
        hits = [
            key
            for key in self._deduplicate(hit_keys, self.max_observation_keys)
            if key in requested_set
        ]
        now = time.monotonic() if timestamp is None else float(timestamp)
        normalized_session_id = str(session_id)
        normalized_request_id = str(request_id) if request_id is not None else None
        normalized_tiers = {
            key: str(tier)
            for key, tier in (hit_tiers or {}).items()
            if key in set(hits)
        }

        with self._lock:
            self._trim_sessions(now)
            previous = self._sessions.get(normalized_session_id)
            if (
                previous is not None
                and normalized_request_id is not None
                and previous.request_id == normalized_request_id
            ):
                previous.updated_at = now
                self._sessions.move_to_end(normalized_session_id)
                return False

            if previous is not None:
                self._increment(self._total, now)
                # Keep transition learning bounded. The recent suffix is most
                # useful for multi-turn KV reuse and avoids quadratic work.
                sources = list(previous.hit_keys[-self.observation_window :])
                target_limit = min(
                    self.max_observation_keys, self.max_targets_per_source * 4
                )
                targets = hits[-target_limit:]
                for source in sources:
                    state = self._sources.get(source)
                    if state is None:
                        state = _SourceState(_Counter(0.0, now), OrderedDict())
                        self._sources[source] = state
                    self._increment(state.support, now)
                    self._sources.move_to_end(source)

                for target in targets:
                    counter = self._key_support.get(target)
                    if counter is None:
                        counter = _Counter(0.0, now)
                        self._key_support[target] = counter
                    self._increment(counter, now)
                    self._key_support.move_to_end(target)
                    tier = normalized_tiers.get(target)
                    if tier is None or tier in {
                        "CxlBackend",
                        "CXL",
                        "CXL_SHARED",
                        "LMCache",
                    }:
                        cxl_counter = self._cxl_support.get(target)
                        if cxl_counter is None:
                            cxl_counter = _Counter(0.0, now)
                            self._cxl_support[target] = cxl_counter
                        self._increment(cxl_counter, now)
                        self._cxl_support.move_to_end(target)

                for source in sources:
                    state = self._sources[source]
                    for target in targets:
                        tier = normalized_tiers.get(target)
                        if tier is not None and tier not in {
                            "CxlBackend",
                            "CXL",
                            "CXL_SHARED",
                            "LMCache",
                        }:
                            continue
                        if target == source:
                            continue
                        pair = state.targets.get(target)
                        if pair is None:
                            pair = _Counter(0.0, now)
                            state.targets[target] = pair
                        self._increment(pair, now)
                        state.targets.move_to_end(target)
                    self._trim_targets(state, now)
                self._trim_sources()

            self._sessions[normalized_session_id] = _SessionObservation(
                request_id=normalized_request_id,
                requested_keys=tuple(requested),
                hit_keys=tuple(hits),
                hit_tiers=normalized_tiers,
                updated_at=now,
            )
            self._sessions.move_to_end(normalized_session_id)
            self._trim_sessions(now)
        return True

    def predict(
        self,
        source_hit_keys: Iterable[KeyT],
        requested_keys: Iterable[KeyT],
        *,
        limit: int = 8,
        timestamp: float | None = None,
    ) -> list[AssociationPrediction[KeyT]]:
        """Predict future hit targets, excluding all chunks in this request."""

        if limit <= 0:
            return []
        sources = self._deduplicate(source_hit_keys, self.max_observation_keys)
        requested = set(
            self._deduplicate(requested_keys, self.max_observation_keys)
        )
        now = time.monotonic() if timestamp is None else float(timestamp)
        best: dict[KeyT, AssociationPrediction[KeyT]] = {}
        with self._lock:
            total = self._decayed(self._total, now)
            if total <= 0:
                return []
            for source in sources:
                state = self._sources.get(source)
                if state is None:
                    continue
                source_support = self._decayed(state.support, now)
                if source_support < self.min_support:
                    continue
                for target, pair_counter in state.targets.items():
                    if target in requested:
                        continue
                    pair_support = self._decayed(pair_counter, now)
                    confidence = pair_support / source_support
                    target_counter = self._key_support.get(target)
                    if target_counter is None:
                        continue
                    cxl_counter = self._cxl_support.get(target)
                    if cxl_counter is None or self._decayed(cxl_counter, now) <= 0:
                        continue
                    target_probability = self._decayed(target_counter, now) / total
                    if target_probability <= 0:
                        continue
                    lift = confidence / target_probability
                    if confidence < self.min_confidence or lift < self.min_lift:
                        continue
                    score = pair_support * confidence * lift
                    prediction = AssociationPrediction(
                        key=target,
                        source=source,
                        confidence=confidence,
                        lift=lift,
                        support=pair_support,
                        score=score,
                    )
                    previous = best.get(target)
                    if previous is None or prediction.score > previous.score:
                        best[target] = prediction
        return sorted(best.values(), key=lambda item: item.score, reverse=True)[:limit]

    def sizes(self) -> tuple[int, int, int]:
        """Return source, pair, and active-session counts for diagnostics."""

        with self._lock:
            return (
                len(self._sources),
                sum(len(state.targets) for state in self._sources.values()),
                len(self._sessions),
            )
