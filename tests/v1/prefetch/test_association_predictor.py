# SPDX-License-Identifier: Apache-2.0

from lmcache.v1.prefetch import (
    KVAssociationPredictor,
    KVSessionAssociationPredictor,
)


def test_predicts_specific_co_access_and_not_current_source_keys():
    predictor = KVAssociationPredictor[str](
        min_support=2.5,
        min_confidence=0.8,
        min_lift=1.2,
        half_life_seconds=1000,
        observation_window=4,
        max_observation_keys=8,
        max_sources=32,
        max_targets_per_source=4,
    )

    for timestamp in range(1, 4):
        predictor.observe(["A", "B"], timestamp=timestamp)
    for timestamp in range(4, 10):
        predictor.observe(["noise"], timestamp=timestamp)

    predictions = predictor.predict(["A"], timestamp=10)
    assert [prediction.key for prediction in predictions] == ["B"]
    assert predictions[0].confidence >= 0.99
    assert predictions[0].lift > 1.2

    # A key already demanded by the triggering request is never predicted.
    assert predictor.predict(["A", "B"], timestamp=10) == []


def test_lift_rejects_globally_hot_target():
    predictor = KVAssociationPredictor[str](
        min_support=1.5,
        min_confidence=0.5,
        min_lift=1.1,
        half_life_seconds=1000,
        observation_window=4,
        max_observation_keys=8,
        max_sources=32,
        max_targets_per_source=4,
    )
    predictor.observe(["A", "B"], timestamp=1)
    predictor.observe(["A", "B"], timestamp=2)
    for timestamp in range(3, 10):
        predictor.observe(["B"], timestamp=timestamp)

    assert predictor.predict(["A"], timestamp=10) == []


def test_new_phase_reduces_stale_confidence():
    predictor = KVAssociationPredictor[str](
        min_support=1.5,
        min_confidence=0.7,
        min_lift=1.0,
        half_life_seconds=5,
        observation_window=4,
        max_observation_keys=8,
        max_sources=32,
        max_targets_per_source=4,
    )
    predictor.observe(["A", "B"], timestamp=1)
    predictor.observe(["A", "B"], timestamp=2)
    assert predictor.predict(["A"], timestamp=3)

    for timestamp in range(20, 24):
        predictor.observe(["A"], timestamp=timestamp)
    assert predictor.predict(["A"], timestamp=24) == []


def test_state_is_bounded():
    predictor = KVAssociationPredictor[str](
        min_support=1,
        min_confidence=0.1,
        min_lift=0.1,
        half_life_seconds=1000,
        observation_window=8,
        max_observation_keys=16,
        max_sources=5,
        max_targets_per_source=2,
    )
    for index in range(20):
        predictor.observe(
            [f"source-{index}", f"target-{index}-1", f"target-{index}-2"],
            timestamp=index + 1,
        )

    sources, pairs = predictor.sizes()
    assert sources <= 5
    assert pairs <= sources * 2


def _session_predictor(**overrides):
    config = {
        "min_support": 1.5,
        "min_confidence": 0.70,
        "min_lift": 1.0,
        "half_life_seconds": 1000,
        "observation_window": 4,
        "max_observation_keys": 8,
        "max_sources": 32,
        "max_targets_per_source": 4,
        "max_sessions": 4,
        "session_ttl_seconds": 1000,
    }
    config.update(overrides)
    return KVSessionAssociationPredictor[str](**config)


def test_session_predictor_learns_only_ordered_cross_request_hits():
    predictor = _session_predictor()

    # A and B coexist in the first request. C is only a hit in the following
    # request, so A/B -> C is temporal evidence rather than same-turn coaccess.
    predictor.observe_request("session-1", "r1", ["A", "B"], ["A", "B"], timestamp=1)
    predictor.observe_request("session-1", "r2", ["C"], ["C"], timestamp=2)
    predictor.observe_request("session-2", "r1", ["A"], ["A"], timestamp=3)
    predictor.observe_request("session-2", "r2", ["C"], ["C"], timestamp=4)

    predictions = predictor.predict(["A"], ["A"], timestamp=5)
    assert [prediction.key for prediction in predictions] == ["C"]
    assert predictions[0].confidence >= 0.99


def test_session_predictor_does_not_turn_requested_miss_into_positive_evidence():
    predictor = _session_predictor(min_confidence=0.75)

    predictor.observe_request("session-1", "r1", ["A"], ["A"], timestamp=1)
    predictor.observe_request("session-1", "r2", ["B"], ["B"], timestamp=2)
    predictor.observe_request("session-2", "r1", ["A"], ["A"], timestamp=3)
    # B was requested but missed. It is recorded in the request matrix, but
    # must not be a positive A -> B destination association.
    predictor.observe_request("session-2", "r2", ["B"], [], timestamp=4)

    assert predictor.predict(["A"], ["A"], timestamp=5) == []


def test_session_predictor_only_prefetches_targets_with_cxl_evidence():
    predictor = _session_predictor(min_support=0.5, min_confidence=0.5)

    predictor.observe_request(
        "session-1", "r1", ["A"], ["A"], hit_tiers={"A": "CxlBackend"}, timestamp=1
    )
    predictor.observe_request(
        "session-1",
        "r2",
        ["B", "C"],
        ["B", "C"],
        hit_tiers={"B": "LocalCPUBackend", "C": "CxlBackend"},
        timestamp=2,
    )

    predictions = predictor.predict(["A"], ["A"], timestamp=3)
    assert [prediction.key for prediction in predictions] == ["C"]


def test_session_predictor_requires_session_and_bounds_session_state():
    predictor = _session_predictor(max_sessions=2)

    assert not predictor.observe_request(None, "r1", ["A"], ["A"], timestamp=1)
    for index in range(3):
        assert predictor.observe_request(
            f"session-{index}", "r1", ["A"], ["A"], timestamp=index + 2
        )
    sources, pairs, sessions = predictor.sizes()
    assert sources == pairs == 0
    assert sessions == 2
