# SPDX-License-Identifier: Apache-2.0
"""Synthetic workload benchmark for the KV association predictor.

Run from LMCache with the repository environment, for example:

    ../dynamo/.venv/bin/python tests/v1/prefetch/benchmark_association_predictor.py

The workload deliberately includes a globally present ``HOT`` key and unstable
``U*`` co-accesses. They exercise lift filtering and false-positive control.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import random
import time

from lmcache.v1.prefetch import KVAssociationPredictor


@dataclass(frozen=True)
class BenchmarkResult:
    seed: int
    train_observations: int
    positive_queries: int
    negative_queries: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    observe_p50_us: float
    observe_p99_us: float
    predict_p50_us: float
    predict_p99_us: float
    sources: int
    pairs: int
    phase_shift_first_new_target: int | None
    phase_shift_last_old_target: int | None


def _percentile(values: list[int], percentile: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = min(len(values) - 1, int(round((len(values) - 1) * percentile)))
    return values[index] / 1000.0


def _build_predictor() -> KVAssociationPredictor[str]:
    return KVAssociationPredictor(
        min_support=20,
        min_confidence=0.70,
        min_lift=1.20,
        half_life_seconds=10_000,
        observation_window=4,
        max_observation_keys=16,
        max_sources=2_000,
        max_targets_per_source=8,
    )


def _train(
    predictor: KVAssociationPredictor[str], rng: random.Random, count: int
) -> tuple[list[int], list[int]]:
    observe_us: list[int] = []
    predict_us: list[int] = []
    sources = [f"A{i}" for i in range(64)]
    targets = {source: f"B{i}" for i, source in enumerate(sources)}
    unstable = [f"U{i}" for i in range(16)]
    random_targets = [f"R{i}" for i in range(64)]

    for timestamp in range(1, count + 1):
        source = rng.choice(sources)
        keys = [source, "HOT"]
        if rng.random() < 0.85:
            keys.append(targets[source])
        start = time.perf_counter_ns()
        predictor.observe(keys, timestamp=2 * timestamp - 1)
        observe_us.append(time.perf_counter_ns() - start)

        # Unstable co-accesses have no fixed target and should not become a
        # confident association despite enough support for the source.
        unstable_source = rng.choice(unstable)
        unstable_keys = [unstable_source, "HOT"]
        if rng.random() < 0.55:
            unstable_keys.append(rng.choice(random_targets))
        start = time.perf_counter_ns()
        predictor.observe(unstable_keys, timestamp=2 * timestamp)
        observe_us.append(time.perf_counter_ns() - start)

        # Keep a representative prediction latency sample in the hot path.
        start = time.perf_counter_ns()
        predictor.predict([source], limit=4, timestamp=2 * timestamp)
        predict_us.append(time.perf_counter_ns() - start)
    return observe_us, predict_us


def _evaluate(
    predictor: KVAssociationPredictor[str], rng: random.Random, start_timestamp: int
) -> tuple[int, int, int, int]:
    true_positives = false_positives = false_negatives = 0
    sources = [f"A{i}" for i in range(64)]
    targets = {source: f"B{i}" for i, source in enumerate(sources)}

    for query_index in range(2_000):
        source = rng.choice(sources)
        predicted = {
            item.key
            for item in predictor.predict(
                [source], limit=4, timestamp=start_timestamp + query_index
            )
        }
        expected = {targets[source]}
        true_positives += len(predicted & expected)
        false_positives += len(predicted - expected)
        false_negatives += len(expected - predicted)

    # U* sources were intentionally trained with random targets. Any returned
    # target is a false positive for this query class.
    for query_index in range(2_000, 4_000):
        source = rng.choice([f"U{i}" for i in range(16)])
        predicted = predictor.predict(
            [source], limit=4, timestamp=start_timestamp + query_index
        )
        false_positives += len(predicted)
    return true_positives, false_positives, false_negatives, 4_000


def _phase_shift() -> tuple[int | None, int | None]:
    predictor = KVAssociationPredictor[str](
        min_support=8,
        min_confidence=0.55,
        min_lift=1.10,
        half_life_seconds=40,
        observation_window=4,
        max_observation_keys=16,
        max_sources=64,
        max_targets_per_source=4,
    )
    for timestamp in range(1, 101):
        predictor.observe(["phase-A", "phase-B"], timestamp=timestamp)
        predictor.observe(["phase-noise"], timestamp=timestamp + 0.5)

    first_new: int | None = None
    last_old: int | None = None
    for timestamp in range(101, 301):
        predictor.observe(["phase-A", "phase-C"], timestamp=timestamp)
        predictor.observe(["phase-noise"], timestamp=timestamp + 0.5)
        predicted = {
            item.key
            for item in predictor.predict(["phase-A"], timestamp=timestamp + 0.5)
        }
        if "phase-C" in predicted and first_new is None:
            first_new = timestamp - 100
        if "phase-B" in predicted:
            last_old = timestamp - 100
    return first_new, last_old


def run(seed: int) -> BenchmarkResult:
    rng = random.Random(seed)
    predictor = _build_predictor()
    observe_us, predict_us = _train(predictor, rng, count=2_000)
    tp, fp, fn, _ = _evaluate(predictor, rng, start_timestamp=5_000)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    first_new, last_old = _phase_shift()
    return BenchmarkResult(
        seed=seed,
        train_observations=4_000,
        positive_queries=2_000,
        negative_queries=2_000,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        observe_p50_us=_percentile(observe_us, 0.50),
        observe_p99_us=_percentile(observe_us, 0.99),
        predict_p50_us=_percentile(predict_us, 0.50),
        predict_p99_us=_percentile(predict_us, 0.99),
        sources=predictor.sizes()[0],
        pairs=predictor.sizes()[1],
        phase_shift_first_new_target=first_new,
        phase_shift_last_old_target=last_old,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    result = run(args.seed)
    print("association predictor synthetic benchmark")
    for key, value in result.__dict__.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
