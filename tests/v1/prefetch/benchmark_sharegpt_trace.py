# SPDX-License-Identifier: Apache-2.0
"""Replay a real multi-round ShareGPT service trace through the predictor.

The guide uses ShareGPT as the request workload for the multi-round service
benchmark.  This tool performs the trace part of that experiment without
requiring a model server: it tokenizes each cumulative chat prompt with the
same local model tokenizer, derives deterministic rolling-prefix chunk keys,
and feeds ordered request hit/miss observations to
``KVSessionAssociationPredictor``. Cache residency is conservatively inferred
from chunks seen in earlier prompts: a newly observed chunk is a miss and a
previously seen chunk is a hit. A prediction made for turn ``n`` is useful when
the predicted chunk is a cache hit in turn ``n + 1``.

Example (from ``LMCache``)::

    ../dynamo/.venv/bin/python tests/v1/prefetch/benchmark_sharegpt_trace.py \
      --trace /tmp/ShareGPT_V3_unfiltered_cleaned_split.json \
      --tokenizer /path/to/Mistral-7B-Instruct-v0.2 \
      --limit 2000 --min-support 2

The raw ShareGPT file is intentionally not checked into the repository.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
import hashlib
import json
import time
from typing import Any

from transformers import AutoTokenizer

from lmcache.v1.prefetch import KVSessionAssociationPredictor


@dataclass(frozen=True)
class TraceResult:
    trace: str
    tokenizer: str
    conversations: int
    requests: int
    evaluated_transitions: int
    demand_chunks: int
    cache_hits: int
    cache_misses: int
    next_eligible_hits: int
    predictions: int
    useful_predictions: int
    useful_eligible_predictions: int
    wasted_predictions: int
    precision: float
    eligible_hit_coverage: float
    observe_p50_us: float
    observe_p99_us: float
    predict_p50_us: float
    predict_p99_us: float
    predictor_sources: int
    predictor_pairs: int


def _percentile_ns(samples: list[int], percentile: float) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * percentile)))
    return ordered[index] / 1_000.0


def _rolling_chunk_keys(tokens: list[int], chunk_size: int) -> list[str]:
    """Return stable LMCache-like rolling-prefix identifiers for token chunks."""

    keys: list[str] = []
    parent = b"\x00" * 32
    for begin in range(0, len(tokens), chunk_size):
        chunk = tokens[begin : begin + chunk_size]
        if not chunk:
            continue
        encoded = b"".join(
            int(token).to_bytes(8, "little", signed=True) for token in chunk
        )
        digest = hashlib.sha256(parent + encoded).digest()
        keys.append(digest.hex()[:32])
        parent = digest
    return keys


def _role_message(message: dict[str, Any]) -> dict[str, str] | None:
    sender = str(message.get("from", "")).lower()
    value = message.get("value")
    if not isinstance(value, str) or not value.strip():
        return None
    if sender in {"human", "user"}:
        role = "user"
    elif sender in {"gpt", "assistant", "bot"}:
        role = "assistant"
    elif sender == "system":
        role = "system"
    else:
        return None
    return {"role": role, "content": value}


def _prompt_keys(
    tokenizer: Any, messages: list[dict[str, str]], chunk_size: int
) -> list[str]:
    try:
        token_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
    except Exception:
        # ShareGPT contains a small number of malformed/non-alternating
        # conversations.  Keep their text in the trace rather than dropping
        # the whole session when a model-specific chat template rejects it.
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        token_ids = tokenizer.encode(text, add_special_tokens=True)
    return _rolling_chunk_keys([int(token) for token in token_ids], chunk_size)


def replay_trace(
    trace_path: str,
    tokenizer_path: str,
    *,
    limit: int,
    min_turns: int,
    chunk_size: int,
    min_support: float,
    min_confidence: float,
    min_lift: float,
    prediction_limit: int,
) -> TraceResult:
    if limit <= 0 or min_turns < 2 or chunk_size <= 0:
        raise ValueError("limit, min_turns and chunk_size must be positive")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    with open(trace_path, encoding="utf-8") as trace_file:
        records = json.load(trace_file)

    predictor = KVSessionAssociationPredictor[str](
        min_support=min_support,
        min_confidence=min_confidence,
        min_lift=min_lift,
        half_life_seconds=10_000,
        observation_window=16,
        max_observation_keys=64,
        max_sources=100_000,
        max_targets_per_source=8,
        max_sessions=max(limit, 1),
        session_ttl_seconds=10_000,
    )
    conversations = requests = transitions = 0
    demand_chunks = cache_hits = cache_misses = next_eligible_hits = predictions = 0
    useful_predictions = useful_eligible_predictions = wasted_predictions = 0
    observe_ns: list[int] = []
    predict_ns: list[int] = []
    timestamp = 1.0
    resident_keys: set[str] = set()

    for record in records:
        if conversations >= limit:
            break
        raw_messages = record.get("conversations", [])
        messages: list[dict[str, str]] = []
        prompts: list[list[str]] = []
        for raw_message in raw_messages:
            message = _role_message(raw_message)
            if message is None:
                continue
            messages.append(message)
            if message["role"] == "user":
                keys = _prompt_keys(tokenizer, messages, chunk_size)
                if keys:
                    prompts.append(keys)
        if len(prompts) < min_turns:
            continue

        conversations += 1
        requests += len(prompts)
        session_id = str(record.get("id", conversations))
        for index, current in enumerate(prompts):
            current_set = set(current)
            demand_chunks += len(current_set)
            current_hits = [key for key in current if key in resident_keys]
            cache_hits += len(current_hits)
            cache_misses += len(current_set) - len(current_hits)
            start = time.perf_counter_ns()
            predicted = predictor.predict(
                current_hits,
                current,
                limit=prediction_limit,
                timestamp=timestamp,
            )
            predict_ns.append(time.perf_counter_ns() - start)
            start = time.perf_counter_ns()
            predictor.observe_request(
                session_id,
                f"{session_id}:{index}",
                current,
                current_hits,
                timestamp=timestamp,
            )
            observe_ns.append(time.perf_counter_ns() - start)
            timestamp += 1.0
            if index + 1 >= len(prompts):
                resident_keys.update(current_set)
                continue
            transitions += 1
            next_set = set(prompts[index + 1])
            next_hit_set = next_set & (resident_keys | current_set)
            eligible_set = next_hit_set - current_set
            next_eligible_hits += len(eligible_set)
            predicted_keys = {item.key for item in predicted}
            predictions += len(predicted_keys)
            useful_predictions += len(predicted_keys & next_hit_set)
            useful_eligible_predictions += len(predicted_keys & eligible_set)
            wasted_predictions += len(predicted_keys - next_hit_set)
            resident_keys.update(current_set)

    precision = useful_predictions / predictions if predictions else 0.0
    coverage = (
        useful_eligible_predictions / next_eligible_hits
        if next_eligible_hits
        else 0.0
    )
    return TraceResult(
        trace=trace_path,
        tokenizer=tokenizer_path,
        conversations=conversations,
        requests=requests,
        evaluated_transitions=transitions,
        demand_chunks=demand_chunks,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        next_eligible_hits=next_eligible_hits,
        predictions=predictions,
        useful_predictions=useful_predictions,
        useful_eligible_predictions=useful_eligible_predictions,
        wasted_predictions=wasted_predictions,
        precision=precision,
        eligible_hit_coverage=coverage,
        observe_p50_us=_percentile_ns(observe_ns, 0.50),
        observe_p99_us=_percentile_ns(observe_ns, 0.99),
        predict_p50_us=_percentile_ns(predict_ns, 0.50),
        predict_p99_us=_percentile_ns(predict_ns, 0.99),
        predictor_sources=predictor.sizes()[0],
        predictor_pairs=predictor.sizes()[1],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--limit", type=int, default=2_000)
    parser.add_argument("--min-turns", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--min-support", type=float, default=2.0)
    parser.add_argument("--min-confidence", type=float, default=0.70)
    parser.add_argument("--min-lift", type=float, default=1.20)
    parser.add_argument("--prediction-limit", type=int, default=4)
    args = parser.parse_args()
    result = replay_trace(
        args.trace,
        args.tokenizer,
        limit=args.limit,
        min_turns=args.min_turns,
        chunk_size=args.chunk_size,
        min_support=args.min_support,
        min_confidence=args.min_confidence,
        min_lift=args.min_lift,
        prediction_limit=args.prediction_limit,
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
