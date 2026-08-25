#!/usr/bin/env python3
"""Replay a real ShareGPT conversation in a two-session association pattern.

The first session creates the transition A -> B.  The second session repeats
the exact A prefix, branches to C, and then requests the B branch again.  The
router should issue a ``global-temporal`` hint for B before C is served; the
last request makes the resulting CPU promotion observable in LMCache logs.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from openai import OpenAI


def _message(record: dict, index: int) -> str:
    conversations = record.get("conversations", [])
    if index >= len(conversations):
        raise ValueError(f"trace record has no conversation at index {index}")
    value = conversations[index].get("value")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"trace conversation {index} is empty")
    return value


def _complete(client: OpenAI, model: str, messages: list[dict[str, str]], session: str,
              user_id: str, max_tokens: int) -> dict[str, object]:
    started = time.perf_counter()
    text = ""
    prompt_tokens = None
    generation_tokens = None
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=max_tokens,
        stream=True,
        stream_options={"include_usage": True},
        extra_headers={"x-user-id": user_id},
        extra_body={"kv_transfer_params": {"lmcache.session_id": session}},
    )
    first_token = None
    for chunk in stream:
        if chunk.choices:
            content = chunk.choices[0].delta.content
            if content:
                if first_token is None:
                    first_token = time.perf_counter()
                text += content
        if chunk.usage is not None:
            prompt_tokens = chunk.usage.prompt_tokens
            generation_tokens = chunk.usage.completion_tokens
    finished = time.perf_counter()
    return {
        "session": session,
        "prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
        "ttft_seconds": (first_token or finished) - started,
        "elapsed_seconds": finished - started,
        "text": text,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8010/v1")
    parser.add_argument("--model", default="Qwen2.5-3B-Instruct")
    parser.add_argument(
        "--trace",
        default="LMCache/benchmarks/multi_round_qa/ShareGPT.json",
    )
    parser.add_argument("--output", default=".runtime/global-association-test/requests.json")
    parser.add_argument("--answer-len", type=int, default=32)
    parser.add_argument("--system-words", type=int, default=512)
    args = parser.parse_args()

    with open(args.trace, encoding="utf-8") as trace_file:
        record = json.load(trace_file)[0]
    a, b, c = (_message(record, index) for index in (0, 2, 4))
    system = " ".join(["shared-context"] * args.system_words)
    base = [{"role": "system", "content": system}]

    client = OpenAI(api_key="EMPTY", base_url=args.base_url)
    requests: list[dict[str, object]] = []

    def send(label: str, session: str, messages: list[dict[str, str]]) -> str:
        result = _complete(client, args.model, messages, session, label, args.answer_len)
        result["label"] = label
        requests.append(result)
        print(
            f"{label}: session={session} prompt_tokens={result['prompt_tokens']} "
            f"ttft={result['ttft_seconds']:.4f}s"
        )
        return str(result["text"])

    # S1 learns A -> B.  S2 repeats A, branches to C (prediction point), then
    # requests the B branch once more to reveal whether promotion completed.
    assistant_a = send("s1-a", "assoc-s1", base + [{"role": "user", "content": a}])
    assistant_b = send(
        "s1-b",
        "assoc-s1",
        base
        + [
            {"role": "user", "content": a},
            {"role": "assistant", "content": assistant_a},
            {"role": "user", "content": b},
        ],
    )
    time.sleep(1.0)
    send("s2-a", "assoc-s2", base + [{"role": "user", "content": a}])
    send(
        "s2-c",
        "assoc-s2",
        base
        + [
            {"role": "user", "content": a},
            {"role": "assistant", "content": assistant_a},
            {"role": "user", "content": c},
        ],
    )
    time.sleep(1.0)
    send(
        "s2-b-again",
        "assoc-s2",
        base
        + [
            {"role": "user", "content": a},
            {"role": "assistant", "content": assistant_a},
            {"role": "user", "content": b},
            {"role": "assistant", "content": assistant_b},
            {"role": "user", "content": c},
        ],
    )
    # The hint for the B branch is issued while serving s2-c.  Repeat the
    # same real-trace prompt after a short drain interval so the next request
    # can observe the prefetched chunk instead of racing the hint dispatch.
    time.sleep(1.0)
    send(
        "s2-b-followup",
        "assoc-s2",
        base
        + [
            {"role": "user", "content": a},
            {"role": "assistant", "content": assistant_a},
            {"role": "user", "content": b},
            {"role": "assistant", "content": assistant_b},
            {"role": "user", "content": c},
        ],
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(requests, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
