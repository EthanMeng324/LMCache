#!/usr/bin/env python3
"""Replay a longer real ShareGPT conversation for chunk association E2E tests.

The trace supplies the user turns and the service supplies the assistant turns.
Two sessions repeat the same A request, then branch to different real user
turns.  With a 192-word context and 128 generated tokens, the branch occupies
the second 256-token chunk, which lets the router learn A -> B and predict B
while serving the C branch.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from openai import OpenAI


def _text(record: dict, index: int) -> str:
    conversations = record.get("conversations", [])
    value = conversations[index].get("value")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"trace conversation {index} is empty")
    return value


def _complete(client: OpenAI, model: str, messages: list[dict[str, str]],
              session: str, label: str, max_tokens: int) -> dict[str, object]:
    started = time.perf_counter()
    first_token = None
    text = ""
    prompt_tokens = generation_tokens = None
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=max_tokens,
        stream=True,
        stream_options={"include_usage": True},
        extra_headers={"x-user-id": label},
        extra_body={"kv_transfer_params": {"lmcache.session_id": session}},
    )
    for chunk in stream:
        if chunk.choices:
            content = chunk.choices[0].delta.content
            if content:
                first_token = first_token or time.perf_counter()
                text += content
        if chunk.usage is not None:
            prompt_tokens = chunk.usage.prompt_tokens
            generation_tokens = chunk.usage.completion_tokens
    finished = time.perf_counter()
    return {
        "label": label,
        "session": session,
        "prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
        "ttft_seconds": (first_token or finished) - started,
        "elapsed_seconds": finished - started,
        "text": text,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="Qwen2.5-3B-Instruct")
    parser.add_argument("--trace", default="LMCache/benchmarks/multi_round_qa/ShareGPT.json")
    parser.add_argument("--record-index", type=int, default=0)
    parser.add_argument("--output", default=".runtime/global-association-test/requests-longtrace.json")
    parser.add_argument("--answer-len", type=int, default=128)
    parser.add_argument("--system-words", type=int, default=192)
    args = parser.parse_args()

    with open(args.trace, encoding="utf-8") as trace_file:
        record = json.load(trace_file)[args.record_index]
    a, b, c, d = (_text(record, index) for index in (0, 2, 4, 6))
    system = " ".join(["shared-context"] * args.system_words)
    base = [{"role": "system", "content": system}]
    client = OpenAI(api_key="EMPTY", base_url=args.base_url)
    requests: list[dict[str, object]] = []

    def send(label: str, session: str, messages: list[dict[str, str]]) -> str:
        result = _complete(client, args.model, messages, session, label, args.answer_len)
        requests.append(result)
        print(f"{label}: session={session} prompt_tokens={result['prompt_tokens']} "
              f"ttft={result['ttft_seconds']:.4f}s")
        return str(result["text"])

    # Learn A -> B in session 1, then create an additional long transition.
    assistant_a = send("s1-a", "long-s1", base + [{"role": "user", "content": a}])
    assistant_b = send(
        "s1-b", "long-s1",
        base + [{"role": "user", "content": a},
                {"role": "assistant", "content": assistant_a},
                {"role": "user", "content": b}],
    )
    send(
        "s1-d", "long-s1",
        base + [{"role": "user", "content": a},
                {"role": "assistant", "content": assistant_a},
                {"role": "user", "content": b},
                {"role": "assistant", "content": assistant_b},
                {"role": "user", "content": d}],
    )
    time.sleep(1.0)

    # Repeat A in a new session and branch to C.  The router should issue a
    # temporal hint for B before the later B demand.
    send("s2-a", "long-s2", base + [{"role": "user", "content": a}])
    c_messages = base + [{"role": "user", "content": a},
                         {"role": "assistant", "content": assistant_a},
                         {"role": "user", "content": c}]
    send("s2-c", "long-s2", c_messages)
    time.sleep(1.0)
    b_messages = base + [{"role": "user", "content": a},
                         {"role": "assistant", "content": assistant_a},
                         {"role": "user", "content": b}]
    send("s2-b", "long-s2", b_messages)
    time.sleep(1.0)
    send("s2-b-followup", "long-s2", b_messages)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(requests, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
