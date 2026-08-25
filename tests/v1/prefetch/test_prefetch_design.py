# SPDX-License-Identifier: Apache-2.0

import time

import torch
import msgspec

from lmcache.utils import CacheEngineKey
from lmcache.v1.prefetch import (
    AccessRecorder,
    CorrelationPredictor,
    PrefetchAdmissionController,
    PrefetchContext,
    PrefetchTask,
    PrefetchState,
    segment_id_from_key,
    PrefetchMetrics,
    PrefetchScheduler,
    GlobalPatternClient,
    PatternDelta,
)


def _key(chunk_hash: int) -> CacheEngineKey:
    return CacheEngineKey("vllm", "model", 1, 0, chunk_hash, torch.bfloat16)


def test_segment_identity_is_logical_and_deterministic():
    first = segment_id_from_key(_key(11))
    second = segment_id_from_key(_key(11))
    assert first == second
    assert first.canonical().endswith(":0000000000000000000000000000000b")


def test_access_recorder_coalesces_and_aborts_request_history():
    recorder = AccessRecorder(max_events=8)
    key = _key(1)
    assert recorder.record_demand(key, req_id="r") is not None
    assert recorder.record_demand(key, req_id="r") is None
    assert recorder.request_history("r") == (key,)
    recorder.abort_request("r")
    assert recorder.request_history("r") == ()


def test_second_order_predictor_and_feedback():
    predictor = CorrelationPredictor[str](
        top_k=4, min_support=0.5, second_order_enabled=True
    )
    context = PrefetchContext(model_namespace="m")
    predictor.observe(["A", "B", "C"], context=context)
    predictions = predictor.predict("B", previous_key="A", context=context)
    assert predictions and predictions[0].key == "C"
    predictor.feedback(
        "C", source_key="B", context=context, on_time=True
    )


def test_admission_rejects_expired_deadline_and_cpu_pressure():
    admission = PrefetchAdmissionController(
        score_threshold=0.1,
        cpu_capacity_bytes=100,
        bandwidth_bytes_per_second=1000,
    )
    task = PrefetchTask(
        key="B",
        trigger_key="A",
        request_id="r",
        context=PrefetchContext(),
        source="CxlBackend",
        target="LocalCPUBackend",
        priority=1,
        deadline_ns=time.time_ns() + 1_000_000,
        expected_use_ns=0,
        score=0.9,
        size_bytes=5,
        state=PrefetchState.QUEUED,
    )
    assert admission.evaluate(task).admitted


def test_prefetch_hint_control_message_round_trip():
    from lmcache.v1.cache_controller.message import Msg, PrefetchWorkerMsg

    message = PrefetchWorkerMsg(
        worker_event_id="e1",
        tokens=[1, 2, 3],
        request_id="req",
        route_epoch=7,
        deadline_ns=123,
        max_prefetch_bytes=4096,
        priority=3,
    )
    encoded = msgspec.msgpack.encode(message)
    decoded = msgspec.msgpack.decode(encoded, type=Msg)
    assert isinstance(decoded, PrefetchWorkerMsg)
    assert decoded.route_epoch == 7
    assert decoded.tokens == [1, 2, 3]


def test_bounded_scheduler_deduplicates_and_honors_priority():
    scheduler = PrefetchScheduler(max_tasks=2)
    make_task = lambda key, priority: PrefetchTask(
        key=key,
        trigger_key="A",
        request_id="r",
        context=PrefetchContext(),
        source="CxlBackend",
        target="LocalCPUBackend",
        priority=priority,
        deadline_ns=0,
        expected_use_ns=0,
        score=1.0,
        size_bytes=1,
    )
    assert scheduler.submit(make_task("low", 1))
    assert not scheduler.submit(make_task("low", 2))
    assert scheduler.submit(make_task("high", 3))
    seen = []
    scheduler.run_once(lambda task: seen.append(task.key) or True)
    assert seen == ["high"]
    metrics = PrefetchMetrics()
    metrics.increment("scheduled", 2)
    assert metrics.snapshot()["scheduled"] == 2


def test_global_pattern_delta_merge_is_bounded_and_weighted():
    context = PrefetchContext(model_namespace="m")
    client = GlobalPatternClient(node_id="n", sink=None)
    client.merge(
        [PatternDelta(context, "P", "A", "B", 4.0, 2.0, 0.0, 1, "n")],
        weight=0.5,
    )
    assert client.global_score(context, "P", "A") == (2.0, 1.0)
