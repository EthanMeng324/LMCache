# SPDX-License-Identifier: Apache-2.0
"""Controller and wire tests for periodic CPU-to-CXL offload commands."""

import asyncio

import msgspec

from lmcache.v1.cache_controller.executor import LMCacheClusterExecutor
from lmcache.v1.cache_controller.message import (
    AbortOffloadWorkerMsg,
    AbortOffloadWorkerRetMsg,
    ErrorMsg,
    Msg,
    CommitOffloadWorkerMsg,
    CommitOffloadWorkerRetMsg,
    FinalizeOffloadWorkerMsg,
    FinalizeOffloadWorkerRetMsg,
    OffloadMsg,
    OffloadRetMsg,
    OffloadWorkerMsg,
    OffloadWorkerRetMsg,
    PinWorkerMsg,
    PinWorkerRetMsg,
)
from lmcache.v1.cache_controller.worker import LMCacheWorker


class _FakeSocket:
    def __init__(
        self,
        worker_id: int,
        *,
        success: bool = True,
        commit_success: bool | None = None,
    ):
        self.worker_id = worker_id
        self.success = success
        self.commit_success = success if commit_success is None else commit_success
        self.last_request = None
        self.requests = []

    async def send(self, payload: bytes) -> None:
        self.last_request = msgspec.msgpack.decode(payload, type=Msg)
        self.requests.append(self.last_request)

    async def recv(self) -> bytes:
        request = self.last_request
        if isinstance(request, CommitOffloadWorkerMsg):
            return msgspec.msgpack.encode(
                CommitOffloadWorkerRetMsg(
                    event_id=request.event_id,
                    worker_id=self.worker_id,
                    success=self.commit_success,
                    published_chunks=2 if self.commit_success else 0,
                    error=None if self.commit_success else "CXL_READY failed",
                )
            )
        if isinstance(request, AbortOffloadWorkerMsg):
            return msgspec.msgpack.encode(
                AbortOffloadWorkerRetMsg(
                    event_id=request.event_id,
                    worker_id=self.worker_id,
                    success=True,
                    removed_chunks=2,
                )
            )
        if isinstance(request, FinalizeOffloadWorkerMsg):
            return msgspec.msgpack.encode(
                FinalizeOffloadWorkerRetMsg(
                    event_id=request.event_id,
                    worker_id=self.worker_id,
                    success=True,
                )
            )
        assert isinstance(request, OffloadWorkerMsg)
        return msgspec.msgpack.encode(
            OffloadWorkerRetMsg(
                event_id=request.event_id,
                worker_id=self.worker_id,
                success=self.success,
                committed_chunks=2 if self.success else 0,
                already_present_chunks=0,
                failed_chunks=0 if self.success else 2,
                bytes_written=4096 if self.success else 0,
                error=None if self.success else "source prefix not found",
            )
        )


class _FakeRegistry:
    def __init__(self, failed_rank=None, commit_failed_rank=None):
        self.worker_ids = [0, 1]
        self.sockets = {
            rank: _FakeSocket(
                rank,
                success=rank != failed_rank,
                commit_success=rank != commit_failed_rank,
            )
            for rank in self.worker_ids
        }

    def get_workers(self, instance_id):
        return self.worker_ids

    def get_socket(self, instance_id, worker_id):
        return self.sockets[worker_id]


def _offload(*, failed_rank=None, commit_failed_rank=None, max_bytes=0):
    registry = _FakeRegistry(
        failed_rank=failed_rank,
        commit_failed_rank=commit_failed_rank,
    )
    result = asyncio.run(
        LMCacheClusterExecutor(registry).offload(
            OffloadMsg(
                event_id="periodic-offload-1",
                instance_id="lmcache-replica-0",
                tokens=[1, 2, 3, 4],
                source="LocalCPUBackend",
                target="CxlBackend",
                copy=True,
                max_chunks=8,
                max_bytes=max_bytes,
            )
        )
    )
    return registry, result


def test_offload_fanouts_one_command_per_tp_rank():
    registry, result = _offload()

    assert isinstance(result, OffloadRetMsg)
    assert result.success
    assert len(result.rank_results) == 2
    for socket in registry.sockets.values():
        assert len(socket.requests) == 3
        request = socket.requests[0]
        assert isinstance(request, OffloadWorkerMsg)
        assert request.source == "LocalCPUBackend"
        assert request.target == "CxlBackend"
        assert request.copy
        assert request.max_bytes == 0
        assert isinstance(socket.requests[1], CommitOffloadWorkerMsg)
        assert isinstance(socket.requests[2], FinalizeOffloadWorkerMsg)


def test_offload_aborts_all_ranks_when_prepare_fails():
    registry, result = _offload(failed_rank=1)

    assert isinstance(result, OffloadRetMsg)
    assert not result.success
    assert [rank.success for rank in result.rank_results] == [True, False]
    assert all(len(socket.requests) == 2 for socket in registry.sockets.values())
    assert all(
        isinstance(socket.requests[1], AbortOffloadWorkerMsg)
        for socket in registry.sockets.values()
    )


def test_offload_aborts_already_published_shards_when_commit_fails():
    registry, result = _offload(commit_failed_rank=1)

    assert isinstance(result, OffloadRetMsg)
    assert not result.success
    assert all(len(socket.requests) == 3 for socket in registry.sockets.values())
    assert all(
        isinstance(socket.requests[2], AbortOffloadWorkerMsg)
        for socket in registry.sockets.values()
    )


def test_offload_splits_global_byte_budget_across_tp_ranks():
    registry, result = _offload(max_bytes=8192)

    assert isinstance(result, OffloadRetMsg)
    assert result.success
    for socket in registry.sockets.values():
        assert isinstance(socket.requests[0], OffloadWorkerMsg)
        assert socket.requests[0].max_bytes == 4096


def test_offload_msg_has_python_rust_golden_wire_shape():
    encoded = msgspec.msgpack.encode(
        OffloadMsg(
            event_id="wire-1",
            instance_id="instance-1",
            tokens=[10, 11],
            source="LocalCPUBackend",
            target="CxlBackend",
            copy=True,
            max_chunks=8,
        )
    )
    decoded = msgspec.msgpack.decode(encoded)
    assert decoded == {
        "type": "OffloadMsg",
        "event_id": "wire-1",
        "instance_id": "instance-1",
        "tokens": [10, 11],
        "source": "LocalCPUBackend",
        "target": "CxlBackend",
        "copy": True,
        "max_chunks": 8,
        "max_bytes": 0,
    }


def test_pin_worker_uses_lookup_id_for_engine_pin_lifecycle():
    request = PinWorkerMsg(
        worker_event_id="pin-event",
        location="LocalCPUBackend",
        tokens=[1, 2, 3],
    )

    class _Engine:
        def __init__(self):
            self.kwargs = None

        def lookup(self, **kwargs):
            self.kwargs = kwargs
            return 3

    class _ReplySocket:
        def __init__(self):
            self.responses = []
            self.reads = 0

        async def recv(self):
            self.reads += 1
            if self.reads == 1:
                return msgspec.msgpack.encode(request)
            raise asyncio.CancelledError()

        async def send(self, payload):
            self.responses.append(msgspec.msgpack.decode(payload, type=Msg))

    worker = LMCacheWorker.__new__(LMCacheWorker)
    worker.lmcache_engine = _Engine()
    worker.reply_socket = _ReplySocket()

    try:
        asyncio.run(worker.handle_request())
    except asyncio.CancelledError:
        pass

    assert worker.lmcache_engine.kwargs == {
        "tokens": [1, 2, 3],
        "search_range": ["LocalCPUBackend"],
        "lookup_id": "pin-event",
        "pin": True,
    }
    assert worker.reply_socket.responses == [PinWorkerRetMsg(num_tokens=3)]
