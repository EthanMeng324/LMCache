# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Union
import asyncio
import uuid

# Third Party
import msgspec
import zmq.asyncio

# First Party
from lmcache.logging import init_logger
from lmcache.v1.cache_controller.message import (  # noqa: E501
    AbortOffloadWorkerMsg,
    AbortOffloadWorkerRetMsg,
    CheckFinishMsg,
    CheckFinishRetMsg,
    ClearMsg,
    ClearRetMsg,
    ClearWorkerMsg,
    CompressMsg,
    CompressRetMsg,
    CompressWorkerMsg,
    DecompressMsg,
    DecompressRetMsg,
    DecompressWorkerMsg,
    ErrorMsg,
    HealthMsg,
    HealthRetMsg,
    HealthWorkerMsg,
    HealthWorkerRetMsg,
    MoveMsg,
    MoveRetMsg,
    MoveWorkerMsg,
    PrefetchHintMsg,
    PrefetchHintRetMsg,
    PrefetchWorkerMsg,
    PrefetchWorkerRetMsg,
    PrefetchStatusMsg,
    PrefetchStatusRetMsg,
    PrefetchStatusWorkerMsg,
    PrefetchStatusWorkerRetMsg,
    CancelPrefetchHintMsg,
    CancelPrefetchHintRetMsg,
    CommitOffloadWorkerMsg,
    CommitOffloadWorkerRetMsg,
    FinalizeOffloadWorkerMsg,
    FinalizeOffloadWorkerRetMsg,
    Msg,
    MsgBase,
    OffloadMsg,
    OffloadRetMsg,
    OffloadWorkerMsg,
    OffloadWorkerRetMsg,
    PinMsg,
    PinRetMsg,
    PinWorkerMsg,
)

logger = init_logger(__name__)


# NOTE (Jiayi): `LMCacheClusterExecutor` might need to be in different processes
# in the future for the sake of performance.
# NOTE (Jiayi): Also, consider scaling up the number of cluster executors
# in the future.
# TODO (Jiayi): need better error handling
class LMCacheClusterExecutor:
    """
    LMCache Cluster Executor class to handle the execution of cache operations.
    """

    def __init__(self, reg_controller):
        """
        Initialize the LMCache Executor with a cache instance.

        :param lmcache_instance_id: lmcache_instance_id
        """
        self.reg_controller = reg_controller

    async def clear(self, msg: ClearMsg) -> Union[ClearRetMsg, ErrorMsg]:
        """
        Execute a clear cache operation with error handling.
        """
        instance_id = msg.instance_id
        location = msg.location

        worker_ids = self.reg_controller.get_workers(instance_id)
        assert worker_ids is not None
        sockets = []
        serialized_msgs = []
        for worker_id in worker_ids:
            socket = self.reg_controller.get_socket(instance_id, worker_id)
            if socket is None:
                return ErrorMsg(
                    error=(
                        f"Worker {worker_id} not registered for instance {instance_id}"
                    )
                )
            sockets.append(socket)

            # TODO(Jiayi): Need a way to trak event_id -> worker_event_id mapping
            # Also, we need to track worker_event_id status
            worker_event_id = f"Worker{worker_id}{msg.event_id}"
            serialized_msg = msgspec.msgpack.encode(
                ClearWorkerMsg(
                    worker_event_id=worker_event_id,
                    location=location,
                )
            )
            serialized_msgs.append(serialized_msg)
        serialized_results = await self.execute_workers(
            sockets=sockets,
            serialized_msgs=serialized_msgs,
        )

        num_tokens_list = []
        for i, serialized_result in enumerate(serialized_results):
            result = msgspec.msgpack.decode(serialized_result, type=Msg)
            num_tokens_list.append(result.num_tokens)

        # TODO(Jiayi): Need to ensure cache consistency across workers.
        assert len(set(num_tokens_list)) == 1, (
            "The number of tokens cleared should be the same across all workers."
        )

        return ClearRetMsg(event_id=msg.event_id, num_tokens=num_tokens_list[0])

    async def pin(self, msg: PinMsg) -> Union[PinRetMsg, ErrorMsg]:
        """
        Execute a pin cache operation with error handling.
        """
        instance_id = msg.instance_id
        tokens = msg.tokens
        location = msg.location

        worker_ids = self.reg_controller.get_workers(instance_id)
        assert worker_ids is not None
        sockets = []
        serialized_msgs = []
        for worker_id in worker_ids:
            socket = self.reg_controller.get_socket(instance_id, worker_id)
            if socket is None:
                return ErrorMsg(
                    error=(
                        f"Worker {worker_id} not registered for instance {instance_id}"
                    )
                )
            sockets.append(socket)

            # TODO(Jiayi): Need a way to trak event_id -> worker_event_id mapping
            # Also, we need to track worker_event_id status
            worker_event_id = f"Worker{worker_id}{msg.event_id}"
            serialized_msg = msgspec.msgpack.encode(
                PinWorkerMsg(
                    worker_event_id=worker_event_id,
                    tokens=tokens,
                    location=location,
                )
            )
            serialized_msgs.append(serialized_msg)
        serialized_results = await self.execute_workers(
            sockets=sockets,
            serialized_msgs=serialized_msgs,
        )

        num_tokens_list = []
        for i, serialized_result in enumerate(serialized_results):
            result = msgspec.msgpack.decode(serialized_result, type=Msg)
            num_tokens_list.append(result.num_tokens)

        # TODO(Jiayi): Need to ensure cache consistency across workers.
        assert len(set(num_tokens_list)) == 1, (
            "The number of tokens pinned should be the same across all workers."
        )

        return PinRetMsg(event_id=msg.event_id, num_tokens=num_tokens_list[0])

    async def compress(self, msg: CompressMsg) -> Union[CompressRetMsg, ErrorMsg]:
        """
        Execute a compress operation with error handling.
        """
        event_id = msg.event_id
        instance_id = msg.instance_id
        method = msg.method
        location = msg.location
        tokens = msg.tokens

        worker_ids = self.reg_controller.get_workers(instance_id)
        assert worker_ids is not None

        # TODO(Jiayi): Currently, we do not support PP or heterogeneous TP.
        # NOTE(Jiayi): The TP ranks are already sorted in registration_controller.

        sockets = []
        serialized_msgs = []
        for worker_id in worker_ids:
            socket = self.reg_controller.get_socket(instance_id, worker_id)

            if socket is None:
                return ErrorMsg(
                    error=(
                        f"Worker {worker_id} not registered for "
                        f"instance {instance_id} or "
                    )
                )
            sockets.append(socket)

            worker_event_id = f"CompressWorker{worker_id}{str(uuid.uuid4())}"
            serialized_msg = msgspec.msgpack.encode(
                CompressWorkerMsg(
                    worker_event_id=worker_event_id,
                    method=method,
                    location=location,
                    tokens=tokens,
                )
            )
            serialized_msgs.append(serialized_msg)
            logger.debug(
                f"Sending compress operation to worker ({instance_id}, {worker_id})"
            )
        serialized_results = await self.execute_workers(
            sockets=sockets,
            serialized_msgs=serialized_msgs,
        )

        num_tokens_list = []
        for serialized_result in serialized_results:
            result = msgspec.msgpack.decode(serialized_result, type=Msg)
            num_tokens_list.append(result.num_tokens)

        # TODO(Jiayi): Need to ensure cache consistency across workers.
        assert len(set(num_tokens_list)) == 1, (
            "The number of tokens compressed should be the same across all workers."
        )

        return CompressRetMsg(
            event_id=event_id,
            num_tokens=num_tokens_list[0],
        )

    async def decompress(self, msg: DecompressMsg) -> Union[DecompressRetMsg, ErrorMsg]:
        """
        Execute a decompress operation with error handling.
        """
        event_id = msg.event_id
        instance_id = msg.instance_id
        method = msg.method
        location = msg.location
        tokens = msg.tokens

        worker_ids = self.reg_controller.get_workers(instance_id)
        assert worker_ids is not None

        sockets = []
        serialized_msgs = []
        for worker_id in worker_ids:
            socket = self.reg_controller.get_socket(instance_id, worker_id)

            if socket is None:
                return ErrorMsg(
                    error=(
                        f"Worker {worker_id} not registered for "
                        f"instance {instance_id} or "
                    )
                )
            sockets.append(socket)

            worker_event_id = f"DecompressWorker{worker_id}{str(uuid.uuid4())}"
            serialized_msg = msgspec.msgpack.encode(
                DecompressWorkerMsg(
                    worker_event_id=worker_event_id,
                    method=method,
                    location=location,
                    tokens=tokens,
                )
            )
            serialized_msgs.append(serialized_msg)
            logger.debug(
                f"Sending decompress operation to worker ({instance_id}, {worker_id})"
            )
        serialized_results = await self.execute_workers(
            sockets=sockets,
            serialized_msgs=serialized_msgs,
        )

        num_tokens_list = []
        for serialized_result in serialized_results:
            result = msgspec.msgpack.decode(serialized_result, type=Msg)
            num_tokens_list.append(result.num_tokens)

        assert len(set(num_tokens_list)) == 1, (
            "The number of tokens decompressed should be the same across all workers."
        )

        return DecompressRetMsg(
            event_id=event_id,
            num_tokens=num_tokens_list[0],
        )

    async def move(self, msg: MoveMsg) -> Union[MoveRetMsg, ErrorMsg]:
        """
        Execute a move cache operation with error handling.
        """
        # NOTE(Jiayi): Currently we assume the transfer is push-based.
        src_instance_id = msg.old_position[0]
        dst_instance_id = msg.new_position[0]

        src_worker_ids = self.reg_controller.get_workers(src_instance_id)
        assert src_worker_ids is not None
        dst_worker_ids = self.reg_controller.get_workers(dst_instance_id)
        assert dst_worker_ids is not None

        # TODO(Jiayi): Currently, we do not support PP or heterogeneous TP.
        # NOTE(Jiayi): The TP ranks are already sorted in registration_controller.

        sockets = []
        serialized_msgs = []
        for src_worker_id, dst_worker_id in zip(
            src_worker_ids, dst_worker_ids, strict=False
        ):
            socket = self.reg_controller.get_socket(src_instance_id, src_worker_id)
            dst_url = self.reg_controller.get_peer_init_url(
                dst_instance_id, dst_worker_id
            )

            if socket is None or dst_url is None:
                return ErrorMsg(
                    error=(
                        f"Src worker {src_worker_id} not registered for "
                        f"instance {src_instance_id} or "
                        f"dst worker {dst_worker_id} not registered for "
                        f"instance {dst_instance_id} or P2P is not enabled."
                    )
                )
            sockets.append(socket)

            worker_event_id = f"MoveWorker{src_worker_id}{str(uuid.uuid4())}"
            serialized_msg = msgspec.msgpack.encode(
                MoveWorkerMsg(
                    worker_event_id=worker_event_id,
                    old_position=msg.old_position[1],
                    new_position=(dst_url, msg.new_position[1]),
                    tokens=msg.tokens,
                    copy=msg.copy,
                )
            )
            serialized_msgs.append(serialized_msg)
            logger.debug(
                f"Sending move operation to worker ({src_instance_id}, {src_worker_id})"
            )
        serialized_results = await self.execute_workers(
            sockets=sockets,
            serialized_msgs=serialized_msgs,
        )

        num_tokens_list = []
        for serialized_result in serialized_results:
            result = msgspec.msgpack.decode(serialized_result, type=Msg)
            num_tokens_list.append(result.num_tokens)

        # TODO(Jiayi): Need to ensure cache consistency across workers.
        assert len(set(num_tokens_list)) == 1, (
            "The number of tokens moved should be the same across all workers."
        )

        return MoveRetMsg(
            event_id=msg.event_id,
            num_tokens=num_tokens_list[0],
        )

    async def prefetch_hint(
        self, msg: PrefetchHintMsg
    ) -> Union[PrefetchHintRetMsg, ErrorMsg]:
        """Fan out a route-time hint to every TP rank without blocking demand."""
        worker_ids = self.reg_controller.get_workers(msg.instance_id)
        if not worker_ids:
            return ErrorMsg(error=f"No workers found for instance {msg.instance_id}")
        sockets = []
        serialized = []
        for worker_id in worker_ids:
            socket = self.reg_controller.get_socket(msg.instance_id, worker_id)
            if socket is None:
                return ErrorMsg(error=f"Worker {worker_id} not registered")
            sockets.append(socket)
            serialized.append(
                msgspec.msgpack.encode(
                    PrefetchWorkerMsg(
                        worker_event_id=f"PrefetchWorker{worker_id}{uuid.uuid4()}",
                        tokens=msg.tokens,
                        request_id=msg.request_id,
                        session_id=msg.session_id,
                        model_id=msg.model_id,
                        route_epoch=msg.route_epoch,
                        deadline_ns=msg.deadline_ns,
                        max_prefetch_bytes=msg.max_prefetch_bytes,
                        priority=msg.priority,
                        task_id=msg.task_id,
                        start_chunk=msg.start_chunk,
                        end_chunk=msg.end_chunk,
                        ttl_ms=msg.ttl_ms,
                    )
                )
            )
        raw_results = await self.execute_workers(sockets, serialized)
        results = [msgspec.msgpack.decode(raw, type=Msg) for raw in raw_results]
        if not all(isinstance(result, PrefetchWorkerRetMsg) for result in results):
            return ErrorMsg(error="Unexpected prefetch hint response")
        typed = [result for result in results if isinstance(result, PrefetchWorkerRetMsg)]
        return PrefetchHintRetMsg(
            event_id=msg.event_id,
            accepted=sum(result.accepted for result in typed),
            scheduled=sum(result.scheduled for result in typed),
            stale=any(result.stale for result in typed),
        )

    async def cancel_prefetch_hint(
        self, msg: CancelPrefetchHintMsg
    ) -> Union[CancelPrefetchHintRetMsg, ErrorMsg]:
        worker_ids = self.reg_controller.get_workers(msg.instance_id)
        if not worker_ids:
            return ErrorMsg(error=f"No workers found for instance {msg.instance_id}")
        sockets = []
        serialized = []
        for worker_id in worker_ids:
            socket = self.reg_controller.get_socket(msg.instance_id, worker_id)
            if socket is None:
                continue
            sockets.append(socket)
            serialized.append(
                msgspec.msgpack.encode(
                    PrefetchWorkerMsg(
                        worker_event_id=f"CancelPrefetch{worker_id}{uuid.uuid4()}",
                        tokens=[],
                        request_id=msg.request_id,
                        route_epoch=msg.route_epoch,
                    )
                )
            )
        if not sockets:
            return CancelPrefetchHintRetMsg(event_id=msg.event_id, cancelled=0)
        raw_results = await self.execute_workers(sockets, serialized)
        results = [msgspec.msgpack.decode(raw, type=Msg) for raw in raw_results]
        cancelled = sum(
            result.cancelled
            for result in results
            if isinstance(result, PrefetchWorkerRetMsg)
        )
        return CancelPrefetchHintRetMsg(event_id=msg.event_id, cancelled=cancelled)

    async def prefetch_status(
        self, msg: PrefetchStatusMsg
    ) -> Union[PrefetchStatusRetMsg, ErrorMsg]:
        worker_ids = self.reg_controller.get_workers(msg.instance_id)
        if not worker_ids:
            return ErrorMsg(error=f"No workers found for instance {msg.instance_id}")
        sockets = []
        serialized = []
        for worker_id in worker_ids:
            socket = self.reg_controller.get_socket(msg.instance_id, worker_id)
            if socket is None:
                continue
            sockets.append(socket)
            serialized.append(
                msgspec.msgpack.encode(
                    PrefetchStatusWorkerMsg(
                        worker_event_id=f"PrefetchStatus{worker_id}{uuid.uuid4()}",
                        task_id=msg.task_id,
                    )
                )
            )
        if not sockets:
            return ErrorMsg(error=f"No workers registered for instance {msg.instance_id}")
        raw_results = await self.execute_workers(sockets, serialized)
        results = [msgspec.msgpack.decode(raw, type=Msg) for raw in raw_results]
        typed = [
            result for result in results if isinstance(result, PrefetchStatusWorkerRetMsg)
        ]
        if not typed:
            return ErrorMsg(error="Unexpected prefetch status response")
        states = {result.state for result in typed}
        if "FAILED" in states:
            state = "FAILED"
        elif "COPYING" in states or "SUBMITTED" in states:
            state = "COPYING"
        elif states == {"READY"}:
            state = "READY"
        else:
            state = "UNKNOWN"
        return PrefetchStatusRetMsg(
            event_id=msg.event_id,
            task_id=msg.task_id,
            state=state,
            requested_chunks=max(result.requested_chunks for result in typed),
            ready_chunks=sum(result.ready_chunks for result in typed),
            bytes_copied=sum(result.bytes_copied for result in typed),
            started_ns=min(
                (result.started_ns for result in typed if result.started_ns is not None),
                default=None,
            ),
            completed_ns=max(
                (result.completed_ns for result in typed if result.completed_ns is not None),
                default=None,
            ),
        )

    async def _abort_prepared_offload(
        self,
        sockets: list[zmq.asyncio.Socket],
        event_id: str,
    ) -> None:
        """Best-effort abort for every rank after prepare/commit failure."""
        if not sockets:
            return
        serialized_messages = [
            msgspec.msgpack.encode(
                AbortOffloadWorkerMsg(
                    worker_event_id=f"AbortOffloadWorker{index}{uuid.uuid4()}",
                    event_id=event_id,
                )
            )
            for index in range(len(sockets))
        ]
        try:
            raw_results = await self.execute_workers(sockets, serialized_messages)
            for raw in raw_results:
                try:
                    result = msgspec.msgpack.decode(raw, type=Msg)
                except Exception:
                    logger.exception(
                        "Invalid offload abort response for operation %s",
                        event_id,
                    )
                    continue
                if not isinstance(result, AbortOffloadWorkerRetMsg):
                    logger.error(
                        "Unexpected offload abort response for operation %s: %s",
                        event_id,
                        type(result).__name__,
                    )
                elif not result.success:
                    logger.error(
                        "Offload abort failed for operation %s on worker %s: %s",
                        event_id,
                        result.worker_id,
                        result.error,
                    )
        except Exception:
            # The original offload already failed.  Preserve that failure while
            # making the cleanup failure visible for operational follow-up.
            logger.exception("Unable to abort prepared offload %s", event_id)

    async def _finalize_committed_offload(
        self,
        sockets: list[zmq.asyncio.Socket],
        event_id: str,
    ) -> None:
        """Release per-rank rollback bookkeeping after global commit success."""
        if not sockets:
            return
        serialized_messages = [
            msgspec.msgpack.encode(
                FinalizeOffloadWorkerMsg(
                    worker_event_id=f"FinalizeOffloadWorker{index}{uuid.uuid4()}",
                    event_id=event_id,
                )
            )
            for index in range(len(sockets))
        ]
        try:
            raw_results = await self.execute_workers(sockets, serialized_messages)
            for raw in raw_results:
                try:
                    result = msgspec.msgpack.decode(raw, type=Msg)
                except Exception:
                    logger.exception(
                        "Invalid offload finalize response for operation %s",
                        event_id,
                    )
                    continue
                if not isinstance(result, FinalizeOffloadWorkerRetMsg):
                    logger.error(
                        "Unexpected offload finalize response for operation %s: %s",
                        event_id,
                        type(result).__name__,
                    )
                elif not result.success:
                    logger.error(
                        "Offload finalize failed for operation %s on worker %s: %s",
                        event_id,
                        result.worker_id,
                        result.error,
                    )
        except Exception:
            # Data is already globally admitted at this point.  A finalize
            # transport failure leaks only bounded bookkeeping, never data.
            logger.exception("Unable to finalize committed offload %s", event_id)

    async def offload(self, msg: OffloadMsg) -> Union[OffloadRetMsg, ErrorMsg]:
        """Fan out one rank-local CPU-to-CXL operation to the TP group.

        The operation has an explicit prepare/commit boundary.  Every rank
        first writes its CXL objects with event publication disabled.  Only
        after every rank has prepared successfully does the executor publish
        the prepared keys, which is the ``CXL_READY`` point observed by the
        router.  A failed prepare therefore cannot make a partial TP prefix
        routable.
        """
        worker_ids = self.reg_controller.get_workers(msg.instance_id)
        if not worker_ids:
            return ErrorMsg(error=f"No workers found for instance {msg.instance_id}")

        sockets = []
        serialized_messages = []
        # ``max_bytes`` is a global operation budget.  Each TP worker receives
        # its rank-local share so the aggregate CXL write cannot multiply the
        # planner's bytes/s reservation by the TP width.
        per_rank_max_bytes = msg.max_bytes // len(worker_ids) if msg.max_bytes > 0 else 0
        if msg.max_bytes > 0 and per_rank_max_bytes == 0:
            return ErrorMsg(
                error=(
                    f"max_bytes={msg.max_bytes} is smaller than the TP width "
                    f"({len(worker_ids)})"
                )
            )
        for worker_id in worker_ids:
            socket = self.reg_controller.get_socket(msg.instance_id, worker_id)
            if socket is None:
                return ErrorMsg(
                    error=f"Worker {worker_id} not registered for {msg.instance_id}"
                )
            sockets.append(socket)
            serialized_messages.append(
                msgspec.msgpack.encode(
                    OffloadWorkerMsg(
                        worker_event_id=f"OffloadWorker{worker_id}{uuid.uuid4()}",
                        event_id=msg.event_id,
                        tokens=msg.tokens,
                        source=msg.source,
                        target=msg.target,
                        copy=msg.copy,
                        max_chunks=msg.max_chunks,
                        max_bytes=per_rank_max_bytes,
                    )
                )
            )

        try:
            raw_results = await self.execute_workers(sockets, serialized_messages)
        except Exception as exc:
            await self._abort_prepared_offload(sockets, msg.event_id)
            return ErrorMsg(error=f"CPU-to-CXL prepare failed: {exc}")
        rank_results: list[OffloadWorkerRetMsg] = []
        try:
            for raw in raw_results:
                result = msgspec.msgpack.decode(raw, type=Msg)
                if not isinstance(result, OffloadWorkerRetMsg):
                    await self._abort_prepared_offload(sockets, msg.event_id)
                    return ErrorMsg(
                        error=f"Unexpected offload response: {type(result).__name__}"
                    )
                rank_results.append(result)
        except Exception as exc:
            await self._abort_prepared_offload(sockets, msg.event_id)
            return ErrorMsg(error=f"Invalid CPU-to-CXL prepare response: {exc}")

        prepared = len(rank_results) == len(worker_ids) and all(
            result.success for result in rank_results
        )
        if not prepared:
            await self._abort_prepared_offload(sockets, msg.event_id)
            return OffloadRetMsg(
                event_id=msg.event_id,
                instance_id=msg.instance_id,
                success=False,
                rank_results=rank_results,
            )

        commit_messages = [
            msgspec.msgpack.encode(
                CommitOffloadWorkerMsg(
                    worker_event_id=f"CommitOffloadWorker{worker_id}{uuid.uuid4()}",
                    event_id=msg.event_id,
                )
            )
            for worker_id in worker_ids
        ]
        try:
            raw_commit_results = await self.execute_workers(sockets, commit_messages)
        except Exception as exc:
            await self._abort_prepared_offload(sockets, msg.event_id)
            return ErrorMsg(error=f"CXL_READY commit failed: {exc}")
        commit_results: list[CommitOffloadWorkerRetMsg] = []
        try:
            for raw in raw_commit_results:
                result = msgspec.msgpack.decode(raw, type=Msg)
                if not isinstance(result, CommitOffloadWorkerRetMsg):
                    await self._abort_prepared_offload(sockets, msg.event_id)
                    return ErrorMsg(
                        error=f"Unexpected CXL_READY response: {type(result).__name__}"
                    )
                commit_results.append(result)
        except Exception as exc:
            await self._abort_prepared_offload(sockets, msg.event_id)
            return ErrorMsg(error=f"Invalid CXL_READY response: {exc}")

        ready = len(commit_results) == len(worker_ids) and all(
            result.success for result in commit_results
        )
        if not ready:
            logger.error(
                "CXL_READY commit failed for offload %s: %s",
                msg.event_id,
                [result.error for result in commit_results if result.error],
            )
            await self._abort_prepared_offload(sockets, msg.event_id)
        else:
            await self._finalize_committed_offload(sockets, msg.event_id)

        return OffloadRetMsg(
            event_id=msg.event_id,
            instance_id=msg.instance_id,
            success=ready,
            rank_results=rank_results,
        )

    async def health(self, msg: HealthMsg) -> Union[HealthRetMsg, ErrorMsg]:
        """
        Execute a compress operation with error handling.
        """
        instance_id = msg.instance_id

        worker_ids = self.reg_controller.get_workers(instance_id)
        if worker_ids is None:
            return ErrorMsg(error=f"No workers found for instance {instance_id}")

        # TODO(Jiayi): Currently, we do not support PP or heterogeneous TP.
        # NOTE(Jiayi): The TP ranks are already sorted in registration_controller.

        sockets = []
        serialized_msgs = []
        for worker_id in worker_ids:
            socket = self.reg_controller.get_socket(instance_id, worker_id)

            if socket is None:
                return ErrorMsg(
                    error=(
                        f"Worker {worker_id} not registered for "
                        f"instance {instance_id} or socket not found"
                    )
                )
            sockets.append(socket)

            worker_event_id = f"HealthWorker{worker_id}{str(uuid.uuid4())}"
            serialized_msg = msgspec.msgpack.encode(
                HealthWorkerMsg(
                    worker_event_id=worker_event_id,
                )
            )
            serialized_msgs.append(serialized_msg)
            logger.debug(
                f"Sending health check operation to worker ({instance_id}, {worker_id})"
            )

        # Collect results from all workers
        serialized_results = await self.execute_workers(
            sockets=sockets,
            serialized_msgs=serialized_msgs,
        )

        # Process results
        error_codes = {}
        for i, serialized_result in enumerate(serialized_results):
            try:
                result = msgspec.msgpack.decode(serialized_result, type=Msg)
                if isinstance(result, HealthWorkerRetMsg):
                    error_codes[worker_ids[i]] = result.error_code
                elif isinstance(result, ErrorMsg):
                    error_codes[worker_ids[i]] = -1001  # Worker returned error
                else:
                    error_codes[worker_ids[i]] = -1002  # Unexpected response
            except Exception as e:
                logger.error(
                    f"Failed to parse health response from worker "
                    f"{worker_ids[i]}: {str(e)}"
                )
                error_codes[worker_ids[i]] = -1003  # Failed to parse response

        return HealthRetMsg(
            event_id=msg.event_id,
            error_codes=error_codes,
        )

    async def check_finish(
        self, msg: CheckFinishMsg
    ) -> Union[CheckFinishRetMsg, ErrorMsg]:
        raise NotImplementedError

    # TODO(Jiayi): need to make the types more specific
    async def execute(self, operation: str, msg: MsgBase) -> MsgBase:
        """
        Execute a cache operation with error handling.

        :param operation: The operation to execute
        (e.g., 'clear').
        :param msg: The message containing the operation details.
        :return: The result of the operation or an error message.
        """
        try:
            method = getattr(self, operation)
            return await method(msg)
        except AttributeError:
            return ErrorMsg(error=f"Operation '{operation}' is not supported.")
        except Exception as e:
            return ErrorMsg(error=str(e))

    async def execute_workers(
        self,
        sockets: list[zmq.asyncio.Socket],
        serialized_msgs: list[bytes],
    ) -> list[bytes]:
        """
        Execute a list of serialized messages on the given sockets.
        :param sockets: The list of sockets to send the messages to.
        :param serialized_msgs: The list of serialized messages to send.
        :return: A list of serialized results received from the sockets.
        """
        tasks = []
        for socket, serialized_msg in zip(sockets, serialized_msgs, strict=False):

            async def send_and_receive(s, msg):
                await s.send(msg)
                return await s.recv()

            tasks.append(send_and_receive(socket, serialized_msg))

        serialized_results = await asyncio.gather(*tasks)
        return serialized_results
