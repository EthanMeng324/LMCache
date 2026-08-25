# SPDX-License-Identifier: Apache-2.0
# Standard
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Tuple
import argparse
import asyncio
import json
import os
import sys
import uuid

# Add project root to Python path for local development
sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
)

# Third Party
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# First Party
from lmcache.logging import init_logger
from lmcache.v1.cache_controller.controller_manager import LMCacheControllerManager
from lmcache.v1.cache_controller.message import (  # noqa: E501
    CheckFinishMsg,
    CheckFinishRetMsg,
    ClearMsg,
    ClearRetMsg,
    CompressMsg,
    CompressRetMsg,
    DecompressMsg,
    DecompressRetMsg,
    ErrorMsg,
    HealthMsg,
    HealthRetMsg,
    LookupMsg,
    LookupRetMsg,
    MoveMsg,
    MoveRetMsg,
    PrefetchHintMsg,
    PrefetchHintRetMsg,
    PrefetchStatusMsg,
    PrefetchStatusRetMsg,
    CancelPrefetchHintMsg,
    CancelPrefetchHintRetMsg,
    PinMsg,
    PinRetMsg,
    QueryInstMsg,
    QueryInstRetMsg,
    QueryWorkerInfoMsg,
    QueryWorkerInfoRetMsg,
)
from lmcache.v1.cache_controller.utils import WorkerInfo
from lmcache.v1.internal_api_server.api_registry import APIRegistry

logger = init_logger(__name__)


def create_app(
    controller_urls: dict[str, str],
    health_check_interval: int,
    lmcache_worker_timeout: int,
) -> FastAPI:
    """
    Create a FastAPI application with endpoints for LMCache operations.
    """
    lmcache_controller_manager = LMCacheControllerManager(
        controller_urls, health_check_interval, lmcache_worker_timeout
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Start background task here
        lmcache_cluster_monitor_task = asyncio.create_task(
            lmcache_controller_manager.start_all()
        )
        yield
        # Optionally cancel the task on shutdown
        lmcache_cluster_monitor_task.cancel()
        try:
            await lmcache_cluster_monitor_task
        except asyncio.CancelledError:
            pass

    app = FastAPI(lifespan=lifespan)
    app.state.lmcache_controller_manager = lmcache_controller_manager

    # Register internal APIs (only common APIs, not vllm-specific ones)
    registry = APIRegistry(app)
    registry.register_all_apis(categories=["common", "controller"])

    # Add static files for frontend
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    static_dir = os.path.join(
        project_root,
        "lmcache",
        "v1",
        "cache_controller",
        "frontend",
        "static",
    )
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
        logger.info("Controller frontend static files mounted at /static")
    else:
        logger.warning("Controller frontend static directory not found: %s", static_dir)

    @app.get("/", response_class=HTMLResponse)
    async def serve_frontend():
        """Serve the Controller frontend HTML page."""
        index_path = os.path.join(static_dir, "index.html")
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                html_content = f.read()
            return HTMLResponse(content=html_content)
        else:
            return HTMLResponse(
                content="<h1>Controller Frontend not found</h1>"
                "<p>Please build the frontend first.</p>",
                status_code=404,
            )

    class QueryInstRequest(BaseModel):
        event_id: str
        ip: str

    class QueryInstResponse(BaseModel):
        event_id: str
        res: str  # the instance id

    @app.post("/query_instance")
    async def query_instance(req: QueryInstRequest):
        try:
            event_id = "QueryInst" + str(uuid.uuid4())
            msg = QueryInstMsg(
                event_id=event_id,
                ip=req.ip,
            )
            ret_msg = await lmcache_controller_manager.handle_orchestration_message(msg)
            assert not isinstance(ret_msg, ErrorMsg), ret_msg.error
            assert isinstance(ret_msg, QueryInstRetMsg)
            return QueryInstResponse(
                event_id=ret_msg.event_id,
                res=ret_msg.instance_id,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    class LookupRequest(BaseModel):
        tokens: List[int]

    class LookupResponse(BaseModel):
        event_id: str
        # a list of (instance_id, location, token_count)
        layout_info: Dict[str, Tuple[str, int]]

    @app.post("/lookup", response_model=LookupResponse)
    async def lookup(req: LookupRequest):
        try:
            event_id = "Lookup" + str(uuid.uuid4())
            msg = LookupMsg(
                event_id=event_id,
                tokens=req.tokens,
            )
            ret_msg = await lmcache_controller_manager.handle_orchestration_message(msg)
            assert not isinstance(ret_msg, ErrorMsg), ret_msg.error
            assert isinstance(ret_msg, LookupRetMsg)
            return LookupResponse(
                event_id=ret_msg.event_id, layout_info=ret_msg.layout_info
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    class ClearRequest(BaseModel):
        instance_id: str
        location: str

    class ClearResponse(BaseModel):
        event_id: str
        num_tokens: int

    @app.post("/clear", response_model=ClearResponse)
    async def clear(req: ClearRequest):
        try:
            event_id = "Clear" + str(uuid.uuid4())
            msg = ClearMsg(
                event_id=event_id,
                instance_id=req.instance_id,
                location=req.location,
            )
            ret_msg = await lmcache_controller_manager.handle_orchestration_message(msg)
            assert not isinstance(ret_msg, ErrorMsg), ret_msg.error
            assert isinstance(ret_msg, ClearRetMsg)
            return ClearResponse(
                event_id=ret_msg.event_id, num_tokens=ret_msg.num_tokens
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    class PinRequest(BaseModel):
        instance_id: str
        location: str
        tokens: list[int]

    class PinResponse(BaseModel):
        event_id: str
        num_tokens: int

    @app.post("/pin", response_model=PinResponse)
    async def pin(req: PinRequest):
        try:
            event_id = "Pin" + str(uuid.uuid4())
            msg = PinMsg(
                event_id=event_id,
                instance_id=req.instance_id,
                location=req.location,
                tokens=req.tokens,
            )
            ret_msg = await lmcache_controller_manager.handle_orchestration_message(msg)
            assert not isinstance(ret_msg, ErrorMsg), ret_msg.error
            assert isinstance(ret_msg, PinRetMsg)
            return PinResponse(event_id=ret_msg.event_id, num_tokens=ret_msg.num_tokens)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    class CompressRequest(BaseModel):
        instance_id: str
        method: str
        location: str
        tokens: Optional[List[int]] = []

    class CompressResponse(BaseModel):
        event_id: str
        num_tokens: int

    class DecompressRequest(BaseModel):
        instance_id: str
        method: str
        location: str
        tokens: Optional[List[int]] = []

    class DecompressResponse(BaseModel):
        event_id: str
        num_tokens: int

    @app.post("/compress", response_model=CompressResponse)
    async def compress(req: CompressRequest):
        try:
            event_id = "Compress" + str(uuid.uuid4())
            msg = CompressMsg(
                event_id=event_id,
                instance_id=req.instance_id,
                method=req.method,
                location=req.location,
                tokens=req.tokens,
            )
            ret_msg = await lmcache_controller_manager.handle_orchestration_message(msg)
            assert not isinstance(ret_msg, ErrorMsg), ret_msg.error
            assert isinstance(ret_msg, CompressRetMsg)
            return CompressResponse(
                event_id=ret_msg.event_id, num_tokens=ret_msg.num_tokens
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/decompress", response_model=DecompressResponse)
    async def decompress(req: DecompressRequest):
        try:
            event_id = "Decompress" + str(uuid.uuid4())
            msg = DecompressMsg(
                event_id=event_id,
                instance_id=req.instance_id,
                method=req.method,
                location=req.location,
                tokens=req.tokens,
            )
            ret_msg = await lmcache_controller_manager.handle_orchestration_message(msg)
            assert isinstance(ret_msg, DecompressRetMsg)
            return DecompressResponse(
                event_id=ret_msg.event_id, num_tokens=ret_msg.num_tokens
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    class MoveRequest(BaseModel):
        # (instance_id, location)
        old_position: Tuple[str, str]
        new_position: Tuple[str, str]
        tokens: Optional[List[int]] = []
        should_copy: Optional[bool] = False

    class MoveResponse(BaseModel):
        event_id: str
        num_tokens: int

    @app.post("/move", response_model=MoveResponse)
    async def move(req: MoveRequest):
        try:
            event_id = "Move" + str(uuid.uuid4())
            msg = MoveMsg(
                event_id=event_id,
                old_position=req.old_position,
                new_position=req.new_position,
                tokens=req.tokens,
                copy=req.should_copy,
            )
            ret_msg = await lmcache_controller_manager.handle_orchestration_message(msg)
            assert not isinstance(ret_msg, ErrorMsg), ret_msg.error
            assert isinstance(ret_msg, MoveRetMsg)
            return MoveResponse(
                event_id=ret_msg.event_id,
                num_tokens=ret_msg.num_tokens,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    class PrefetchHintRequest(BaseModel):
        instance_id: str
        tokens: List[int]
        request_id: Optional[str] = None
        session_id: Optional[str] = None
        model_id: Optional[str] = None
        task_id: Optional[str] = None
        route_epoch: int = 0
        deadline_ns: Optional[int] = None
        max_prefetch_bytes: Optional[int] = None
        priority: int = 0
        start_chunk: Optional[int] = None
        end_chunk: Optional[int] = None
        ttl_ms: int = 5000

    class PrefetchHintResponse(BaseModel):
        event_id: str
        accepted: int
        scheduled: int
        stale: bool = False

    @app.post("/prefetch_hint", response_model=PrefetchHintResponse)
    @app.post("/v1/cxl/prefetch", response_model=PrefetchHintResponse)
    async def prefetch_hint(req: PrefetchHintRequest):
        """Submit a non-blocking route-time LMCache prefetch hint."""
        if not req.tokens:
            raise HTTPException(status_code=422, detail="tokens must not be empty")
        try:
            event_id = "PrefetchHint" + str(uuid.uuid4())
            ret_msg = await lmcache_controller_manager.handle_orchestration_message(
                PrefetchHintMsg(
                    event_id=event_id,
                    instance_id=req.instance_id,
                    tokens=req.tokens,
                    request_id=req.request_id,
                    session_id=req.session_id,
                    model_id=req.model_id,
                    route_epoch=req.route_epoch,
                    deadline_ns=req.deadline_ns,
                    max_prefetch_bytes=req.max_prefetch_bytes,
                    priority=req.priority,
                    task_id=req.task_id,
                    start_chunk=req.start_chunk,
                    end_chunk=req.end_chunk,
                    ttl_ms=req.ttl_ms,
                )
            )
            # Route-time prefetch is best-effort.  A worker may finish or
            # transition its control socket while this hint is being queued;
            # surface that race as a stale no-op instead of HTTP 500.  The
            # router can then continue serving the demand request normally.
            if isinstance(ret_msg, ErrorMsg):
                logger.warning(
                    "Ignoring prefetch hint race: instance_id=%s request_id=%s "
                    "tokens=%d route_epoch=%d error=%s",
                    req.instance_id,
                    req.request_id,
                    len(req.tokens),
                    req.route_epoch,
                    ret_msg.error,
                )
                return PrefetchHintResponse(
                    event_id=event_id,
                    accepted=0,
                    scheduled=0,
                    stale=True,
                )
            assert isinstance(ret_msg, PrefetchHintRetMsg)
            return PrefetchHintResponse(
                event_id=ret_msg.event_id,
                accepted=ret_msg.accepted,
                scheduled=ret_msg.scheduled,
                stale=ret_msg.stale,
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(
                "LMCache prefetch hint failed: instance_id=%s request_id=%s "
                "tokens=%d route_epoch=%d",
                req.instance_id,
                req.request_id,
                len(req.tokens),
                req.route_epoch,
            )
            raise HTTPException(status_code=500, detail=str(e)) from e

    class PrefetchStatusResponse(BaseModel):
        event_id: str
        task_id: str
        state: str
        requested_chunks: int = 0
        ready_chunks: int = 0
        bytes_copied: int = 0
        started_ns: Optional[int] = None
        completed_ns: Optional[int] = None

    @app.get("/v1/cxl/prefetch/{task_id}", response_model=PrefetchStatusResponse)
    async def prefetch_status(task_id: str, instance_id: str):
        """Return the worker-aggregated state of a global prefetch task."""
        try:
            event_id = "PrefetchStatus" + str(uuid.uuid4())
            ret_msg = await lmcache_controller_manager.handle_orchestration_message(
                PrefetchStatusMsg(
                    event_id=event_id,
                    instance_id=instance_id,
                    task_id=task_id,
                )
            )
            if isinstance(ret_msg, ErrorMsg):
                raise HTTPException(status_code=404, detail=ret_msg.error)
            assert isinstance(ret_msg, PrefetchStatusRetMsg)
            return PrefetchStatusResponse(
                event_id=ret_msg.event_id,
                task_id=ret_msg.task_id,
                state=ret_msg.state,
                requested_chunks=ret_msg.requested_chunks,
                ready_chunks=ret_msg.ready_chunks,
                bytes_copied=ret_msg.bytes_copied,
                started_ns=ret_msg.started_ns,
                completed_ns=ret_msg.completed_ns,
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    class CancelPrefetchHintRequest(BaseModel):
        instance_id: str
        request_id: str
        route_epoch: int

    class CancelPrefetchHintResponse(BaseModel):
        event_id: str
        cancelled: int

    @app.post("/cancel_prefetch_hint", response_model=CancelPrefetchHintResponse)
    async def cancel_prefetch_hint(req: CancelPrefetchHintRequest):
        try:
            event_id = "CancelPrefetchHint" + str(uuid.uuid4())
            ret_msg = await lmcache_controller_manager.handle_orchestration_message(
                CancelPrefetchHintMsg(
                    event_id=event_id,
                    instance_id=req.instance_id,
                    request_id=req.request_id,
                    route_epoch=req.route_epoch,
                )
            )
            # Cancellation is best-effort and idempotent.  The route may have
            # already completed, expired, or lost its worker socket by the
            # time the frontend sends the cancellation request.  Treat those
            # control-plane races as a successful no-op instead of exposing a
            # spurious HTTP 500 to the router.
            if isinstance(ret_msg, ErrorMsg):
                logger.warning(
                    "Ignoring prefetch cancellation race: instance_id=%s "
                    "request_id=%s route_epoch=%d error=%s",
                    req.instance_id,
                    req.request_id,
                    req.route_epoch,
                    ret_msg.error,
                )
                return CancelPrefetchHintResponse(
                    event_id=event_id,
                    cancelled=0,
                )
            assert isinstance(ret_msg, CancelPrefetchHintRetMsg)
            return CancelPrefetchHintResponse(
                event_id=ret_msg.event_id,
                cancelled=ret_msg.cancelled,
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(
                "LMCache prefetch cancellation failed: instance_id=%s request_id=%s "
                "route_epoch=%d",
                req.instance_id,
                req.request_id,
                req.route_epoch,
            )
            raise HTTPException(status_code=500, detail=str(e)) from e

    class HealthRequest(BaseModel):
        instance_id: str

    class HealthResponse(BaseModel):
        event_id: str
        # worker_id -> error_code
        error_codes: dict[int, int]

    @app.post("/health", response_model=HealthResponse)
    async def health(req: HealthRequest):
        try:
            event_id = "health" + str(uuid.uuid4())
            msg = HealthMsg(
                event_id=event_id,
                instance_id=req.instance_id,
            )
            ret_msg = await lmcache_controller_manager.handle_orchestration_message(msg)
            assert not isinstance(ret_msg, ErrorMsg), ret_msg.error
            assert isinstance(ret_msg, HealthRetMsg)
            return HealthResponse(
                event_id=ret_msg.event_id, error_codes=ret_msg.error_codes
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    class CheckFinishRequest(BaseModel):
        event_id: str

    class CheckFinishResponse(BaseModel):
        status: str

    @app.post("/check_finish", response_model=CheckFinishResponse)
    async def check_finish(req: CheckFinishRequest):
        try:
            msg = CheckFinishMsg(
                event_id=req.event_id,
            )
            ret_msg = await lmcache_controller_manager.handle_orchestration_message(msg)
            assert not isinstance(ret_msg, ErrorMsg), ret_msg.error
            assert isinstance(ret_msg, CheckFinishRetMsg)
            return CheckFinishResponse(status=ret_msg.status)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    class QueryWorkerInfoRequest(BaseModel):
        instance_id: str
        worker_ids: Optional[list[int]] = None

    class QueryWorkerInfoResponse(BaseModel):
        event_id: str
        worker_infos: list[WorkerInfo]

    @app.post("/query_worker_info", response_model=QueryWorkerInfoResponse)
    async def query_worker_info(req: QueryWorkerInfoRequest):
        try:
            event_id = "QueryWorkerInfo" + str(uuid.uuid4())
            msg = QueryWorkerInfoMsg(
                event_id=event_id,
                instance_id=req.instance_id,
                worker_ids=req.worker_ids,
            )
            ret_msg = await lmcache_controller_manager.handle_orchestration_message(msg)
            assert not isinstance(ret_msg, ErrorMsg), ret_msg.error
            assert isinstance(ret_msg, QueryWorkerInfoRetMsg)
            return QueryWorkerInfoResponse(
                event_id=ret_msg.event_id, worker_infos=ret_msg.worker_infos
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument(
        "--monitor-ports",
        type=json.loads,
        default=None,
        help='JSON string of monitor ports, e.g. \'{"pull": 8300, "reply": 8400}\'',
    )
    parser.add_argument(
        "--monitor-port",
        type=int,
        default=9001,
        help="The controller pull port to maintain backward compatibility.",
    )
    parser.add_argument(
        "--health-check-interval",
        type=int,
        default=-1,
        help="Health check interval in secs, default is -1, which means disabled.",
    )
    parser.add_argument(
        "--lmcache-worker-timeout",
        type=int,
        default=300,
        help="The lmcache worker timeout in seconds.",
    )

    args = parser.parse_args()

    try:
        if args.monitor_ports is not None:
            controller_urls = {
                "pull": f"{args.host}:{args.monitor_ports['pull']}",
                "reply": f"{args.host}:{args.monitor_ports['reply']}",
            }
        else:
            logger.warning(
                "Argument --monitor-port will be deprecated soon. "
                "Please use --monitor-ports instead."
            )
            controller_urls = {
                "pull": f"{args.host}:{args.monitor_port}",
                "reply": None,
            }
        app = create_app(
            controller_urls, args.health_check_interval, args.lmcache_worker_timeout
        )

        logger.info(f"Starting LMCache controller at {args.host}:{args.port}")
        logger.info(f"Monitoring lmcache workers at ports {args.monitor_ports}")

        uvicorn.run(app, host=args.host, port=args.port)
    except TimeoutError as e:
        logger.error(e)


if __name__ == "__main__":
    main()
