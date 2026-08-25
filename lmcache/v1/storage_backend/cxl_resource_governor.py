# SPDX-License-Identifier: Apache-2.0
"""Serving-priority arbitration for shared CXL bandwidth.

The CXL backend is synchronous and is used by both the serving data path and
the best-effort CPU->CXL offload path.  An asyncio task or a background thread
does not, by itself, give the two paths different resource priority.  This
small, process-local governor provides that missing admission control:

* serving operations may wait for one already-running background chunk, then
  take priority over future background work;
* background work is admitted only when there is no serving operation active
  or waiting, and never waits for the resource;
* background callers can therefore defer a chunk without blocking the serving
  event loop or consuming an unbounded queue of pinned buffers.

The governor deliberately limits only one background operation at a time per
backend process.  The native CXL shared-memory lock still provides the
cross-process correctness guarantee; this class is the local latency and
bandwidth-priority layer.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import errno
import hashlib
import os
from threading import Condition
from typing import Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - CXL deployments are Linux-based.
    fcntl = None


@dataclass(frozen=True, slots=True)
class CxlResourceSnapshot:
    """A bounded diagnostic snapshot of the local CXL arbiter."""

    serving_active: int
    serving_waiting: int
    background_active: bool
    background_admitted: int
    background_deferred: int
    serving_waited: int


class CxlSharedResourceLock:
    """Coordinate CXL payload access between local worker processes.

    The Python governor is process-local.  vLLM/LMCache normally starts one
    EngineCore process per GPU, so a process-local lock alone would still let
    two idle-looking workers write the shared CXL mapping at the same time.
    A short-lived POSIX advisory lock extends the same policy to processes on
    this host:

    * serving uses a shared lock and can proceed concurrently with other
      serving operations;
    * background offload uses a non-blocking exclusive lock and is deferred if
      any serving process currently owns the shared lock.

    The lock coordinates payload copies only.  CXL metadata correctness
    remains the responsibility of the native CXL shared-memory locks.
    """

    def __init__(
        self,
        resource_id: str,
        *,
        lock_dir: str | None = None,
        enabled: bool = True,
    ) -> None:
        self.enabled = bool(enabled and fcntl is not None)
        if not self.enabled:
            self.path = None
            return

        directory = lock_dir or os.getenv(
            "LMCACHE_CXL_RESOURCE_LOCK_DIR", "/tmp"
        )
        if not os.path.isdir(directory):
            self.enabled = False
            self.path = None
            return
        digest = hashlib.sha256(resource_id.encode("utf-8")).hexdigest()[:24]
        self.path = os.path.join(directory, f"lmcache-cxl-resource-{digest}.lock")

    def _open(self) -> int | None:
        if not self.enabled or self.path is None:
            return None
        return os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)

    @staticmethod
    def _close(fd: int | None) -> None:
        if fd is None:
            return
        try:
            os.close(fd)
        except OSError:
            pass

    @contextmanager
    def serving(self) -> Iterator[None]:
        """Hold a process-shared read lock for one serving CXL operation."""

        fd = self._open()
        try:
            if fd is not None:
                assert fcntl is not None
                fcntl.flock(fd, fcntl.LOCK_SH)
            yield
        finally:
            if fd is not None:
                assert fcntl is not None
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                finally:
                    self._close(fd)

    @contextmanager
    def try_background(self) -> Iterator[bool]:
        """Try to hold a process-shared exclusive lock without waiting."""

        fd = self._open()
        if fd is None:
            yield True
            return

        admitted = False
        try:
            assert fcntl is not None
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                admitted = True
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
            yield admitted
        finally:
            if admitted:
                assert fcntl is not None
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                finally:
                    self._close(fd)
            else:
                self._close(fd)


class CxlResourceGovernor:
    """Give serving CXL operations priority over background copies.

    ``serving()`` is a normal blocking context manager because a demand read
    or write must eventually complete.  ``try_background()`` is intentionally
    non-blocking: an offload that loses the race is deferred to the next
    planner round instead of waiting behind serving work.
    """

    def __init__(self) -> None:
        self._condition = Condition()
        self._serving_active = 0
        self._serving_waiting = 0
        self._background_active = False
        self._background_admitted = 0
        self._background_deferred = 0
        self._serving_waited = 0

    @contextmanager
    def serving(self) -> Iterator[None]:
        """Enter a high-priority serving CXL operation."""

        waited = False
        with self._condition:
            self._serving_waiting += 1
            try:
                while self._background_active:
                    waited = True
                    self._condition.wait()
                self._serving_active += 1
            finally:
                self._serving_waiting -= 1
            if waited:
                self._serving_waited += 1

        try:
            yield
        finally:
            with self._condition:
                self._serving_active -= 1
                self._condition.notify_all()

    @contextmanager
    def try_background(self) -> Iterator[bool]:
        """Try to enter one low-priority background CXL operation.

        This method never waits.  In particular, a waiting serving caller
        prevents a new background operation from starting, which avoids
        starving demand traffic between two background chunks.
        """

        admitted = False
        with self._condition:
            if (
                not self._background_active
                and self._serving_active == 0
                and self._serving_waiting == 0
            ):
                self._background_active = True
                self._background_admitted += 1
                admitted = True
            else:
                self._background_deferred += 1

        try:
            yield admitted
        finally:
            if admitted:
                with self._condition:
                    self._background_active = False
                    self._condition.notify_all()

    def snapshot(self) -> CxlResourceSnapshot:
        """Return counters suitable for logs or benchmark telemetry."""

        with self._condition:
            return CxlResourceSnapshot(
                serving_active=self._serving_active,
                serving_waiting=self._serving_waiting,
                background_active=self._background_active,
                background_admitted=self._background_admitted,
                background_deferred=self._background_deferred,
                serving_waited=self._serving_waited,
            )
