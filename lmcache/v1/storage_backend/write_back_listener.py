# SPDX-License-Identifier: Apache-2.0
# Standard
from __future__ import annotations

from typing import List, Tuple

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.storage_backend.storage_backend_listener import StorageBackendListener

logger = init_logger(__name__)


class WriteBackEvictionListener(StorageBackendListener):
    """Write-back policy listener.

    When the L1 hot cache evicts (key, MemoryObj), we "spill" the evicted items
    to a configured set of lower-tier backends.

    This is intentionally best-effort: failures in any target backend should not
    crash the engine and should not block eviction progress.
    """

    def __init__(self, spill_targets):
        # spill_targets: list[(name, backend)] where backend has batched_submit_put_task
        self._spill_targets = spill_targets

    def on_evict(
        self,
        backend,
        items: List[Tuple[CacheEngineKey, MemoryObj]],
    ) -> None:
        if not items or not self._spill_targets:
            return

        keys = [k for k, _ in items]
        mem_objs = [m for _, m in items]

        # Spill to all configured targets. Each target backend is responsible for
        # taking ownership (ref_count_up) if it uses MemoryObj asynchronously.
        for name, target_backend in self._spill_targets:
            try:
                # Provide KV schema early (dtype/fmt/layer/hidden_dim) so a backend
                # can reconstruct metadata for objects created by other nodes.
                if mem_objs and hasattr(target_backend, "set_kv_template_from_memory_obj"):
                    try:
                        target_backend.set_kv_template_from_memory_obj(mem_objs[0])
                    except Exception:
                        # best-effort; never fail spill due to template inference
                        pass
                target_backend.batched_submit_put_task(keys, mem_objs, transfer_spec=None)
            except Exception:
                logger.exception(
                    "write-back spill: backend %s failed to store %d objects; skipping",
                    name,
                    len(keys),
                )
        # NOTE: This listener does not take ownership refs for `mem_objs`.
        # The backend that triggered eviction keeps its own ref alive during this
        # callback and will release it after returning. If a spill target needs to
        # retain `mem_obj` asynchronously, it must take ownership by calling
        # `ref_count_up()` and later release via `ref_count_down()`.

