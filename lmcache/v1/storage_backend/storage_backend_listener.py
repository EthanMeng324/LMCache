# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, List, Tuple
import abc

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryObj

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface


class StorageBackendListener(metaclass=abc.ABCMeta):
    """Listener for events happen inside storage backend."""

    @abc.abstractmethod
    def on_evict(
        self,
        backend: "StorageBackendInterface",
        items: List[Tuple[CacheEngineKey, MemoryObj]],
    ) -> None:
        """Called when a backend evicts items.

        Args:
            backend: the backend that evicted items.
            items: list of (key, memory_obj) pairs for evicted items. The caller
                guarantees `memory_obj` remains valid for the duration of this
                callback. If the listener schedules asynchronous work using
                the object, it should take ownership via `ref_count_up()` and
                later release via `ref_count_down()`.
        """
        raise NotImplementedError
