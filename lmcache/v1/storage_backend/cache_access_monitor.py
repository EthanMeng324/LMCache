# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Dict, List, Optional, Tuple
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.v1.storage_backend.cache_policy.lfu import LFUCachePolicy
from lmcache.v1.storage_backend.cache_policy.lru import LRUCachePolicy

logger = init_logger(__name__)


class CacheAccessMonitor:
    """
    Monitor access statistics for KV cache in a storage backend.
    Supports both LFU and LRU cache policies.
    """

    def __init__(self, backend: StorageBackendInterface):
        """
        Initialize the monitor for a storage backend.

        Args:
            backend: The storage backend to monitor
        """
        self.backend = backend
        self.lock = threading.Lock()

    def get_access_counts(self) -> Dict[CacheEngineKey, int]:
        """
        Get access counts for all keys in the backend.

        Returns:
            A dictionary mapping keys to their access counts
        """
        with self.lock:
            if not hasattr(self.backend, "cache_policy"):
                logger.warning(
                    f"Backend {type(self.backend).__name__} does not have cache_policy attribute"
                )
                return {}

            cache_policy = self.backend.cache_policy

            if isinstance(cache_policy, LFUCachePolicy):
                # LFU already tracks frequency
                return cache_policy.get_all_access_counts()
            elif isinstance(cache_policy, LRUCachePolicy):
                # LRU now tracks access count
                return cache_policy.get_all_access_counts()
            else:
                logger.warning(
                    f"Cache policy {type(cache_policy).__name__} does not support access count tracking"
                )
                return {}

    def get_top_keys(self, top_n: int = 10) -> List[Tuple[CacheEngineKey, int]]:
        """
        Get the top N most accessed keys.

        Args:
            top_n: Number of top keys to return

        Returns:
            A list of (key, access_count) tuples, sorted by access count (descending)
        """
        access_counts = self.get_access_counts()

        if not access_counts:
            return []

        # Sort by access count (descending)
        sorted_keys = sorted(
            access_counts.items(), key=lambda x: x[1], reverse=True
        )

        return sorted_keys[:top_n]

    def get_access_count(self, key: CacheEngineKey) -> int:
        """
        Get the access count for a specific key.

        Args:
            key: The cache key

        Returns:
            The access count, or 0 if not found
        """
        if not hasattr(self.backend, "cache_policy"):
            return 0

        cache_policy = self.backend.cache_policy

        if isinstance(cache_policy, (LFUCachePolicy, LRUCachePolicy)):
            return cache_policy.get_access_count(key)
        else:
            return 0

