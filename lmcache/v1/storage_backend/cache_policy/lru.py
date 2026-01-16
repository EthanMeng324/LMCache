# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import OrderedDict
from typing import Any, Dict
import time

# First Party
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor
from lmcache.utils import CacheEngineKey
from lmcache.v1.storage_backend.cache_policy.base_policy import BaseCachePolicy, KeyType

logger = init_logger(__name__)

KeyType = CacheEngineKey


class LRUCachePolicy(BaseCachePolicy[KeyType, OrderedDict[KeyType, Any]]):
    """
    LRU cache policy.
    """

    def __init__(self):
        logger.info("Initializing LRUCachePolicy")
        self.chunk_hash_to_init_timestamp: Dict[Any, float] = {}
        self.stats_monitor = LMCStatsMonitor.GetOrCreate()
        self.max_num_chunk_hash = 12500000
        # Track access count for each key
        self.key_to_access_count: Dict[KeyType, int] = {}

    def init_mutable_mapping(self) -> OrderedDict[KeyType, Any]:
        return OrderedDict()

    def update_chunk_hash_dict(self, key: KeyType) -> None:
        curr_time = time.time()
        # HACK: doing type conversion here
        key_hash: Any = key
        if isinstance(key, CacheEngineKey):
            key_hash = key.chunk_hash

        if init_timestamp := self.chunk_hash_to_init_timestamp.get(key_hash, None):
            time_interval = curr_time - init_timestamp
            self.stats_monitor.on_chunk_reuse(time_interval)
        else:
            if len(self.chunk_hash_to_init_timestamp) >= self.max_num_chunk_hash:
                self.chunk_hash_to_init_timestamp.clear()
            self.chunk_hash_to_init_timestamp[key_hash] = curr_time

    def update_on_hit(
        self,
        key: KeyType,
        cache_dict: OrderedDict[KeyType, Any],
    ) -> None:
        self.update_chunk_hash_dict(key)
        # Increment access count
        self.key_to_access_count[key] = self.key_to_access_count.get(key, 0) + 1
        cache_dict.move_to_end(key)

    def update_on_put(
        self,
        key: KeyType,
    ) -> None:
        self.update_chunk_hash_dict(key)
        # Initialize access count for new key
        if key not in self.key_to_access_count:
            self.key_to_access_count[key] = 0

    def update_on_force_evict(
        self,
        key: KeyType,
    ) -> None:
        # Remove access count when evicted
        self.key_to_access_count.pop(key, None)

    # NOTE(Jiayi): We do best effort to get eviction candidates so the number
    # of returned keys mignt be smaller than num_candidates.
    def get_evict_candidates(
        self,
        cache_dict: OrderedDict[KeyType, Any],
        num_candidates: int = 1,
    ) -> list[KeyType]:
        evict_keys = []
        for key, cache in cache_dict.items():
            if not cache.can_evict:
                continue
            evict_keys.append(key)
            if len(evict_keys) == num_candidates:
                break

        return evict_keys

    def get_access_count(self, key: KeyType) -> int:
        """
        Get the access count for a given key.
        
        Args:
            key: The cache key
            
        Returns:
            The number of times this key has been accessed
        """
        return self.key_to_access_count.get(key, 0)
    
    def get_all_access_counts(self) -> Dict[KeyType, int]:
        """
        Get access counts for all keys.
        
        Returns:
            A dictionary mapping keys to their access counts
        """
        return self.key_to_access_count.copy()
