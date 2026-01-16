# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.v1.storage_backend.cache_access_monitor import CacheAccessMonitor

logger = init_logger(__name__)


class CacheMigrationService:
    """
    Service to migrate top N most accessed KV cache items
    from one storage backend to another.
    
    This service runs in a background thread and periodically checks
    if migration is needed based on the configured interval.
    """

    def __init__(
        self,
        source_backend: StorageBackendInterface,
        target_backend: StorageBackendInterface,
        top_n: int = 10,
        migration_interval: float = 60.0,  # seconds
        copy_mode: bool = True,
    ):
        """
        Initialize the migration service.

        Args:
            source_backend: The backend to migrate from
            target_backend: The backend to migrate to
            top_n: Number of top accessed keys to migrate
            migration_interval: Minimum interval between migrations (in seconds)
            copy_mode: If True, keep the original in source_backend; if False, move it
        """
        self.source_backend = source_backend
        self.target_backend = target_backend
        self.top_n = top_n
        self.migration_interval = migration_interval
        self.copy_mode = copy_mode

        self.monitor = CacheAccessMonitor(source_backend)
        self.last_migration_time: float = 0.0
        self.lock = threading.Lock()
        
        # Background thread for periodic migration
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._migration_worker,
            name="cache-migration-worker",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            f"Cache migration service started background thread "
            f"(interval={migration_interval}s)"
        )
    
    def _migration_worker(self):
        """Background worker thread that periodically performs migration."""
        while not self._stop_event.is_set():
            try:
                # Wait for migration interval or until stop event
                if self._stop_event.wait(timeout=self.migration_interval):
                    # Stop event was set, exit
                    break
                
                # Check if migration is needed
                if self.should_migrate():
                    num_migrated = self.migrate()
                    if num_migrated is not None and num_migrated > 0:
                        logger.debug(
                            f"Background migration completed: {num_migrated} keys migrated"
                        )
            except Exception as e:
                logger.error(
                    f"Error in migration worker thread: {e}", exc_info=True
                )
                # Continue running even if there's an error
                time.sleep(1)
    
    def stop(self):
        """Stop the background migration thread."""
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)
        logger.info("Cache migration service stopped")

    def should_migrate(self) -> bool:
        """
        Check if it's time to perform a migration based on the interval.

        Returns:
            True if enough time has passed since last migration
        """
        current_time = time.time()
        return (current_time - self.last_migration_time) >= self.migration_interval

    def migrate(self) -> int:
        """
        Perform a migration cycle (synchronous, call from main program).

        This method should be called from your main program's call stack,
        for example in a periodic task or event loop.

        Returns:
            Number of keys migrated
        """
        with self.lock:
            return self._perform_migration()

    def _perform_migration(self) -> int:
        """Perform one migration cycle."""
        logger.debug("Starting migration cycle")

        # Get top N keys
        top_keys = self.monitor.get_top_keys(self.top_n)

        if not top_keys:
            logger.debug("No keys to migrate")
            return 0

        logger.info(
            f"Migrating top {len(top_keys)} keys from {self.source_backend} "
            f"to {self.target_backend}"
        )

        migrated_count = 0
        failed_count = 0

        for key, access_count in top_keys:
            try:
                # Check if key still exists in source backend
                if not self.source_backend.contains(key):
                    logger.debug(f"Key {key} no longer exists in source backend")
                    continue

                # Get the memory object from source backend
                memory_obj = self.source_backend.get_blocking(key)
                if memory_obj is None:
                    logger.warning(f"Failed to get memory object for key {key}")
                    failed_count += 1
                    continue

                # Put into target backend
                # Use batched_submit_put_task for compatibility
                if hasattr(self.target_backend, "submit_put_task"):
                    self.target_backend.submit_put_task(key, memory_obj)
                else:
                    self.target_backend.batched_submit_put_task([key], [memory_obj])

                # If not copy mode, remove from source backend
                if not self.copy_mode:
                    self.source_backend.batched_remove([key], force=False)

                migrated_count += 1
                logger.debug(
                    f"Migrated key {key} (access_count={access_count}) "
                    f"from {self.source_backend} to {self.target_backend}"
                )

            except Exception as e:
                logger.error(
                    f"Failed to migrate key {key}: {e}", exc_info=True
                )
                failed_count += 1

        logger.info(
            f"Migration cycle completed: {migrated_count} migrated, "
            f"{failed_count} failed"
        )
        
        # Update last migration time
        self.last_migration_time = time.time()
        
        return migrated_count

