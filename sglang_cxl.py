import hashlib
import logging
import mmap
import os
from typing import Any, List, Optional

import numpy as np
import torch

# DAX devices typically require 2MB alignment
DAX_ALIGNMENT = 2 * 1024 * 1024  # 2MB

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
)

logger = logging.getLogger(__name__)

# Default DAX device path
DEFAULT_DAX_DEVICE = "/dev/dax1.0"

# Use a simple directory-like structure on DAX device
# Each key maps to a fixed-size slot based on its hash
# Slot size: 8MB (enough for most KV cache pages)
SLOT_SIZE = 8 * 1024 * 1024  # 8MB per slot
# MAX_SLOTS will be calculated dynamically based on device size
# Default to a reasonable value, but will be adjusted
DEFAULT_MAX_SLOTS = 100000  # Default maximum number of slots
METADATA_SIZE = 16  # [magic:4B][version:4B][slot_size:4B][max_slots:4B]
SLOT_METADATA_SIZE = 12  # [key_len:4B][value_size:8B] per slot


class CXLMemoryPool(HiCacheStorage):
    """
    CXL Memory Pool backend for HiCache storage using DAX device.
    Uses /dev/dax1.0 as the storage device, similar to HiCacheFile but on DAX device.
    Each key is stored in a fixed-size slot based on its hash.
    """

    def __init__(
        self,
        storage_config: HiCacheStorageConfig,
        dax_device: str = DEFAULT_DAX_DEVICE,
    ):
        """Initialize CXL Memory Pool storage backend.

        Args:
            storage_config: Storage configuration
            dax_device: Path to DAX device (default: /dev/dax1.0)
        """
        # Allow override via environment variable
        self.dax_device = os.getenv("SGLANG_CXL_DAX_DEVICE", dax_device)

        if not os.path.exists(self.dax_device):
            raise FileNotFoundError(
                f"DAX device {self.dax_device} not found. "
                f"Please ensure the device exists and is accessible."
            )

        # Get device size
        # For DAX devices (character devices), os.stat().st_size typically returns 0
        # We need to get size from environment variable or ndctl
        try:
            # First, try to get size from environment variable (most reliable)
            env_size = os.getenv("SGLANG_CXL_DAX_DEVICE_SIZE", 38654705664)
            if env_size:
                self.device_size = int(env_size)
            else:
                # Try os.stat (may return 0 for character devices)
                stat_info = os.stat(self.dax_device)
                self.device_size = stat_info.st_size
                
                # If st_size is 0, try to get size from ndctl
                if self.device_size == 0:
                    import subprocess
                    import re
                    try:
                        # Get device name from path (e.g., /dev/dax1.0 -> dax1.0)
                        device_name = os.path.basename(self.dax_device)
                        # Run ndctl list -X to get device-dax info
                        result = subprocess.run(
                            ["ndctl", "list", "-X"],
                            capture_output=True,
                            text=True,
                            timeout=5,
                        )
                        if result.returncode == 0:
                            # Parse output to find device size
                            # Look for lines containing the device name and size
                            lines = result.stdout.split("\n")
                            for i, line in enumerate(lines):
                                if device_name in line:
                                    # Look for size in nearby lines
                                    for j in range(max(0, i - 5), min(len(lines), i + 5)):
                                        size_match = re.search(r'size[:\s]+(\d+)', lines[j], re.IGNORECASE)
                                        if size_match:
                                            self.device_size = int(size_match.group(1))
                                            break
                                    if self.device_size > 0:
                                        break
                    except (subprocess.TimeoutExpired, subprocess.SubprocessError, ValueError, AttributeError):
                        # If ndctl fails, we need user to specify size
                        pass
                
                # If still 0, we cannot proceed without size
                if self.device_size == 0:
                    raise ValueError(
                        f"Cannot determine device size for {self.dax_device}. "
                        f"DAX devices (character devices) don't report size via os.stat(). "
                        f"Please set SGLANG_CXL_DAX_DEVICE_SIZE environment variable. "
                        f"For example: export SGLANG_CXL_DAX_DEVICE_SIZE=8589934592  # 8GB"
                    )
        except ValueError:
            # Re-raise ValueError as-is
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to get device size: {e}") from e

        # Calculate MAX_SLOTS based on available device size
        # Account for DAX alignment requirements
        # First, calculate what size we can actually map (after alignment)
        # Align device size down to DAX_ALIGNMENT boundary
        aligned_device_size = (self.device_size // DAX_ALIGNMENT) * DAX_ALIGNMENT
        
        # Reserve some space for metadata and ensure we have at least a few slots
        available_for_slots = aligned_device_size - METADATA_SIZE
        slot_overhead = SLOT_METADATA_SIZE + SLOT_SIZE
        self.max_slots = max(1, available_for_slots // slot_overhead)
        
        # Use a reasonable maximum to avoid excessive memory usage
        # Cap at DEFAULT_MAX_SLOTS to prevent extremely large allocations
        self.max_slots = min(self.max_slots, DEFAULT_MAX_SLOTS)
        
        # Verify we have at least minimal space
        required_size = METADATA_SIZE + self.max_slots * slot_overhead
        if aligned_device_size < required_size:
            raise ValueError(
                f"Aligned device size {aligned_device_size} is too small. "
                f"Minimum required: {required_size} (for {self.max_slots} slots)"
            )
        
        logger.info(
            f"Configured {self.max_slots} slots for device size {self.device_size} bytes "
            f"({self.device_size / (1024**3):.2f} GB)"
        )

        # Open and mmap the DAX device
        # For character devices, we must specify the size explicitly
        # DAX devices require proper alignment (typically 2MB)
        aligned_size = None
        map_size = None
        try:
            self.dax_fd = os.open(self.dax_device, os.O_RDWR)
            # Calculate the total size we need to map
            total_size = METADATA_SIZE + self.max_slots * (SLOT_METADATA_SIZE + SLOT_SIZE)
            # Ensure we don't exceed device size
            map_size = min(total_size, self.device_size)
            
            # Align map_size to DAX_ALIGNMENT boundary (round up)
            # This is required for DAX devices
            aligned_size = ((map_size + DAX_ALIGNMENT - 1) // DAX_ALIGNMENT) * DAX_ALIGNMENT
            # But don't exceed device size
            aligned_size = min(aligned_size, self.device_size)
            
            # Ensure aligned_size is at least total_size to fit all slots
            # If not, we need to reduce max_slots (but we already calculated it above)
            if aligned_size < total_size:
                # Recalculate max_slots based on actual aligned_size
                available_for_slots = aligned_size - METADATA_SIZE
                slot_overhead = SLOT_METADATA_SIZE + SLOT_SIZE
                actual_max_slots = available_for_slots // slot_overhead
                if actual_max_slots < self.max_slots:
                    logger.warning(f"Reducing max_slots from {self.max_slots} to {actual_max_slots} "
                                 f"due to alignment constraints (aligned_size={aligned_size} < total_size={total_size})")
                    self.max_slots = actual_max_slots
            
            # Store the actual mapped size for bounds checking
            self.mmap_size = aligned_size
            # Calculate maximum slot index that fits in the mapped region
            max_slot_idx = (aligned_size - METADATA_SIZE) // (SLOT_METADATA_SIZE + SLOT_SIZE) - 1
            if max_slot_idx < self.max_slots - 1:
                logger.warning(f"Maximum slot index is {max_slot_idx}, but max_slots is {self.max_slots}. "
                             f"Reducing max_slots to {max_slot_idx + 1}")
                self.max_slots = max_slot_idx + 1
            logger.info(f"Mapped {aligned_size} bytes (requested {total_size}, device size {self.device_size}, max_slots={self.max_slots})")
            
            # For DAX devices, use simple mmap call with aligned size
            # The access parameter should be sufficient
            # Note: offset parameter defaults to 0, which is already aligned
            # For DAX devices, mmap offset must be aligned to DAX_ALIGNMENT (2MB)
            self.mmap_obj = mmap.mmap(
                self.dax_fd, aligned_size, access=mmap.ACCESS_WRITE, offset=0
            )
        except PermissionError as e:
            if hasattr(self, "dax_fd"):
                try:
                    os.close(self.dax_fd)
                except Exception:
                    pass
            raise RuntimeError(
                f"Permission denied when opening DAX device {self.dax_device}. "
                f"Please ensure you have read/write permissions. "
                f"You may need to run with sudo or add your user to the appropriate group. "
                f"Original error: {e}"
            ) from e
        except Exception as e:
            if hasattr(self, "dax_fd"):
                try:
                    os.close(self.dax_fd)
                except Exception:
                    pass
            # Build error message with available information
            error_msg = f"Failed to mmap DAX device: {e}. "
            if aligned_size is not None and map_size is not None:
                error_msg += f"Tried to map {aligned_size} bytes (aligned from {map_size}). "
            error_msg += f"Device size: {self.device_size} bytes."
            raise RuntimeError(error_msg) from e

        # Initialize suffix based on storage config (same as HiCacheFile)
        tp_rank, tp_size, model_name, is_mla_model = (
            storage_config.tp_rank,
            storage_config.tp_size,
            storage_config.model_name,
            storage_config.is_mla_model,
        )
        model_name = "-".join(model_name.split("/")) if model_name else ""
        if is_mla_model:
            self.config_suffix = f"_{model_name}"
        else:
            self.config_suffix = f"_{model_name}_{tp_rank}_{tp_size}"

        logger.info(
            f"CXL Memory Pool initialized: device={self.dax_device}, "
            f"size={self.device_size}"
        )

    def _get_suffixed_key(self, key: str) -> str:
        """Add suffix to key based on config (same as HiCacheFile)."""
        return key + self.config_suffix

    def _get_slot_offset(self, key: str) -> int:
        """Get slot offset for a key based on its hash."""
        # Hash the key to get a slot index
        key_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
        slot_idx = int(key_hash[:8], 16) % self.max_slots
        
        # Ensure slot_idx is within bounds (should be, but double-check)
        if slot_idx >= self.max_slots:
            slot_idx = slot_idx % self.max_slots
        
        # Calculate offset: metadata + slot_idx * (slot_metadata + slot_size)
        slot_offset = (
            METADATA_SIZE
            + slot_idx * (SLOT_METADATA_SIZE + SLOT_SIZE)
            + SLOT_METADATA_SIZE  # Skip slot metadata, point to data area
        )
        return slot_offset, slot_idx

    def _get_slot_metadata_offset(self, slot_idx: int) -> int:
        """Get metadata offset for a slot."""
        return METADATA_SIZE + slot_idx * (SLOT_METADATA_SIZE + SLOT_SIZE)

    def _read_slot_metadata(self, slot_idx: int) -> tuple:
        """Read slot metadata: (key_len, value_size)."""
        offset = self._get_slot_metadata_offset(slot_idx)
        key_len = int.from_bytes(self.mmap_obj[offset : offset + 4], "little")
        value_size = int.from_bytes(self.mmap_obj[offset + 4 : offset + 12], "little")
        return key_len, value_size

    def _write_slot_metadata(self, slot_idx: int, key_len: int, value_size: int):
        """Write slot metadata."""
        offset = self._get_slot_metadata_offset(slot_idx)
        self.mmap_obj[offset : offset + 4] = key_len.to_bytes(4, "little")
        self.mmap_obj[offset + 4 : offset + 12] = value_size.to_bytes(8, "little")

    def _get_slot_key(self, slot_idx: int) -> Optional[str]:
        """Get the key stored in a slot (for verification)."""
        key_len, _ = self._read_slot_metadata(slot_idx)
        if key_len == 0:
            return None
        # Key is stored at the beginning of slot data area
        key_offset = METADATA_SIZE + slot_idx * (SLOT_METADATA_SIZE + SLOT_SIZE) + SLOT_METADATA_SIZE
        if key_len > SLOT_SIZE:
            return None
        key_bytes = self.mmap_obj[key_offset : key_offset + key_len]
        return key_bytes.decode("utf-8", errors="ignore")

    def get(
        self,
        key: str,
        target_location: Optional[torch.Tensor] = None,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        """Get value for a key (same interface as HiCacheFile)."""
        if target_location is None:
            return None
        
        key = self._get_suffixed_key(key)
        slot_offset, slot_idx = self._get_slot_offset(key)
        
        # Read metadata
        key_len, value_size = self._read_slot_metadata(slot_idx)
        if key_len == 0 or value_size == 0:
            return None
        
        # Verify key matches (simple check: compare stored key)
        stored_key = self._get_slot_key(slot_idx)
        if stored_key != key:
            return None
        
        expected_size = target_location.numel() * target_location.element_size()
        if value_size != expected_size:
            logger.warning(
                f"Size mismatch for key {key}: expected {expected_size}, got {value_size}"
            )
            return None

        # Read data from slot (skip key, read value)
        # Use the same approach as HiCacheFile: memoryview for direct memory access
        try:
            # Key is stored first, then value
            key_len_bytes = len(key.encode("utf-8"))
            data_offset = slot_offset + key_len_bytes
            
            # Ensure target is contiguous and get memoryview (exactly like HiCacheFile)
            target_contiguous = target_location.contiguous()
            target_buf = memoryview(target_contiguous.view(torch.uint8).numpy())
            
            # Read from mmap - mmap slice returns bytes
            mmap_bytes = bytes(self.mmap_obj[data_offset : data_offset + value_size])
            
            # Create a temporary numpy array from mmap bytes
            mmap_np = np.frombuffer(mmap_bytes, dtype=np.uint8)
            
            # Get target as numpy array and flatten
            target_np = np.frombuffer(target_buf, dtype=np.uint8)
            target_flat = target_np.reshape(-1)
            
            # Copy data using copyto
            np.copyto(target_flat[:value_size], mmap_np)
            
            return target_location
        except Exception as e:
            logger.warning(f"Failed to fetch {key} from CXL Memory Pool: {e}")
            return None

    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[List[torch.Tensor]] = None,
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None]:
        """Batch get values (same interface as HiCacheFile)."""
        if not target_locations:
            return [None] * len(keys)
        
        return [
            self.get(key, target_location)
            for key, target_location in zip(keys, target_locations)
        ]

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        """Set value for a key (same behavior as HiCacheFile: skip if exists)."""
        if value is None:
            return False
        
        # Check if key already exists (same as HiCacheFile)
        if self.exists(key):
            logger.debug(f"Key {key} already exists. Skipped.")
            return True

        key = self._get_suffixed_key(key)
        slot_offset, slot_idx = self._get_slot_offset(key)
        
        value_size = value.numel() * value.element_size()
        if value_size > SLOT_SIZE:
            logger.error(f"Value size {value_size} exceeds slot size {SLOT_SIZE}")
            return False

        try:
            # Write metadata
            self._write_slot_metadata(slot_idx, len(key), value_size)
            
            # Write key at the beginning of slot (for verification)
            key_bytes = key.encode("utf-8")
            self.mmap_obj[slot_offset : slot_offset + len(key_bytes)] = key_bytes
            
            # Write value data
            # Use the same approach as HiCacheFile: convert to uint8 view and get bytes
            value_contiguous = value.contiguous()
            value_np = value_contiguous.view(torch.uint8).numpy()
            # Flatten to 1D to ensure consistent byte order (same as get method)
            value_flat = value_np.ravel()
            data_offset = slot_offset + len(key_bytes)
            
            # Check bounds to ensure we don't exceed mmap size
            mmap_size = len(self.mmap_obj)
            if data_offset + value_size > mmap_size:
                logger.error(f"Write offset {data_offset} + size {value_size} exceeds mmap size {mmap_size}. "
                           f"slot_idx={slot_idx}, slot_offset={slot_offset}, key_len={len(key_bytes)}, max_slots={self.max_slots}")
                return False
            
            # Use tobytes() to get bytes representation (same as HiCacheFile's tofile)
            # Get bytes from the flattened array
            value_bytes = value_flat[:value_size].tobytes()
            
            # Write to mmap - use slice assignment
            # For DAX devices, we use mmap slice assignment directly
            # Note: DAX devices require mmap offset to be aligned, but once mmap is created,
            # accessing via mmap object slices doesn't require alignment
            try:
                self.mmap_obj[data_offset : data_offset + value_size] = value_bytes
            except (OSError, ValueError) as e:
                # If direct slice assignment fails, try writing in smaller chunks
                # This can help with some DAX device configurations
                chunk_size = 4096  # 4KB chunks
                try:
                    for i in range(0, value_size, chunk_size):
                        chunk_end = min(i + chunk_size, value_size)
                        self.mmap_obj[data_offset + i : data_offset + chunk_end] = value_bytes[i:chunk_end]
                except (OSError, ValueError) as e2:
                    # If chunked write also fails, log the error
                    logger.error(f"Failed to write to DAX device at offset {data_offset}: {e2}. "
                               f"Original error: {e}. data_offset={data_offset}, value_size={value_size}, "
                               f"mmap_size={mmap_size}, slot_idx={slot_idx}")
                    raise e from e2
            
            # Flush to ensure persistence
            # Flush the entire slot region (from slot metadata start to end of slot)
            # Note: For DAX devices, writes are already persistent, so flush may fail but that's okay
            slot_metadata_offset = self._get_slot_metadata_offset(slot_idx)
            try:
                self.mmap_obj.flush(slot_metadata_offset, SLOT_METADATA_SIZE + SLOT_SIZE)
            except OSError:
                # Flush failure is acceptable for DAX devices as writes are already persistent
                pass
            return True
        except Exception as e:
            logger.error(f"Failed to save tensor {key}: {e}")
            return False

    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        """Batch set values (same interface as HiCacheFile)."""
        if not values:
            return False
        
        for key, value in zip(keys, values):
            if not self.set(key, value):
                return False
        return True

    def exists(self, key: str) -> bool:
        """Check if key exists (same interface as HiCacheFile)."""
        key = self._get_suffixed_key(key)
        slot_offset, slot_idx = self._get_slot_offset(key)
        
        # Read metadata
        key_len, value_size = self._read_slot_metadata(slot_idx)
        if key_len == 0 or value_size == 0:
            return False
        
        # Verify key matches
        stored_key = self._get_slot_key(slot_idx)
        return stored_key == key

    def clear(self) -> None:
        """Clear all entries (similar to HiCacheFile)."""
        # Zero out all slot metadata
        for slot_idx in range(self.max_slots):
            self._write_slot_metadata(slot_idx, 0, 0)
        
        # Note: For DAX devices, writes are already persistent, so flush may not be needed
        # If flush is needed, it should be done per-slot like in set() method
        # But for now, skip flush to avoid errors on some DAX device configurations
        logger.info("Cleared all entries in CXL Memory Pool")

    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, "mmap_obj") and self.mmap_obj:
            try:
                self.mmap_obj.flush()
                self.mmap_obj.close()
            except Exception:
                pass
        if hasattr(self, "dax_fd"):
            try:
                os.close(self.dax_fd)
            except Exception:
                pass