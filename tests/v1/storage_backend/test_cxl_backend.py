# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for CXL Backend using cxl_shm.c C library.

Before running tests, compile the C library:
    make -f Makefile.cxl_shm
    cp libcxl_shm.so lmcache/v1/storage_backend/

Then run:
    python tests/v1/storage_backend/test_cxl_backend.py
    # or
    pytest tests/v1/storage_backend/test_cxl_backend.py -v
"""
import asyncio
import sys
from unittest.mock import Mock, patch

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

import torch

from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat, TensorMemoryObj, MemoryObjMetadata
from lmcache.v1.storage_backend.cxl_backend import CxlBackend, CXL_SHM_ONAME_LEN
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend


def create_mock_config():
    """Create a mock LMCacheEngineConfig."""
    config = Mock(spec=LMCacheEngineConfig)
    config.cache_policy = "LRU"
    config.local_cpu = True
    config.lmcache_instance_id = "test_instance"
    config.extra_config = {
        "cxl_num_procs": 1,
        "cxl_rank": 0,
        "max_cxl_size": 32.0,
    }
    return config


def create_mock_local_cpu_backend():
    """Create a mock LocalCPUBackend."""
    backend = Mock(spec=LocalCPUBackend)
    backend.allocate = Mock(return_value=None)
    backend.contains = Mock(return_value=False)
    backend.submit_put_task = Mock()
    return backend


def create_sample_key():
    """Create a sample CacheEngineKey."""
    return CacheEngineKey(
        fmt="vllm",
        model_name="test_model",
        world_size=1,
        worker_id=0,
        chunk_hash=hash("test_hash"),
        dtype=torch.bfloat16,
    )


def create_sample_memory_obj():
    """Create a sample MemoryObj."""
    shape = torch.Size([2, 4])
    dtype = torch.float32
    tensor = torch.randn(shape, dtype=dtype)
    
    metadata = MemoryObjMetadata(
        shape=shape,
        dtype=dtype,
        address=0,
        phy_size=tensor.numel() * dtype.itemsize,
        ref_count=1,
        pin_count=0,
        fmt=MemoryFormat.KV_T2D,
    )
    
    return TensorMemoryObj(
        raw_data=tensor,
        metadata=metadata,
        parent_allocator=None,
    )


if HAS_PYTEST:
    @pytest.fixture
    def mock_config():
        return create_mock_config()

    @pytest.fixture
    def mock_local_cpu_backend():
        return create_mock_local_cpu_backend()

    @pytest.fixture
    def sample_key():
        return create_sample_key()

    @pytest.fixture
    def sample_memory_obj():
        return create_sample_memory_obj()


class TestCxlBackendKeyHandling:
    """Test key handling and length limits."""
    
    def test_get_cxl_key_short(self, mock_config=None, mock_local_cpu_backend=None):
        """Test _get_cxl_key with short key."""
        if mock_config is None:
            mock_config = create_mock_config()
        if mock_local_cpu_backend is None:
            mock_local_cpu_backend = create_mock_local_cpu_backend()
        
        with patch('lmcache.v1.storage_backend.cxl_backend.CxlShmWrapper') as mock_cxl_shm:
            mock_wrapper = Mock()
            mock_wrapper.init.return_value = 0
            mock_cxl_shm.return_value = mock_wrapper
            
            loop = asyncio.new_event_loop()
            backend = CxlBackend(
                config=mock_config,
                loop=loop,
                local_cpu_backend=mock_local_cpu_backend,
            )
            
            key = CacheEngineKey(
                fmt="vllm",
                model_name="test",
                world_size=1,
                worker_id=0,
                chunk_hash=hash("short"),
                dtype=torch.bfloat16,
            )
            
            try:
                cxl_key = backend._get_cxl_key(key)
                assert len(cxl_key) <= CXL_SHM_ONAME_LEN
                assert cxl_key == key.to_string() or cxl_key.startswith("H")
            finally:
                backend.close()
                loop.close()
    
    def test_get_cxl_key_long(self, mock_config=None, mock_local_cpu_backend=None):
        """Test _get_cxl_key with long key (uses hash)."""
        if mock_config is None:
            mock_config = create_mock_config()
        if mock_local_cpu_backend is None:
            mock_local_cpu_backend = create_mock_local_cpu_backend()
        
        with patch('lmcache.v1.storage_backend.cxl_backend.CxlShmWrapper') as mock_cxl_shm:
            mock_wrapper = Mock()
            mock_wrapper.init.return_value = 0
            mock_cxl_shm.return_value = mock_wrapper
            
            loop = asyncio.new_event_loop()
            backend = CxlBackend(
                config=mock_config,
                loop=loop,
                local_cpu_backend=mock_local_cpu_backend,
            )
            
            # Create a key that will produce a long string
            key = CacheEngineKey(
                fmt="vllm",
                model_name="very_long_model_name_that_exceeds_limit",
                world_size=1,
                worker_id=0,
                chunk_hash=hash("very_long_chunk_hash_that_exceeds_limit"),
                dtype=torch.bfloat16,
            )
            
            try:
                key_str = key.to_string()
                if len(key_str) > CXL_SHM_ONAME_LEN:
                    cxl_key = backend._get_cxl_key(key)
                    assert len(cxl_key) <= CXL_SHM_ONAME_LEN
                    assert cxl_key.startswith("H")
                    # Should be consistent
                    assert backend._get_cxl_key(key) == cxl_key
            finally:
                backend.close()
                loop.close()
    
    def test_get_cxl_key_caching(self, mock_config=None, mock_local_cpu_backend=None):
        """Test that _get_cxl_key caches the mapping."""
        if mock_config is None:
            mock_config = create_mock_config()
        if mock_local_cpu_backend is None:
            mock_local_cpu_backend = create_mock_local_cpu_backend()
        
        with patch('lmcache.v1.storage_backend.cxl_backend.CxlShmWrapper') as mock_cxl_shm:
            mock_wrapper = Mock()
            mock_wrapper.init.return_value = 0
            mock_cxl_shm.return_value = mock_wrapper
            
            loop = asyncio.new_event_loop()
            backend = CxlBackend(
                config=mock_config,
                loop=loop,
                local_cpu_backend=mock_local_cpu_backend,
            )
            
            key = CacheEngineKey(
                fmt="vllm",
                model_name="test",
                world_size=1,
                worker_id=0,
                chunk_hash=hash("hash"),
                dtype=torch.bfloat16,
            )
            
            try:
                cxl_key1 = backend._get_cxl_key(key)
                cxl_key2 = backend._get_cxl_key(key)
                
                assert cxl_key1 == cxl_key2
                assert key in backend.key_to_cxl_key
            finally:
                backend.close()
                loop.close()


class TestCxlBackendBasicOperations:
    """Test basic backend operations."""
    
    def test_initialization(self, mock_config=None, mock_local_cpu_backend=None):
        """Test backend initialization."""
        if mock_config is None:
            mock_config = create_mock_config()
        if mock_local_cpu_backend is None:
            mock_local_cpu_backend = create_mock_local_cpu_backend()
        
        with patch('lmcache.v1.storage_backend.cxl_backend.CxlShmWrapper') as mock_cxl_shm:
            mock_wrapper = Mock()
            mock_wrapper.init.return_value = 0
            mock_cxl_shm.return_value = mock_wrapper
            
            loop = asyncio.new_event_loop()
            backend = CxlBackend(
                config=mock_config,
                loop=loop,
                local_cpu_backend=mock_local_cpu_backend,
            )
            
            try:
                assert backend.cxl_shm == mock_wrapper
                assert backend.max_cache_size == 32 * 1024**3  # 32GB
                assert len(backend.key_handles) == 0
                assert len(backend.key_to_cxl_key) == 0
            finally:
                backend.close()
                loop.close()
    
    def test_initialization_failure(self, mock_config=None, mock_local_cpu_backend=None):
        """Test backend initialization failure."""
        if mock_config is None:
            mock_config = create_mock_config()
        if mock_local_cpu_backend is None:
            mock_local_cpu_backend = create_mock_local_cpu_backend()
        
        with patch('lmcache.v1.storage_backend.cxl_backend.CxlShmWrapper') as mock_cxl_shm:
            mock_wrapper = Mock()
            mock_wrapper.init.return_value = -1
            mock_cxl_shm.return_value = mock_wrapper
            
            loop = asyncio.new_event_loop()
            try:
                CxlBackend(
                    config=mock_config,
                    loop=loop,
                    local_cpu_backend=mock_local_cpu_backend,
                )
                assert False, "Should have raised RuntimeError"
            except RuntimeError as e:
                assert "Failed to initialize CXL shared memory" in str(e)
            finally:
                loop.close()
    
    def test_pin_unpin(self, mock_config=None, mock_local_cpu_backend=None, sample_key=None):
        """Test pin and unpin operations."""
        if mock_config is None:
            mock_config = create_mock_config()
        if mock_local_cpu_backend is None:
            mock_local_cpu_backend = create_mock_local_cpu_backend()
        if sample_key is None:
            sample_key = create_sample_key()
        
        with patch('lmcache.v1.storage_backend.cxl_backend.CxlShmWrapper') as mock_cxl_shm:
            mock_wrapper = Mock()
            mock_wrapper.init.return_value = 0
            mock_cxl_shm.return_value = mock_wrapper
            
            loop = asyncio.new_event_loop()
            backend = CxlBackend(
                config=mock_config,
                loop=loop,
                local_cpu_backend=mock_local_cpu_backend,
            )
            
            try:
                # Pin non-existent key should return False
                assert backend.pin(sample_key) == False
                
                # Add key to dict
                from lmcache.utils import DiskCacheMetadata
                backend.dict[sample_key] = DiskCacheMetadata(
                    path="test_path",
                    size=100,
                    shape=torch.Size([2, 4]),
                    dtype=torch.float32,
                    fmt=MemoryFormat.KV_T2D,
                )
                
                # Pin existing key should return True
                assert backend.pin(sample_key) == True
                assert backend.dict[sample_key].is_pinned == True
                
                # Unpin should return True
                assert backend.unpin(sample_key) == True
                assert backend.dict[sample_key].is_pinned == False
            finally:
                backend.close()
                loop.close()


if __name__ == "__main__":
    if HAS_PYTEST:
        pytest.main([__file__, "-v"])
    else:
        # Run tests directly without pytest
        print("Running CXL Backend tests (without pytest)...")
        print("=" * 60)
        
        passed = 0
        failed = 0
        
        # Test key handling
        print("\n1. Testing key handling...")
        test = TestCxlBackendKeyHandling()
        
        try:
            test.test_get_cxl_key_short()
            print("   ✓ test_get_cxl_key_short passed")
            passed += 1
        except Exception as e:
            print(f"   ✗ test_get_cxl_key_short failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        
        try:
            test.test_get_cxl_key_long()
            print("   ✓ test_get_cxl_key_long passed")
            passed += 1
        except Exception as e:
            print(f"   ✗ test_get_cxl_key_long failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        
        try:
            test.test_get_cxl_key_caching()
            print("   ✓ test_get_cxl_key_caching passed")
            passed += 1
        except Exception as e:
            print(f"   ✗ test_get_cxl_key_caching failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        
        # Test basic operations
        print("\n2. Testing basic operations...")
        test = TestCxlBackendBasicOperations()
        
        try:
            test.test_initialization()
            print("   ✓ test_initialization passed")
            passed += 1
        except Exception as e:
            print(f"   ✗ test_initialization failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        
        try:
            test.test_initialization_failure()
            print("   ✓ test_initialization_failure passed")
            passed += 1
        except Exception as e:
            print(f"   ✗ test_initialization_failure failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        
        try:
            test.test_pin_unpin()
            print("   ✓ test_pin_unpin passed")
            passed += 1
        except Exception as e:
            print(f"   ✗ test_pin_unpin failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        
        print("\n" + "=" * 60)
        print(f"Tests completed: {passed} passed, {failed} failed")
        sys.exit(0 if failed == 0 else 1)
