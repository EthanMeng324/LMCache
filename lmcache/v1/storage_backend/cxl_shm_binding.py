# SPDX-License-Identifier: Apache-2.0
"""
Python bindings for cxl_shm.c C library.
Uses ctypes to call C functions directly.
"""
import ctypes
import os
from typing import Optional, Tuple
from ctypes import Structure, POINTER, c_int, c_char_p, c_size_t, c_void_p, c_uint64, c_uint8, c_uint32, c_char

from lmcache.logging import init_logger

logger = init_logger(__name__)

# Try to load the shared library
_lib_paths = [
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "libcxl_shm.so"),
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "libcxl_shm.so"),
    "libcxl_shm.so",
    "./libcxl_shm.so",
]

_lib = None
for path in _lib_paths:
    try:
        _lib = ctypes.CDLL(path)
        logger.info(f"Loaded cxl_shm library from {path}")
        break
    except OSError:
        continue

if _lib is None:
    raise RuntimeError(
        "Failed to load libcxl_shm.so. "
        "Please compile cxl_shm.c to a shared library first:\n"
        "  gcc -shared -fPIC -o libcxl_shm.so cxl_shm.c -lrt"
    )


# Define C structures based on cxl_shm.h
# Constants from cxl_shm.h
CXL_SHM_ONAME_LEN = 20
CXL_SHM_MAX_OBJS = 1 << 21  # 2097152

class CxlShmObjMeta(Structure):
    """cxl_shm_obj_meta_t structure"""
    _fields_ = [
        ("name", c_char * CXL_SHM_ONAME_LEN),  # CXL_SHM_ONAME_LEN = 20
        ("offset", c_uint64),  # cxl_shm_obj_offset_t
        ("size", c_uint64),    # cxl_shm_obj_size_t
        ("in_use", ctypes.c_uint8),  # uint8_t (changed from int)
    ]
    
    def __repr__(self):
        name_str = self.name.decode('utf-8', errors='ignore').rstrip('\x00')
        return f"CxlShmObjMeta(name='{name_str}', offset={self.offset}, size={self.size}, in_use={self.in_use})"


class CxlShmHnd(Structure):
    """cxl_shm_hnd_t structure"""
    _fields_ = [
        ("obj", POINTER(CxlShmObjMeta)),
        ("mapped_addr", c_void_p),
    ]
    
    @property
    def obj_contents(self):
        """Get the contents of obj pointer."""
        if self.obj:
            return self.obj.contents
        return None
    
    def __repr__(self):
        return f"CxlShmHnd(obj={self.obj_contents}, mapped_addr={hex(self.mapped_addr) if self.mapped_addr else None})"


# Define function signatures
_lib.cxl_shm_init.argtypes = [c_int, c_int]
_lib.cxl_shm_init.restype = c_int

_lib.cxl_shm_finalize.argtypes = []
_lib.cxl_shm_finalize.restype = c_int

_lib.cxl_shm_create.argtypes = [c_char_p, c_size_t, POINTER(CxlShmHnd)]
_lib.cxl_shm_create.restype = c_int

_lib.cxl_shm_open_obj.argtypes = [c_char_p, POINTER(CxlShmHnd)]
_lib.cxl_shm_open_obj.restype = c_int

_lib.cxl_shm_close.argtypes = [POINTER(CxlShmHnd)]
_lib.cxl_shm_close.restype = c_int

_lib.cxl_shm_destroy_from_hnd.argtypes = [POINTER(CxlShmHnd)]
_lib.cxl_shm_destroy_from_hnd.restype = c_int

_lib.clflush_region_with_mfence.argtypes = [c_void_p, c_size_t]
_lib.clflush_region_with_mfence.restype = c_int


class CxlShmWrapper:
    """Wrapper class for cxl_shm C functions"""
    
    def __init__(self, num_procs: int = 1, rank: int = 0):
        """
        Initialize CXL shared memory.
        
        Args:
            num_procs: Number of processes
            rank: Process rank (0-based)
        """
        self.num_procs = num_procs
        self.rank = rank
        self.initialized = False
        
    def init(self) -> int:
        """Initialize CXL shared memory."""
        result = _lib.cxl_shm_init(self.num_procs, self.rank)
        if result == 0:
            self.initialized = True
        return result
    
    def finalize(self) -> int:
        """Finalize CXL shared memory."""
        if self.initialized:
            result = _lib.cxl_shm_finalize()
            self.initialized = False
            return result
        return 0
    
    def create(self, name: str, size: int) -> Tuple[int, Optional[CxlShmHnd]]:
        """
        Create a shared memory object.
        
        Args:
            name: Object name
            size: Object size in bytes
            
        Returns:
            (return_code, handle) where return_code is 0 on success, -1 on failure
        """
        hnd = CxlShmHnd()
        name_bytes = name.encode('utf-8')
        result = _lib.cxl_shm_create(name_bytes, size, ctypes.byref(hnd))
        if result == 0:
            return (0, hnd)
        return (result, None)
    
    def open_obj(self, name: str) -> Tuple[int, Optional[CxlShmHnd]]:
        """
        Open an existing shared memory object.
        
        Args:
            name: Object name
            
        Returns:
            (return_code, handle) where return_code is 0 on success, -1 on failure
        """
        hnd = CxlShmHnd()
        name_bytes = name.encode('utf-8')
        result = _lib.cxl_shm_open_obj(name_bytes, ctypes.byref(hnd))
        if result == 0:
            return (0, hnd)
        return (result, None)
    
    def close(self, hnd: CxlShmHnd) -> int:
        """
        Close a shared memory object handle.
        
        Args:
            hnd: Handle to close
            
        Returns:
            0 on success, -1 on failure
        """
        return _lib.cxl_shm_close(ctypes.byref(hnd))
    
    def destroy(self, hnd: CxlShmHnd) -> int:
        """
        Destroy a shared memory object.
        
        Args:
            hnd: Handle to destroy
            
        Returns:
            0 on success, -1 on failure
        """
        return _lib.cxl_shm_destroy_from_hnd(ctypes.byref(hnd))
    
    def flush(self, addr: c_void_p, size: int) -> int:
        """
        Flush a memory region with memory fence.
        
        Args:
            addr: Memory address
            size: Size in bytes
            
        Returns:
            0 on success
        """
        return _lib.clflush_region_with_mfence(addr, size)

