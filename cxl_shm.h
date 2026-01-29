/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */
#ifndef CXL_SHM_H_INCLUDED
#define CXL_SHM_H_INCLUDED

#include <fcntl.h>
#include <stdint.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <immintrin.h>
#include <x86intrin.h>
#include <assert.h>


#ifdef MPL_HAVE_SYS_MMAN_H
#include <sys/mman.h>
#endif

#define CXL_SHM_MAX_OBJS (1 << 21) // Avg 1KB per object
#define CXL_SHM_ONAME_LEN 20 

typedef uint64_t cxl_shm_obj_offset_t;
typedef uint64_t cxl_shm_obj_size_t;

// sizeof(cxl_lock_t) should be 16 bytes.
typedef struct {
    volatile int owner_id;
    volatile uint64_t seq;
    volatile uint32_t  locked;
} cxl_lock_t;

// Metadata structure for each shared object
typedef struct {
    char name[CXL_SHM_ONAME_LEN];
    cxl_shm_obj_offset_t offset; // Offset within the DAX device
    // Allocated size (bytes) in the DAX mapping. This may be a padded/fixed size.
    cxl_shm_obj_size_t size;
    // Logical payload size (bytes) written by the caller (<= size).
    cxl_shm_obj_size_t actual_size;
    uint8_t in_use;  // Flag indicating if the entry is used
} cxl_shm_obj_meta_t;
 
typedef struct {
    // Simple spinlock for synchronization
    cxl_lock_t lock;
    volatile _Atomic int initialized;
    // Allocation cursor within the DAX mapping.
    // Must be 64-bit; otherwise it overflows at 4GB and causes data overlap.
    uint64_t curr_offset;
} cxl_shm_head_t;

// Metadata region structure
typedef struct {
    cxl_shm_head_t head;
    cxl_shm_obj_meta_t objs[CXL_SHM_MAX_OBJS];
} cxl_shm_metadata_t;

// Handle to a shared memory object
typedef struct {
    cxl_shm_obj_meta_t *obj;
    void *mapped_addr;
} cxl_shm_hnd_t;

typedef struct {
    uint64_t* level;       // Shared array: level[i] is the current level of process i (0 means not interested)
    uint64_t* victim;      // Shared array: victim[k] is the process designated at level k
} cxl_shm_lock_t;

int cxl_shm_init(int num_procs, int rank); 
int cxl_shm_finalize();
// Create a shared memory object
// `size` is allocated bytes; `actual_size` is logical payload bytes (<= size).
int cxl_shm_create(const char *name, size_t size, size_t actual_size, cxl_shm_hnd_t *hnd);
// Open an existing shared memory object
int cxl_shm_open_obj(const char *name, cxl_shm_hnd_t *hnd);
// Close a shared memory object handle
int cxl_shm_close(cxl_shm_hnd_t *hnd);
// Destroy a shared memory object
// int cxl_shm_destroy_from_name(const char *name);
int cxl_shm_destroy_from_hnd(cxl_shm_hnd_t *hnd);

// Debug utilities (best-effort; intended for testing/diagnosis)
// Reset only metadata slots and allocator cursor (does NOT zero the whole DAX).
// WARNING: This makes existing objects unreachable by name and may cause data overlap
// if you continue allocating without a full device reset. Use only for debugging.
int cxl_shm_reset_metadata();
// Count how many slots are currently in_use.
int cxl_shm_debug_count_in_use(uint64_t *reachable_in_use,
                               uint64_t *reachable_max,
                               uint64_t *total_in_use,
                               uint64_t *total_max,
                               uint64_t *curr_offset,
                               uint32_t *bucket_levels);
// For a given name, report the candidate slot indices (one per bucket level)
// and whether each slot is in_use.
int cxl_shm_debug_candidate_slots(const char *name,
                                 uint32_t *out_idxs,
                                 uint8_t *out_in_use,
                                 uint32_t max_out);
int cxl_release_lock(cxl_lock_t *lock, int my_id);
int cxl_acquire_lock(cxl_lock_t *lock, int my_id);
int cxl_acquire_smart_lock();
int cxl_release_smart_lock();

int clflush_region_with_mfence(void *addr, size_t size);
void clflush_region_with_sfence(void *addr, size_t size);
void clwb_region_with_barrier(void *addr, size_t size);
void clflush_region(void *addr, size_t size);
void clwb_region(void *addr, size_t size);
void* cxl_nt_load_ptr(const void *addr);
void cxl_nt_store_ptr(void *addr, const void *value);
void* cxl_nt_cas_ptr(void *addr, const void *old, const void *new);
void* cxl_nt_swap_ptr(void *addr, const void *new);
uint64_t cxl_nt_load_uint64(const void *addr);
void cxl_nt_store_uint64(void *addr, uint64_t value);
#endif /* CXL_SHM_H_INCLUDED */