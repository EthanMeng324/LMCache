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
    cxl_shm_obj_size_t size;   // Size of the memory object
    uint8_t in_use;  // Flag indicating if the entry is used
} cxl_shm_obj_meta_t;
 
typedef struct {
    // Simple spinlock for synchronization
    cxl_lock_t lock;
    volatile _Atomic int initialized;
    uint32_t curr_offset;
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
int cxl_shm_create(const char *name, size_t size, cxl_shm_hnd_t *hnd);
// Open an existing shared memory object
int cxl_shm_open_obj(const char *name, cxl_shm_hnd_t *hnd);
// Close a shared memory object handle
int cxl_shm_close(cxl_shm_hnd_t *hnd);
// Destroy a shared memory object
// int cxl_shm_destroy_from_name(const char *name);
int cxl_shm_destroy_from_hnd(cxl_shm_hnd_t *hnd);
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