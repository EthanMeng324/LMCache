#define _POSIX_C_SOURCE 200809L
#include "cxl_shm.h"
#include "mem_hash.h"
#define CXL_SHM_DAX_SIZE (1UL << 35) // 32GB
#define CXL_SHM_DAX_PATH_DEFAULT "/dev/dax0.0"
#define CACHELINE_SIZE 64

static cxl_shm_metadata_t *meta = NULL;
static mem_hash_t *mh = NULL;
static void *dax_addr = NULL;
static pid_t my_pid = 0;
static int my_rank = 0;
static int num_ranks = 0;
static cxl_shm_lock_t *cxl_shm_lock = NULL;
static char dax_path[256] = {0}; // Store DAX device path

// Spinlock functions
static void acquire_lock(atomic_flag *lock) {
    while (atomic_flag_test_and_set(lock)) {
        // Busy wait
    }
}
static uint64_t str2hash(const char *str);
static int find_meta_hash_idx(const char *name);
static int find_avail_hash_idx(const char *name);

#define LOCK_TIMEOUT_NS 6000000000  // 6 second timeout

static inline uint64_t align_to_cacheline(uint64_t addr) {
    return (addr + CACHELINE_SIZE - 1) & ~(CACHELINE_SIZE - 1);
}

static void release_lock(atomic_flag *lock) {
    // struct timeval tv;
    // gettimeofday(&tv, NULL);
    // printf("release_lock pid [%d] Timestamp: %ld.%06ld\n", getpid(), tv.tv_sec, tv.tv_usec);
    atomic_flag_clear(lock);
    clflush_region_with_mfence(lock, sizeof (atomic_flag));
}

static int acquire_lock_with_timeout(atomic_flag *lock) {
    struct timespec start, current;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    clflush_region_with_mfence(lock, sizeof (atomic_flag));
    while (atomic_flag_test_and_set(lock)) {
        clock_gettime(CLOCK_MONOTONIC, &current);
        if ((current.tv_sec - start.tv_sec) * 1000000000 + 
            (current.tv_nsec - start.tv_nsec) > LOCK_TIMEOUT_NS) {
            // release_lock(lock);
            return -1;
        }
        clflush_region_with_mfence(lock, sizeof (atomic_flag));
    }
    clflush_region_with_mfence(lock, sizeof (atomic_flag));
    return 0;
}

// Export functions for Python binding
__attribute__((visibility("default"))) int clflush_region_with_mfence(void *addr, size_t size) {
    uintptr_t p = (uintptr_t)addr & ~(CACHELINE_SIZE - 1);
    uintptr_t end = ((uintptr_t)addr + size + CACHELINE_SIZE - 1) & ~(CACHELINE_SIZE - 1);
    do {
        _mm_clflushopt((void*)p);
        p += CACHELINE_SIZE;
    } while (p < end);
    _mm_mfence();
    return 0;
}

void clflush_region_with_sfence(void *addr, size_t size) {
    uintptr_t p = (uintptr_t)addr & ~(CACHELINE_SIZE - 1);
    uintptr_t end = ((uintptr_t)addr + size + CACHELINE_SIZE - 1) & ~(CACHELINE_SIZE - 1);
    do {
        _mm_clflushopt((void*)p);
        p += CACHELINE_SIZE;
    } while (p < end);
    _mm_sfence();
}

void clwb_region_with_barrier(void *addr, size_t size) {
    uintptr_t p = (uintptr_t)addr & ~(CACHELINE_SIZE - 1);
    uintptr_t end = ((uintptr_t)addr + size + CACHELINE_SIZE - 1) & ~(CACHELINE_SIZE - 1);
    do {
        _mm_clwb((void*)p);
        p += CACHELINE_SIZE;
    } while (p < end);
    _mm_mfence();
}

void clflush_region(void *addr, size_t size) {
    uintptr_t p = (uintptr_t)addr & ~(CACHELINE_SIZE - 1);
    uintptr_t end = ((uintptr_t)addr + size + CACHELINE_SIZE - 1) & ~(CACHELINE_SIZE - 1);
    do {
        _mm_clflushopt((void*)p);
        p += CACHELINE_SIZE;
    } while (p < end);
}

void clwb_region(void *addr, size_t size) {
    uintptr_t p = (uintptr_t)addr & ~(CACHELINE_SIZE - 1);
    uintptr_t end = ((uintptr_t)addr + size + CACHELINE_SIZE - 1) & ~(CACHELINE_SIZE - 1);
    do {
        _mm_clwb((void*)p);
        p += CACHELINE_SIZE;
    } while (p < end);
}

void debug_print_shm_head() {
    const char *env_dax_path = getenv("LMCACHE_CXL_DAX_DEVICE");
    const char *dax_device_path = env_dax_path ? env_dax_path : CXL_SHM_DAX_PATH_DEFAULT;
    printf("Trying to open file: %s\n", dax_device_path);
    
    int fd = open(dax_device_path, O_RDONLY);
    if (fd < 0) {
        perror("Failed to open shm file for debug");
        return;
    }
    printf("File opened successfully, fd=%d\n", fd);

    void *debug_addr = mmap(NULL, 64, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    
    if (debug_addr == MAP_FAILED) {
        perror("Failed to mmap for debug");
        return;
    }
    printf("Memory mapped successfully at %p\n", debug_addr);

    unsigned char *buffer = (unsigned char *)debug_addr;
    printf("First 64 bytes of shm file:\n");
    
    for (size_t row = 0; row < 4; row++) {
        printf("\n%04zx: ", row * 16);
        for (size_t col = 0; col < 16; col++) {
            size_t i = row * 16 + col;
            printf("%02x ", buffer[i]);
            fflush(stdout);
        }
        printf(" (row %zu complete)\n", row);
        fflush(stdout);
    }
    printf("\nDebug print complete\n");
    fflush(stdout);

    if (munmap(debug_addr, 64) != 0) {
        perror("Failed to munmap debug memory");
    }
    printf("Debug function finished\n");
    fflush(stdout);
}

 
__attribute__((visibility("default"))) int cxl_shm_finalize() {
    clflush_region_with_mfence(&meta->head.initialized, sizeof(meta->head.initialized));
    if (my_rank ==0) {
        if (atomic_load(&meta->head.initialized)) {
            // if (cxl_acquire_lock(&meta->head.lock, my_pid)) {
            //     fprintf(stderr, "cxl_shm_finalize failed to acquire the lock\n");
            //     return -1;
            // }
            // clflush_region_with_mfence(&meta->head.initialized, sizeof(meta->head.initialized));
            // if (atomic_load(&meta->head.initialized)) {
                // memset((char*)dax_addr + sizeof(cxl_lock_t), 0, CXL_SHM_DAX_SIZE - sizeof(cxl_lock_t));
                memset((char*)dax_addr, 0, CXL_SHM_DAX_SIZE);
                // clflush_region_with_mfence((char*)dax_addr + sizeof(cxl_lock_t), CXL_SHM_DAX_SIZE - sizeof(cxl_lock_t));
                atomic_store(&meta->head.initialized, 0);
                clflush_region_with_mfence(&meta->head.initialized, sizeof(meta->head.initialized));
                fprintf(stderr, "cxl shm finalize done\n");
            // }
            // cxl_release_lock(&meta->head.lock, my_pid);
        }
    }
    while (atomic_load(&meta->head.initialized)) {
        clflush_region_with_mfence(&meta->head.initialized, sizeof(meta->head.initialized));
    }

    clflush_region_with_mfence((char*)dax_addr, CXL_SHM_DAX_SIZE);

    return 0;
}

// Initialize the shared memory management layer
__attribute__((visibility("default"))) int cxl_shm_init(int num_procs, int rank) {
    my_pid = getpid();
    num_ranks = num_procs;
    my_rank = rank;
    
    // Get DAX device path from environment variable or use default
    const char *env_dax_path = getenv("LMCACHE_CXL_DAX_DEVICE");
    const char *dax_device_path = env_dax_path ? env_dax_path : CXL_SHM_DAX_PATH_DEFAULT;
    
    // Store the path for later use
    strncpy(dax_path, dax_device_path, sizeof(dax_path) - 1);
    dax_path[sizeof(dax_path) - 1] = '\0';
    
    int fd = open(dax_device_path, O_RDWR);
    if (fd < 0) {
        fprintf(stderr, "Failed to open DAX device: %s\n", dax_device_path);
        perror("open");
        return -1;
    }
 
    // Memory-map the entire DAX device
    void *addr = mmap(NULL, CXL_SHM_DAX_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) {
        fprintf(stderr, "Failed to mmap DAX device\n");
        close(fd);
        return -1;
    }
 
    close(fd); // File descriptor no longer needed after mmap
 
    mh = malloc(sizeof(mem_hash_t));
    if (!mh) {
        fprintf(stderr, "Failed to allocate memory for mem_hash_t\n");
        return -1;
    }
    if (mem_hash_init(mh, 10, 200000)) {
        fprintf(stderr, "Failed to initialize mem_hash_t\n");
        return -1;
    }

    meta = (cxl_shm_metadata_t *)addr;
    dax_addr = addr;
    cxl_shm_lock = (cxl_shm_lock_t*)malloc(sizeof(cxl_shm_lock_t));
    cxl_shm_lock->victim = NULL;
    cxl_shm_lock->level = NULL;

    clflush_region_with_mfence(&meta->head.initialized, sizeof(meta->head.initialized));
    if (my_rank == 0) {
        if (!atomic_load(&meta->head.initialized)) {
            // if (cxl_acquire_lock(&meta->head.lock, my_pid)) {
            //     fprintf(stderr, "cxl_shm_init new failed to acquire the lock\n");
            //     return -1;
            // }
            // fprintf(stderr, "cxl_shm_init acquire the lock\n");
            // clflush_region_with_mfence(&meta->head.initialized, sizeof(meta->head.initialized));
            // if (!atomic_load(&meta->head.initialized)) {
                // Too slow for 128G shared memory, so comment it
                // memset((char*)addr + sizeof(cxl_lock_t), 0, CXL_SHM_DAX_SIZE - sizeof(cxl_lock_t));
                // clflush_region_with_mfence((char*)addr + sizeof(cxl_lock_t), CXL_SHM_DAX_SIZE - sizeof(cxl_lock_t));
                meta->head.curr_offset = mh->total_size + 2 * sizeof(uint64_t) * num_procs;
                clflush_region_with_mfence(&meta->head.curr_offset, sizeof(meta->head.curr_offset));
                atomic_store(&meta->head.initialized, 1);
                clflush_region_with_mfence(&meta->head.initialized, sizeof(meta->head.initialized));
                fprintf(stderr, "cxl shm init done\n");
            // } 
            // fprintf(stderr, "cxl_shm_init release the lock\n");
            // cxl_release_lock(&meta->head.lock, my_pid);
        }
    }
    while (!atomic_load(&meta->head.initialized)) {
        clflush_region_with_mfence(&meta->head.initialized, sizeof(meta->head.initialized));
    }

    clflush_region_with_mfence(&meta->head, sizeof(cxl_shm_head_t));
    // Too slow for 128G shared memory, so comment it
    // clflush_region_with_mfence((char*)addr, CXL_SHM_DAX_SIZE);
    cxl_shm_lock->level = (uint64_t *)((char *)dax_addr + mh->total_size + sizeof(uint64_t) * num_procs);
    cxl_shm_lock->victim = (uint64_t *)((char *)dax_addr + mh->total_size + 2 * sizeof(uint64_t) * num_procs);

    return 0;
}
 
// Create a shared memory object
__attribute__((visibility("default"))) int cxl_shm_create(const char *name, size_t size, cxl_shm_hnd_t *hnd) {
    if (!name || !hnd) return -1;

    // struct timeval tv;
    // gettimeofday(&tv, NULL);
    // printf("cxl_shm_create pid [%d] Timestamp: %ld.%06ld\n", getpid(), tv.tv_sec, tv.tv_usec);

    // TODO: optimize avoid using global locks
    // if (cxl_acquire_lock(&meta->head.lock, my_pid)) {
    if (cxl_acquire_smart_lock()) {
        fprintf(stderr, "cxl_shm_create failed to acquire the lock for '%s'\n", name);
        return -1;
    }
 
    clflush_region_with_mfence(&meta->objs, sizeof(meta->objs));
    int idx = find_meta_hash_idx(name);
    if (idx != -1) {
        fprintf(stderr, "Object with name '%s' already exists\n", name);
        // cxl_release_lock(&meta->head.lock, my_pid);
        cxl_release_smart_lock();
        return -1;
    }
 
    // Find a free slot
    int free_idx = find_avail_hash_idx(name);
    if (free_idx == -1) {
        fprintf(stderr, "No free slots available for shared objects\n");
        // cxl_release_lock(&meta->head.lock, my_pid);
        cxl_release_smart_lock();
        return -1;
    }
 
    // Simple allocation: find the end of allocated space
    clflush_region_with_mfence(&meta->head, sizeof(meta->head));
    uint64_t current_offset = meta->head.curr_offset;
    current_offset = align_to_cacheline(current_offset);
 
    // Check if enough space is available
    size_t dax_size = CXL_SHM_DAX_SIZE; 
    if (current_offset + size > dax_size) {
        fprintf(stderr, "Not enough space in DAX device, device size %zu, data size: %zu, current_offset: %lu, meta->head.curr_offset: %u \n", dax_size, size, current_offset, meta->head.curr_offset);
        // cxl_release_lock(&meta->head.lock, my_pid);
        cxl_release_smart_lock();
        abort();
        return -1;
    }
 
    // Initialize the shared object metadata
    cxl_shm_obj_meta_t *obj = &meta->objs[free_idx];
    strncpy(obj->name, name, CXL_SHM_ONAME_LEN - 1);
    obj->name[CXL_SHM_ONAME_LEN - 1] = '\0';
    obj->offset = current_offset;
    obj->size = size;
    obj->in_use = 1;
    clflush_region_with_mfence(obj, sizeof(cxl_shm_obj_meta_t));
 
    // Prepare the handle
    hnd->obj = obj;
    hnd->mapped_addr = (char *)dax_addr + obj->offset;

    meta->head.curr_offset = current_offset + size;
    clflush_region_with_mfence(&meta->head, sizeof(meta->head));
 
    // cxl_release_lock(&meta->head.lock, my_pid);
    cxl_release_smart_lock();
 
    return 0;
}
 
// Open an existing shared memory object
__attribute__((visibility("default"))) int cxl_shm_open_obj(const char *name, cxl_shm_hnd_t *hnd) {
    if (!name || !hnd) return -1;
    // if (cxl_acquire_lock(&meta->head.lock, my_pid)) {
    if (cxl_acquire_smart_lock()) {
        fprintf(stderr, "cxl_shm_open_obj failed to acquire the lock\n");
        return -1;
    }

    clflush_region_with_mfence(&meta->objs, sizeof(meta->objs));
    int idx = find_meta_hash_idx(name);
    if (idx == -1) {
        fprintf(stderr, "Shared object '%s' not found\n", name);
        // cxl_release_lock(&meta->head.lock, my_pid);
        cxl_release_smart_lock();
        return -1;
    }
    cxl_shm_obj_meta_t *found = NULL;
    found = &meta->objs[idx];
    hnd->obj = found;
    hnd->mapped_addr = (char *)dax_addr + found->offset;
    clflush_region_with_mfence(hnd->mapped_addr, found->size);
    // cxl_release_lock(&meta->head.lock, my_pid);
    cxl_release_smart_lock();
 
    return 0;
}
 
// Close a shared memory object handle
__attribute__((visibility("default"))) int cxl_shm_close(cxl_shm_hnd_t *hnd) {
    if (!hnd) return -1;
 
    // TODO: release hander pointer?
    // In this simple implementation, nothing is needed
    // In a more complex setup, reference counts or other cleanup might be necessary
    hnd->obj = NULL;
    hnd->mapped_addr = NULL;
 
    return 0;
}
 
// // Destroy a shared memory object
// int cxl_shm_destroy_from_name(const char *name) {
//     if (!name) return -1;

//     cxl_acquire_lock(&meta->lock, my_pid);
 
//     int found_idx = find_meta_hash_idx(name);
//     if (found_idx == -1) {
//         fprintf(stderr, "Shared object '%s' not found\n", name);
//         cxl_release_lock(&meta->lock, my_pid);
//         return -1;
//     }
 
//     // Clear the object
//     void* mapped_addr = (char *)dax_addr + meta->objects[found_idx].offset;
//     memset(mapped_addr, 0, meta->objects[found_idx].size);

//     // Clear the metadata entry
//     meta->objects[found_idx].in_use = 0;
//     memset(&meta->objects[found_idx], 0, sizeof(cxl_shm_obj_entry_t));
 
//     cxl_release_lock(&meta->lock, my_pid);
 
//     return 0;
// }

// Destroy a shared memory object
__attribute__((visibility("default"))) int cxl_shm_destroy_from_hnd(cxl_shm_hnd_t *hnd) {
    if (!hnd || !hnd->obj || !hnd->mapped_addr) return -1;
    // if (cxl_acquire_lock(&meta->head.lock, my_pid)) {
    if (cxl_acquire_smart_lock()) {
        fprintf(stderr, "cxl_shm_destroy_from_hnd failed to acquire the lock\n");
        return -1;
    }
 
    // Clear the object
    void* mapped_addr = hnd->mapped_addr;
    memset(mapped_addr, 0, hnd->obj->size);
    clflush_region_with_mfence(mapped_addr, hnd->obj->size);

    // Clear the metadata
    hnd->obj->in_use = 0;
    memset(hnd->obj, 0, sizeof(cxl_shm_obj_meta_t));
    clflush_region_with_mfence(hnd->obj,  sizeof(cxl_shm_obj_meta_t));


    // close handle
    cxl_shm_close(hnd);
    // cxl_release_lock(&meta->head.lock, my_pid);
    cxl_release_smart_lock();
 
    return 0;
}

int find_meta_arr_idx(const char *name) {
   for (int i = 0; i < CXL_SHM_MAX_OBJS; ++i) {
       if (meta->objs[i].in_use && strncmp(meta->objs[i].name, name, CXL_SHM_ONAME_LEN) == 0) {
           return i;
       }
   }
   return -1;
}

static int find_meta_hash_idx(const char *name) {
    int idx = -1;
    uint32_t base_pos = 0;
    uint64_t key = str2hash(name);
	for (uint32_t i = 0; i < mh->bucket_levels; i++) {
		if (i > 0) base_pos += mh->bucket[i-1];
		idx =  base_pos + (key % mh->bucket[i]);
        if (meta->objs[idx].in_use && strncmp(meta->objs[idx].name, name, CXL_SHM_ONAME_LEN) == 0) {
           return idx;
        }
	}
    return -1;
}

static int find_avail_hash_idx(const char *name) {
    int idx = -1;
    uint32_t base_pos = 0;
    uint64_t key = str2hash(name);
	for (uint32_t i = 0; i < mh->bucket_levels; i++) {
		if (i > 0) base_pos += mh->bucket[i-1];
		idx =  base_pos + (key % mh->bucket[i]);
        if (!meta->objs[idx].in_use) {
           return idx;
        }
	}
    return -1;
}

static inline void flush_cxl_lock(cxl_lock_t *lock)
{
    clflush_region_with_mfence(lock, sizeof(cxl_lock_t));
}

int cxl_acquire_smart_lock() {
    // struct timespec start, current;
    // clock_gettime(CLOCK_MONOTONIC, &start);
    // int print_only = 0;

    for (int k = 1; k < num_ranks; k++) {  // k = level from 1 to num_ranks-1

        cxl_nt_store_uint64(&cxl_shm_lock->level[my_rank], k); // Announce intent to enter at level k
        cxl_nt_store_uint64(&cxl_shm_lock->victim[k], my_rank);// Set self as victim for level k
        // Wait until no other process is at a level >= k and my_rank is not the victim anymore
        int conflict = 0;
        do {
            conflict = 0;
            for (int j = 0; j < num_ranks; j++) {
                if (j == my_rank) continue;
                if ((cxl_nt_load_uint64(&cxl_shm_lock->level[j]) >= k) && (cxl_nt_load_uint64(&cxl_shm_lock->victim[k]) == my_rank)) {
                    conflict = 1;
                    // clock_gettime(CLOCK_MONOTONIC, &current);
                    // if ((current.tv_sec - start.tv_sec) * 1000000000 + 
                    //     (current.tv_nsec - start.tv_nsec) > LOCK_TIMEOUT_NS) {
                    //         if (!print_only) {
                    //             fprintf(stderr, "I %d am still waiting rank %d occupied the lock!!!!\n", j, my_rank);
                    //             print_only = 1;
                    //         }
                    // }

                    break;
                }
            }


        } while (conflict);
    }
    return 0;
}

int cxl_release_smart_lock() {
    cxl_nt_store_uint64(&cxl_shm_lock->level[my_rank], 0);  // Reset level to indicate exit from the critical section
    return 0;
}

int cxl_acquire_lock(cxl_lock_t *lock, int my_id)
{
    struct timespec start, current;
    clock_gettime(CLOCK_MONOTONIC, &start);
    while (1) {
        flush_cxl_lock(lock);

        // 1. Read the current lock state
        uint64_t old_seq    = lock->seq;
        // int old_owner  = lock->owner_id;  // Unused variable
        uint8_t  old_locked = lock->locked;

        // 2. If it's locked, someone else might own it => spin or yield
        if (old_locked == 1) {
            // Could do an exponential backoff or short sleep here
            clock_gettime(CLOCK_MONOTONIC, &current);
            if ((current.tv_sec - start.tv_sec) * 1000000000 + 
                (current.tv_nsec - start.tv_nsec) > LOCK_TIMEOUT_NS) {
                return -1;
            }
            continue;
        }

        // 3. Try to claim the lock:
        uint64_t new_seq = old_seq + 1;
        lock->owner_id = my_id;
        lock->seq      = new_seq;
        lock->locked   = 1;

        flush_cxl_lock(lock);
        int check_owner  = lock->owner_id;
        uint64_t check_seq    = lock->seq;
        uint8_t  check_locked = lock->locked;

        if (check_owner == my_id && check_seq == new_seq && check_locked == 1) {
            // return 0;

            // double check
            flush_cxl_lock(lock);
            check_owner  = lock->owner_id;
            check_seq    = lock->seq;
            check_locked = lock->locked;
            if (check_owner == my_id && check_seq == new_seq && check_locked == 1) { 
                return 0;
            }
        } 
            // // A mismatch suggests a collision or stale read, keep trying
            // fprintf(stderr,
            //         "[%llu] Collision or mismatch acquiring lock! "
            //         "Expected (owner=%llu, seq=%llu, locked=1), "
            //         "got (owner=%llu, seq=%llu, locked=%u)\n",
            //         (unsigned long long)my_id,
            //         (unsigned long long)my_id,
            //         (unsigned long long)new_seq,
            //         (unsigned long long)check_owner,
            //         (unsigned long long)check_seq,
            //         check_locked);
            continue;
    }
    return -1;
}

int cxl_release_lock(cxl_lock_t *lock, int my_id)
{
    flush_cxl_lock(lock);
    lock->owner_id = 0;
    lock->locked   = 0;
    flush_cxl_lock(lock);

    // uint64_t current_owner = lock->owner_id;
    // uint8_t  current_locked = lock->locked;

    // if (current_locked == 1 && current_owner == my_id) {
    //     lock->owner_id = 0;
    //     lock->locked   = 0;
    //     flush_cxl_lock(lock);
    // } else {
    //     fprintf(stderr,
    //             "[%llu] Attempt to release lock but not owner. "
    //             "(owner_id=%llu, locked=%u)\n",
    //             (unsigned long long)my_id,
    //             (unsigned long long)current_owner,
    //             current_locked);
    //     return -1;
    // }
    return 0;
}

// static void* cxl_mm_stream_load_si64(const void *addr) {
//     __m128i loaded = _mm_stream_load_si128((const __m128i*)addr);
//     return (void*)_mm_cvtsi128_si64(loaded);
// }

void* cxl_nt_load_ptr(const void *addr) {
    if  (((uintptr_t)addr % 8) != 0) {
        fprintf(stderr, "Address must be 8-byte aligned");
        assert(0);
    }
    _mm_clflush(addr);
    _mm_mfence(); 
    void* value = (void *)(*(volatile intptr_t *)addr);
    _mm_mfence();
    return value;
}

void cxl_nt_store_ptr(void *addr, const void *value) {
    if  (((uintptr_t)addr % 8) != 0) {
        fprintf(stderr, "Address must be 8-byte aligned");
        assert(0);
    }
    _mm_mfence();  
    _mm_stream_si64(addr, (intptr_t)value);
    _mm_mfence();
}

uint64_t cxl_nt_load_uint64(const void *addr) {
    if (((uintptr_t)addr % 8) != 0) {
        fprintf(stderr, "Address must be 8-byte aligned\n");
        assert(0);
    }

    _mm_clflush((void *)addr);                // flush cache line
    _mm_mfence();                             // memory fence
    // uint64_t value = __atomic_load_n((const uint64_t *)addr, __ATOMIC_ACQUIRE);
    uint64_t value = (uint64_t)(*(volatile uint64_t *)addr);
    return value;
}

void cxl_nt_store_uint64(void *addr, uint64_t value) {
    if (((uintptr_t)addr % 8) != 0) {
        fprintf(stderr, "Address must be 8-byte aligned\n");
        assert(0);
    }

    // __atomic_store_n((uint64_t *)addr, value, __ATOMIC_RELEASE);
    _mm_stream_si64(addr, value);
    _mm_clflush(addr);                        // flush after write
    _mm_mfence();                             // ensure visibility
}

// // Non temporal store of a pointer value by packing it into a 128 bit vector.
// void cxl_nt_store_ptr(void *addr, const void *value) {
//     assert(((uintptr_t)addr % 16) == 0 && "Address must be 16-byte aligned");
//     _mm_sfence();  // Ensure previous operations complete
//     // Pack the pointer value into a __m128i.
//     __m128i to_store = _mm_set_epi64x(0, (uintptr_t)value);
//     _mm_stream_si128((__m128i*)addr, to_store);
//     _mm_sfence();  // Ensure the store is complete
// }

// void* cxl_nt_load_ptr(const void *addr) {
//     assert(((uintptr_t)addr % 16) == 0 && "Address must be 16-byte aligned");
//     _mm_sfence();
//     void* value = (void *)_mm_stream_load_si128((__m128i *)addr);
//     _mm_sfence();
//     return value;
// }
    
// void cxl_nt_store_ptr(void *addr, const void *value) {
//     assert(((uintptr_t)addr % 16) == 0 && "Address must be 16-byte aligned");
//     _mm_sfence(); 
//     _mm_stream_si128((__m128i *)addr, (intptr_t)value);
//     _mm_sfence();
// }

void* cxl_nt_cas_ptr(void *addr, const void *old, const void *new) {
    assert(((uintptr_t)addr % 8) == 0 && "Address must be 8-byte aligned");
    _mm_mfence(); 
    void* old_val = cxl_nt_load_ptr(addr);
    if (old_val == old) {
        _mm_stream_si64(addr, (intptr_t)new);
    }
    _mm_mfence(); 
    return old_val;
}

void* cxl_nt_swap_ptr(void *addr, const void *new) {
    assert(((uintptr_t)addr % 8) == 0 && "Address must be 8-byte aligned");
    _mm_mfence(); 
    void* old_val = cxl_nt_load_ptr(addr);
    _mm_stream_si64(addr, (intptr_t)new);
    _mm_mfence(); 
    return old_val;
}

// Function to convert a string to a hash key using djb2 algorithm
/*
 The djb2 hash function was created by Daniel J. Bernstein, a renowned computer scientist. Through empirical testing, Bernstein found that multiplying by 33 provided a good balance between speed and distribution quality for a wide variety of input strings. 
 */
static uint64_t str2hash(const char *str) {
    uint64_t hash = 5381; // Initialize hash to a large prime number
    uint64_t c;
    while ((c = *str++)) {
        // hash * 33 + c
        hash = ((hash << 5) + hash) + c;
    }
    return hash;
}