#ifndef MEM_HASH_H_INCLUDED
#define MEM_HASH_H_INCLUDED

#include <stdint.h>
#include <time.h>
#include <sys/mman.h>

#define MAX_LEVELS  200           
#define OPEN_MLOCK      1              
#define CLOSE_MLOCK     0

typedef struct mem_hash {
    uint32_t bucket[MAX_LEVELS];  /* primes array */
    uint32_t bucket_levels;              
    uint32_t bucket_len;               /* max length of one bucket */
    uint32_t max_node;                 /* the number of nodes */
    size_t total_size;                
} mem_hash_t;

int mem_hash_init(mem_hash_t* mh, uint32_t bucket_levels, uint32_t bucket_len);
uint32_t mem_hash_generate_primes(uint32_t* primes, uint32_t  max, uint32_t  num);


#endif /* MEM_HASH_H_INCLUDED */ 