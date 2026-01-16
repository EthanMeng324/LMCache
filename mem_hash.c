#define _GNU_SOURCE
#include "mem_hash.h"
#include "cxl_shm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <math.h>

static int mem_hash_node_init(mem_hash_t* mh);

static int mem_hash_bucket_init(mem_hash_t* mh,
                               uint32_t bucket_levels,
                               uint32_t bucket_len);
static int mem_hash_size_init(mem_hash_t* mh);
static int print_mf_info(mem_hash_t *mh);
static int mem_hash_is_prime(uint32_t value);

int mem_hash_init(mem_hash_t* mh, uint32_t bucket_levels, uint32_t bucket_len) {
    int ret = 0;
    ret = mem_hash_bucket_init(mh, bucket_levels, bucket_len);
    if (ret != 0) {
        fprintf(stderr, "mem_hash_bucket_init failed\n");
        return -1;
    }
    ret = mem_hash_node_init(mh);
    if (ret != 0) {
        fprintf(stderr, "mem_hash_node_init failed\n");
        return -1;
    }
    ret = mem_hash_size_init(mh);
    if (ret != 0) {
        fprintf(stderr, "mem_hash_size_init failed\n");
        return -1;
    }
    return 0;
}


static int mem_hash_size_init(mem_hash_t* mh) {
    if (!mh) return -1;
    mh->total_size = sizeof(cxl_shm_head_t) * 1  + sizeof(cxl_shm_obj_meta_t) * mh->max_node;
    return 0;
}


static int mem_hash_node_init(mem_hash_t* mh) {
    if (!mh) return -1;
    
    int node_count = 0;
    for (uint32_t i = 0; i < mh->bucket_levels; i++) {
        node_count += mh->bucket[i];
    }
    mh->max_node = node_count;
    return 0;
}

static int mem_hash_bucket_init(mem_hash_t* mh,
                               uint32_t bucket_levels,
                               uint32_t bucket_len) {
    if (!mh) return - 1;
    
    if (bucket_levels > MAX_LEVELS) {
        fprintf(stderr, "bucket_levels[%u] > MAX_LEVELS[%d]\n", bucket_levels, MAX_LEVELS);
        return -1;
    }

    mh->bucket_levels = bucket_levels;
    mh->bucket_len = bucket_len;
    if ((mh->bucket_levels == 10) && (mh->bucket_len == 200000)) {
        mh->bucket[0] = 199999;
        mh->bucket[1] = 199967;
        mh->bucket[2] = 199961;
        mh->bucket[3] = 199933;
        mh->bucket[4] = 199931;
        mh->bucket[5] = 199921;
        mh->bucket[6] = 199909;
        mh->bucket[7] = 199889;
        mh->bucket[8] = 199877;
        mh->bucket[9] = 199873;
    } else if (mem_hash_generate_primes(mh->bucket, bucket_len, bucket_levels) != bucket_levels) {
        fprintf(stderr, "GeneratePrimes < bucket_levels[%u]\n", bucket_levels);
        return -1;
    }
    // print_mf_info(mh);
    return 0;
}

static int print_mf_info(mem_hash_t *mh) {
    printf("mem_hash info: bucket_levels: %d, bucket_len: %d\n", mh->bucket_levels, mh->bucket_len);
    for (int i = 0; i < mh->bucket_levels; i++) {
        printf("mem_hash info: buckets[%d] = %d\n", i, mh->bucket[i]);
    }
    return 0;
}


uint32_t mem_hash_generate_primes(uint32_t* primes,
			    uint32_t  max,
		   	    uint32_t  num)
{
	uint32_t i = 0, j = 0;
	for (i = max; i > 1; i--) {
		if (mem_hash_is_prime(i)) {
			primes[j] = i;
			j++;
		}
		if (j == num) return j;
	}
	return j;
}

static int mem_hash_is_prime(uint32_t value)
{
	uint32_t square = (uint32_t)sqrt(value);
	uint32_t i;

	for(i = 2; i <= square; i++) {
		if(value % i == 0)
			return 0;
	}

	return 1;
}