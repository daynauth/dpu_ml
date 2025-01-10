#ifndef ARENA_H
#define ARENA_H

#include <stdlib.h>
#include <mram.h>

#define MRAM_MAX (64 * 1024 * 1024) // 64 KB

typedef struct Arena Arena;
struct Arena {
	unsigned char *buf;
	size_t         buf_len;
	size_t         offset; 
};

static Arena * arena;

void * arena_alloc(size_t size);

#endif