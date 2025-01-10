#include "arena.h"
#include <assert.h>
#include <stdbool.h>
#include <alloc.h>
#include <stdio.h>

void arena_init(){
    arena = (Arena *)mem_alloc(sizeof(Arena));
    arena->buf = (unsigned char *)DPU_MRAM_HEAP_POINTER;
    arena->buf_len = MRAM_MAX;
    arena->offset = 0;
}


bool is_power_of_two(uintptr_t x) {
  return (x & (x-1)) == 0;
}

uintptr_t align_forward(uintptr_t ptr, size_t align) {
	uintptr_t p, a, modulo;

	assert(is_power_of_two(align));

	p = ptr;

    // cast to do pointer arithematic
	a = (uintptr_t)align; 
  
	// Same as (p % a) but faster as 'a' is a power of two
	modulo = p & (a-1);

	if (modulo != 0) {
		// If 'p' address is not aligned, push the address to the
		// next value which is aligned
		p += a - modulo; // pad the address with the remaining bits
	}
	return p;
}


void * arena_alloc(size_t size){
    if(arena == NULL){
        arena_init();
    }
    
    uintptr_t curr_ptr = (uintptr_t)arena->buf + (uintptr_t)arena->offset;
    uintptr_t offset = align_forward(curr_ptr, 8);

    offset -= (uintptr_t)arena->buf;

	if (offset + size <= arena->buf_len) {
		void *ptr = &arena->buf[offset];
		arena->offset = offset + size;
		return ptr;
	}

    

    return NULL;
}