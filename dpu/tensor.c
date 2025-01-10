#include "tensor.h"
#include "arena.h"

#include <alloc.h>
#include <stdio.h>

Tensor * Tensor_init(size_t size){
    Tensor * self = (Tensor *)mem_alloc(sizeof(Tensor));
    float * data = (float *)arena_alloc(size * sizeof(float));

    if(data == NULL){
        printf("memory could not be written\n");
        exit(EXIT_FAILURE);
    }

    self->size = size;
    self->data = data;

    return self;
}

void Tensor_store(Tensor * self, float * buffer, size_t size){
    if(size > self->size){
        printf("Buffer size too langer to copy\n");
    }

    mram_write(buffer, (__mram_ptr void *)(self->data), (unsigned int)(size * sizeof(float)));
}

// Load data from MRAM to WRAM
void Tensor_load(Tensor * self, float *wram_buffer, size_t size) {
    if (size > self->size) {
        printf("Error: Requested size exceeds tensor size in MRAM.\n");
        return;
    }

    // Use mram_read to move data from MRAM to WRAM
    mram_read((__mram_ptr void const *)self->data, wram_buffer, size * sizeof(float));
}