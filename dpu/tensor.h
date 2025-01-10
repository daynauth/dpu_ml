#ifndef TENSOR_H
#define TENSOR_H

#include<stdlib.h>

typedef struct {
    float * data;
    size_t size;
} Tensor;

Tensor * Tensor_init(size_t size);
void Tensor_store(Tensor * self, float * buffer, size_t size);
void Tensor_load(Tensor * self, float *wram_buffer, size_t size);
#endif