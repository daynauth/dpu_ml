#ifndef TENSOR_H
#define TENSOR_H

#include <dpu.h>

typedef struct {
    double * data;
    int nd;
    int64_t n_elems;
    int64_t * dims;
} Tensor;

Tensor * Tensor_init(int nd, int64_t * dims);
Tensor * Tensor_randn(int nd, int64_t * dims);
Tensor * Tensor_zeros(int nd, int64_t * dims);
Tensor * Tensor_append(Tensor * a, Tensor * b);
Tensor * Tensor_matmul(Tensor * a, Tensor * b);
#endif