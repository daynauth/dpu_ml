#include "Tensor.h"

#include <time.h>


Tensor * Tensor_init(int nd, int64_t * dims){
    Tensor * tensor = (Tensor *)malloc(sizeof(Tensor));

    tensor->n_elems = 1;
    for(int64_t i = 0; i < nd; i++){
        tensor->n_elems *= dims[i];
    }

    tensor->data = (double *)malloc(sizeof(double) * tensor->n_elems);
    tensor->dims = (int64_t *)malloc(sizeof(int64_t) * nd);

    for(int i = 0; i < nd; i++){
        tensor->dims[i] = dims[i];
    }

    return tensor;
}

Tensor * Tensor_randn(int nd, int64_t * dims){
    Tensor * tensor = Tensor_init(nd, dims);

    for(int64_t i = 0; i < tensor->n_elems; i++){
        tensor->data[i] = (double)rand() / (double)RAND_MAX;
    }

    return tensor;
}