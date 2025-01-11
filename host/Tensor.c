#include "Tensor.h"

#include <time.h>

void matrix_multiply(double *A, double *B, double *C, int rowsA, int colsA, int colsB) {
    // Initialize the result matrix C to zero
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            C[i * colsB + j] = 0.0;
        }
    }

    // Perform the matrix multiplication
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            for (int k = 0; k < colsA; k++) {
                C[i * colsB + j] += A[i * colsA + k] * B[k * colsB + j];
            }
        }
    }
}

Tensor * Tensor_init(int nd, int64_t * dims){
    Tensor * tensor = (Tensor *)malloc(sizeof(Tensor));
    tensor->nd = nd;
    
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

Tensor * Tensor_zeros(int nd, int64_t * dims){
    Tensor * tensor = Tensor_init(nd, dims);

    for(int64_t i = 0; i < tensor->n_elems; i++){
        tensor->data[i] = 0.0;
    }

    return tensor;
}

Tensor * Tensor_append(Tensor * a, Tensor * b){
    int64_t axis = 1;


    if(a->nd != b->nd){
        printf( "Dimension mismatch");
        exit(EXIT_FAILURE);
    }

    for(int64_t i = 0; i < a->nd; i++){
        if(i == axis){
            continue;
        }

        if(a->dims[i] != b->dims[i]){
            printf("Shape Mismatch\n");
            exit(EXIT_FAILURE);
        }
    }

    int64_t * new_shape = (int64_t *)malloc(sizeof(int64_t) * a->nd);

    for(int i = 0; i < a->nd; i++){
        if(i == axis){
            new_shape[i] = a->dims[i] + b->dims[i];
        }
        else{
            new_shape[i] = a->dims[i];
        }
    }

    Tensor * tensor = Tensor_init(a->nd, new_shape);

    for(int64_t i = 0; i < a->n_elems; i++){
        tensor->data[i] = a->data[i];
    }

    for(int64_t i = 0; i < b->n_elems; i++){
        tensor->data[i + a->n_elems] = b->data[i];
    }

    free(new_shape);

    return tensor;
}

Tensor * Tensor_matmul(Tensor * a, Tensor * b){
    if(a->nd != 2){
        printf("Incorrect shape for matrix multiplication\n");
        exit(EXIT_FAILURE);
    }

    if(a->nd != b->nd){
        printf( "Dimension mismatch");
        exit(EXIT_FAILURE);
    }

    if(a->dims[1] != b->dims[0]){
        printf("Shape Mismatch for multiplication");
        exit(EXIT_FAILURE);
    }

    int64_t new_shape[2] = {a->dims[0], b->dims[1]};
    
    Tensor * tensor = Tensor_init(2, new_shape);
    matrix_multiply(a->data, b->data, tensor->data, a->dims[0], a->dims[1], b->dims[1]);

    return tensor;
}