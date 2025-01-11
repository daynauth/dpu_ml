#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <alloc.h>
#include <string.h>

#include <mram.h>

#ifndef NUM_TASKLETS
#define NUM_TASKLETS 1
#endif

#define BUFFER_SIZE (1 << 16)

typedef struct {
    int64_t input_size;
    int64_t hidden_size;
} lstm_config;

static uint32_t __mram_offset = 0;

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

__mram_noinit uint8_t buffer[BUFFER_SIZE];

void load_weights_from_host(void * weights, int64_t size){
    mram_read(
        (__mram_ptr void const*)(buffer + __mram_offset), 
        weights, 
        size
    );

    __mram_offset += size;
}

void concat_array(double *a, double *b, double *c, int64_t a_size, int64_t b_size){
    for(int64_t i = 0; i < a_size; i++){
        c[i] = a[i];
    }

    for(int64_t i = 0; i < b_size; i++){
        c[i + a_size] = b[i];
    }
}

int main(){
    __dma_aligned lstm_config * config;
    __dma_aligned double * W_i;
    __dma_aligned double * h_t;
    __dma_aligned double * x;
    __dma_aligned double * concat;
    __dma_aligned double * output;

    load_weights_from_host(config, sizeof(lstm_config));


    int64_t shape[2] = {config->input_size + config->hidden_size, config->hidden_size};
    int64_t weight_size = sizeof(double) * shape[0] * shape[1];
    W_i = mem_alloc(weight_size);
    h_t = mem_alloc(sizeof(double) * config->hidden_size);
    x = mem_alloc(sizeof(double) * config->input_size);
    concat = mem_alloc(sizeof(double) * (config->input_size + config->hidden_size));
    output = mem_alloc(sizeof(double) * config->hidden_size);

    load_weights_from_host(W_i, weight_size);
    load_weights_from_host(h_t, sizeof(double) * config->hidden_size);
    load_weights_from_host(x, sizeof(double) * config->input_size);


    concat_array(x, h_t, concat, config->input_size, config->hidden_size);
    matrix_multiply(concat, W_i, output, 1, shape[0], shape[1]);


    for(int64_t i = 0; i < config->hidden_size; i++){
        printf("%lf, ", output[i]);
    }

    printf("\n");

    return 0;
}