#include <assert.h>
#include <dpu.h>
#include <dpu_log.h>
#include <stdio.h>
#include <time.h>

#include <stdarg.h>

#include "Tensor.h"

#ifndef DPU_BINARY
#define DPU_BINARY "./task"
#endif

static uint32_t __mram_offset = 0;

void dpu_memcpy(struct dpu_set_t set, void * src, int64_t length){
    DPU_ASSERT(dpu_broadcast_to(set, "buffer", __mram_offset, src, length, DPU_XFER_DEFAULT));
    __mram_offset += length;
}

typedef struct {
    int64_t input_size;
    int64_t hidden_size;   
} LSTM_config;

typedef struct {
    LSTM_config config;

    Tensor * W_i;
    Tensor * b_i;
} LSTM;


LSTM * LSTM_init(int64_t input_size, int64_t hidden_size){
    LSTM * lstm = (LSTM*)malloc(sizeof(LSTM));
    lstm->config = (LSTM_config){input_size, hidden_size};

    int64_t shape[2] = {input_size + hidden_size, hidden_size};


    lstm->W_i = Tensor_randn(2, shape);


    return lstm;
}

Tensor * LSTM_forward(LSTM * self, Tensor * input, Tensor * h_t, Tensor * c_t){
    Tensor * concat_dataset = Tensor_append(input, h_t);

    //input gate calculation
    Tensor * ia = Tensor_matmul(concat_dataset, self->W_i);

    return ia;
}

int main(){
    struct dpu_set_t set, dpu;
    
    srand((unsigned int)time(NULL));
    
    int64_t input_size = 5;
    int64_t hidden_size = 3;
    int64_t sequence_length = 10;
    int64_t batch_size = 1;

    LSTM * lstm = LSTM_init(input_size, hidden_size);

    // generate random input
    int64_t input_shape[2] = {batch_size, input_size};
    Tensor * x = Tensor_randn(2, input_shape);

    //initial hidden state and cell state
    int64_t state_shape[2] = {batch_size, hidden_size};
    Tensor * h_t = Tensor_zeros(2, state_shape);
    Tensor * c_t = Tensor_zeros(2, state_shape);

    Tensor * output = LSTM_forward(lstm, x, h_t, c_t);


    printf("LSTM Output\n");
    for(int64_t i = 0; i < output->n_elems; i++){
        printf("%lf, ", output->data[i]);
    }
    printf("\n");

    DPU_ASSERT(dpu_alloc(1, NULL, &set));
    DPU_ASSERT(dpu_load(set, DPU_BINARY, NULL));


    dpu_memcpy(set, (int64_t *)lstm, sizeof(int64_t) * 2);
    dpu_memcpy(set, lstm->W_i->data, lstm->W_i->n_elems * sizeof(double));
    dpu_memcpy(set, h_t->data, h_t->n_elems * sizeof(double));
    dpu_memcpy(set, x->data, x->n_elems * sizeof(double));



    DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));

    DPU_FOREACH(set, dpu) {
        DPU_ASSERT(dpu_log_read(dpu, stdout));
    }

    DPU_ASSERT(dpu_free(set));
    return 0;
}
