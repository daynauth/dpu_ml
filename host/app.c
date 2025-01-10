#include <assert.h>
#include <dpu.h>
#include <dpu_log.h>
#include <stdio.h>
#include <time.h>

#include "Tensor.h"

#ifndef DPU_BINARY
#define DPU_BINARY "./task"
#endif

static uint32_t __mram_offset = 0;

void dpu_memcpy(struct dpu_set_t set, void * src, int64_t length){
    DPU_ASSERT(dpu_broadcast_to(set, "buffer", __mram_offset, src, length, DPU_XFER_DEFAULT));
    __mram_offset += length;
}

int main(){
    struct dpu_set_t set, dpu;
    
    srand((unsigned int)time(NULL));
    

    int64_t dims[2] = {5, 3}; // input size, hidden size


    Tensor * W_f = Tensor_randn(2, dims);
    Tensor * W_i = Tensor_randn(2, dims);


    DPU_ASSERT(dpu_alloc(1, NULL, &set));
    DPU_ASSERT(dpu_load(set, DPU_BINARY, NULL));


    dpu_memcpy(set, dims, sizeof(int64_t) * 2);
    dpu_memcpy(set, W_f->data, W_f->n_elems * sizeof(double));
    dpu_memcpy(set, W_i->data, W_i->n_elems * sizeof(double));


    DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));

    DPU_FOREACH(set, dpu) {
        DPU_ASSERT(dpu_log_read(dpu, stdout));
    }

    DPU_ASSERT(dpu_free(set));
    return 0;
}
