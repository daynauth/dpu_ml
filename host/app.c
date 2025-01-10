#include <assert.h>
#include <dpu.h>
#include <dpu_log.h>
#include <stdio.h>
#include <time.h>

#ifndef DPU_BINARY
#define DPU_BINARY "./task"
#endif

double * generate_weights(int64_t length){
    double * weight = malloc(sizeof(double) * length);

    for(int64_t i = 0; i < length; i++){
        weight[i] = (double)rand() / (double)RAND_MAX;
    }

    return weight;
}

void copy_weights_to_dpu(struct dpu_set_t set, double * weights, int64_t length, uint32_t * offset){
    DPU_ASSERT(dpu_broadcast_to(set, "buffer", *offset, weights, length * sizeof(double), DPU_XFER_DEFAULT));
    *offset += length * sizeof(double);
}

int main(){
    struct dpu_set_t set, dpu;
    srand((unsigned int)time(NULL));

    int64_t config[2] = {5, 3}; // input size, hidden size
    int64_t length = config[0] * config[1];

    double * W_f = generate_weights(length);
    double * W_i = generate_weights(length);


    DPU_ASSERT(dpu_alloc(1, NULL, &set));
    DPU_ASSERT(dpu_load(set, DPU_BINARY, NULL));

    DPU_ASSERT(dpu_broadcast_to(set, "buffer", 0, config, sizeof(int64_t) * 2, DPU_XFER_DEFAULT));
    uint32_t offset = 128;

    copy_weights_to_dpu(set, W_f, length, &offset);
    copy_weights_to_dpu(set, W_i, length, &offset);


    DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));

    DPU_FOREACH(set, dpu) {
        DPU_ASSERT(dpu_log_read(dpu, stdout));
    }

    DPU_ASSERT(dpu_free(set));
    return 0;
}
