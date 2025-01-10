#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <alloc.h>
#include <string.h>

#include <mram.h>

#define BUFFER_SIZE (1 << 16)

typedef struct {
    int64_t input_size;
    int64_t hidden_size;
} lstm_config;


// Approximation of exp(x)
float exp_approx(float x) {
    float result = 1.0f + x + (x * x) / 2.0f + (x * x * x) / 6.0f;
    return result > 100.0f ? 100.0f : result; // Prevent overflow
}

// Approximation of sigmoid(x) = 1 / (1 + exp(-x))
float sigmoid(float x) {
    return 1.0f / (1.0f + exp_approx(-x));
}

// Approximation of tanh(x)
float tanh_approx(float x) {
    float x2 = x * x;
    return x * (27.0f + x2) / (27.0f + 9.0f * x2);
}

// Manual array copy function
void array_copy(float *dest, const float *src, int size) {
    for (int i = 0; i < size; i++) {
        dest[i] = src[i];
    }
}

// Vocabulary-to-index mapping
int get_vocab_index(char c) {
    if (c >= 'a' && c <= 'z') return c - 'a';
    if (c >= 'A' && c <= 'Z') return c - 'A';
    return 26; // Unknown token
}

__mram_noinit uint8_t buffer[BUFFER_SIZE];

void load_weights_from_host(lstm_config * config, double * weights, uint32_t * offset){
    mram_read(
        (__mram_ptr void const*)(buffer + *offset), 
        weights, 
        sizeof(double) * config->input_size * config->hidden_size
    );

    *offset += sizeof(double) * config->input_size * config->hidden_size;
}


int main(){
    __dma_aligned lstm_config * config;
    __dma_aligned double * W_f, * W_i;
    uint32_t offset = 0;

    mram_read((__mram_ptr void const*)buffer, config, sizeof(lstm_config));
    offset += sizeof(lstm_config);

    printf("%lld\n", config->input_size);
    printf("%lld\n", config->hidden_size);

    W_f = mem_alloc(sizeof(double) * config->input_size * config->hidden_size);
    W_i = mem_alloc(sizeof(double) * config->input_size * config->hidden_size);


    load_weights_from_host(config, W_f, &offset);
    load_weights_from_host(config, W_i, &offset);

    for(int64_t i = 0; i < config->input_size * config->hidden_size; i++){
        printf("%lf, ", W_f[i]);
    }

    printf("\n");

    for(int64_t i = 0; i < config->input_size * config->hidden_size; i++){
        printf("%lf, ", W_i[i]);
    }

    printf("\n");

    return 0;
}