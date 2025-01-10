#ifndef LSTM_H
#define LSTM_H

#include <mram.h>

#include "tensor.h"

typedef struct {
    int64_t input_size;
    int64_t hidden_size;
} lstm_config;

typedef struct {
    lstm_config * config;
}LSTM;

int LSTM_init(LSTM * self, lstm_config * config);


#endif