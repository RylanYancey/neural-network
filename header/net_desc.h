
#pragma once

#include "activation.h"

struct NetDescriptor {
    uint inputs;
    uint outputs;
    uint layers;
    uint layer_size;
    uint epochs;
    uint log_freq;
    double target_accuracy;
    double learning_rate;
    bool linear_output;
    Actv activation;
    string name;
};