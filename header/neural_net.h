
#pragma once

namespace NeuralNet {
    #include "network.h"

    enum Activation {
        Linear,
        ELU,
        ReLU,
        LeakyReLU,
        Sigmoid,
        Tanh,
        Softmax,
    };

    struct NetConfig {
        short  inputs;
        short  outputs;
        short  layers;
        short  layer_size;
        float  learning_rate;
        Activation activation;
        int    epochs;
        string name;
    };
}
