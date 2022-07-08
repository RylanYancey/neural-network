
#include <iostream>
using std::cout;
using std::endl;

#include "network.h"

int main() {
    
    NetConfig config;
        config.inputs = 2;
        config.outputs = 1;
        config.layers = 1;
        config.layer_size = 3;
        config.activation = ActFn::ReLU;
        config.learning_rate = 0.1;
        config.name = "test1";

    Network net(config);

}