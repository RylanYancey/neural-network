
#pragma once

#include "neuron.h"

class Network {
public:

    Network(int inputs, int layers, int layer_size, int outputs);
    ~Network();

    int inputs;
    int layers;
    int layer_size;
    int outputs;

    vector<vector<unique_ptr<Neuron>> *> net;

    vector<float> train(vector<float> & inputs, vector<float> & target);
    vector<float> run(vector<float> & inputs);
    void backpropogate(vector<float> & output);

private:


};