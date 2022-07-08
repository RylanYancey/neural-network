
// Preprocessor
#pragma once

// STL
#include <vector>
using std::vector;

#include <string>
using std::string;

// External Libraries
#include "json/json.h"

// Local Includes
#include "layer.h"

#include "neural_net.h"
using NeuralNet::NetConfig;
using NeuralNet::Activation;

class Network {
public:

    Network (NetConfig configuration);
    Network (string from_json);

    vector<float> run   (vector<float> & input);
    vector<float> train (vector<float> & input, vector<float> & target);

    NetConfig config;

private:

    vector<Layer> layers;

};