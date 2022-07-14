
#pragma once

#include "jsonifier.h"
#include "layer.h"

class Network {
public:

    Network (NetDescriptor description);
    Network (string from_json);

    vector<double> run (vector<double> & input);
    void train (vector<vector<double>> & inputs, vector<vector<double>> & targets);

    void save();

    NetDescriptor desc;

private:

    void set_pointers();

    vector<Layer> layers;

};