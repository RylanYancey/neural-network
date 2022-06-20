
#include <vector>
using std::vector;

#include "layer.h"

class Network {
public:

    Network (int inputs, int num_layers, int layer_size, int outputs);

    vector<double> run   (vector<double> & input);
    vector<double> train (vector<double> & input, vector<double> & target);

private:

    int inputs;
    int outputs;

    vector<Layer> layers;

};