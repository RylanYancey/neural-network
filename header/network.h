
#include "layer.h"

class Network {
public:

    Network(int observations, int input, int layers, int layer_size, int outputs);

    arma::Mat<double> train();
    arma::Mat<double> run(arma::Mat<double> & input_values);

private:

    arma::Mat<double> inputs;
    vector<Layer> layers;

};