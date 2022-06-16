
#include "network.h"

Network::Network(int observations, int inputs, int layers, int layer_size, int outputs) {
    arma::arma_rng::set_seed_random();

    this -> inputs = arma::zeros(observations, inputs);

    this -> layers.push_back(Layer(0, false, layer_size, inputs));
    for (int i = 1; i < layers; i++)
        this -> layers.push_back(Layer(i, false, layer_size, layer_size));
    this -> layers.push_back(Layer(layers, true, outputs, layer_size));

}

arma::Mat<double> Network::run(arma::Mat<double> & input_values) {

    if (input_values.n_rows != inputs.n_rows && input_values.n_cols != inputs.n_cols) {
        cout << "Invalid input. Does not match observation count." << endl;
        return NULL;
    }

    arma::Mat<double> & prev_out = layers.at(0).forward(inputs);
    for (int i = 1; i < layers.size(); i++)
        prev_out = layers.at(i).forward(prev_out);

    return prev_out;
}
