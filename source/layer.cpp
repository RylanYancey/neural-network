
#include "layer.h"

Layer::Layer(int id, bool output, int layer_size, int prev_layer_size) {
    this -> id = id;
    this -> is_output_layer = output;

    weight = 2 * arma::randu(prev_layer_size, layer_size) - 1;
    bias = arma::randu(1, layer_size);
}

arma::Mat<double> & Layer::forward(arma::Mat<double> & x) {
    output = x * weight;
    add_bias();
    derivative();
    activation();
    return output;
}

void Layer::add_bias() {
    output.each_row() += bias;
}

void Layer::activation() {
    arma::mat::iterator it = output.begin();
    arma::mat::iterator it_end = output.end();

    for (; it != it_end; ++it)
        *it = sigmoid(*it);
}

void Layer::derivative() {
    deltas = output;
    arma::mat::iterator it = deltas.begin();
    arma::mat::iterator it_end = deltas.end();

    for (; it != it_end; ++it)
        *it = sigmoid_der(*it);
}

double Layer::sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

double Layer::sigmoid_der(double x) {
    return x * (1 - x);
}