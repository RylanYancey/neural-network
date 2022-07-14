
#pragma once

#include <memory>
using std::shared_ptr;

#include <armadillo>
using arma::Mat;
using arma::randu;

#include "activation.h"

class Layer {
public:

    Layer (int layer_size, int prev_layer_size, Actv activatioin, double lr);
    Layer (Mat<double> & weights, Mat<double> & biases, Actv activation, double lr);

    void forward  (Mat<double> & prev_output);
    void backward (Mat<double> & prev_delta, Mat<double> & prev_weight);
    void update   (Mat<double> & prev_out, double learning_rate);

    void set_layer_ptr (shared_ptr<Layer> prev, shared_ptr<Layer> next);

    Mat<double> output;
    Mat<double> weight;
    Mat<double> zprime;
    Mat<double> delta;
    Mat<double> bias;

    Actv activation;

private:

    double learning_rate;

    shared_ptr<Layer> prev = nullptr;
    shared_ptr<Layer> next = nullptr;

};