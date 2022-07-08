
// Preprocessor
#pragma once

// STL 
#include<memory>
using std::shared_ptr;

// External Libraries
#include <armadillo>
using arma::Mat;
using arma::randu;

// Local Includes
#include "neural_net.h"
using NeuralNet::Activation;

class Layer {

public:

    Layer (int layer_size, int prev_layer_size, Activation activation);
    Layer (Mat<float> & weights, Mat<float> & biases);

    void forward  (Mat<float> & prev_output);
    void backward (Mat<float> & prev_delta, Mat<float> & prev_weight);
    void update   (Mat<float> & prev_out, float learning_rate);

    void set_layer_ptr (shared_ptr<Layer> prev, shared_ptr<Layer> next);

    Mat<float> output;

private:

    void activate   (Mat<float> & z);
    void derivative (Mat<float> & z);

    Mat<float> weight;
    Mat<float> zprime;
    Mat<float> delta;
    Mat<float> bias;

    Activation activation;

    shared_ptr<Layer> prev = nullptr;
    shared_ptr<Layer> next = nullptr;

};