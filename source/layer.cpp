
#include "layer.h"

Layer::Layer (int layer_size, int prev_layer_size, Actv activation, double lr) {
    weight = randu (prev_layer_size, layer_size);
    bias =   randu (1, layer_size);
    this -> activation = activation;
    learning_rate = lr;
}

Layer::Layer (Mat<double> & weights, Mat<double> & biases, Actv activation, double lr) {
    weight = weights;
    bias = biases;
    this -> activation = activation;
    learning_rate = lr;
}

void Layer::set_layer_ptr(shared_ptr<Layer> prev, shared_ptr<Layer> next) {
    this -> prev = prev;
    this -> next = next;
}

void Layer::forward (Mat<double> & prev_output) {
    
    Mat<double> z = (prev_output * weight) + bias;

    activate (z, output, activation, learning_rate);
    derivative(z, zprime, activation, learning_rate);

    if (next != nullptr)
        next -> forward(output);
}

void Layer::backward (Mat<double> & prev_delta, Mat<double> & prev_weight) {
    
    if (next == nullptr)
        delta = (2 * (output - prev_delta)) % zprime;

    else
        delta = (prev_delta * prev_weight.t()) % zprime;

    if (prev != nullptr)
        prev -> backward (delta, weight);
}

void Layer::update(Mat<double> & prev_output, double lr) {
    weight -= (prev_output.t() * delta) * lr;
    bias -= lr * delta;

    if (next != nullptr)
        next -> update (output, lr);   
}