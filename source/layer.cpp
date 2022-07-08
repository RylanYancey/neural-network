
#include "layer.h"

Layer::Layer (int layer_size, int prev_layer_size, Activation activation) {
    weight = randu (prev_layer_size, layer_size);
    bias   = randu (1, layer_size);

    this -> activation = activation;
}

void Layer::set_layer_ptr (shared_ptr<Layer> prev, shared_ptr<Layer> next) {
    this -> prev = prev;
    this -> next = next;
}

void Layer::forward (Mat<float> & prev_output) {
    Mat<float> z = (prev_output * weight) + bias;

    activation (z); // Creates output
    derivative (z); // Creates zprime

    if (next != nullptr)
        next -> forward (output);   

}

void Layer::activate (Mat<float> & z) {
    
}

void Layer::derivative (Mat<float> & z) {

}

void Layer::backward (Mat<float> & prev_delta, Mat<float> & prev_weight) {

    if (next == nullptr) 
        delta = (output - prev_delta) % zprime;

    else 
        delta = (prev_delta * prev_weight.t()) % zprime;

    if (prev != nullptr)
        prev -> backward(delta, weight);
}

void Layer::update (Mat<float> & prev_output, float lr) {

    weight -= (prev_output.t() * delta) * lr;
    bias -= lr * delta;

    if (next != nullptr)
        next -> update (output, lr);

}