
#include "layer.h"

// - // - Constructors - // - //

Layer::Layer (int layer_size, int prev_layer_size) {

    this -> layer_size = layer_size;

    weight = 1.0 - (arma::randu(prev_layer_size, layer_size) * 2.0);
}

// - // - Public Methods - // - //

void Layer::forward(Matrix & prev_output) {
    Matrix z = prev_output * weight;
    output = 1 / ( 1 + arma::exp(-(z)));
    zprime = (z % (1 - z));

    if (next_layer != nullptr) 
        next_layer -> forward(output);
}

void Layer::backward(Matrix & prev_delta, Matrix & prev_weight) {

    if (next_layer == nullptr) {
        delta = (2 * (output - prev_delta)) % zprime;
    }

    else {
        delta = (prev_delta * prev_weight.t()) % zprime;
    }

    if (prev_layer != nullptr)
        prev_layer -> backward(delta, weight);

}

void Layer::update(double lr, Matrix & prev_out) {

    weight -= (prev_out.t() * delta) * lr;

    if (next_layer != nullptr)
        next_layer -> update(lr, output);
}

// - // - Public Utility Methods - // - //

void Layer::set_layer_ptr(Layer * prev, Layer * next) {
    next_layer = next;
    prev_layer = prev;
}

