
#include "layer.h"

// - // - Constructors - // - //

Layer::Layer (int layer_size, int prev_layer_size) {

    this -> layer_size = layer_size;

    weight = 1.0 - (arma::randu(prev_layer_size, layer_size) * 2.0);
}

// - // - Public Methods - // - //

// Feed-Forward the algorithm to produce an output. 
void Layer::forward(Matrix & prev_output) {

    // Z = Input Matrix dot Weight Matrix
    Matrix z = prev_output * weight;

    // Output = Relu Activation of Z
    // (this used sigmoid before but I updated it to use ReLU)
    output = arma::max(arma::zeros(z.n_rows, z.n_cols), z);
    
    // Zprime = Relu Derivative
    zprime = z;
    zprime.elem(find(zprime >  0)).fill(1);
    zprime.elem(find(zprime <= 0)).fill(0);

    // Continue to the Next Layer, if there is one
    if (next_layer != nullptr) 
        next_layer -> forward(output);
}

// Backward propogate this layer
void Layer::backward(Matrix & prev_delta, Matrix & prev_weight) {

    // if this is the output layer
    if (next_layer == nullptr) {
        // Delta of Output = 2 * (output - previous_delta) * Sigmoid Prime
        // or, (del CO / del Ao) * (del Ao / del Zo)
        // Previous Delta here is the Target, the known good data. 
        delta = (output - prev_delta) % zprime;
    }

    // if this is a hidden layer
    else {
        // Delta of Hidden = Previous Delta * Previous Weight * Sigmoid Prime
        // or, (del A / del Z) * (del Z / del A^L-1)
        delta = (prev_delta * prev_weight.t()) % zprime;
    }

    // Continue with backpropogation if this is not the first layer.
    if (prev_layer != nullptr)
        prev_layer -> backward(delta, weight);
}

// Update the current Layer with the calculated delta. 
void Layer::update(double lr, Matrix & prev_out) {

    // weight -= delta * (del Z / del w) * lr
    weight -= (prev_out.t() * delta) * lr;

    // Continue updating unless last layer.
    if (next_layer != nullptr)
        next_layer -> update(lr, output);
}

// - // - Public Utility Methods - // - //

void Layer::set_layer_ptr(Layer * prev, Layer * next) {
    next_layer = next;
    prev_layer = prev;
}

