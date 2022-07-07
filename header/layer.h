
#pragma once

#include <iostream>
using std::cout;
using std::endl;

#include <armadillo>
typedef arma::Mat<double> Matrix;

class Layer {
public:

    // Constructor for Input Layer
    Layer (int layer_size, int prev_layer_size);

    Matrix output;

    void forward  (Matrix & prev_output);
    void backward (Matrix & prev_delta, Matrix & prev_weight);
    void update   (double lr, Matrix & prev_out);

    void set_layer_ptr(Layer * prev, Layer * next);

    int layer_size;

private:

    Matrix weight;
    Matrix zprime;
    Matrix delta;
    Matrix bias;

    Layer * prev_layer = nullptr;
    Layer * next_layer = nullptr;

};



