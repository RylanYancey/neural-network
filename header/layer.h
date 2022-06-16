
#include <stdlib.h>
#include <cmath>
using namespace std;

#include <armadillo>

class Layer {
public:

    Layer(int id, bool output, int layer_size, int prev_layer_size);

    int id;
    bool is_output_layer;

    arma::Mat<double> bias;

    arma::Mat<double> weight;
    arma::Mat<double> output;
    arma::Mat<double> deltas;

    arma::Mat<double> & forward(arma::Mat<double> & x);
    void backward();
    void update();

private:

    void add_bias();

    // Run the activation function for each element of the matrix //
    void activation();

    double sigmoid(double x);
    double sigmoid_der(double x);

    // Fills Deltas with Output, before activation // 
    void derivative();

};