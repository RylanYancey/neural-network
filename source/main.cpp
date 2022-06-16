
#include "network.h"

int main() {
    
    Network net(4, 2, 1, 3, 2);

    arma::Mat<double> inputs = arma::randu(4, 2);

    arma::Mat<double> outputs = net.run(inputs);

    for (auto a : outputs)
        cout << a << endl;

}