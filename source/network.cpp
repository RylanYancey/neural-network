
#include "network.h"

Network::Network (int inputs, int num_layers, int layer_size, int outputs) {

    this -> inputs  = inputs;
    this -> outputs = outputs;

    this -> layers.push_back(Layer(layer_size, inputs));
    for (int i = 1; i < num_layers; i++)
        this -> layers.push_back(Layer(layer_size, layer_size));
    this -> layers.push_back(Layer(outputs, layer_size));

    auto layer_data = layers.data();

    for (int i = 0; i < layers.size(); i++) {
        if (i == 0)
            layers.at(i).set_layer_ptr(nullptr, &layer_data[i + 1]);
        else if (i == layers.size() - 1)
            layers.at(i).set_layer_ptr(&layer_data[i - 1], nullptr);
        else
            layers.at(i).set_layer_ptr(&layer_data[i - 1], &layer_data[i + 1]);
    }
}

vector<double> Network::run (vector<double> & input) {

    Matrix in(input);
    in = in.t();

    layers.front().forward(in);
    auto output = arma::conv_to<vector<double>>::from(layers.back().output);

    return output;
}

vector<double> Network::train (vector<double> & input, vector<double> & target) {

    if (input.size() != inputs) cout << "Incorrect input size" << endl;
    if (target.size() != outputs) cout << "Incorrect target size" << endl;

    //cout << "Z" << endl;

    Matrix in  (input);
    Matrix tar (target);
                                                                                                                                                                                                                                                                                                     
    in  =  in.t();
    tar = tar.t();

    //cout << "A" << endl;

    layers.front().forward(in);
    
    //cout << "B" << endl;

    layers.back().backward(tar, tar);

    //cout << "C" << endl;

    layers.front().update(0.1, in);

    //cout << "D" << endl;

    auto output = arma::conv_to<vector<double>>::from(layers.back().output);
    return output;

}