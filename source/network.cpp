
#include "network.h"

Network::Network (NetDescriptor description) {
    desc = description;

    layers.push_back(Layer(desc.layer_size, desc.inputs, desc.activation, desc.learning_rate));
    for (int i = 1; i < desc.layers; i++)
        layers.push_back(Layer(desc.layer_size, desc.layer_size, desc.activation, desc.learning_rate));
    layers.push_back(Layer(desc.outputs, desc.layer_size, desc.activation, desc.learning_rate));

    set_pointers();

    create_json(desc);

    if (desc.linear_output == true)
        layers.back().activation = Actv::Linear;
}

Network::Network (string from_json) {

    NetDescriptor description = get_desc(from_json);
    desc = description;

    auto weights = get_weight(from_json);
    auto biases = get_biases(from_json);

    for (int i = 0; i <= desc.layers; i++) {
        layers.push_back(Layer(weights[i], biases[i], desc.activation, desc.learning_rate));
    }

    set_pointers();

    if (desc.linear_output == true)
        layers.back().activation = Actv::Linear;

}

vector<double> Network::run (vector<double> & input) {
    Mat<double> in(input);
    in = in.t();

    layers.front().forward(in);
    auto output = arma::conv_to<vector<double>>::from(layers.back().output);

    return output;
}

void Network::train (vector<vector<double>> & inputs, vector<vector<double>> & targets) {
    if (inputs[0].size() != desc.inputs) throw invalid_argument("Incorrect Input Size");
    if (targets[0].size() != desc.outputs) throw invalid_argument("Incorrect Target Size");

    vector<Mat<double>> input;
    vector<Mat<double>> target;

    for (auto i : inputs) {
        Mat<double> a(i);
        a = a.t();
        input.push_back(a);
    }

    for (auto t : targets) {
        Mat<double> a(t);
        a = a.t();
        target.push_back(a);
    }

    int e = 1;
    double accuracy = 1000.0;
    while (accuracy > desc.target_accuracy && e <= desc.epochs) {

        accuracy = 0;

        for (int i = 0; i < input.size(); i++) {
            layers.front().forward(input[i]);
            layers.back().backward(target[i], target[i]);
            layers.front().update(input[i], desc.learning_rate);

            Mat<double> temp = layers.back().output;
            Mat<double> loss = arma::abs(target[i] - temp);
            accuracy += arma::sum(arma::sum(loss));
        }

        accuracy /= desc.layers;

        if (e % desc.log_freq == 0) {
            cout << accuracy << endl;
        }
        e++;
    }
    
    string save_net;
    cout << " Sure you want to save? y / n ";
    cin >> save_net;
    if (save_net == "y")
        save();

}

void Network::save() {

    vector<Mat<double>> weights;
    vector<Mat<double>> biases;

    for (int i = 0; i < layers.size(); i++) {
        weights.push_back(layers[i].weight);
        biases.push_back(layers[i].bias);
    }   

    save_json(desc, weights, biases);
}

void Network::set_pointers() {

    auto layer_data = layers.data();

    for (int i = 0; i < layers.size(); i++) {
        if (i == 0) {
            shared_ptr<Layer> right (&layer_data[i + 1]);
            layers.at(i).set_layer_ptr(nullptr, right);
        }
        else if (i == layers.size() - 1) {
            shared_ptr<Layer> left (&layer_data[i - 1]);
            layers.at(i).set_layer_ptr(left, nullptr);
        }
        else {
            shared_ptr<Layer> right (&layer_data[i + 1]);
            shared_ptr<Layer> left (&layer_data[i - 1]);
            layers.at(i).set_layer_ptr(left, right);
        }
    }
}