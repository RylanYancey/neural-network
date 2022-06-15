
#include "network.h"
#include "neuron.h"

// - Constructors - // --------------------------------

Network::Network(int inputs, int layers, int layer_size, int outputs) {

    srand(time(NULL));

    this -> inputs = inputs;
    this -> layers = layers;
    this -> layer_size = layer_size;
    this -> outputs = outputs;

    // for every layer
    for (int i = 0; i < layers + 2; i++) {

        // add a new layer to the net. 
        net.push_back(new vector<unique_ptr<Neuron>>());

        // how many neurons should we make
        int size = inputs;
        if (i > layers) size = outputs;
        else if (i != 0) size = layer_size;

        auto net_data = net.data();

        // Fill the layer
        for (int k = 0; k < size; k++) {
            // Create a new Neuron and give it a reference to the previous and next layer. 
            if (i == 0) net.at(i) -> push_back(unique_ptr<Input>(new Input(0, net_data[i + 1])));
            else if (i > layers) net.at(i) -> push_back(unique_ptr<Output>(new Output(layers + 1, net_data[i - 1])));
            else net.at(i) -> push_back(unique_ptr<Hidden>(new Hidden(i, net_data[i - 1], net_data[i + 1])));
        }
    }
}

Network::~Network() {
    for (int i = 0; i < net.size(); i++)
        free(net.at(i));
}

// - Public Methods - // --------------------------------

vector<float> Network::train(vector<float> & inputs, vector<float> & target) {
    vector<float> output = run(inputs);

    float loss = 0;
    for (int i = 0; i < target.size(); i++)
        loss += (1/2) * ((1 - output.at(i)) * (1 - output.at(i)));

    backpropogate(output);

    return output;
}

void Network::backpropogate(vector<float> & output) {

}

vector<float> Network::run(vector<float> & inputs) {

    for (int i = 0; i < net.at(0) -> size(); i++)
        net.at(0) -> at(i) -> output = inputs[i];

    for (int x = 1; x < net.size(); x++)
        for (int y = 0; y < net.at(x) -> size(); y++)
            net.at(x) -> at(y) -> activate();

    vector<float> result;
    for (int i = 0; i < outputs; i++)
        result.push_back(net.back() -> at(i) -> output);

    return result;
}