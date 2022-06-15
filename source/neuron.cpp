
#include "neuron.h"

// - Base Class Implementation - // --------------------------

Neuron::Neuron(int layer, vector<unique_ptr<Neuron>> * prev, vector<unique_ptr<Neuron>> * next) {
    this -> prev = prev;
    this -> next = next;
    this -> layer = layer;
}

void Neuron::activate() {
    output = sigmoid(summation());
}

float Neuron::sigmoid(float x) {
    return 1 / (1 + exp(-x));
}

float Neuron::sig_der() {
    return output * (1 - output);
}

float Neuron::summation() {
    cout << "This should not be activated" << endl;
    return 0;
}

// - Input Neuron - // --------------------------

Input::Input(int layer, vector<unique_ptr<Neuron>> * next) 
    : Neuron(layer, next, next) { }

float Input::summation() {
    cout << "Summation not defined for Input Neuron. No Summation possible" << endl;
    return 0;
}

// - Hidden Neuron - // --------------------------

Hidden::Hidden(int layer, vector<unique_ptr<Neuron>> * prev, vector<unique_ptr<Neuron>> * next) 
    : Neuron(layer, prev, next) {
        for (int i = 0; i < prev -> size(); i++)
            weights.push_back(((double)rand()) / ((double)RAND_MAX) * 2.0 - 1.0);
    }

float Hidden::summation() {
    float sum = 0;
    for (int i = 0; i < prev -> size(); i++)
        sum += prev -> at(i) -> output * weights[i];
    return sum;
}

// - Output Neuron - // --------------------------

Output::Output(int layer, vector<unique_ptr<Neuron>> * prev)
    : Neuron(layer, prev, prev) {
        for (int i = 0; i < prev -> size(); i++)
            weights.push_back(((double)rand()) / ((double)RAND_MAX) * 2.0 - 1.0);
    }

float Output::summation() {
    float sum = 0;
    for (int i = 0; i < prev -> size(); i++)
        sum += prev -> at(i) -> output * weights[i];
    return sum;
}