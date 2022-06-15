
#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <time.h>
#include <stdlib.h>
using namespace std;

class Neuron {
public:

    Neuron(int layer, vector<unique_ptr<Neuron>> * prev, vector<unique_ptr<Neuron>> * next);

    vector<unique_ptr<Neuron>> * prev;
    vector<unique_ptr<Neuron>> * next;

    int layer;

    float output;

    void activate();
    float sigmoid(float x);
    float sig_der();

private:

    virtual float summation();

};

// - Classes Which Inherit from Neuron - //

class Input : public Neuron {
public:

    Input(int layer, vector<unique_ptr<Neuron>> * next);

private: 

    virtual float summation();

};

class Hidden : public Neuron {
public:

    Hidden(int layer, vector<unique_ptr<Neuron>> * prev, vector<unique_ptr<Neuron>> * next);

    vector<float> weights;

private:

    virtual float summation();

};

class Output : public Neuron {
public:

    Output(int layer, vector<unique_ptr<Neuron>> * prev);

    vector<float> weights;

private:

    virtual float summation();

};