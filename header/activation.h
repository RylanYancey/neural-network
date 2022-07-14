
#pragma once

#include <string>
using std::string;

#include <stdexcept>
using std::invalid_argument;

#include <iostream>
using std::endl;
using std::cout;
using std::cin;

#include <armadillo>
using arma::Mat;
using arma::mat;

enum Actv {
    Linear,
    ELU,
    ReLU,
    LeakyReLU,
    Sigmoid,
    Tanh,
    Softmax,
};

void activate (Mat<double> & z, Mat<double> & output, Actv activation, double lr);
void derivative (Mat<double> & z, Mat<double> & zprime, Actv activation, double lr);

Actv   str_to_actv (string from);
string actv_to_str(Actv from);

