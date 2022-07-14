
#include "activation.h"

void activate (Mat<double> & z, Mat<double> & output, Actv activation, double lr) {
    output = z;

    switch (activation) {
        case Actv::Linear: {
            // do nothing
        } break;

        case Actv::ELU: {
            output.for_each( [lr](mat::elem_type& val) {
                if (val <= 0)
                    val = lr * (exp(val) - 1);
            });
        } break;

        case Actv::ReLU: {
            output.for_each( [lr](mat::elem_type& val) {
                if (val < 0)
                    val = 0;
            });
        } break;

        case Actv::LeakyReLU: {
            output.for_each( [lr](mat::elem_type& val) {
                if (val > 0)
                    val *= lr;
            });
        } break;

        case Actv::Sigmoid: {
            output = 1 / (1 + exp(-output));
        } break;

        case Actv::Tanh: {
            output = (exp(output) - exp(-output)) / (exp(output) + exp(-output));
        } break;

        case Actv::Softmax: {
            throw invalid_argument("Softmax has not been implemented. Sorry.");
        } break;
    } 
}

void derivative (Mat<double> & z, Mat<double> & zprime, Actv activation, double lr) {
    zprime = z;

    switch (activation) {
        case Actv::Linear: {
            zprime.for_each( [lr](mat::elem_type& val) {
                val = 1;
            });
        } break;

        case Actv::ELU: {
            zprime.for_each( [lr](mat::elem_type& val) {
                if (val < 0)
                    val = lr * exp(val);
            });
        } break;

        case Actv::ReLU: {
            zprime.elem(find(zprime >  0)).fill(1);
            zprime.elem(find(zprime <= 0)).fill(0);
        } break;

        case Actv::LeakyReLU: {
            zprime.for_each( [lr](mat::elem_type& val) {
                if (val > 0) val = 1;
                else         val = lr;
            });
        } break;

        case Actv::Sigmoid: {
            zprime = (1 / (1 + exp(-zprime))) % (1 - (1 / (1 + exp(-zprime))));
        } break;

        case Actv::Tanh: {
            zprime = 1 - pow(((exp(zprime) - exp(-zprime)) / (exp(zprime) + exp(-zprime))), 2);
        } break;

        case Actv::Softmax: {
            throw invalid_argument("Softmax has not been implemented. Sorry.");
        } break;
    }
}

Actv str_to_actv (string from) {
    if (from == "Linear") return Actv::Linear;
    if (from == "ELU") return Actv::ELU;
    if (from == "ReLU") return Actv::ReLU;
    if (from == "LeakyReLU") return Actv::LeakyReLU;
    if (from == "Sigmoid") return Actv::Sigmoid;
    if (from == "Tanh") return Actv::Tanh;
    if (from == "Softmax") return Actv::Softmax;

    throw invalid_argument("Unable to parse Activation from string. Please input a valid Activation. ex: \"Sigmoid\"");
}

string actv_to_str (Actv from) {
    if (from == Actv::Linear) return "Linear";
    if (from == Actv::ELU) return "ELU";
    if (from == Actv::ReLU) return "ReLU";
    if (from == Actv::LeakyReLU) return "LeakyReLU";
    if (from == Actv::Sigmoid) return "Sigmoid";
    if (from == Actv::Tanh) return "Tanh";
    if (from == Actv::Softmax) return "Softmax";

    throw invalid_argument("Unable to parse String from provided Activation."); 
}