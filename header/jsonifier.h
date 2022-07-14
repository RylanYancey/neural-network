
#pragma once

#include <string>
using std::string;
using std::to_string;

#include <fstream>
using std::fstream;
using std::ofstream;
using std::ifstream;
using std::stringstream;

#include <vector>
using std::vector;

#include <regex>
using std::regex;

#include <armadillo>
using arma::Mat;

#include <jsoncpp/json/json.h>

#include "net_desc.h"

void create_json (NetDescriptor desc);
void save_json (NetDescriptor desc, vector<Mat<double>> weights, vector<Mat<double>> biases);
NetDescriptor get_desc (string filename);

vector<Mat<double>> get_weight (string filename);
vector<Mat<double>> get_biases (string filename);


