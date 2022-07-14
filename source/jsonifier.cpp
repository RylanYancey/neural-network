
#include "jsonifier.h"

void create_json (NetDescriptor desc) {
    string temp =
        "{\n"
            "\"inputs\" : " + std::to_string(desc.inputs) + ",\n"
            "\"outputs\" : " + std::to_string(desc.outputs) + ",\n"
            "\"layers\" : " + std::to_string(desc.layers) + ",\n"
            "\"layer_size\" : " + std::to_string(desc.layer_size) + ",\n"
            "\"epochs\" : " + std::to_string(desc.epochs) + ",\n"
            "\"log_freq\" : " + std::to_string(desc.log_freq) + ",\n"
            "\"target_accuracy\" : " + std::to_string(desc.target_accuracy) + ",\n"
            "\"learning_rate\" : " + std::to_string(desc.learning_rate) + ",\n"
            "\"linear_output\" : " + std::to_string(desc.linear_output) + ",\n"
            "\"activation\" : \"" + actv_to_str(desc.activation) + "\",\n"
            "\"name\" : \"" + desc.name + "\",\n"
            "\"weights\" : [\n\n"

            "],\n"
            "\"biases\" : [\n\n"
                
            "]\n"
        "}\n";

    ofstream outfile(desc.name);
    outfile << temp;
    outfile.close();

}

void save_json (NetDescriptor desc, vector<Mat<double>> weights, vector<Mat<double>> biases) {
    ifstream ifs (desc.name);

    Json::Reader reader;
    Json::StyledWriter writer;
    Json::Value root;
    reader.parse(ifs, root);

    root["inputs"] = desc.inputs;
    root["outputs"] = desc.outputs;
    root["layers"] = desc.layers;
    root["layer_size"] = desc.layer_size;
    root["epochs"] = desc.epochs;
    root["log_freq"] = desc.log_freq;
    root["target_accuracy"] = desc.target_accuracy;
    root["learning_rate"] = desc.learning_rate;
    root["activation"] = actv_to_str(desc.activation);
    root["name"] = desc.name;

    root["weights"] = Json::arrayValue;
    root["biases"] = Json::arrayValue;

    for (int i = 0; i <= desc.layers; i++) {
        stringstream w;
        stringstream b;

        weights[i].raw_print(w);
        biases[i].raw_print(b);

        std::string const wresult =
            regex_replace(w.str(), regex("\n"), "; ");

        std::string const bresult =
            regex_replace(b.str(), regex("\n"), "; ");

        root["weights"].append(wresult);
        root["biases"].append(bresult);
    }

    writer.write(root);

    ofstream outfile(desc.name);
    outfile << root;
    outfile.close();

    ifs.close();
}

NetDescriptor get_desc (string filename) {
    ifstream ifs (filename);
    Json::Reader reader;
    Json::Value root;

    reader.parse(ifs, root);

    NetDescriptor nd {
        root["inputs"].asUInt(),
        root["outputs"].asUInt(),
        root["layers"].asUInt(),
        root["layer_size"].asUInt(),
        root["epochs"].asUInt(),
        root["log_freq"].asUInt(),
        root["target_accuracy"].asDouble(),
        root["learning_rate"].asDouble(),
        root["linear_output"].asBool(),
        str_to_actv(root["activation"].asString()),
        root["name"].asString(),
    };

    return nd;
}

vector<Mat<double>> get_weight (string filename) {

    ifstream ifs (filename);
    Json::Reader reader;
    Json::Value root;

    reader.parse(ifs, root);

    Json::Value& weights = root["weights"];

    vector<Mat<double>> result;

    for (int i = 0; i < weights.size(); i++) {
        string w = weights[i].asString();
        Mat<double> wMat(w);

        result.push_back(wMat);
    }

    ifs.close();

    return result;

}

vector<Mat<double>> get_biases (string filename) {
    ifstream ifs (filename);
    Json::Reader reader;
    Json::Value root;

    reader.parse(ifs, root);

    Json::Value& biases = root["biases"];

    vector<Mat<double>> result;

    for (int i = 0; i < biases.size(); i++) {
        string b = biases[i].asString();
        Mat<double> bMat(b);

        result.push_back(bMat);
    }

    ifs.close();
    
    return result;

}
