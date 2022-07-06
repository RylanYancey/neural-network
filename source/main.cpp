
#include "network.h"

#include <vector>
using std::vector;

#include <stdlib.h>
#include <time.h>

int main() {

    Network net(2, 3, 3, 2);

    std::srand(std::time(NULL));

    vector<vector<double>> input {
        { 0, 0 },
        { 0, 1 },
        { 1, 0 },
        { 1, 1 },
    };

    vector<vector<double>> target {
        { 0, 0 },
        { 1, 1 },
        { 1, 0 },
        { 0, 0 },
    };
    
    //vector<double> in2 = { 0.1, 0.9 };
    //vector<double> tar2 = { 0.7, 0.7 };

    int i = 0;
    while (true) {

        //vector<double> out = net.train(in2, tar2);
        //cout << out[0] << " " << out[1] << endl;

        int rand = std::rand() % 4;

        vector<double> output = net.train(input[rand], target[rand]);

        //if (output[0] > 0.5) output[0] = 1;
        //else output[0] = 0;

        //if (output[1] > 0.5) output[1] = 1;
        //else output[1] = 0;

        cout << input[rand][0] << " " << input[rand][1] << " " << output[0]  << " " << target[rand][0] << endl;

        if (i > 1000)
            return 0;
        i++;
    }

}