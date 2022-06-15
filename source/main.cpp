
#include "network.h"

int main() {

    Network net(3, 3, 3, 3);

    vector<float> inputs {1, 2, 3};

    vector<float> outputs = net.run(inputs);

    for (auto i : outputs)
        cout << i << endl;

}