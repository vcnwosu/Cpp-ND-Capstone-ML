/**
 * C++ Capstone: A Basic Feedforward Neural Network
 * main.cpp
 *
 * @author Victor Nwosu
 * @version 1.0
 */

#include <iostream>
#include <vector>
#include "network/network.h"
#include "iris/dataset.h"

int main() {
    // load the iris dataset
    iris::dataset data("../data/iris.data");

    // set up the multilayer perceptron network
    network::architecture::multilayer_perceptron mlp;

    // add desired layers for the network
    mlp.add_layer(16, network::activation::kind::SIGMOID);
    mlp.add_layer(3, network::activation::kind::SIGMOID);

    // train the network
    // 5 epochs
    // 0.01 learning rate
    // true for verbosity/output to std::cout
    mlp.train(data, 1000, 0.10, false);

    // test "new data"
    // true for verbosity/output to std::cout
    for (std::vector<float> example : data.testing) {
        std::cout << "Testing new flower: " << data.name(example[example.size() - 1]) << " ... ";
        std::vector<float> res = mlp.predict(example);

        int max = 0;
        for (int i = 1; i < res.size(); i++) {
            max = res[max] > res[i] ? max : i;
        }

        std::cout << data.name(max) << "(" << (res[max] * 100) << "%)";

        std::cout << std::endl;
    }

    return 0;
}
