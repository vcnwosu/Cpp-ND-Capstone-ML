#include <iostream>
#include <cassert>
#include <vector>
#include <math.h>
#include "network/network.h"

int main(void) {
    std::cout << "Testing Universal Approximation Unit implementation";

    std::vector<float> v{1, -20, 14, 4, -7};
    network::unit r{network::activation::kind::RELU};
    network::unit s{network::activation::kind::SIGMOID};

    /**
     * Assuming a weight vector of (.25, .75) and bias of -.75...
     * For input vector (1, 2), we expect 1 as the output for relu activation
     *
     * Since we also know that the derivative of a ReLU activation > 0 == 1,
     * we expect the derivative to also be 1 after calculating the activation.
     */
    float activation = network::activation::activate((.25 * 1) + (.75 * 2) - .75, r.kind);
    float derivative = network::activation::derivative(activation, r.kind);
    assert(fabs(activation - 1) == 0.0);
    assert(fabs(derivative - 1) == 0.0);

    /**
     * Assuming a weight vector of (.25, .75) and a bias of -.75...
     * For input vector (1, 2), we expect a 0.731059 as the sigmoid activation
     *
     * Since we know that the derivative of any Sigmoid activation is equal
     * to the activation * (1 - activation), we expect the derivative to also
     * be 0.196612
     */
    activation = network::activation::activate((.25 * 1) + (.75 * 2) - .75, s.kind);
    derivative = network::activation::derivative(activation, s.kind);
    assert(fabs(activation - 0.731059) < 0.0001);
    assert(fabs(derivative - 0.196612) < 0.0001);

    std::cout << " [PASS] " << std::endl;

    std::cout << "Testing Multilayer Perceptron architecture implementation";

    /**
     * Construct multilayer perceptron with one hidden layer of 10 ReLU units
     * and a final layer of one Sigmoid unit as the output
     */
    network::architecture::multilayer_perceptron brain;
    brain.add_layer(10, network::activation::kind::RELU);
    brain.add_layer(3, network::activation::kind::SIGMOID);

    std::vector<std::vector<float>> data{
        {5.1, 3.5, 1.4, 0.2}, {1, 0, 0},
        {7.0, 3.2, 4.7, 1.4}, {0, 1, 0},
        {6.3, 3.3, 6.0, 2.5}, {0, 0, 1},
        {4.6, 3.6, 1.0, 0.2}, {1, 0, 0},
        {5.7, 2.6, 3.5, 1.0}, {0, 1, 0},
        {7.7, 3.8, 6.7, 2.2}, {0, 0, 1},
    };

    brain.train(data, 5, 0.01, false);

    std::cout << " [PASS] " << std::endl;

    return 0;
}
