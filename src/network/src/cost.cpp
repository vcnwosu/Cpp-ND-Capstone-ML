/**
 * cost.cpp
 *
 * Implementation of miscellaneous free members/functions within
 * the network::cost namespace
 *
 * @author Victor Nwosu
 */

#include <cmath>
#include "network/cost.h"

/**
 * Compute the cost function for the network using
 * cross entropy for two classes
 *
 * @param const float y the known label for the input
 * @param const float y_hat the probability predicted by the network
 *
 * @return float the cross entropy result
 */
float network::cost::binary_cross_entropy(const float y, const float y_hat) {
    if (y == 1) {
        return log(y_hat);
    }

    return log(1 - y_hat);
}

/**
 * Compute the cost function for the network using
 * cross entropy for multiple classes
 * 
 * @param const std::vector<float> &y_hat the network output
 * @param const std::vector<float> &y the known labels
 *
 * @return float the cross entropy result
 */
float network::cost::cross_entropy(const std::vector<int> &y, const std::vector<float> &y_hat) {
    float ce = 0.0f;

    for (int i = 0; i < y_hat.size(); i++) {
        ce += y[i] * log(y_hat[i]);
    }

    return -ce;
}

/**
 * Simple error calculation for each neuron in the output
 *
 * @param const std::vector<int> &y the known labels
 * @param const std::vector<int> &y_hat the network output
 *
 * @return std::vector<float> the error for each output
 */
std::vector<float> network::cost::simple_error(const std::vector<int> &y, const std::vector<float> &y_hat) {
    std::vector<float> res;

    for (int i = 0; i < y.size(); i++) {
        res.push_back(y[i] - y_hat[i]);
    }

    return res;
}
