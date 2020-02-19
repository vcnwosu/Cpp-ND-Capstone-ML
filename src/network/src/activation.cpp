/**
 * activation.cpp
 *
 * Definition of free members within the network::activation namespace
 *
 * @author Victor Nwosu
 */

#include <math.h>
#include "network/activation.h"


/**
 * Compute the activation of an incoming signal using
 * a designated function
 *
 * @param const float signal the incoming signal
 * @param const kind func the activation function to compute
 *
 * @return float the activation output
 */
float network::activation::activate(const float signal, const kind func) {
    float value = 0.0f;

    switch (func) {
        case kind::RELU:
            value = signal > 0 ? signal : 0;
            break;
        case kind::SIGMOID:
            value = std::min((float) 1 / (1 + exp(-signal)), 0.9999f);
            value = std::max(value, 0.0001f);
            break;
    }

    return value;
}

/**
 * Compute the derivative of the incoming signal using
 * a designated function
 *
 * @param const float signal the incoming signal
 * @param const kind func the function to compute the derivative
 *
 * @return float the derivative of the incoming signal
 */
float network::activation::derivative(const float signal, const kind func) {
    float value = 0.0;

    switch (func) {
        case kind::RELU:
            value = signal > 0 ? 1 : 0;
            break;
        case kind::SIGMOID:
            value = signal * (1 - signal);
            break;
    }

    return value;
}
