/**
 * activation.h
 *
 * Declaration of free members within the network::activation namespace
 *
 * @author Victor Nwosu
 */

#ifndef VICTOR_NETWORK_ACTIVATION_H
#define VICTOR_NETWORK_ACTIVATION_H

namespace network {
    /**
     * Collection of activation members/functions
     */
    namespace activation {
        /**
         * enum struct kind the kind of activation
         */
        enum struct kind {
            RELU, // ReLU activation kind
            SIGMOID, // Sigmoid activation kind
        };

        /**
         * Compute the activation of an incoming signal using
         * a designated function
         *
         * @param const float signal the incoming signal
         * @param const kind func the activation function to compute
         *
         * @return float the activation output
         */
        float activate(const float signal, const kind func);

        /**
         * Compute the derivative of the incoming signal using
         * a designated function
         *
         * @param const float signal the incoming signal
         * @param const kind func the function to compute the derivative
         *
         * @return float the derivative of the incoming signal
         */
        float derivative(const float signal, const kind func);
    } // namespace activation
} // namespace network

#endif // VICTOR_NETWORK_ACTIVATION_H
