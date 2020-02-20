/**
 * multilayer_perceptron.h
 *
 * Declaration of the multilayer perceptron architecture. It can be used to
 * construct an arbitrary number of hidden layers. The current implementation
 * only supports one activation type per layer. Also, by default, the final
 * layer is treated as the output layer.
 *
 * @author Victor Nwosu
 */

#ifndef VICTOR_NETWORK_MULTILAYER_PERCEPTRON_H
#define VICTOR_NETWORK_MULTILAYER_PERCEPTRON_H

#include <random>
#include <vector>
#include <functional>
#include "network/unit.h"
#include "iris/dataset.h"

namespace network {
    /**
     * the architecture namespace
     */
    namespace architecture {
        /**
         * class multilayer_perceptron
         */
        class multilayer_perceptron {
            public:
                /**
                 * Constructor
                 */
                multilayer_perceptron();

                /**
                 * Add a new layer to the network.
                 *
                 * @param const int nodes the number of nodes for this layer
                 * @param const network::activation kind the activation kind
                 */
                void add_layer(const int nodes, const network::activation::kind kind);

                /**
                 * Train the network using the provided input data. This method
                 * expects the data to have been preproccessed/one-hot encoded.
                 *
                 * @param std::vector<std::vector<float>>
                 */
                void train(iris::dataset &data, int epochs, float l_rate, bool verbose);

                /**
                 * Predict the label far a provided vector
                 *
                 * @param std::vector<float> &input
                 *
                 * @return float the prediction
                 */
                std::vector<float> predict(std::vector<float> &input);
            private:
                /**
                 * @var std::mt19937 _gen to generate random numbers for weights
                 */
                std::mt19937 _gen;

                /**
                 * @var std::uniform_real_distribution<> _dist to generate
                 *      random numbers within a uniform distribution
                 */
                std::uniform_real_distribution<> _dist;

                /**
                 * @var std::vector<std::vector<network::unit>> _layers
                 *      a structure to hold computation data for the layers
                 */
                std::vector<std::vector<network::unit>> _layers;

                /**
                 * Perform a one-time feedforward pass on an incoming vector
                 *
                 * @param std::vector<float> &v the incoming vector
                 * @param bool training flag the execution as training or not
                 */
                void _feedforward(std::vector<float> &v);

                /**
                 * Perform a one-time backpropagation after a feedforward pass
                 *
                 * @param std::vector<float> &v the vector used during
                 *        feedforwarding
                 * @param float error the cross entropy error
                 * @param float l_rate the learning rate with which to apply
                 *        the gradient
                 */
                void _backpropagate(std::vector<float> &v, float error, float l_rate);
                void _backpropagate(std::vector<float> &v, std::vector<float> &e, float l_rate);

                /**
                 * Compute the dot product on two vectors
                 *
                 * @param std::vector<float> &w the weight vector of a
                 *        network::unit within the _layers matrix
                 * @param std::vector<float> &v an incoming vector
                 *
                 * @return float the dot product
                 */
                float _dot(std::vector<float> &w, std::vector<float> &v);

                /**
                 * Compute the dot product on a network::unit using a layer as
                 * an incoming vector
                 *
                 * @param std::vector<float> &w the weight vector of a
                 *        network::unit
                 * @param std::vector<network::unit> &x a network layer as an
                 *        incoming vector
                 */
                float _dot(std::vector<float> &w, std::vector<network::unit> &x);
        }; // class multilayer_perceptron
    } // namespace architecture
} // namespace network

#endif // VICTOR_NETWORK_MULTILAYER_PERCEPTRON_H
