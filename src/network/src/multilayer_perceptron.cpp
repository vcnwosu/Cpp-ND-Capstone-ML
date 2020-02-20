/**
 * multilayer_perceptron.cpp
 *
 * Definitions for the multilayer perceptron architecture. It can be used to
 * construct an arbitrary number of hidden layers. The current implementation
 * only supports one activation type per layer. Also, by default, the final
 * layer is treated as the output layer.
 *
 * @author Victor Nwosu
 */

#include <iostream>
#include "network/cost.h"
#include "network/multilayer_perceptron.h"
#include "iris/dataset.h"

/**
 * The fully-qualified name for all methods can get long; too long
 * so I'll typedef it for this file
 */
typedef network::architecture::multilayer_perceptron mlp;

/**
 * Constructor
 */
mlp::multilayer_perceptron() {
    std::random_device rd;

    _gen = std::mt19937(rd());
    _dist = std::uniform_real_distribution<>(-1.0, 1.0);
}

/**
 * Add a new layer to the network
 *
 * @param const int nodes the number of nodes for this layer
 * @param const network::activation kind the activation kind
 */
void mlp::add_layer(const int nodes, const network::activation::kind kind) {
    _layers.emplace_back(std::vector<network::unit>(nodes, {kind}));
}

/**
 * Train the network using the provided input data. This method
 * expects the data to have been preproccessed/one-hot encoded.
 *
 * @param dataloader::dataset &data
 * @param int epochs the number of times to run the training for the set
 * @param float l_rate ratio used for applying the gradient
 * @param bool verbose to output progress to std::cout
 */
void mlp::train(iris::dataset &data, int epochs, float l_rate, bool verbose) {
    float error = 0.0f;
    float change = 0.0f;

    std::vector<std::vector<float>> &v = data.training;

    for (int i = 0; i < epochs; i++) {
        float curr_error = 0.0f;
        for (int j = 0; j < v.size(); j += 2) {
            std::vector<float> a(v[j].begin(), v[j].begin() + 4);

            if (verbose) {
                std::cout << "Training new example: " << data.name(v[j][4]) << " (";

                for (int y_val : data.expected_vector(v[j][4])) {
                    std::cout << y_val << ", ";
                }

                std::cout << ") ... ";
            }

            _feedforward(a);

            if (verbose) { std::cout << "("; }

            std::vector<float> y_hat;
            for (network::unit u : _layers[_layers.size() - 1]) {
                if (verbose) {
                    std::cout << u.activation << ", ";
                }

                y_hat.push_back(u.activation);
            }

            if (verbose) { std::cout << std::endl; }

            // float sum =  network::cost::cross_entropy(data.expected_vector(v[j][4]), y_hat);
            std::vector<float> e_vector = network::cost::simple_error(data.expected_vector(v[j][4]), y_hat);
            float sum = std::accumulate(e_vector.begin(), e_vector.end(), 0.0f);

            if (verbose) {
                for (auto f : e_vector) {
                    std::cout << f << ", ";
                }
                std::cout << std::endl;

                std::cout << "E: " << sum << std::endl;
            }

            if (verbose) {
                std::cout << "Error: " << sum << std::endl;
            }

            curr_error += sum;

            _backpropagate(a, e_vector, l_rate);

            verbose = false;
        }

        change = error == 0.0f ? curr_error : curr_error - error;
        error = curr_error; 

        if (verbose) {
            std::cout << "Epoch " << (i + 1) << ": Error: " << error << " Change: " << change << std::endl;
        }
    }
}

/**
 * Predict the label far a provided vector
 *
 * @param std::vector<float> &input
 *
 * @return std::vector<float> the prediction
 */
std::vector<float> mlp::predict(std::vector<float> &input) {
    std::vector<float> a(input.begin(), input.begin() + 4);
    _feedforward(a);

    std::vector<float> res;

    for (network::unit u : _layers[_layers.size() - 1]) {
        res.push_back(u.activation);
    }

    return res;
}

/**
 * Perform a one-time feedforward pass on an incoming vector
 *
 * @param std::vector<float> &v the incoming vector
 */
void mlp::_feedforward(std::vector<float> &v) {
    for (network::unit &u : _layers[0]) {
        float output = _dot(u.weights, v) + u.bias;
        u.activation = network::activation::activate(output, u.kind);
    }

    for (int i = 1; i < _layers.size(); i++) {
        for (network::unit &u : _layers[i]) {
            float output = _dot(u.weights, _layers[i - 1]) + u.bias;
            u.activation = network::activation::activate(output, u.kind);
        }
    }

/*    for (auto layer : _layers) {
        std::cout << "Layer: ";
        for (auto u : layer) {
            std::cout << u.activation << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
*/}


/**
 * Perform a one-time backpropagation after a feedforward pass
 *
 * @param const std::vector<float> &v the vector used during feedforwarding
 * @param float error the cross entropy error
 * @param float l_rate the learning rate with which to apply the gradient
 */
void mlp::_backpropagate(std::vector<float> &v, float error, float l_rate) {
    for (int j = _layers.size() - 1; j > -1; j--) {
        for (int i = 0; i < _layers[j].size(); i++) {
            if (j < _layers.size() - 1) {
                float error = 0.0f;

                for (int k = 0; k < _layers[j + 1].size(); k++) {
                    error += _layers[j + 1][k].weights[i] * _layers[j + 1][k].gradient;
                }
            }

            _layers[j][i].gradient = error * network::activation::derivative(_layers[j][i].activation, _layers[j][i].kind);
        }
    }

    for (int j = 0; j < _layers.size(); j++) {
        for (int i = 0; i < (j == 0 ? v.size() : _layers[j].size()); i++) {
            for (int k = 0; k < _layers[j][i].weights.size(); k++) {
                _layers[j][i].weights[i] += l_rate * _layers[j][i].gradient * (j == 0 ? v[i] : _layers[j - 1][i].activation);
            }

            // _layers[j][i].bias += l_rate * _layers[j][i].gradient;
        }
    }
}

/**
 * Perform a one-time backpropagation after a feedforward pass
 *
 * @param const std::vector<float> &v the vector used during feedforwarding
 * @param float error the cross entropy error
 * @param float l_rate the learning rate with which to apply the gradient
 */
void mlp::_backpropagate(std::vector<float> &v, std::vector<float> &e, float l_rate) {
    for (int j = _layers.size() - 1; j > -1; j--) {
        for (int i = 0; i < _layers[j].size(); i++) {
            float error = 0.0f;

            if (j == _layers.size() - 1) {
                error = e[i];
            } else if (j < _layers.size() - 1) {
                for (int k = 0; k < _layers[j + 1].size(); k++) {
                    error += _layers[j + 1][k].weights[i] * _layers[j + 1][k].gradient;
                }
            }

            _layers[j][i].gradient = error * network::activation::derivative(_layers[j][i].activation, _layers[j][i].kind);
        }
    }

    for (int j = 0; j < _layers.size(); j++) {
        for (int i = 0; i < (j == 0 ? v.size() : _layers[j].size()); i++) {
            for (int k = 0; k < _layers[j][i].weights.size(); k++) {
                _layers[j][i].weights[i] += l_rate * _layers[j][i].gradient * (j == 0 ? v[i] : _layers[j - 1][i].activation);
            }

            _layers[j][i].bias += l_rate * _layers[j][i].gradient;
        }
    }
}

/**
 * Compute the dot product on two vectors
 *
 * @param std::vector<float> &w the weight vector of a network::unit
          in the _layers matrix
 * @param std::vector<float> &x an incoming vector
 *
 * @return float the dot product
 */
float mlp::_dot(std::vector<float> &w, std::vector<float> &x) {
    bool is_empty = w.empty();
    float dot_product = 0.0f;

    for (int i = 0; i < x.size(); i++) {
        if (is_empty) {
            w.push_back(_dist(_gen));
        }

        dot_product += w[i] * x[i]; 
    }

    return dot_product;
}

/**
 * Compute the dot product on a network::unit, using a layer as
 * an incoming vector
 *
 * @param std::vector<float> &w the weight vector of a network::unit
 * @param std::vector<network::unit> &x a network layer as an incoming vector
 *
 * @return float the dot product
 */
float mlp::_dot(std::vector<float> &w, std::vector<network::unit> &x) {
    bool is_empty = w.empty();
    float dot_product = 0.0f;

    for (int i = 0; i < x.size(); i++) {
        if (is_empty) {
            w.push_back(_dist(_gen));
        }

        dot_product += w[i] * x[i].activation;
    }

    return dot_product;
}
