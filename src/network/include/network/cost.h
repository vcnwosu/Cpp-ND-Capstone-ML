/**
 * cost.h
 *
 * The declaration for miscellaneous free members/functions
 * within the network::cost namespace
 *
 * @author Victor Nwosu
 */

#ifndef VICTOR_NETWORK_COST_H
#define VICTOR_NETWORK_COST_H

#include <vector>

namespace network {
    /**
     * Collection of cost members/functions for the network
     */
    namespace cost {
        /**
         * Compute the cost function for the network using
         * cross entropy for two classes
         *
         * @param const float y the known label for the input
         * @param const float y_hat the probability predicted by the network
         *
         * @return float the cross entropy result
         */
        float binary_cross_entropy(const float y, const float y_hat);

        /**
         * Compute the cost function for the network using
         * cross entropy for multiple classes
         *
         * @param const std::vector<float> &y the known labels
         * @param const std::vector<float> &y_hat the network output
         *
         * @return float the cross entropy result
         */
        float cross_entropy(const std::vector<float> &y, const std::vector<float> &y_hat);
    } // namespace cost
} // namspace network

#endif // VICTOR_NETWORK_COST_H
