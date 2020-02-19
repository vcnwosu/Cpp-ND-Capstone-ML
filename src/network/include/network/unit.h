/**
 * unit.h
 *
 * Declaration of the structure for a universal approximation unit.
 *
 * @author Victor Nwosu
 */

#ifndef VICTOR_NETWORK_UNIT_H
#define VICTOR_NETWORK_UNIT_H

#include <vector>
#include "network/activation.h"

namespace network {
    /**
     * struct network::unit 
     */
    struct unit {
        /**
         * @var network::activation kind the type of
         *      universal approximation unit
         */
        network::activation::kind kind;
 
        /**
         * @var float activation the latest output from the unit
         */
        float activation;

        /**
         * @var float gradient the amount to change the weights/bias
         */
        float gradient;

        /**
         * @var float bias the bias for the unit
         */
        float bias;

        /**
         * @var std::vector<float> weights the unit's own set of weights
         */
        std::vector<float> weights;
    }; // class unit
} // namepace network

#endif // VICTOR_NETWORK_UNIT_H
