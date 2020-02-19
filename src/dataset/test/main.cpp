#include <cassert>
#include <iostream>
#include "dataset/dataset.h"

int main(void) {
    std::cout << "Testing dataset's ability to parse the Iris data... ";

    dataset data("../../../data/iris.data");

    /**
     * there should be a total of 150 testing and training examples
     */
    assert(data.training.size() + data.testing.size() == 150);

    /**
     * An example from data and testing should have five elements
     */
    assert(data.training[0].size() == 5);
    assert(data.testing[0].size() == 5);

    /**
     * training data size should be more than testing
     */
    assert(data.training.size() > data.testing.size());
    assert((150 * 0.60) <= data.training.size());
    assert((150 * 0.40) >= data.testing.size());

    std::cout << " [PASS] " << std::endl;

    return 0;
}
