#include <cassert>
#include <iostream>
#include "iris/dataset.h"

int main(void) {
    std::cout << "Testing dataset's ability to parse the Iris data... ";

    iris::dataset data("../../../data/iris.data");

    /**
     * expect the labels to be organized correclty and can be retrieved correctly
     */
    assert(data.label("Iris-setosa") == 0);
    assert(data.label("Iris-versicolor") == 1);
    assert(data.label("Iris-virginica") == 2);
    assert(data.name(0) == "Iris-setosa");
    assert(data.name(1) == "Iris-versicolor");
    assert(data.name(2) == "Iris-virginica");

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
