/**
 * dataset.cpp
 *
 * Definition of dataset class to load the Iris dataset
 *
 * @author Victor Nwosu
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include "iris/dataset.h"

/**
 * Constructor
 */
iris::dataset::dataset(std::string filename) {
    std::ifstream file(filename);
    std::string line;

    _expected_vectors.emplace_back(std::vector<int>{1, 0, 0}); // Iris-setosa
    _expected_vectors.emplace_back(std::vector<int>{0, 1, 0}); // Iris-veriscolor
    _expected_vectors.emplace_back(std::vector<int>{0, 0, 1}); // Iris-virginica

    _labels[0] = "Iris-setosa";
    _labels[1] = "Iris-versicolor";
    _labels[2] = "Iris-virginica";

    std::vector<std::vector<float>> observations;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<float> v;
        std::string cell;

        while (std::getline(ss, cell, ',')) {
            try {
                float val = std::stof(cell);
                v.push_back(val);
            } catch (std::invalid_argument e) {
                v.push_back(label(cell));
            }
        }

        if (!v.empty()) {
            observations.push_back(std::move(v));
        }
    }

    std::random_device rd;
    std::mt19937 rands(rd());

    std::shuffle(observations.begin(), observations.end(), rands);

    int sixty_percent = observations.size() * 0.60;

    std::vector<std::vector<float>> train(observations.begin(), observations.begin() + sixty_percent);
    std::vector<std::vector<float>> test(observations.begin() + sixty_percent, observations.end());

    for (std::vector<float> ex : train) {
        training.push_back(std::move(ex));
    }

    for (std::vector<float> ex : test) {
        testing.push_back(std::move(ex));
    }

    file.close();
}

/**
 * Get the label name for a particular observation
 * ex. name(0) -> "Iris-setosa"
 *
 * @param int observation label
 */
std::string iris::dataset::name(int label) {
    return _labels[label];
}

/**
 * Get label code for an Iris example name
 * ex. label("Iris-setosa") -> 0
 *
 * @param std::string name
 *
 * @return int the label code
 */
int iris::dataset::label(std::string name) {
    int res;

    for (std::pair<int, std::string> el : _labels) {
        if (el.second == name) {
            res = el.first;
            break;
        }
    }

    return res;
}

/**
 * Get the expected output vector for a label, using either the
 * name or the label code
 *
 * @param int label | std::string name
 *
 * @return std::vector<int> the label vector
 */
std::vector<int> iris::dataset::expected_vector(int label) {
    return _expected_vectors[label];
}

std::vector<int> iris::dataset::expected_vector(std::string name) {
    return _expected_vectors[label(name)];
}
