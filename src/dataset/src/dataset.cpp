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
#include "dataset/dataset.h"

/**
 * Constructor
 */
dataset::dataset(std::string filename) {
    std::ifstream file(filename);
    std::string line;

    std::vector<std::vector<float>> observations;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<float> v;
        std::string cell;

        while (std::getline(ss, cell, ',')) {
            if (cell == "Iris-setosa") {
                v.push_back(0.0f);
            } else if (cell == "Iris-versicolor") {
                v.push_back(1.0f);
            } else if (cell == "Iris-virginica") {
                v.push_back(2.0f);
            } else if (!cell.empty()) {
                v.push_back(std::stof(cell));
            }
        }

        if (!v.empty()) {
            observations.push_back(std::move(v));
        }
    }

    labels.emplace_back(std::vector<int>{1, 0, 0}); // Iris-setosa
    labels.emplace_back(std::vector<int>{0, 1, 0}); // Iris-veriscolor
    labels.emplace_back(std::vector<int>{0, 0, 1}); // Iris-verginica

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
