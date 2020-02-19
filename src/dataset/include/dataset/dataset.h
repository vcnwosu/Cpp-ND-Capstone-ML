/**
 * dataset.h
 *
 * Declaration of dataset class to load the Iris dataset
 *
 * @author Victor Nwosu
 */

#ifndef VICTOR_DATASET_H
#define VICTOR_DATASET_H

#include <string>
#include <vector>

/**
 * class dataset
 */
class dataset {
    public:
        /**
         * Constructor
         *
         * @param std::string filename the path to the file
         */
        dataset(std::string filename);

        /**
         * @var std::vector<std::vector<float>> training the dataset that can be
         *      for training against the Iris dataset
         */
        std::vector<std::vector<float>> training;

        /**
         * @var std::vector<std::vector<float>> testing the dataset to test
         */
        std::vector<std::vector<float>> testing;

        /**
         * @var std::vector<std::vector<int>> labels the labels of the dataset
         */
        std::vector<std::vector<int>> labels;
}; // class dataset

#endif // VICTOR_DATASET_H
