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
#include <map>

namespace iris {
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
             * Get the label name for a particular observation
             * ex. name(0) -> "Iris-setosa"
             *
             * @param int observation label
             */
            std::string name(int label);

            /**
             * Get label code for an Iris example name
             * ex. label("Iris-setosa") -> 0
             *
             * @param std::string name
             *
             * @return int the label code
             */
            int label(std::string name);

            /**
             * Get the expected output vector for a label, using either the
             * name or the label code
             *
             * @param int label | std::string name
             *
             * @return std::vector<int> the label vector
             */
            std::vector<int> expected_vector(int label);
            std::vector<int> expected_vector(std::string name);
        private:
            /**
             * @var std::map<int, std::string> _labels the labels and names for
             *      the Iris dataset
             */
            std::map<int, std::string> _labels;

            /**
             * @var std::vector<std::vector<int>> _expected_vectors for the labels
             */
            std::vector<std::vector<int>> _expected_vectors;
    }; // class dataset
} // namespace iris

#endif // VICTOR_DATASET_H
