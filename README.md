# C++ Capstone: A Basic Feedforward Neural Network

This repo contains the code for the C++ capstone project for the Udacity C++ Nanodegree. I chose to build something to reinforce my knowledge of machine learning by implementing a neural network. I also wanted to practice modular C++ development, which is something I feel we did not get much of in the Nanodegree program. In the interest of time for the project, I chose to implement a basic multilayer perceptron/feedforward neural network.

The following resources were used as references in this implementation:
- [Artificial Nerual Networks](https://brilliant.org/wiki/artificial-neural-network/)
- [Feedforward Neural Networks](https://brilliant.org/wiki/feedforward-neural-networks/)
- [Multilayer Perceptron - Backpropagation](https://www.cse.unsw.edu.au/~cs9417ml/MLP2/BackPropagation.html)
- [Iris Data Set](http://archive.ics.uci.edu/ml/datasets/Iris)

## Expectations

This project build three things, a wrapper for working the the Iris data set. The wrapper is in the `iris::dataset` namespace. It loads up the Iris dataset and performs all the necessary parsing and encoding upon instantiation. It provides convenient methods to accessing and manipulating the data. By default, upon instantiation, the `dataset` class will always shuffle the Iris dataset and split it into two groups `training` and `testing`, of which 60% of the samples should fall under `training`. So each instantiation should provide a new ordering of the data, meaning testing and training should be "fresh" each time.

Running the `./capstone` executable will simply run through an automated process of loading the Iris dataset, setting up a feedforward neural network, training it and then finally testing it and display the results. By default, the only output will be results of the testing phase in the following format:

`Testing new flower: {flower name} ... {flower prediction} ({estimated confidence})`

`flower name` is the name of the Iris flower being tested. The network will output values for all three classes, and `flower prediction` is the output with the highest `estimated confidence`. `estimated confidence` is the network's estimate of how sure it is about its predictions.

Since the data is shuffled on each instantiation, running `./capstone` should result in a different output each time, while still performing the same automated load, train test process.

## Dependencies

* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Building

1. Clone this repo.
2. Make a build directory at the top level (same level as this README).
3. `cd` into the build directory and run `cmake .. && make`.
4. Run the program by executing `./capstone`.

## Testing

To test, you will need to `cd` into the respective `test` directories for the `iris` and `network` subdirectories and then run their executables locally from those directories. This ensures that hardcoded file paths to the `data` directory can be "found".
