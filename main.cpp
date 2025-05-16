// main.cpp
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <cstdlib>    // std::srand, std::rand
#include <ctime>      // std::time


#include "layer.h"
#include "activationfunctions.h"
#include "neural_network.h"
#include "utils.h"

// bring in just the names you need
using Eigen::VectorXd;
using activationfunctions::sigmoid;
using activationfunctions::sigmoid_derivative;

int main() {
    // Set random seed for reproducibility
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    // 1) Load MNIST data
    std::vector<VectorXd> trainData, trainLabels;
    std::vector<VectorXd> testData,  testLabels;

    // Paths to MNIST data files
    // Note: Adjust these paths according to your system
    const std::string trainDataPath = "C:\\Neural Network C++\\data\\MNIST\\raw\\train-images.idx3-ubyte";
    const std::string trainLabelsPath = "C:\\Neural Network C++\\data\\MNIST\\raw\\train-labels.idx1-ubyte";
    const std::string testDataPath =  "C:\\Neural Network C++\\data\\MNIST\\raw\\t10k-images.idx3-ubyte";
    const std::string testLabelsPath = "C:\\Neural Network C++\\data\\MNIST\\raw\\t10k-labels.idx1-ubyte";
    
    // Load training and test data
    utils::read_mnist_train_data(trainDataPath, trainData);
    utils::read_mnist_train_label(trainLabelsPath, trainLabels);
    utils::read_mnist_test_data(testDataPath,  testData);
    utils::read_mnist_test_label(testLabelsPath,  testLabels);

    if (trainData.empty() || trainLabels.empty()) {
        std::cerr << "Error loading MNIST training data.\n";
        return 1;
    }

    // 2) Build layers & network
    int inputSize  = static_cast<int>(trainData[0].size());
    int hiddenSize = 64;
    int outputSize = static_cast<int>(trainLabels[0].size());

    Layer hiddenLayer(inputSize,  hiddenSize, sigmoid, sigmoid_derivative);
    Layer outputLayer(hiddenSize, outputSize, sigmoid, sigmoid_derivative);

    // Initialize weights and biases
    NeuralNetwork nn({ hiddenLayer, outputLayer });

    // 3) Train & test
    const double learningRate = 0.1;
    const int    epochs       = 3;

    nn.train(trainData, trainLabels, learningRate, epochs);
    nn.test( testData,  testLabels);

    return 0;
}
