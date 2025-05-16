#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <iostream>
#include <Eigen/Dense>
#include <functional>

#include "layer.h"
#include "activationfunctions.h"

using namespace Eigen;

class NeuralNetwork {
private: 
    std::vector<Layer> layers; // Layers of the neural network
public:
    // Construct from list of layers (must be non-empty) 
    explicit NeuralNetwork(std::vector<Layer> layers) : layers(std::move(layers)) 
    {
        if (this->layers.empty()) {
            throw std::invalid_argument("Neural network must have at least one layer.");
        }
    } // Constructor

    void forward(const VectorXd& input) {
        // First layer consumes raw input 
        layers[0].forward(input); // Forward pass through the first layer
        // Forward pass through all remaining layers 
        for (int i = 1; i < layers.size(); ++i) {
            layers[i].forward(layers[i - 1].getNeuronValuesActivate());
        }
    }

    VectorXd getOutput() const {
        // Get the output of the last layer
        return layers.back().getNeuronValuesActivate();
    }

    void backpropagation(VectorXd& input, VectorXd& targetOutputs, double learningRate) {
        // Calculate the output error and delta for the last layer
        VectorXd outputError = activationfunctions::error_function_derivative(getOutput(), targetOutputs); 
        VectorXd outputDerivative = layers.back().derivative_of_activation_function();
        VectorXd delta = outputError.cwiseProduct(outputDerivative);
        layers.back().set_delta(delta);

        // Backpropagate the error through the network
        for (int i = static_cast<int>(layers.size()) - 2; i >= 0; i--) {
            MatrixXd next_weights = layers[i + 1].getWeights();
            VectorXd next_delta = layers[i + 1].get_delta();
            VectorXd hiddenError = next_weights * next_delta;
            VectorXd hiddenDerivative = layers[i].derivative_of_activation_function();
            VectorXd hiddenDelta = hiddenError.cwiseProduct(hiddenDerivative);
            layers[i].set_delta(hiddenDelta);
        }

        // Update weights for each layer 
        for (int i = 0; i < layers.size(); ++i) {
            VectorXd input_; 
            if (i == 0) {
                input_ = input;
            } else {
                input_ = layers[i - 1].getNeuronValuesActivate();
            }

            layers[i].updateWeights(input_, learningRate);
        }
    }
    
    // Train the neural network
    void train(const std::vector<VectorXd>& inputs, const std::vector<VectorXd>& targetOutputs, double learningRate, int epochs) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            double epochLoss = 0.0; 
            for (size_t i = 0; i < inputs.size(); ++i) {
                VectorXd input = inputs[i]; 
                VectorXd target = targetOutputs[i];
                forward(input); // Forward pass
                VectorXd output = getOutput(); // Get output
                double loss = activationfunctions::error_function(output, target); // Compute loss
                epochLoss += loss; // Accumulate loss
                backpropagation(input, target, learningRate); // Backpropagation
            }
            epochLoss /= inputs.size(); // Average loss for the epoch
            std::cout << "Epoch " << epoch + 1 << "/" << epochs << ", Loss: " << epochLoss << std::endl; // Print loss
        }
    }

    void test(const std::vector<VectorXd>& inputs, const std::vector<VectorXd>& targetOutputs) {
        int correctPredictions = 0; // Counter for correct predictions
        for (size_t i = 0; i < inputs.size(); ++i) {
            VectorXd input = inputs[i]; // Input vector
            VectorXd targetOutput = targetOutputs[i]; // Target vector
            forward(input); // Forward pass
            VectorXd output = getOutput(); // Get output
            int predictedIndex; 
            int targetIndex; 
            output.maxCoeff(&predictedIndex); // Get index of max value in output
            targetOutput.maxCoeff(&targetIndex); // Get index of max value in target
            if (predictedIndex == targetIndex) {
                correctPredictions++; // Increment counter if prediction is correct
            }
        }

        double accuracy = static_cast<double>(correctPredictions) / inputs.size(); // Calculate accuracy
        std::cout << "Accuracy: " << accuracy * 100.0 << "%" << std::endl; // Print accuracy

    }
}; 

#endif // NEURAL_NETWORK_H






