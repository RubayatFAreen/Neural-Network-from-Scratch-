#ifndef LAYER_H
#define LAYER_H

#include <iostream>
#include <Eigen/Dense>
#include <functional>

#include "activationfunctions.h"

class Layer {
private:
    MatrixXd weights; // Weights of the layer
    VectorXd biases; // Biases of the layer
    VectorXd neuronValues; // Values of the neurons in the layer
    VectorXd neuron_values_activate; // Activated values of the neurons in the layer
    std::function<VectorXd(const VectorXd&)> activationFunction; // Activation function
    std::function<VectorXd(const VectorXd&)> activationFunctionDerivative; // Derivative of the activation function
    VectorXd delta; // Delta values for backpropagation
public: 
    // Constructor
    Layer(int inputSize, int neurons, std::function<VectorXd(const VectorXd&)> activationFunction, std::function<VectorXd(const VectorXd&)> activationFunctionDerivative) {
        this->activationFunction = activationFunction;
        this->activationFunctionDerivative = activationFunctionDerivative;
        weights = MatrixXd::Random(inputSize, neurons);
        biases = VectorXd::Zero(neurons);
        neuronValues = VectorXd::Zero(neurons);
        neuron_values_activate = VectorXd::Zero(neurons);
        delta = VectorXd::Zero(neurons);
    }

    // Forward pass
    void forward(const VectorXd& input) {
        // Compute the weighted sum of inputs and biases
        neuronValues = (weights.transpose() * input + biases);
        // Apply the activation function
        neuron_values_activate = activationFunction(neuronValues);
    }

    VectorXd getNeuronValues() const {
        return neuronValues;
    }

    VectorXd getNeuronValuesActivate() const {
        return neuron_values_activate;
    }

    void set_delta(const VectorXd& delta) {
        this->delta = std::move(delta);
    }

    VectorXd get_delta() const {
        return delta;
    }

    MatrixXd getWeights() const {
        return weights;
    }   

    void updateWeights(const VectorXd& input, double learningRate) {
        // Update weights using the delta and the input
        weights -= learningRate * (input * delta.transpose());
        // Update biases using the delta
        biases -= learningRate * delta;
    }

    VectorXd derivative_of_activation_function() const {
        // Compute the derivative of the activation function
        return activationFunctionDerivative(neuronValues);
    }

}; 

#endif