#ifndef ACTIVATIONFUNCTIONS_H
#define ACTIVATIONFUNCTIONS_H

#include <Eigen/Dense>

using namespace Eigen; 

namespace activationfunctions {
    // Sigmoid activation function
    // Input: Pre-activation vector x (z = W*a + b)
    // Output: Activation vector a (a = σ(z)), computing each component 1/(1 + exp(-z))
    inline VectorXd sigmoid(const VectorXd& x) {
        return 1.0 / (1.0 + (-x).array().exp());
    }

    inline VectorXd sigmoid_derivative(const VectorXd& x) {
        // Derivative of sigmoid function: σ'(x) = σ(x) * (1 - σ(x))
        return sigmoid(x).array() * (1.0 - sigmoid(x).array());
    }

    // Hyperbolic tangent activation function

    // Input: Pre-activation vector x (z = W*a + b)
    // Output: Activation vector a (a = tanh(z)), computing each component tanh(z)
    inline VectorXd tanh(const VectorXd& x) {
        return x.array().tanh();
    }
    
    inline VectorXd tanh_derivative(const VectorXd& x) {
        // Derivative of tanh function: tanh'(x) = 1 - tanh^2(x)
        return 1.0 - (tanh(x).array() * tanh(x).array());
    }

    // ReLU activation function
    inline VectorXd relu(const VectorXd& x) {
        // ReLU function: f(x) = max(0, x)
        return x.array().max(0);
    }

    inline VectorXd relu_derivative(const VectorXd& x) {
        // Derivative of ReLU function: f'(x) = 1 if x > 0 else 0
        // We can compute it using a conditional expression
        return (x.array() > 0).cast<double>();
    }

    inline double error_function(const VectorXd& output, const VectorXd& target) {
        // Mean Squared Error (MSE) function: E = 1/2 * ||output - target||^2
        return (output - target).squaredNorm() / 2.0;
    }

    inline VectorXd error_function_derivative(const VectorXd& output, const VectorXd& target) {
        // Derivative of the Mean Squared Error (MSE) function: ∂E/∂output = output - target
        return output - target;
    }
}

#endif 
