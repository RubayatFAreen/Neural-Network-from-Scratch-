# C++ Neural Network From Scratch

&#x20;

A comprehensive, header-only C++ implementation of a fully-connected neural network—built from first principles, using Eigen for linear algebra, and trained end-to-end on the MNIST dataset. Perfect for learning the nuts-and-bolts of forward/backpropagation, vectorized operations, and network architecture.

---

## 🚀 Table of Contents

* [✨ Features](#✨-features)
* [📐 Architecture](#📐-architecture)
* [⚙️ Prerequisites](#⚙️-prerequisites)
* [🔧 Installation & Build](#🔧-installation--build)
* [🎯 Usage](#🎯-usage)
* [⚙️ Configuration](#⚙️-configuration)
* [📊 Performance](#📊-performance)
* [📁 Project Structure](#📁-project-structure)
* [🛠️ Roadmap](#🛠️-roadmap)
* [🤝 Contributing](#🤝-contributing)
* [📄 License](#📄-license)

---

## ✨ Features

* **From-scratch implementation** of forward propagation, backpropagation, and gradient descent in modern C++
* **Layer abstraction**: easily stack `Layer` objects with pluggable activation functions
* A suite of **activation functions**: Sigmoid, ReLU, Tanh, and their derivatives
* **Mean Squared Error** loss, with hooks to swap in Cross-Entropy + Softmax
* **MNIST loader** utilities (train/test data & one-hot labels) for hands-on experimentation
* **No external dependencies** besides [Eigen](https://eigen.tuxfamily.org) (header-only)
* **Configurable** hyperparameters: learning rate, epochs, hidden size
* **Training & testing** pipelines with console logging for loss and accuracy

---

## 📐 Architecture

This project is organized around three core components:

1. \`\`: Represents a fully-connected layer, encapsulating weights, biases, forward pass, and local δ computation for backprop.
2. \`\`: Manages a sequence of `Layer`s, implements global forward/backward loops, training and evaluation routines.
3. \`\`: MNIST-specific readers that parse the IDX binary format into `Eigen::VectorXd` and perform normalization & one-hot encoding.

Key algorithms:

* **Forward pass**: $a^0 = x,$
  $z^l = W^l a^{\,l-1} + b^l,$
  $a^l = \sigma(z^l)$ or ReLU/Tanh as configured

* **Backpropagation**: $\delta^L = (a^L - y) \odot \sigma'(z^L)$
  $\delta^l = (W^{l+1})^T \delta^{l+1} \odot \sigma'(z^l)$

* **Updates**: $W^l \gets W^l - \eta\, \delta^l (a^{\,l-1})^T,$
  $b^l \gets b^l - \eta\, \delta^l.$

---

## ⚙️ Prerequisites

* A C++17-compatible compiler (MSVC, g++, or clang++)
* [Eigen 3](https://gitlab.com/libeigen/eigen) (download and unzip)
* CMake (optional, for multi-file projects)

---

## 🔧 Installation & Build

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/cpp-neural-network.git
cd cpp-neural-network
```

### 2. Install Eigen

Unzip Eigen to a folder, e.g., `C:/libs/eigen-3.4.0` (Windows) or `/usr/local/include/eigen-3.4.0` (Linux/macOS).

### 3A. Build with MSVC (Developer Command Prompt)

```bat
cd "C:\path\to\cpp-neural-network"
cl /EHsc /std:c++17 /I "C:\libs\eigen-3.4.0" main.cpp
main.exe
```

### 3B. Build with g++

```bash
cd /path/to/cpp-neural-network
g++ -std=c++17 -I "C:/libs/eigen-3.4.0" main.cpp -o nn
./nn
```

### 3C. (Optional) Build with CMake

```bash
mkdir build && cd build
cmake .. -DEIGEN3_INCLUDE_DIR="C:/libs/eigen-3.4.0"
cmake --build .
./NeuralNetwork
```

---

## 🎯 Usage

After building, simply run the executable. The program will:

1. Load MNIST train/test images & labels
2. Construct a 784→64→10 sigmoid network
3. Train for 3 epochs (configurable)
4. Print per-epoch loss and final test accuracy

```bash
./nn  # or main.exe on Windows
```

Sample output:

```
Epoch 1/3 | Loss: 1.2345
Epoch 2/3 | Loss: 0.5678
Epoch 3/3 | Loss: 0.3456
Accuracy: 92.30%
```

---

## ⚙️ Configuration

You can tweak:

* **Hidden layer size**: adjust `hiddenSize` in `main.cpp`
* **Learning rate** & **epochs**: passed to `nn.train(...)`
* **Activation functions**: swap `sigmoid` for `relu` or `tanh` in layer construction
* **Batch size** & **optimizers**: extend `NeuralNetwork` with mini-batching or Adam optimizer

---

## 📊 Performance

| Hidden Size | Learning Rate | Epochs | Accuracy |
| ----------- | ------------- | ------ | -------- |
| 64          | 0.1           | 3      | \~92.3%  |
| 128         | 0.05          | 5      | \~94.1%  |

*Tip: Increasing epochs and switching to cross-entropy + softmax can push >97% on MNIST.*

---

## 📁 Project Structure

```
cpp-neural-network/
├─ include/
│   ├─ Layer.h
│   ├─ activationfunctions.h
│   ├─ neural_network.h
│   └─ utils.h
├─ src/
│   └─ main.cpp
├─ CMakeLists.txt       # optional
├─ data/                # MNIST .idx files (not checked in)
└─ README.md
```

---

## 🛠️ Roadmap

*

## 🤝 Contributing

Contributions are welcome! Feel free to:

* Open issues for bugs or feature requests
* Submit pull requests with enhancements
* Suggest optimizations, new layers, or training tricks

Please respect the [Code of Conduct](CODE_OF_CONDUCT.md).

---

## 📄 License

Distributed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

*Crafted with ❤️ to demystify neural networks in C++.*
