#pragma once

#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>
#include <fstream>
#include <cstdint>

#include <Eigen/Dense>

namespace utils {

// Helper: read a big-endian unsigned 32-bit integer
inline uint32_t readBigEndianUInt32(std::ifstream& file) {
    unsigned char bytes[4];
    file.read(reinterpret_cast<char*>(bytes), 4);
    return (static_cast<uint32_t>(bytes[0]) << 24) |
           (static_cast<uint32_t>(bytes[1]) << 16) |
           (static_cast<uint32_t>(bytes[2]) << 8)  |
           (static_cast<uint32_t>(bytes[3]));
}

// Read MNIST training images (IDX3 format)
inline void read_mnist_train_data(const std::string& path,
                                  std::vector<Eigen::VectorXd>& data) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return;
    uint32_t magic    = readBigEndianUInt32(file);
    uint32_t count    = readBigEndianUInt32(file);
    uint32_t rows     = readBigEndianUInt32(file);
    uint32_t cols     = readBigEndianUInt32(file);
    data.clear(); data.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        Eigen::VectorXd vec(rows * cols);
        for (uint32_t r = 0; r < rows; ++r) {
            for (uint32_t c = 0; c < cols; ++c) {
                unsigned char px;
                file.read(reinterpret_cast<char*>(&px), 1);
                vec[r * cols + c] = static_cast<double>(px) / 255.0;
            }
        }
        data.push_back(std::move(vec));
    }
}

// Read MNIST training labels (IDX1 format) and one-hot encode
inline void read_mnist_train_label(const std::string& path,
                                   std::vector<Eigen::VectorXd>& data) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return;
    uint32_t magic = readBigEndianUInt32(file);
    uint32_t count = readBigEndianUInt32(file);
    data.clear(); data.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        unsigned char lbl;
        file.read(reinterpret_cast<char*>(&lbl), 1);
        Eigen::VectorXd vec = Eigen::VectorXd::Zero(10);
        vec[static_cast<int>(lbl)] = 1.0;
        data.push_back(std::move(vec));
    }
}

// Read MNIST test images (IDX3 format)
inline void read_mnist_test_data(const std::string& path,
                                 std::vector<Eigen::VectorXd>& data) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return;
    uint32_t magic    = readBigEndianUInt32(file);
    uint32_t count    = readBigEndianUInt32(file);
    uint32_t rows     = readBigEndianUInt32(file);
    uint32_t cols     = readBigEndianUInt32(file);
    data.clear(); data.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        Eigen::VectorXd vec(rows * cols);
        for (uint32_t r = 0; r < rows; ++r) {
            for (uint32_t c = 0; c < cols; ++c) {
                unsigned char px;
                file.read(reinterpret_cast<char*>(&px), 1);
                vec[r * cols + c] = static_cast<double>(px) / 255.0;
            }
        }
        data.push_back(std::move(vec));
    }
}

// Read MNIST test labels (IDX1 format) and one-hot encode
inline void read_mnist_test_label(const std::string& path,
                                  std::vector<Eigen::VectorXd>& data) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return;
    uint32_t magic = readBigEndianUInt32(file);
    uint32_t count = readBigEndianUInt32(file);
    data.clear(); data.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        unsigned char lbl;
        file.read(reinterpret_cast<char*>(&lbl), 1);
        Eigen::VectorXd vec = Eigen::VectorXd::Zero(10);
        vec[static_cast<int>(lbl)] = 1.0;
        data.push_back(std::move(vec));
    }
}

} // namespace utils

#endif // UTILS_H
