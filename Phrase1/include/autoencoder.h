#pragma once

#include "../include/tensor.h"
#include "../include/layers.h"
#include <string>

class Autoencoder {
private:
    Conv2D conv1, conv2, conv3, conv4, conv5;
    ReLU relu1, relu2, relu3, relu4, relu5;
    MaxPool2D pool;
    Upsample2D upsample;
    
public:
    Autoencoder();
    Tensor forward(const Tensor& input);
    void backward(const Tensor& grad_output, float learning_rate);
    Tensor extractFeatures(const Tensor& input);
    void saveWeights(const std::string& path);
    void loadWeights(const std::string& path);
};
