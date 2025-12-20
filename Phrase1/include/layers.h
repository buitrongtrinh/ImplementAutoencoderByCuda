#pragma once

#include "../include/tensor.h"

class Conv2D {
private:
    Tensor weights, bias, grad_weights, grad_bias;
    Tensor last_input;
    int in_channels, out_channels, kernel_size, stride, padding;
    
public:
    Conv2D(int in_ch, int out_ch, int k_size = 3, int s = 1, int p = 1);
    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& grad_output);
    void updateWeights(float learning_rate);
    
    Tensor& getWeights() { return weights; }
    Tensor& getBias()    { return bias; }
    const Tensor& getWeights() const { return weights; }
    const Tensor& getBias() const { return bias; }
};

class ReLU {
private:
    Tensor last_input;
    
public:
    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& grad_output);
};

class MaxPool2D {
private:
    Tensor last_input, mask;
    int pool_size, stride;
    
public:
    MaxPool2D(int p_size = 2, int s = 2);
    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& grad_output);
};

class Upsample2D {
private:
    int scale_factor;
    
public:
    Upsample2D(int scale = 2);
    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& grad_output);
};