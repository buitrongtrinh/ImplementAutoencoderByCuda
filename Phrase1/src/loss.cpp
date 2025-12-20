#include "../include/loss.h"
#include <algorithm>

float MSELoss::forward(const Tensor& output, const Tensor& target) {
    float sum = 0.0f;
    for (size_t i = 0; i < output.data.size(); i++) {
        float diff = output.data[i] - target.data[i];
        sum += diff * diff;
    }
    return sum / output.data.size();
}

Tensor MSELoss::backward(const Tensor& output, const Tensor& target) {
    Tensor grad = output;
    float scale = 2.0f / output.data.size();
    
    for (size_t i = 0; i < output.data.size(); i++) {
        grad.data[i] = scale * (output.data[i] - target.data[i]);
    }
    
    return grad;
}