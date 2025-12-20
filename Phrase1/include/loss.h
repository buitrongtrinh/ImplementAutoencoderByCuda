#pragma once

#include "../include/tensor.h"

class MSELoss {
public:
    float forward(const Tensor& output, const Tensor& target);
    Tensor backward(const Tensor& output, const Tensor& target);
};