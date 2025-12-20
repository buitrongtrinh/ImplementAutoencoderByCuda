#include "../include/tensor.h"
#include <algorithm>
#include <random>

Tensor::Tensor() : batch(0), channels(0), height(0), width(0) {}

Tensor::Tensor(int b, int c, int h, int w) 
    : batch(b), channels(c), height(h), width(w) {
    data.resize(b * c * h * w, 0.0f);
}

float& Tensor::at(int b, int c, int h, int w) {
    return data[b * channels * height * width + 
               c * height * width + 
               h * width + w];
}

const float& Tensor::at(int b, int c, int h, int w) const {
    return data[b * channels * height * width + 
               c * height * width + 
               h * width + w];
}

void Tensor::fill(float value) {
    std::fill(data.begin(), data.end(), value);
}

void Tensor::zeros() { 
    fill(0.0f); 
}

void Tensor::randomize(float min_val, float max_val) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);
    for (auto& val : data) {
        val = dis(gen);
    }
}

int Tensor::size() const { 
    return data.size(); 
}