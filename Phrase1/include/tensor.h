#pragma once

#include <vector>

class Tensor {
public:
    std::vector<float> data;
    int batch, channels, height, width;
    
    Tensor();
    Tensor(int b, int c, int h, int w);
    
    float& at(int b, int c, int h, int w);
    const float& at(int b, int c, int h, int w) const;
    
    void fill(float value);
    void zeros();
    void randomize(float min_val, float max_val);
    int size() const;
};