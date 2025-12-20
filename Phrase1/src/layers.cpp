#include "../include/layers.h"
#include <cmath>
#include <algorithm>

// ==================== CONV2D ====================
Conv2D::Conv2D(int in_ch, int out_ch, int k_size, int s, int p)
    : in_channels(in_ch), out_channels(out_ch), 
      kernel_size(k_size), stride(s), padding(p) {
    
    weights = Tensor(out_ch, in_ch, k_size, k_size);
    bias = Tensor(1, out_ch, 1, 1);
    grad_weights = Tensor(out_ch, in_ch, k_size, k_size);
    grad_bias = Tensor(1, out_ch, 1, 1);
    
    float std = std::sqrt(2.0f / (in_ch * k_size * k_size));
    weights.randomize(-std, std);
    bias.zeros();
}

Tensor Conv2D::forward(const Tensor& input) {
    last_input = input;
    
    int batch = input.batch;
    int in_h = input.height;
    int in_w = input.width;
    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;
    
    Tensor output(batch, out_channels, out_h, out_w);
    
    for (int b = 0; b < batch; b++) {
        for (int oc = 0; oc < out_channels; oc++) {
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    float sum = bias.at(0, oc, 0, 0);
                    
                    for (int ic = 0; ic < in_channels; ic++) {
                        for (int kh = 0; kh < kernel_size; kh++) {
                            for (int kw = 0; kw < kernel_size; kw++) {
                                int ih = oh * stride - padding + kh;
                                int iw = ow * stride - padding + kw;
                                
                                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                    sum += input.at(b, ic, ih, iw) * 
                                           weights.at(oc, ic, kh, kw);
                                }
                            }
                        }
                    }
                    
                    output.at(b, oc, oh, ow) = sum;
                }
            }
        }
    }
    
    return output;
}

Tensor Conv2D::backward(const Tensor& grad_output) {
    grad_weights.zeros();
    grad_bias.zeros();
    
    Tensor grad_input(last_input.batch, in_channels, 
                     last_input.height, last_input.width);
    grad_input.zeros();
    
    int batch = grad_output.batch;
    int out_h = grad_output.height;
    int out_w = grad_output.width;
    
    for (int b = 0; b < batch; b++) {
        for (int oc = 0; oc < out_channels; oc++) {
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    float grad = grad_output.at(b, oc, oh, ow);
                    grad_bias.at(0, oc, 0, 0) += grad;
                    
                    for (int ic = 0; ic < in_channels; ic++) {
                        for (int kh = 0; kh < kernel_size; kh++) {
                            for (int kw = 0; kw < kernel_size; kw++) {
                                int ih = oh * stride - padding + kh;
                                int iw = ow * stride - padding + kw;
                                
                                if (ih >= 0 && ih < last_input.height && 
                                    iw >= 0 && iw < last_input.width) {
                                    grad_weights.at(oc, ic, kh, kw) += 
                                        grad * last_input.at(b, ic, ih, iw);
                                    grad_input.at(b, ic, ih, iw) += 
                                        grad * weights.at(oc, ic, kh, kw);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    return grad_input;
}

void Conv2D::updateWeights(float learning_rate) {
    for (size_t i = 0; i < weights.data.size(); i++) {
        weights.data[i] -= learning_rate * grad_weights.data[i];
    }
    for (size_t i = 0; i < bias.data.size(); i++) {
        bias.data[i] -= learning_rate * grad_bias.data[i];
    }
}

// ==================== RELU ====================
Tensor ReLU::forward(const Tensor& input) {
    last_input = input;
    Tensor output = input;
    
    for (auto& val : output.data) {
        val = std::max(0.0f, val);
    }
    
    return output;
}

Tensor ReLU::backward(const Tensor& grad_output) {
    Tensor grad_input = grad_output;
    
    for (size_t i = 0; i < grad_input.data.size(); i++) {
        if (last_input.data[i] <= 0) {
            grad_input.data[i] = 0;
        }
    }
    
    return grad_input;
}

// ==================== MAXPOOL2D ====================
MaxPool2D::MaxPool2D(int p_size, int s) : pool_size(p_size), stride(s) {}

Tensor MaxPool2D::forward(const Tensor& input) {
    last_input = input;
    
    int batch = input.batch;
    int channels = input.channels;
    int out_h = (input.height - pool_size) / stride + 1;
    int out_w = (input.width - pool_size) / stride + 1;
    
    Tensor output(batch, channels, out_h, out_w);
    mask = Tensor(batch, channels, input.height, input.width);
    mask.zeros();
    
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    float max_val = -1e9f;
                    int max_h = 0, max_w = 0;
                    
                    for (int ph = 0; ph < pool_size; ph++) {
                        for (int pw = 0; pw < pool_size; pw++) {
                            int ih = oh * stride + ph;
                            int iw = ow * stride + pw;
                            float val = input.at(b, c, ih, iw);
                            
                            if (val > max_val) {
                                max_val = val;
                                max_h = ih;
                                max_w = iw;
                            }
                        }
                    }
                    
                    output.at(b, c, oh, ow) = max_val;
                    mask.at(b, c, max_h, max_w) = 1.0f;
                }
            }
        }
    }
    
    return output;
}

Tensor MaxPool2D::backward(const Tensor& grad_output) {
    Tensor grad_input(last_input.batch, last_input.channels,
                     last_input.height, last_input.width);
    grad_input.zeros();
    
    int out_h = grad_output.height;
    int out_w = grad_output.width;
    
    for (int b = 0; b < grad_output.batch; b++) {
        for (int c = 0; c < grad_output.channels; c++) {
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    float grad = grad_output.at(b, c, oh, ow);
                    
                    for (int ph = 0; ph < pool_size; ph++) {
                        for (int pw = 0; pw < pool_size; pw++) {
                            int ih = oh * stride + ph;
                            int iw = ow * stride + pw;
                            grad_input.at(b, c, ih, iw) += 
                                grad * mask.at(b, c, ih, iw);
                        }
                    }
                }
            }
        }
    }
    
    return grad_input;
}

// ==================== UPSAMPLE2D ====================
Upsample2D::Upsample2D(int scale) : scale_factor(scale) {}

Tensor Upsample2D::forward(const Tensor& input) {
    int out_h = input.height * scale_factor;
    int out_w = input.width * scale_factor;
    
    Tensor output(input.batch, input.channels, out_h, out_w);
    
    for (int b = 0; b < input.batch; b++) {
        for (int c = 0; c < input.channels; c++) {
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    int ih = oh / scale_factor;
                    int iw = ow / scale_factor;
                    output.at(b, c, oh, ow) = input.at(b, c, ih, iw);
                }
            }
        }
    }
    
    return output;
}

Tensor Upsample2D::backward(const Tensor& grad_output) {
    int in_h = grad_output.height / scale_factor;
    int in_w = grad_output.width / scale_factor;
    
    Tensor grad_input(grad_output.batch, grad_output.channels, in_h, in_w);
    grad_input.zeros();
    
    for (int b = 0; b < grad_output.batch; b++) {
        for (int c = 0; c < grad_output.channels; c++) {
            for (int oh = 0; oh < grad_output.height; oh++) {
                for (int ow = 0; ow < grad_output.width; ow++) {
                    int ih = oh / scale_factor;
                    int iw = ow / scale_factor;
                    grad_input.at(b, c, ih, iw) += 
                        grad_output.at(b, c, oh, ow);
                }
            }
        }
    }
    
    return grad_input;
}