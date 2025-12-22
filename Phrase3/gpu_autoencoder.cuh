#pragma once
#include <cuda_runtime.h>
#include "kernels_forward.cuh"
#include <string>
#include "kernels_backward.cuh"

class GPUAutoencoder {
public:
    int batch_size;
    float learning_rate;

    // ================= WEIGHTS =================
    float *d_w1, *d_w2, *d_w3, *d_w4, *d_w5;
    float *d_b1, *d_b2, *d_b3, *d_b4, *d_b5;

    // ================= ACTIVATIONS =================
    float *d_input;
    float *d_o1, *d_o2, *d_o3, *d_o4;
    float *d_o5, *d_o6, *d_o7, *d_o8;
    float *d_output;

    // ================= GRADIENTS =================
    float *d_dw1, *d_dw2, *d_dw3, *d_dw4, *d_dw5;
    float *d_db1, *d_db2, *d_db3, *d_db4, *d_db5;

    // ================= HOST WEIGHTS (INIT) =================
    float *h_w1, *h_w2, *h_w3, *h_w4, *h_w5;
    float *h_b1, *h_b2, *h_b3, *h_b4, *h_b5;
    
    // THÊM: Pre-activation buffers cho backward pass
    float *d_o1_pre;   // Pre-ReLU for Conv1
    float *d_o3_pre;   // Pre-ReLU for Conv2
    float *d_o5_pre;   // Pre-ReLU for Conv3
    float *d_o7_pre;   // Pre-ReLU for Conv4

    GPUAutoencoder(int batch, float lr = 0.001f);
    ~GPUAutoencoder();
    float forward(float* h_input, float* h_output);
    void backward();
    void saveWeights(const std::string& path);
    void loadWeights(const std::string& path);
    void extractFeatures(float* h_input, float* h_latent);
private:
    void allocateDeviceMemory();
    void allocateHostMemory();
    void initializeWeights();
    void copyWeightsToGPU();
};
