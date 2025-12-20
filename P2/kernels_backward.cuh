#pragma once
#include <cuda_runtime.h>

/*
 Layout: BCHW
 index = ((b * C + c) * H + y) * W + x
*/

// ================= ReLU BACKWARD =================
__global__ void relu_backward(
    const float* input,     // forward input
    const float* d_out,     // grad from above
    float* d_in,
    int n
);

// ================= MAXPOOL BACKWARD =================
__global__ void maxpool2x2_backward(
    const float* input,     // forward input
    const float* output,    // forward output
    const float* d_out,
    float* d_in,
    int B, int C, int H, int W
);

// ================= UPSAMPLE BACKWARD =================
__global__ void upsample2x_backward(
    const float* d_out,
    float* d_in,
    int B, int C, int H, int W
);

// ================= CONV BACKWARD =================

// bias gradient
__global__ void conv_bias_backward(
    const float* d_out,
    float* d_b,
    int B, int C, int H, int W
);

// weight gradient
__global__ void conv_weight_backward(
    const float* input,
    const float* d_out,
    float* d_w,
    int B, int Cin, int H, int W, int Cout
);

// input gradient
__global__ void conv_input_backward(
    const float* d_out,
    const float* weight,
    float* d_in,
    int B, int Cin, int H, int W, int Cout
);
__global__ void mse_grad(
    const float* pred,
    const float* target,
    float* d_grad,
    int n
);
// ================= SGD UPDATE =================
__global__ void sgd_update(
    float* param,          // weight or bias
    const float* grad,     // corresponding gradient
    float lr,
    int n                  // number of parameters
);
