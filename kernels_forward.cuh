#pragma once
#include <cuda_runtime.h>

/*
 Layout: BCHW
 index = ((b * C + c) * H + y) * W + x
*/

// ================= CONVOLUTION =================
__global__ void conv2d(
    const float* input,    // [B, Cin, H, W]
    const float* weight,   // [Cout, Cin, 3, 3]
    const float* bias,     // [Cout]
    float* output,         // [B, Cout, H, W]
    int B, int Cin, int H, int W, int Cout
);

// ================= RELU =================
__global__ void relu(
    float* x,
    int n
);

// ================= MAXPOOL 2x2 =================
__global__ void maxpool2x2(
    const float* input,    // [B, C, H, W]
    float* output,         // [B, C, H/2, W/2]
    int B, int C, int H, int W
);

// ================= UPSAMPLE 2x =================
__global__ void upsample2x(
    const float* input,    // [B, C, H, W]
    float* output,         // [B, C, 2H, 2W]
    int B, int C, int H, int W
);

// ================= MSE LOSS =================
__global__ void mse_loss(
    const float* pred,     // reconstructed output
    const float* target,   // original input
    float* loss,           // single float on device
    int n                  // total elements
);