#pragma once
#include <cuda_runtime.h>
__global__ void conv2d(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int B, int Cin, int H, int W, int Cout
);
/*
 * HIGHLY OPTIMIZED FORWARD KERNELS
 * - Shared Memory Tiling
 * - Loop Unrolling
 * - Multiple Output Channels per Block
 * Layout: BCHW
 */

// ================= MULTI-CHANNEL CONVOLUTIONS =================

// General: mỗi block tính OC_PER_BLOCK output channels
template<int OC_PER_BLOCK>
__global__ void conv2d_multi_oc(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int Cin, int H, int W, int Cout
);

// Fused với ReLU
template<int OC_PER_BLOCK>
__global__ void conv2d_multi_oc_relu(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int Cin, int H, int W, int Cout
);

// Specialized cho 8x8 spatial
template<int OC_PER_BLOCK>
__global__ void conv2d_8x8_multi_oc(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int Cin, int Cout
);

template<int OC_PER_BLOCK>
__global__ void conv2d_8x8_multi_oc_relu(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int Cin, int Cout
);

// ================= SPECIALIZED KERNELS =================

// Conv1: Cin=3, tính 4 output channels per block
__global__ void conv2d_cin3_oc4_relu(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int H, int W, int Cout
);

// Conv5: Cout=3, tính tất cả 3 output channels per block
__global__ void conv2d_cout3_all(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int Cin, int H, int W
);

// ================= OTHER LAYERS =================
__global__ void relu_opt(float* x, int n);

__global__ void maxpool2x2_opt(
    const float* __restrict__ input,
    float* __restrict__ output,
    int B, int C, int H, int W
);

__global__ void upsample2x_opt(
    const float* __restrict__ input,
    float* __restrict__ output,
    int B, int C, int H, int W
);

__global__ void mse_loss_opt(
    const float* __restrict__ pred,
    const float* __restrict__ target,
    float* loss,
    int n
);