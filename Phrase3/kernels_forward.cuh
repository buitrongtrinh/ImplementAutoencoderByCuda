#ifndef KERNELS_FORWARD_CUH
#define KERNELS_FORWARD_CUH

#include <cuda_runtime.h>

// Basic convolution
__global__ void conv2d(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int B, int Cin, int H, int W, int Cout
);

// Multi-OC without ReLU
template<int OC_PER_BLOCK>
__global__ void conv2d_multi_oc(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int Cin, int H, int W, int Cout
);

// Multi-OC with fused ReLU (saves pre-activation)
template<int OC_PER_BLOCK>
__global__ void conv2d_multi_oc_relu(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ pre_relu,    // ✅ THÊM
    float* __restrict__ output,
    int B, int Cin, int H, int W, int Cout
);

// 8x8 specialized without ReLU
template<int OC_PER_BLOCK>
__global__ void conv2d_8x8_multi_oc(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int Cin, int Cout
);

// 8x8 specialized with fused ReLU (saves pre-activation)
template<int OC_PER_BLOCK>
__global__ void conv2d_8x8_multi_oc_relu(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ pre_relu,    // ✅ THÊM
    float* __restrict__ output,
    int B, int Cin, int Cout
);

// Specialized Cin=3 with fused ReLU (saves pre-activation)
__global__ void conv2d_cin3_oc4_relu(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ pre_relu,    // ✅ THÊM
    float* __restrict__ output,
    int B, int H, int W, int Cout
);

// Specialized Cout=3 (no ReLU)
__global__ void conv2d_cout3_all(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int Cin, int H, int W
);

// Other operations
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

#endif // KERNELS_FORWARD_CUH