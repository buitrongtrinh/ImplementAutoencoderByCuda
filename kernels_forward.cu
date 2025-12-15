#include "kernels_forward.cuh"
#include <math.h>

// =======================================================
// Convolution 3x3 - Naive - BCHW
// Each thread computes ONE output pixel
// =======================================================
__global__ void conv2d(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int B, int Cin, int H, int W, int Cout
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * Cout * H * W;
    if (idx >= total) return;

    int x  = idx % W;
    int y  = (idx / W) % H;
    int oc = (idx / (W * H)) % Cout;
    int b  = idx / (W * H * Cout);

    float sum = bias[oc];

    for (int ic = 0; ic < Cin; ic++) {
        for (int ky = -1; ky <= 1; ky++) {
            int iy = y + ky;
            if (iy < 0 || iy >= H) continue;

            for (int kx = -1; kx <= 1; kx++) {
                int ix = x + kx;
                if (ix < 0 || ix >= W) continue;

                int in_idx =
                    ((b * Cin + ic) * H + iy) * W + ix;

                int w_idx =
                    ((oc * Cin + ic) * 3 + (ky + 1)) * 3 + (kx + 1);

                sum += input[in_idx] * weight[w_idx];
            }
        }
    }

    int out_idx =
        ((b * Cout + oc) * H + y) * W + x;

    output[out_idx] = sum;
}

// =======================================================
// ReLU - Naive
// =======================================================
__global__ void relu(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        x[i] = fmaxf(0.0f, x[i]);
}

// =======================================================
// MaxPooling 2x2 - Naive - BCHW
// =======================================================
__global__ void maxpool2x2(
    const float* input,
    float* output,
    int B, int C, int H, int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int H2 = H / 2;
    int W2 = W / 2;
    int total = B * C * H2 * W2;
    if (idx >= total) return;

    int x = idx % W2;
    int y = (idx / W2) % H2;
    int c = (idx / (W2 * H2)) % C;
    int b = idx / (W2 * H2 * C);

    int base =
        ((b * C + c) * H + y * 2) * W + x * 2;

    float m = input[base];
    m = fmaxf(m, input[base + 1]);
    m = fmaxf(m, input[base + W]);
    m = fmaxf(m, input[base + W + 1]);

    int out_idx =
        ((b * C + c) * H2 + y) * W2 + x;

    output[out_idx] = m;
}

// =======================================================
// Upsampling 2x - Nearest Neighbor - BCHW
// =======================================================
__global__ void upsample2x(
    const float* input,
    float* output,
    int B, int C, int H, int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int H2 = H * 2;
    int W2 = W * 2;
    int total = B * C * H2 * W2;
    if (idx >= total) return;

    int x = idx % W2;
    int y = (idx / W2) % H2;
    int c = (idx / (W2 * H2)) % C;
    int b = idx / (W2 * H2 * C);

    int in_idx =
        ((b * C + c) * H + (y >> 1)) * W + (x >> 1);

    int out_idx =
        ((b * C + c) * H2 + y) * W2 + x;

    output[out_idx] = input[in_idx];
}

// =======================================================
// MSE Loss - Parallel Reduction + Shared + atomicAdd
// =======================================================
__global__ void mse_loss(
    const float* pred,
    const float* target,
    float* loss,
    int n
) {
    __shared__ float buf[256];   // one block partial sum

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    float val = 0.0f;
    if (idx < n) {
        float diff = pred[idx] - target[idx];
        val = diff * diff;
    }

    buf[tid] = val;
    __syncthreads();

    // reduction inside block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            buf[tid] += buf[tid + s];
        __syncthreads();
    }

    // accumulate block result
    if (tid == 0)
        atomicAdd(loss, buf[0]);
}
