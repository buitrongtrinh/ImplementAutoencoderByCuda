#include "kernels_backward.cuh"
#include <math.h>

// =======================================================
// ReLU backward
// =======================================================
__global__ void relu_backward(
    const float* input,
    const float* d_out,
    float* d_in,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        d_in[i] = (input[i] > 0.0f) ? d_out[i] : 0.0f;
}

// =======================================================
// MaxPool 2x2 backward
// =======================================================
__global__ void maxpool2x2_backward(
    const float* input,
    const float* output,
    const float* d_out,
    float* d_in,
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

    float grad = d_out[idx];

    int base =
        ((b * C + c) * H + y * 2) * W + x * 2;

    for (int dy = 0; dy < 2; dy++) {
        for (int dx = 0; dx < 2; dx++) {
            int in_idx = base + dy * W + dx;
            if (input[in_idx] == output[idx]) {
                d_in[in_idx] += grad;
            }
        }
    }
}

// =======================================================
// Upsample backward (atomic required)
// =======================================================
__global__ void upsample2x_backward(
    const float* d_out,
    float* d_in,
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

    atomicAdd(&d_in[in_idx], d_out[idx]);
}

// =======================================================
// Conv bias backward
// =======================================================
__global__ void conv_bias_backward(
    const float* d_out,
    float* d_b,
    int B, int C, int H, int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * H * W;
    if (idx >= total) return;

    int c = (idx / (W * H)) % C;
    atomicAdd(&d_b[c], d_out[idx]);
}

// =======================================================
// Conv weight backward (VERY NAIVE)
// =======================================================
__global__ void conv_weight_backward(
    const float* input,
    const float* d_out,
    float* d_w,
    int B, int Cin, int H, int W, int Cout
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Cout * Cin * 3 * 3;
    if (idx >= total) return;

    int kx = idx % 3;
    int ky = (idx / 3) % 3;
    int ic = (idx / 9) % Cin;
    int oc = idx / (9 * Cin);

    float sum = 0.0f;

    for (int b = 0; b < B; b++) {
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                int iy = y + ky - 1;
                int ix = x + kx - 1;
                if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                    int in_idx =
                        ((b * Cin + ic) * H + iy) * W + ix;
                    int out_idx =
                        ((b * Cout + oc) * H + y) * W + x;
                    sum += input[in_idx] * d_out[out_idx];
                }
            }
        }
    }

    d_w[idx] = sum;
}

// =======================================================
// Conv input backward
// =======================================================
__global__ void conv_input_backward(
    const float* d_out,
    const float* weight,
    float* d_in,
    int B, int Cin, int H, int W, int Cout
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * Cin * H * W;
    if (idx >= total) return;

    int x = idx % W;
    int y = (idx / W) % H;
    int ic = (idx / (W * H)) % Cin;
    int b  = idx / (W * H * Cin);

    float sum = 0.0f;

    for (int oc = 0; oc < Cout; oc++) {
        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                int oy = y - ky;
                int ox = x - kx;
                if (oy >= 0 && oy < H && ox >= 0 && ox < W) {
                    int out_idx =
                        ((b * Cout + oc) * H + oy) * W + ox;
                    int w_idx =
                        ((oc * Cin + ic) * 3 + (ky + 1)) * 3 + (kx + 1);
                    sum += d_out[out_idx] * weight[w_idx];
                }
            }
        }
    }

    d_in[idx] = sum;
}

__global__ void mse_grad(
    const float* pred,
    const float* target,
    float* d_grad,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        d_grad[i] = 2.0f * (pred[i] - target[i]) / n;
}

// =======================================================
// SGD weight update: param -= lr * grad
// =======================================================
__global__ void sgd_update(
    float* param,
    const float* grad,
    float lr,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        param[i] -= lr * grad[i];
    }
}
