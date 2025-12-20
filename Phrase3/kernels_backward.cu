#include "kernels_backward.cuh"
#include <math.h>

// =======================================================
// ReLU backward - Optimized
// =======================================================
__global__ void relu_backward(
    const float* __restrict__ input,
    const float* __restrict__ d_out,
    float* __restrict__ d_in,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        d_in[i] = (input[i] > 0.0f) ? d_out[i] : 0.0f;
}

// =======================================================
// MaxPool 2x2 backward - Optimized
// =======================================================
__global__ void maxpool2x2_backward(
    const float* __restrict__ input,
    const float* __restrict__ output,
    const float* __restrict__ d_out,
    float* __restrict__ d_in,
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
    float max_val = output[idx];

    int base = ((b * C + c) * H + y * 2) * W + x * 2;

    // Unroll 2x2 loop for better performance
    int in_idx0 = base;
    int in_idx1 = base + 1;
    int in_idx2 = base + W;
    int in_idx3 = base + W + 1;
    
    if (input[in_idx0] == max_val) d_in[in_idx0] += grad;
    if (input[in_idx1] == max_val) d_in[in_idx1] += grad;
    if (input[in_idx2] == max_val) d_in[in_idx2] += grad;
    if (input[in_idx3] == max_val) d_in[in_idx3] += grad;
}

// =======================================================
// Upsample backward - Optimized (atomic required but optimized)
// =======================================================
__global__ void upsample2x_backward(
    const float* __restrict__ d_out,
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

    // Use bit shift for division by 2 (faster)
    int in_y = y >> 1;
    int in_x = x >> 1;
    int in_idx = ((b * C + c) * H + in_y) * W + in_x;

    // Atomic add is necessary here, but compiler will optimize
    atomicAdd(&d_in[in_idx], d_out[idx]);
}

// =======================================================
// Conv bias backward - Optimized with Warp-level Reduction
// =======================================================
__global__ void conv_bias_backward(
    const float* __restrict__ d_out,
    float* d_b,
    int B, int C, int H, int W
) {
    // Each block processes one channel
    int c = blockIdx.x;
    if (c >= C) return;
    
    int tid = threadIdx.x;
    int lane_id = tid % 32;
    int warp_id = tid / 32;
    
    float sum = 0.0f;
    int total = B * H * W;
    
    // Each thread processes multiple elements with stride
    for (int i = tid; i < total; i += blockDim.x) {
        int b = i / (H * W);
        int pos = i % (H * W);
        int y = pos / W;
        int x = pos % W;
        int idx = (b * C + c) * H * W + y * W + x;
        sum += d_out[idx];
    }
    
    // Warp-level reduction
    float warp_sum = sum;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, offset);
    }
    
    // Shared memory for final reduction
    __shared__ float s_sum[8];
    if (lane_id == 0) {
        s_sum[warp_id] = warp_sum;
    }
    __syncthreads();
    
    // Final reduction
    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x + 31) / 32) ? s_sum[lane_id] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        if (lane_id == 0) {
            atomicAdd(&d_b[c], val);
        }
    }
}

// =======================================================
// Conv weight backward - Optimized with Shared Memory
// Each block processes one weight, uses shared memory for reduction
// =======================================================
__global__ void conv_weight_backward(
    const float* __restrict__ input,
    const float* __restrict__ d_out,
    float* __restrict__ d_w,
    int B, int Cin, int H, int W, int Cout
) {
    // Shared memory for reduction
    __shared__ float s_sum[256];
    
    // Each thread handles one weight gradient: (oc, ic, ky, kx)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Cout * Cin * 3 * 3;
    if (idx >= total) return;

    int kx = idx % 3;
    int ky = (idx / 3) % 3;
    int ic = (idx / 9) % Cin;
    int oc = idx / (9 * Cin);

    int tid = threadIdx.x;
    float sum = 0.0f;
    int ky_offset = ky - 1;
    int kx_offset = kx - 1;
    
    // Pre-compute valid range
    int y_start = (ky_offset < 0) ? -ky_offset : 0;
    int y_end = (ky_offset > 0) ? H - ky_offset : H;
    int x_start = (kx_offset < 0) ? -kx_offset : 0;
    int x_end = (kx_offset > 0) ? W - kx_offset : W;
    
    // Optimize: process spatial dimensions with better memory access pattern
    int spatial_size = (y_end - y_start) * (x_end - x_start);
    for (int b = 0; b < B; b++) {
        int b_input_base = (b * Cin + ic) * H * W;
        int b_output_base = (b * Cout + oc) * H * W;
        
        // Each thread processes multiple spatial positions with stride
        for (int i = tid; i < spatial_size; i += blockDim.x) {
            int y = y_start + i / (x_end - x_start);
            int x = x_start + i % (x_end - x_start);
            int iy = y + ky_offset;
            int ix = x + kx_offset;
            int in_idx = b_input_base + iy * W + ix;
            int out_idx = b_output_base + y * W + x;
            sum += input[in_idx] * d_out[out_idx];
        }
    }
    
    s_sum[tid] = sum;
    __syncthreads();
    
    // Optimized reduction: use warp-level primitives first, then shared memory
    // Warp-level reduction (32 threads)
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    float warp_sum = sum;
    
    // Shuffle reduction within warp (faster than shared memory)
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, offset);
    }
    
    // Store warp sum to shared memory
    if (lane_id == 0) {
        s_sum[warp_id] = warp_sum;
    }
    __syncthreads();
    
    // Final reduction across warps (if needed)
    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x + 31) / 32) ? s_sum[lane_id] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        if (lane_id == 0) {
            d_w[idx] = val;
        }
    }
}

// =======================================================
// Conv input backward - Optimized
// =======================================================
__global__ void conv_input_backward(
    const float* __restrict__ d_out,
    const float* __restrict__ weight,
    float* __restrict__ d_in,
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
    
    // Pre-compute base indices
    int d_out_base = b * Cout * H * W;

    // Unroll kernel loops for better performance
    for (int oc = 0; oc < Cout; oc++) {
        int w_base = (oc * Cin + ic) * 9;
        int d_out_oc_base = d_out_base + oc * H * W;
        
        // Unroll 3x3 kernel - conv input backward: d_in[y,x] = sum over oc,ky,kx of d_out[oy,ox] * w[oc,ic,ky,kx]
        // where oy = y - ky, ox = x - kx (transposed convolution)
        // Top row (ky = -1, so oy = y + 1)
        if (y < H-1) {
            int oy = y + 1;
            if (x < W-1) {
                sum += d_out[d_out_oc_base + oy * W + (x+1)] * weight[w_base + 0];
            }
            sum += d_out[d_out_oc_base + oy * W + x] * weight[w_base + 1];
            if (x > 0) {
                sum += d_out[d_out_oc_base + oy * W + (x-1)] * weight[w_base + 2];
            }
        }
        
        // Middle row (ky = 0, so oy = y)
        if (x < W-1) {
            sum += d_out[d_out_oc_base + y * W + (x+1)] * weight[w_base + 3];
        }
        sum += d_out[d_out_oc_base + y * W + x] * weight[w_base + 4];
        if (x > 0) {
            sum += d_out[d_out_oc_base + y * W + (x-1)] * weight[w_base + 5];
        }
        
        // Bottom row (ky = 1, so oy = y - 1)
        if (y > 0) {
            int oy = y - 1;
            if (x < W-1) {
                sum += d_out[d_out_oc_base + oy * W + (x+1)] * weight[w_base + 6];
            }
            sum += d_out[d_out_oc_base + oy * W + x] * weight[w_base + 7];
            if (x > 0) {
                sum += d_out[d_out_oc_base + oy * W + (x-1)] * weight[w_base + 8];
            }
        }
    }

    d_in[idx] = sum;
}

__global__ void mse_grad(
    const float* __restrict__ pred,
    const float* __restrict__ target,
    float* __restrict__ d_grad,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        d_grad[i] = 2.0f * (pred[i] - target[i]) / n;
}

// =======================================================
// SGD weight update: param -= lr * grad - Optimized
// =======================================================
__global__ void sgd_update(
    float* __restrict__ param,
    const float* __restrict__ grad,
    float lr,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        param[i] -= lr * grad[i];
    }
}
