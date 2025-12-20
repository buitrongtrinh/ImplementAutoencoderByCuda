#include "kernels_forward.cuh"

#define TILE_SIZE 16
#define TILE_HALO (TILE_SIZE + 2)
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
// MACRO: UNROLLED 3x3 CONVOLUTION
// =======================================================
#define CONV3x3(s_in, w_base, local_x, local_y, HALO_W, sum) \
    do { \
        sum += s_in[(local_y - 1) * HALO_W + (local_x - 1)] * w_base[0]; \
        sum += s_in[(local_y - 1) * HALO_W + (local_x    )] * w_base[1]; \
        sum += s_in[(local_y - 1) * HALO_W + (local_x + 1)] * w_base[2]; \
        sum += s_in[(local_y    ) * HALO_W + (local_x - 1)] * w_base[3]; \
        sum += s_in[(local_y    ) * HALO_W + (local_x    )] * w_base[4]; \
        sum += s_in[(local_y    ) * HALO_W + (local_x + 1)] * w_base[5]; \
        sum += s_in[(local_y + 1) * HALO_W + (local_x - 1)] * w_base[6]; \
        sum += s_in[(local_y + 1) * HALO_W + (local_x    )] * w_base[7]; \
        sum += s_in[(local_y + 1) * HALO_W + (local_x + 1)] * w_base[8]; \
    } while(0)


// =======================================================
// CONV2D MULTI OUTPUT CHANNELS PER BLOCK
// Mỗi block tính OC_PER_BLOCK output channels
// Reuse input tile cho tất cả output channels
// =======================================================
template<int OC_PER_BLOCK>
__global__ void conv2d_multi_oc(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int Cin, int H, int W, int Cout
) {
    // Shared memory cho input tile
    __shared__ float s_input[TILE_HALO * TILE_HALO];
    // Shared memory cho weights của OC_PER_BLOCK output channels
    __shared__ float s_weight[OC_PER_BLOCK * 9];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * TILE_SIZE + tx;
    
    const int out_x = blockIdx.x * TILE_SIZE + tx;
    const int out_y = blockIdx.y * TILE_SIZE + ty;
    const int oc_base = blockIdx.z % ((Cout + OC_PER_BLOCK - 1) / OC_PER_BLOCK) * OC_PER_BLOCK;
    const int b = blockIdx.z / ((Cout + OC_PER_BLOCK - 1) / OC_PER_BLOCK);
    
    const int threads_per_block = TILE_SIZE * TILE_SIZE;
    const int local_x = tx + 1;
    const int local_y = ty + 1;
    
    const bool valid = (out_x < W && out_y < H && b < B);
    
    // Registers cho OC_PER_BLOCK output sums
    float sums[OC_PER_BLOCK];
    
    #pragma unroll
    for (int i = 0; i < OC_PER_BLOCK; i++) {
        int oc = oc_base + i;
        sums[i] = (valid && oc < Cout) ? bias[oc] : 0.0f;
    }
    
    // Loop qua input channels
    for (int ic = 0; ic < Cin; ic++) {
        __syncthreads();
        
        // ========== LOAD INPUT TILE (1 lần cho tất cả output channels) ==========
        #pragma unroll 2
        for (int i = tid; i < TILE_HALO * TILE_HALO; i += threads_per_block) {
            int ly = i / TILE_HALO;
            int lx = i % TILE_HALO;
            int gx = blockIdx.x * TILE_SIZE + lx - 1;
            int gy = blockIdx.y * TILE_SIZE + ly - 1;
            
            float val = 0.0f;
            if (gx >= 0 && gx < W && gy >= 0 && gy < H && b < B) {
                val = input[((b * Cin + ic) * H + gy) * W + gx];
            }
            s_input[i] = val;
        }
        
        // ========== LOAD WEIGHTS cho OC_PER_BLOCK output channels ==========
        #pragma unroll
        for (int i = tid; i < OC_PER_BLOCK * 9; i += threads_per_block) {
            int oc_offset = i / 9;
            int k_idx = i % 9;
            int oc = oc_base + oc_offset;
            
            float val = 0.0f;
            if (oc < Cout) {
                val = weight[((oc * Cin + ic) * 3 + k_idx / 3) * 3 + k_idx % 3];
            }
            s_weight[i] = val;
        }
        
        __syncthreads();
        
        // ========== COMPUTE cho tất cả OC_PER_BLOCK output channels ==========
        if (valid) {
            // Load input values vào registers (reuse cho tất cả output channels)
            float in00 = s_input[(local_y - 1) * TILE_HALO + (local_x - 1)];
            float in01 = s_input[(local_y - 1) * TILE_HALO + (local_x    )];
            float in02 = s_input[(local_y - 1) * TILE_HALO + (local_x + 1)];
            float in10 = s_input[(local_y    ) * TILE_HALO + (local_x - 1)];
            float in11 = s_input[(local_y    ) * TILE_HALO + (local_x    )];
            float in12 = s_input[(local_y    ) * TILE_HALO + (local_x + 1)];
            float in20 = s_input[(local_y + 1) * TILE_HALO + (local_x - 1)];
            float in21 = s_input[(local_y + 1) * TILE_HALO + (local_x    )];
            float in22 = s_input[(local_y + 1) * TILE_HALO + (local_x + 1)];
            
            // Compute cho mỗi output channel
            #pragma unroll
            for (int i = 0; i < OC_PER_BLOCK; i++) {
                float* w = &s_weight[i * 9];
                sums[i] += in00 * w[0] + in01 * w[1] + in02 * w[2];
                sums[i] += in10 * w[3] + in11 * w[4] + in12 * w[5];
                sums[i] += in20 * w[6] + in21 * w[7] + in22 * w[8];
            }
        }
    }
    
    // ========== WRITE OUTPUT ==========
    if (valid) {
        #pragma unroll
        for (int i = 0; i < OC_PER_BLOCK; i++) {
            int oc = oc_base + i;
            if (oc < Cout) {
                output[((b * Cout + oc) * H + out_y) * W + out_x] = sums[i];
            }
        }
    }
}


// =======================================================
// CONV2D MULTI OC + RELU FUSED
// =======================================================
template<int OC_PER_BLOCK>
__global__ void conv2d_multi_oc_relu(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int Cin, int H, int W, int Cout
) {
    __shared__ float s_input[TILE_HALO * TILE_HALO];
    __shared__ float s_weight[OC_PER_BLOCK * 9];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * TILE_SIZE + tx;
    
    const int out_x = blockIdx.x * TILE_SIZE + tx;
    const int out_y = blockIdx.y * TILE_SIZE + ty;
    const int oc_base = blockIdx.z % ((Cout + OC_PER_BLOCK - 1) / OC_PER_BLOCK) * OC_PER_BLOCK;
    const int b = blockIdx.z / ((Cout + OC_PER_BLOCK - 1) / OC_PER_BLOCK);
    
    const int threads_per_block = TILE_SIZE * TILE_SIZE;
    const int local_x = tx + 1;
    const int local_y = ty + 1;
    
    const bool valid = (out_x < W && out_y < H && b < B);
    
    float sums[OC_PER_BLOCK];
    
    #pragma unroll
    for (int i = 0; i < OC_PER_BLOCK; i++) {
        int oc = oc_base + i;
        sums[i] = (valid && oc < Cout) ? bias[oc] : 0.0f;
    }
    
    for (int ic = 0; ic < Cin; ic++) {
        __syncthreads();
        
        // Load input
        #pragma unroll 2
        for (int i = tid; i < TILE_HALO * TILE_HALO; i += threads_per_block) {
            int ly = i / TILE_HALO;
            int lx = i % TILE_HALO;
            int gx = blockIdx.x * TILE_SIZE + lx - 1;
            int gy = blockIdx.y * TILE_SIZE + ly - 1;
            
            float val = 0.0f;
            if (gx >= 0 && gx < W && gy >= 0 && gy < H && b < B) {
                val = input[((b * Cin + ic) * H + gy) * W + gx];
            }
            s_input[i] = val;
        }
        
        // Load weights
        #pragma unroll
        for (int i = tid; i < OC_PER_BLOCK * 9; i += threads_per_block) {
            int oc_offset = i / 9;
            int k_idx = i % 9;
            int oc = oc_base + oc_offset;
            
            float val = 0.0f;
            if (oc < Cout) {
                val = weight[((oc * Cin + ic) * 3 + k_idx / 3) * 3 + k_idx % 3];
            }
            s_weight[i] = val;
        }
        
        __syncthreads();
        
        if (valid) {
            float in00 = s_input[(local_y - 1) * TILE_HALO + (local_x - 1)];
            float in01 = s_input[(local_y - 1) * TILE_HALO + (local_x    )];
            float in02 = s_input[(local_y - 1) * TILE_HALO + (local_x + 1)];
            float in10 = s_input[(local_y    ) * TILE_HALO + (local_x - 1)];
            float in11 = s_input[(local_y    ) * TILE_HALO + (local_x    )];
            float in12 = s_input[(local_y    ) * TILE_HALO + (local_x + 1)];
            float in20 = s_input[(local_y + 1) * TILE_HALO + (local_x - 1)];
            float in21 = s_input[(local_y + 1) * TILE_HALO + (local_x    )];
            float in22 = s_input[(local_y + 1) * TILE_HALO + (local_x + 1)];
            
            #pragma unroll
            for (int i = 0; i < OC_PER_BLOCK; i++) {
                float* w = &s_weight[i * 9];
                sums[i] += in00 * w[0] + in01 * w[1] + in02 * w[2];
                sums[i] += in10 * w[3] + in11 * w[4] + in12 * w[5];
                sums[i] += in20 * w[6] + in21 * w[7] + in22 * w[8];
            }
        }
    }
    
    // Write với ReLU
    if (valid) {
        #pragma unroll
        for (int i = 0; i < OC_PER_BLOCK; i++) {
            int oc = oc_base + i;
            if (oc < Cout) {
                output[((b * Cout + oc) * H + out_y) * W + out_x] = fmaxf(0.0f, sums[i]);
            }
        }
    }
}


// =======================================================
// CONV2D 8x8 MULTI OUTPUT CHANNELS
// =======================================================
template<int OC_PER_BLOCK>
__global__ void conv2d_8x8_multi_oc(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int Cin, int Cout
) {
    __shared__ float s_input[10 * 10];  // 8x8 + halo
    __shared__ float s_weight[OC_PER_BLOCK * 9];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * 8 + tx;
    
    const int oc_base = blockIdx.y * OC_PER_BLOCK;
    const int b = blockIdx.x;
    
    if (b >= B) return;
    
    const int local_x = tx + 1;
    const int local_y = ty + 1;
    
    float sums[OC_PER_BLOCK];
    
    #pragma unroll
    for (int i = 0; i < OC_PER_BLOCK; i++) {
        int oc = oc_base + i;
        sums[i] = (oc < Cout) ? bias[oc] : 0.0f;
    }
    
    for (int ic = 0; ic < Cin; ic++) {
        __syncthreads();
        
        // Load 10x10 input
        for (int i = tid; i < 100; i += 64) {
            int ly = i / 10;
            int lx = i % 10;
            int gx = lx - 1;
            int gy = ly - 1;
            
            float val = 0.0f;
            if (gx >= 0 && gx < 8 && gy >= 0 && gy < 8) {
                val = input[((b * Cin + ic) * 8 + gy) * 8 + gx];
            }
            s_input[i] = val;
        }
        
        // Load weights cho OC_PER_BLOCK channels
        for (int i = tid; i < OC_PER_BLOCK * 9; i += 64) {
            int oc_offset = i / 9;
            int k_idx = i % 9;
            int oc = oc_base + oc_offset;
            
            float val = 0.0f;
            if (oc < Cout) {
                val = weight[((oc * Cin + ic) * 3 + k_idx / 3) * 3 + k_idx % 3];
            }
            s_weight[i] = val;
        }
        
        __syncthreads();
        
        // Load input vào registers
        float in00 = s_input[(local_y - 1) * 10 + (local_x - 1)];
        float in01 = s_input[(local_y - 1) * 10 + (local_x    )];
        float in02 = s_input[(local_y - 1) * 10 + (local_x + 1)];
        float in10 = s_input[(local_y    ) * 10 + (local_x - 1)];
        float in11 = s_input[(local_y    ) * 10 + (local_x    )];
        float in12 = s_input[(local_y    ) * 10 + (local_x + 1)];
        float in20 = s_input[(local_y + 1) * 10 + (local_x - 1)];
        float in21 = s_input[(local_y + 1) * 10 + (local_x    )];
        float in22 = s_input[(local_y + 1) * 10 + (local_x + 1)];
        
        #pragma unroll
        for (int i = 0; i < OC_PER_BLOCK; i++) {
            float* w = &s_weight[i * 9];
            sums[i] += in00 * w[0] + in01 * w[1] + in02 * w[2];
            sums[i] += in10 * w[3] + in11 * w[4] + in12 * w[5];
            sums[i] += in20 * w[6] + in21 * w[7] + in22 * w[8];
        }
    }
    
    // Write output
    #pragma unroll
    for (int i = 0; i < OC_PER_BLOCK; i++) {
        int oc = oc_base + i;
        if (oc < Cout) {
            output[((b * Cout + oc) * 8 + ty) * 8 + tx] = sums[i];
        }
    }
}


// =======================================================
// CONV2D 8x8 MULTI OC + RELU
// =======================================================
template<int OC_PER_BLOCK>
__global__ void conv2d_8x8_multi_oc_relu(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int Cin, int Cout
) {
    __shared__ float s_input[10 * 10];
    __shared__ float s_weight[OC_PER_BLOCK * 9];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * 8 + tx;
    
    const int oc_base = blockIdx.y * OC_PER_BLOCK;
    const int b = blockIdx.x;
    
    if (b >= B) return;
    
    const int local_x = tx + 1;
    const int local_y = ty + 1;
    
    float sums[OC_PER_BLOCK];
    
    #pragma unroll
    for (int i = 0; i < OC_PER_BLOCK; i++) {
        int oc = oc_base + i;
        sums[i] = (oc < Cout) ? bias[oc] : 0.0f;
    }
    
    for (int ic = 0; ic < Cin; ic++) {
        __syncthreads();
        
        for (int i = tid; i < 100; i += 64) {
            int ly = i / 10;
            int lx = i % 10;
            int gx = lx - 1;
            int gy = ly - 1;
            
            float val = (gx >= 0 && gx < 8 && gy >= 0 && gy < 8) ?
                input[((b * Cin + ic) * 8 + gy) * 8 + gx] : 0.0f;
            s_input[i] = val;
        }
        
        for (int i = tid; i < OC_PER_BLOCK * 9; i += 64) {
            int oc_offset = i / 9;
            int k_idx = i % 9;
            int oc = oc_base + oc_offset;
            
            s_weight[i] = (oc < Cout) ?
                weight[((oc * Cin + ic) * 3 + k_idx / 3) * 3 + k_idx % 3] : 0.0f;
        }
        
        __syncthreads();
        
        float in00 = s_input[(local_y - 1) * 10 + (local_x - 1)];
        float in01 = s_input[(local_y - 1) * 10 + (local_x    )];
        float in02 = s_input[(local_y - 1) * 10 + (local_x + 1)];
        float in10 = s_input[(local_y    ) * 10 + (local_x - 1)];
        float in11 = s_input[(local_y    ) * 10 + (local_x    )];
        float in12 = s_input[(local_y    ) * 10 + (local_x + 1)];
        float in20 = s_input[(local_y + 1) * 10 + (local_x - 1)];
        float in21 = s_input[(local_y + 1) * 10 + (local_x    )];
        float in22 = s_input[(local_y + 1) * 10 + (local_x + 1)];
        
        #pragma unroll
        for (int i = 0; i < OC_PER_BLOCK; i++) {
            float* w = &s_weight[i * 9];
            sums[i] += in00 * w[0] + in01 * w[1] + in02 * w[2];
            sums[i] += in10 * w[3] + in11 * w[4] + in12 * w[5];
            sums[i] += in20 * w[6] + in21 * w[7] + in22 * w[8];
        }
    }
    
    #pragma unroll
    for (int i = 0; i < OC_PER_BLOCK; i++) {
        int oc = oc_base + i;
        if (oc < Cout) {
            output[((b * Cout + oc) * 8 + ty) * 8 + tx] = fmaxf(0.0f, sums[i]);
        }
    }
}


// =======================================================
// CONV1 SPECIALIZED: Cin=3, OC_PER_BLOCK=4
// Load tất cả 3 input channels + 4 output channels weights
// =======================================================
__global__ void conv2d_cin3_oc4_relu(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int H, int W, int Cout
) {
    // 3 input channels
    __shared__ float s_input0[TILE_HALO * TILE_HALO];
    __shared__ float s_input1[TILE_HALO * TILE_HALO];
    __shared__ float s_input2[TILE_HALO * TILE_HALO];
    // 4 output channels * 3 input channels * 9 weights = 108
    __shared__ float s_weight[4 * 3 * 9];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * TILE_SIZE + tx;
    
    const int out_x = blockIdx.x * TILE_SIZE + tx;
    const int out_y = blockIdx.y * TILE_SIZE + ty;
    const int oc_base = (blockIdx.z % ((Cout + 3) / 4)) * 4;
    const int b = blockIdx.z / ((Cout + 3) / 4);
    
    const int threads_per_block = TILE_SIZE * TILE_SIZE;
    const int local_x = tx + 1;
    const int local_y = ty + 1;
    
    const bool valid = (out_x < W && out_y < H && b < B);
    
    // Load all 3 input channels
    #pragma unroll 2
    for (int i = tid; i < TILE_HALO * TILE_HALO; i += threads_per_block) {
        int ly = i / TILE_HALO;
        int lx = i % TILE_HALO;
        int gx = blockIdx.x * TILE_SIZE + lx - 1;
        int gy = blockIdx.y * TILE_SIZE + ly - 1;
        
        float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f;
        if (gx >= 0 && gx < W && gy >= 0 && gy < H && b < B) {
            int base = (b * 3 * H + gy) * W + gx;
            v0 = input[base];
            v1 = input[base + H * W];
            v2 = input[base + 2 * H * W];
        }
        s_input0[i] = v0;
        s_input1[i] = v1;
        s_input2[i] = v2;
    }
    
    // Load weights: 4 oc * 3 ic * 9 = 108 elements
    for (int i = tid; i < 108; i += threads_per_block) {
        int oc_offset = i / 27;      // 0-3
        int ic = (i % 27) / 9;       // 0-2
        int k = i % 9;               // 0-8
        int oc = oc_base + oc_offset;
        
        float val = 0.0f;
        if (oc < Cout) {
            val = weight[((oc * 3 + ic) * 3 + k / 3) * 3 + k % 3];
        }
        s_weight[i] = val;
    }
    
    __syncthreads();
    
    if (valid) {
        // Load input vào registers
        float in0[9], in1[9], in2[9];
        
        in0[0] = s_input0[(local_y - 1) * TILE_HALO + (local_x - 1)];
        in0[1] = s_input0[(local_y - 1) * TILE_HALO + (local_x    )];
        in0[2] = s_input0[(local_y - 1) * TILE_HALO + (local_x + 1)];
        in0[3] = s_input0[(local_y    ) * TILE_HALO + (local_x - 1)];
        in0[4] = s_input0[(local_y    ) * TILE_HALO + (local_x    )];
        in0[5] = s_input0[(local_y    ) * TILE_HALO + (local_x + 1)];
        in0[6] = s_input0[(local_y + 1) * TILE_HALO + (local_x - 1)];
        in0[7] = s_input0[(local_y + 1) * TILE_HALO + (local_x    )];
        in0[8] = s_input0[(local_y + 1) * TILE_HALO + (local_x + 1)];
        
        in1[0] = s_input1[(local_y - 1) * TILE_HALO + (local_x - 1)];
        in1[1] = s_input1[(local_y - 1) * TILE_HALO + (local_x    )];
        in1[2] = s_input1[(local_y - 1) * TILE_HALO + (local_x + 1)];
        in1[3] = s_input1[(local_y    ) * TILE_HALO + (local_x - 1)];
        in1[4] = s_input1[(local_y    ) * TILE_HALO + (local_x    )];
        in1[5] = s_input1[(local_y    ) * TILE_HALO + (local_x + 1)];
        in1[6] = s_input1[(local_y + 1) * TILE_HALO + (local_x - 1)];
        in1[7] = s_input1[(local_y + 1) * TILE_HALO + (local_x    )];
        in1[8] = s_input1[(local_y + 1) * TILE_HALO + (local_x + 1)];
        
        in2[0] = s_input2[(local_y - 1) * TILE_HALO + (local_x - 1)];
        in2[1] = s_input2[(local_y - 1) * TILE_HALO + (local_x    )];
        in2[2] = s_input2[(local_y - 1) * TILE_HALO + (local_x + 1)];
        in2[3] = s_input2[(local_y    ) * TILE_HALO + (local_x - 1)];
        in2[4] = s_input2[(local_y    ) * TILE_HALO + (local_x    )];
        in2[5] = s_input2[(local_y    ) * TILE_HALO + (local_x + 1)];
        in2[6] = s_input2[(local_y + 1) * TILE_HALO + (local_x - 1)];
        in2[7] = s_input2[(local_y + 1) * TILE_HALO + (local_x    )];
        in2[8] = s_input2[(local_y + 1) * TILE_HALO + (local_x + 1)];
        
        // Compute 4 output channels
        #pragma unroll
        for (int oc_off = 0; oc_off < 4; oc_off++) {
            int oc = oc_base + oc_off;
            if (oc < Cout) {
                float sum = bias[oc];
                
                // Channel 0
                float* w0 = &s_weight[oc_off * 27];
                #pragma unroll
                for (int k = 0; k < 9; k++) {
                    sum += in0[k] * w0[k];
                }
                
                // Channel 1
                float* w1 = &s_weight[oc_off * 27 + 9];
                #pragma unroll
                for (int k = 0; k < 9; k++) {
                    sum += in1[k] * w1[k];
                }
                
                // Channel 2
                float* w2 = &s_weight[oc_off * 27 + 18];
                #pragma unroll
                for (int k = 0; k < 9; k++) {
                    sum += in2[k] * w2[k];
                }
                
                output[((b * Cout + oc) * H + out_y) * W + out_x] = fmaxf(0.0f, sum);
            }
        }
    }
}


// =======================================================
// CONV5 SPECIALIZED: Cout=3, tính tất cả 3 output cùng lúc
// =======================================================
__global__ void conv2d_cout3_all(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int Cin, int H, int W
) {
    __shared__ float s_input[TILE_HALO * TILE_HALO];
    __shared__ float s_weight[3 * 9];  // 3 output channels * 9
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * TILE_SIZE + tx;
    
    const int out_x = blockIdx.x * TILE_SIZE + tx;
    const int out_y = blockIdx.y * TILE_SIZE + ty;
    const int b = blockIdx.z;
    
    const int threads_per_block = TILE_SIZE * TILE_SIZE;
    const int local_x = tx + 1;
    const int local_y = ty + 1;
    
    const bool valid = (out_x < W && out_y < H && b < B);
    
    float sum0 = valid ? bias[0] : 0.0f;
    float sum1 = valid ? bias[1] : 0.0f;
    float sum2 = valid ? bias[2] : 0.0f;
    
    for (int ic = 0; ic < Cin; ic++) {
        __syncthreads();
        
        // Load input
        #pragma unroll 2
        for (int i = tid; i < TILE_HALO * TILE_HALO; i += threads_per_block) {
            int ly = i / TILE_HALO;
            int lx = i % TILE_HALO;
            int gx = blockIdx.x * TILE_SIZE + lx - 1;
            int gy = blockIdx.y * TILE_SIZE + ly - 1;
            
            float val = 0.0f;
            if (gx >= 0 && gx < W && gy >= 0 && gy < H && b < B) {
                val = input[((b * Cin + ic) * H + gy) * W + gx];
            }
            s_input[i] = val;
        }
        
        // Load weights cho 3 output channels
        if (tid < 27) {
            int oc = tid / 9;
            int k = tid % 9;
            s_weight[tid] = weight[((oc * Cin + ic) * 3 + k / 3) * 3 + k % 3];
        }
        
        __syncthreads();
        
        if (valid) {
            float in00 = s_input[(local_y - 1) * TILE_HALO + (local_x - 1)];
            float in01 = s_input[(local_y - 1) * TILE_HALO + (local_x    )];
            float in02 = s_input[(local_y - 1) * TILE_HALO + (local_x + 1)];
            float in10 = s_input[(local_y    ) * TILE_HALO + (local_x - 1)];
            float in11 = s_input[(local_y    ) * TILE_HALO + (local_x    )];
            float in12 = s_input[(local_y    ) * TILE_HALO + (local_x + 1)];
            float in20 = s_input[(local_y + 1) * TILE_HALO + (local_x - 1)];
            float in21 = s_input[(local_y + 1) * TILE_HALO + (local_x    )];
            float in22 = s_input[(local_y + 1) * TILE_HALO + (local_x + 1)];
            
            // Output 0
            sum0 += in00 * s_weight[0] + in01 * s_weight[1] + in02 * s_weight[2];
            sum0 += in10 * s_weight[3] + in11 * s_weight[4] + in12 * s_weight[5];
            sum0 += in20 * s_weight[6] + in21 * s_weight[7] + in22 * s_weight[8];
            
            // Output 1
            sum1 += in00 * s_weight[9]  + in01 * s_weight[10] + in02 * s_weight[11];
            sum1 += in10 * s_weight[12] + in11 * s_weight[13] + in12 * s_weight[14];
            sum1 += in20 * s_weight[15] + in21 * s_weight[16] + in22 * s_weight[17];
            
            // Output 2
            sum2 += in00 * s_weight[18] + in01 * s_weight[19] + in02 * s_weight[20];
            sum2 += in10 * s_weight[21] + in11 * s_weight[22] + in12 * s_weight[23];
            sum2 += in20 * s_weight[24] + in21 * s_weight[25] + in22 * s_weight[26];
        }
    }
    
      if (valid) {
          int base = (b * 3 * H + out_y) * W + out_x;
          output[base]             = sum0;  // Channel 0
          output[base + H * W]     = sum1;  // Channel 1  
          output[base + 2 * H * W] = sum2;  // Channel 2
      }
}


// =======================================================
// OTHER KERNELS
// =======================================================
__global__ void relu_opt(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = fmaxf(0.0f, x[i]);
}

__global__ void maxpool2x2_opt(
    const float* __restrict__ input,
    float* __restrict__ output,
    int B, int C, int H, int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int H2 = H / 2, W2 = W / 2;
    int total = B * C * H2 * W2;
    if (idx >= total) return;
    
    int x = idx % W2;
    int y = (idx / W2) % H2;
    int c = (idx / (W2 * H2)) % C;
    int b = idx / (W2 * H2 * C);
    
    int base = ((b * C + c) * H + y * 2) * W + x * 2;
    float m = fmaxf(fmaxf(input[base], input[base + 1]),
                    fmaxf(input[base + W], input[base + W + 1]));
    output[((b * C + c) * H2 + y) * W2 + x] = m;
}

__global__ void upsample2x_opt(
    const float* __restrict__ input,
    float* __restrict__ output,
    int B, int C, int H, int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int H2 = H * 2, W2 = W * 2;
    int total = B * C * H2 * W2;
    if (idx >= total) return;
    
    int x = idx % W2;
    int y = (idx / W2) % H2;
    int c = (idx / (W2 * H2)) % C;
    int b = idx / (W2 * H2 * C);
    
    output[idx] = input[((b * C + c) * H + (y >> 1)) * W + (x >> 1)];
}

__global__ void mse_loss_opt(
    const float* __restrict__ pred,
    const float* __restrict__ target,
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


// =======================================================
// EXPLICIT TEMPLATE INSTANTIATIONS
// =======================================================
template __global__ void conv2d_multi_oc<4>(const float*, const float*, const float*, float*, int, int, int, int, int);
template __global__ void conv2d_multi_oc<8>(const float*, const float*, const float*, float*, int, int, int, int, int);
template __global__ void conv2d_multi_oc_relu<4>(const float*, const float*, const float*, float*, int, int, int, int, int);
template __global__ void conv2d_multi_oc_relu<8>(const float*, const float*, const float*, float*, int, int, int, int, int);
template __global__ void conv2d_8x8_multi_oc<4>(const float*, const float*, const float*, float*, int, int, int);
template __global__ void conv2d_8x8_multi_oc<8>(const float*, const float*, const float*, float*, int, int, int);
template __global__ void conv2d_8x8_multi_oc_relu<4>(const float*, const float*, const float*, float*, int, int, int);
template __global__ void conv2d_8x8_multi_oc_relu<8>(const float*, const float*, const float*, float*, int, int, int);