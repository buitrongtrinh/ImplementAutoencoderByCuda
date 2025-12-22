#include "gpu_autoencoder.cuh"
#include <iostream>
#include <random>
#include <cmath>
#include <fstream>

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

// ================= CONSTRUCTOR =================
GPUAutoencoder::GPUAutoencoder(int batch, float lr)
    : batch_size(batch), learning_rate(lr)
{
    allocateHostMemory();
    initializeWeights();
    allocateDeviceMemory();
    copyWeightsToGPU();

    std::cout << "[GPUAutoencoder] Initialized (Phase 2.1 – standalone)\n";
}

// ================= HOST MEMORY =================
void GPUAutoencoder::allocateHostMemory() {
    h_w1 = new float[3*3*3*256];
    h_w2 = new float[3*3*256*128];
    h_w3 = new float[3*3*128*128];
    h_w4 = new float[3*3*128*256];
    h_w5 = new float[3*3*256*3];

    h_b1 = new float[256];
    h_b2 = new float[128];
    h_b3 = new float[128];
    h_b4 = new float[256];
    h_b5 = new float[3];
}

// ================= WEIGHT INITIALIZATION =================
void GPUAutoencoder::initializeWeights() {
    std::mt19937 gen(42);

    auto init_w = [&](float* w, int n, int fan_in) {
        float std = sqrt(2.0f / fan_in);
        std::normal_distribution<float> dist(0.0f, std);
        for (int i = 0; i < n; i++)
            w[i] = dist(gen);
    };

    init_w(h_w1, 3*3*3*256, 3*3*3);
    init_w(h_w2, 3*3*256*128, 3*3*256);
    init_w(h_w3, 3*3*128*128, 3*3*128);
    init_w(h_w4, 3*3*128*256, 3*3*128);
    init_w(h_w5, 3*3*256*3, 3*3*256);

    std::fill(h_b1, h_b1 + 256, 0.01f);
    std::fill(h_b2, h_b2 + 128, 0.01f);
    std::fill(h_b3, h_b3 + 128, 0.01f);
    std::fill(h_b4, h_b4 + 256, 0.01f);
    std::fill(h_b5, h_b5 + 3,   0.0f);
}

// ================= DEVICE MEMORY =================
void GPUAutoencoder::allocateDeviceMemory() {

    // WEIGHTS
    CUDA_CHECK(cudaMalloc(&d_w1, 3*3*3*256*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w2, 3*3*256*128*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w3, 3*3*128*128*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w4, 3*3*128*256*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w5, 3*3*256*3*sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_b1, 256*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b2, 128*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b3, 128*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b4, 256*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b5, 3*sizeof(float)));

    // ACTIVATIONS
    CUDA_CHECK(cudaMalloc(&d_input,  batch_size*32*32*3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_o1,     batch_size*32*32*256*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_o2,     batch_size*16*16*256*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_o3,     batch_size*16*16*128*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_o4,     batch_size*8*8*128*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_o5,     batch_size*8*8*128*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_o6,     batch_size*16*16*128*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_o7,     batch_size*16*16*256*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_o8,     batch_size*32*32*256*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, batch_size*32*32*3*sizeof(float)));

    // GRADIENTS
    CUDA_CHECK(cudaMalloc(&d_dw1, 3*3*3*256*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dw2, 3*3*256*128*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dw3, 3*3*128*128*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dw4, 3*3*128*256*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dw5, 3*3*256*3*sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_db1, 256*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db2, 128*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db3, 128*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db4, 256*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db5, 3*sizeof(float)));
}

// ================= COPY TO GPU =================
void GPUAutoencoder::copyWeightsToGPU() {
    cudaMemcpy(d_w1, h_w1, 3*3*3*256*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w2, h_w2, 3*3*256*128*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w3, h_w3, 3*3*128*128*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w4, h_w4, 3*3*128*256*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w5, h_w5, 3*3*256*3*sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_b1, h_b1, 256*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, h_b2, 128*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b3, h_b3, 128*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b4, h_b4, 256*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b5, h_b5, 3*sizeof(float), cudaMemcpyHostToDevice);
}

// ================= DESTRUCTOR =================
GPUAutoencoder::~GPUAutoencoder() {

    delete[] h_w1; delete[] h_w2; delete[] h_w3;
    delete[] h_w4; delete[] h_w5;
    delete[] h_b1; delete[] h_b2; delete[] h_b3;
    delete[] h_b4; delete[] h_b5;

    cudaFree(d_w1); cudaFree(d_w2); cudaFree(d_w3);
    cudaFree(d_w4); cudaFree(d_w5);
    cudaFree(d_b1); cudaFree(d_b2); cudaFree(d_b3);
    cudaFree(d_b4); cudaFree(d_b5);

    cudaFree(d_input);
    cudaFree(d_o1); cudaFree(d_o2); cudaFree(d_o3); cudaFree(d_o4);
    cudaFree(d_o5); cudaFree(d_o6); cudaFree(d_o7); cudaFree(d_o8);
    cudaFree(d_output);

    cudaFree(d_dw1); cudaFree(d_dw2); cudaFree(d_dw3);
    cudaFree(d_dw4); cudaFree(d_dw5);
    cudaFree(d_db1); cudaFree(d_db2); cudaFree(d_db3);
    cudaFree(d_db4); cudaFree(d_db5);

    std::cout << "[GPUAutoencoder] Destroyed\n";
}
float GPUAutoencoder::forward(float* h_input, float* h_output) {
    int B = batch_size;
    int threads = 256;
    float h_loss = 0.0f;
    
    dim3 block16(16, 16);
    dim3 block8(8, 8);
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, B * 3 * 32 * 32 * sizeof(float), cudaMemcpyHostToDevice));
    
    // ================= ENCODER =================
    
    // Conv1 + ReLU: [B,3,32,32] -> [B,256,32,32]
    // Cin=3 specialized, 4 output channels per block
    // Grid: 2x2 tiles * (B * 256/4) = 2*2*B*64
    dim3 grid1(2, 2, B * (256 / 4));
    conv2d_cin3_oc4_relu<<<grid1, block16>>>(d_input, d_w1, d_b1, d_o1, B, 32, 32, 256);
    
    // MaxPool1: [B,256,32,32] -> [B,256,16,16]
    int n_pool1 = B * 256 * 16 * 16;
    maxpool2x2_opt<<<(n_pool1 + threads - 1) / threads, threads>>>(d_o1, d_o2, B, 256, 32, 32);
    
    // Conv2 + ReLU: [B,256,16,16] -> [B,128,16,16]
    // 8 output channels per block -> 128/8 = 16 oc groups
    dim3 grid2(1, 1, B * (128 / 8));
    conv2d_multi_oc_relu<8><<<grid2, block16>>>(d_o2, d_w2, d_b2, d_o3, B, 256, 16, 16, 128);
    
    // MaxPool2: [B,128,16,16] -> [B,128,8,8]
    int n_pool2 = B * 128 * 8 * 8;
    maxpool2x2_opt<<<(n_pool2 + threads - 1) / threads, threads>>>(d_o3, d_o4, B, 128, 16, 16);
    
    // ================= DECODER =================
    
    // Conv3 + ReLU: [B,128,8,8] -> [B,128,8,8]
    // 8 output channels per block -> 128/8 = 16 oc groups
    dim3 grid3(B, 128 / 8);
    conv2d_8x8_multi_oc_relu<8><<<grid3, block8>>>(d_o4, d_w3, d_b3, d_o5, B, 128, 128);
    
    // Upsample1: [B,128,8,8] -> [B,128,16,16]
    int n_up1 = B * 128 * 16 * 16;
    upsample2x_opt<<<(n_up1 + threads - 1) / threads, threads>>>(d_o5, d_o6, B, 128, 8, 8);
    
    // Conv4 + ReLU: [B,128,16,16] -> [B,256,16,16]
    // 8 output channels per block -> 256/8 = 32 oc groups
    dim3 grid4(1, 1, B * (256 / 8));
    conv2d_multi_oc_relu<8><<<grid4, block16>>>(d_o6, d_w4, d_b4, d_o7, B, 128, 16, 16, 256);
    
    // Upsample2: [B,256,16,16] -> [B,256,32,32]
    int n_up2 = B * 256 * 32 * 32;
    upsample2x_opt<<<(n_up2 + threads - 1) / threads, threads>>>(d_o7, d_o8, B, 256, 16, 16);
    
    // Conv5 (NO ReLU): [B,256,32,32] -> [B,3,32,32]
    // Cout=3 specialized, tính tất cả 3 output cùng lúc
    dim3 grid5(2, 2, B);
    conv2d_cout3_all<<<grid5, block16>>>(d_o8, d_w5, d_b5, d_output, B, 256, 32, 32);
    
    // ================= LOSS =================
    int n_out = B * 3 * 32 * 32;
    
    float* d_loss;
    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));
    
    mse_loss_opt<<<(n_out + threads - 1) / threads, threads>>>(d_output, d_input, d_loss, n_out);
    
    CUDA_CHECK(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_loss));

    h_loss /= n_out;
    
    CUDA_CHECK(cudaMemcpy(h_output, d_output, n_out * sizeof(float), cudaMemcpyDeviceToHost));
    return h_loss;
}
void GPUAutoencoder::saveWeights(const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "Cannot open file to save weights\n";
        return;
    }

    // copy weights back to host
    cudaMemcpy(h_w1, d_w1, 3*3*3*256*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_w2, d_w2, 3*3*256*128*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_w3, d_w3, 3*3*128*128*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_w4, d_w4, 3*3*128*256*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_w5, d_w5, 3*3*256*3*sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemcpy(h_b1, d_b1, 256*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b2, d_b2, 128*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b3, d_b3, 128*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b4, d_b4, 256*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b5, d_b5, 3*sizeof(float), cudaMemcpyDeviceToHost);

    // write to file
    f.write((char*)h_w1, 3*3*3*256*sizeof(float));
    f.write((char*)h_w2, 3*3*256*128*sizeof(float));
    f.write((char*)h_w3, 3*3*128*128*sizeof(float));
    f.write((char*)h_w4, 3*3*128*256*sizeof(float));
    f.write((char*)h_w5, 3*3*256*3*sizeof(float));

    f.write((char*)h_b1, 256*sizeof(float));
    f.write((char*)h_b2, 128*sizeof(float));
    f.write((char*)h_b3, 128*sizeof(float));
    f.write((char*)h_b4, 256*sizeof(float));
    f.write((char*)h_b5, 3*sizeof(float));

    f.close();
    std::cout << "[SAVE] Weights saved to " << path << "\n";
}

void GPUAutoencoder::backward() {

    int B = batch_size;
    int threads = 256;

    // ==================================================
    // 0. Clear ALL gradients
    // ==================================================
    cudaMemset(d_dw1, 0, 3*3*3*256*sizeof(float));
    cudaMemset(d_dw2, 0, 3*3*256*128*sizeof(float));
    cudaMemset(d_dw3, 0, 3*3*128*128*sizeof(float));
    cudaMemset(d_dw4, 0, 3*3*128*256*sizeof(float));
    cudaMemset(d_dw5, 0, 3*3*256*3*sizeof(float));

    cudaMemset(d_db1, 0, 256*sizeof(float));
    cudaMemset(d_db2, 0, 128*sizeof(float));
    cudaMemset(d_db3, 0, 128*sizeof(float));
    cudaMemset(d_db4, 0, 256*sizeof(float));
    cudaMemset(d_db5, 0, 3*sizeof(float));

    // ==================================================
    // 1. dL/dOutput (MSE)
    // ==================================================
    int n_out = B * 3 * 32 * 32;
    mse_grad<<<(n_out + threads - 1)/threads, threads>>>(
        d_output,     // pred
        d_input,      // target
        d_output,     // reuse as grad buffer
        n_out
    );
    // ✅ CRITICAL: Sync vì reuse d_output buffer
    CUDA_CHECK(cudaDeviceSynchronize());

    // ==================================================
    // 2. Conv5 backward
    // ==================================================
    conv_bias_backward<<<(n_out+threads-1)/threads, threads>>>(
        d_output, d_db5, B, 3, 32, 32
    );

    conv_weight_backward<<<(3*3*256*3+threads-1)/threads, threads>>>(
        d_o8, d_output, d_dw5, B, 256, 32, 32, 3
    );

    conv_input_backward<<<(B*256*32*32+threads-1)/threads, threads>>>(
        d_output, d_w5, d_o8, B, 256, 32, 32, 3
    );
    // ✅ Sync vì d_o8 được ghi, sẽ được đọc bởi upsample_backward
    CUDA_CHECK(cudaDeviceSynchronize());

    // ==================================================
    // 3. Upsample backward
    // ==================================================
    upsample2x_backward<<<(B*256*32*32+threads-1)/threads, threads>>>(
        d_o8, d_o7, B, 256, 16, 16
    );
    // ✅ Sync vì d_o7 được ghi (với atomic), sẽ được đọc/ghi bởi relu_backward
    CUDA_CHECK(cudaDeviceSynchronize());

    // ==================================================
    // 4. Conv4 backward
    // ==================================================
    relu_backward<<<(B*256*16*16+threads-1)/threads, threads>>>(
        d_o7, d_o7, d_o7, B*256*16*16
    );
    // ✅ Sync vì d_o7 in-place operation
    CUDA_CHECK(cudaDeviceSynchronize());

    conv_bias_backward<<<(B*256*16*16+threads-1)/threads, threads>>>(
        d_o7, d_db4, B, 256, 16, 16
    );

    conv_weight_backward<<<(3*3*128*256+threads-1)/threads, threads>>>(
        d_o6, d_o7, d_dw4, B, 128, 16, 16, 256
    );

    conv_input_backward<<<(B*128*16*16+threads-1)/threads, threads>>>(
        d_o7, d_w4, d_o6, B, 128, 16, 16, 256
    );
    // ✅ Sync vì d_o6 được ghi
    CUDA_CHECK(cudaDeviceSynchronize());

    // ==================================================
    // 5. Upsample backward
    // ==================================================
    upsample2x_backward<<<(B*128*16*16+threads-1)/threads, threads>>>(
        d_o6, d_o5, B, 128, 8, 8
    );
    // ✅ Sync vì d_o5 được ghi (với atomic)
    CUDA_CHECK(cudaDeviceSynchronize());

    // ==================================================
    // 6. Conv3 backward
    // ==================================================
    relu_backward<<<(B*128*8*8+threads-1)/threads, threads>>>(
        d_o5, d_o5, d_o5, B*128*8*8
    );
    // ✅ Sync vì d_o5 in-place operation
    CUDA_CHECK(cudaDeviceSynchronize());

    conv_bias_backward<<<(B*128*8*8+threads-1)/threads, threads>>>(
        d_o5, d_db3, B, 128, 8, 8
    );

    conv_weight_backward<<<(3*3*128*128+threads-1)/threads, threads>>>(
        d_o4, d_o5, d_dw3, B, 128, 8, 8, 128
    );

    conv_input_backward<<<(B*128*8*8+threads-1)/threads, threads>>>(
        d_o5, d_w3, d_o4, B, 128, 8, 8, 128
    );
    // ✅ Sync vì d_o4 được ghi
    CUDA_CHECK(cudaDeviceSynchronize());

    // ==================================================
    // 7. MaxPool backward
    // ==================================================
    maxpool2x2_backward<<<(B*128*8*8+threads-1)/threads, threads>>>(
        d_o3, d_o4, d_o4, d_o3, B, 128, 16, 16
    );
    // ✅ Sync vì d_o3 được ghi (với atomic)
    CUDA_CHECK(cudaDeviceSynchronize());

    // ==================================================
    // 8. Conv2 backward
    // ==================================================
    relu_backward<<<(B*128*16*16+threads-1)/threads, threads>>>(
        d_o3, d_o3, d_o3, B*128*16*16
    );
    // ✅ Sync vì d_o3 in-place operation
    CUDA_CHECK(cudaDeviceSynchronize());

    conv_bias_backward<<<(B*128*16*16+threads-1)/threads, threads>>>(
        d_o3, d_db2, B, 128, 16, 16
    );

    conv_weight_backward<<<(3*3*256*128+threads-1)/threads, threads>>>(
        d_o2, d_o3, d_dw2, B, 256, 16, 16, 128
    );

    conv_input_backward<<<(B*256*16*16+threads-1)/threads, threads>>>(
        d_o3, d_w2, d_o2, B, 256, 16, 16, 128
    );
    // ✅ Sync vì d_o2 được ghi
    CUDA_CHECK(cudaDeviceSynchronize());

    // ==================================================
    // 9. MaxPool backward
    // ==================================================
    maxpool2x2_backward<<<(B*256*16*16+threads-1)/threads, threads>>>(
        d_o1, d_o2, d_o2, d_o1, B, 256, 32, 32
    );
    // ✅ Sync vì d_o1 được ghi (với atomic)
    CUDA_CHECK(cudaDeviceSynchronize());

    // ==================================================
    // 10. Conv1 backward
    // ==================================================
    relu_backward<<<(B*256*32*32+threads-1)/threads, threads>>>(
        d_o1, d_o1, d_o1, B*256*32*32
    );
    // ✅ Sync vì d_o1 in-place operation
    CUDA_CHECK(cudaDeviceSynchronize());

    conv_bias_backward<<<(B*256*32*32+threads-1)/threads, threads>>>(
        d_o1, d_db1, B, 256, 32, 32
    );

    conv_weight_backward<<<(3*3*3*256+threads-1)/threads, threads>>>(
        d_input, d_o1, d_dw1, B, 3, 32, 32, 256
    );

    // ==================================================
    // 11. SGD UPDATE
    // ==================================================
    // ✅ CRITICAL: Sync before SGD updates to ensure all gradients are computed
    CUDA_CHECK(cudaDeviceSynchronize());

    // Now launch all SGD updates - they can run in parallel since they're independent
    sgd_update<<<(3*3*3*256+threads-1)/threads, threads>>>(d_w1, d_dw1, learning_rate, 3*3*3*256);
    sgd_update<<<(3*3*256*128+threads-1)/threads, threads>>>(d_w2, d_dw2, learning_rate, 3*3*256*128);
    sgd_update<<<(3*3*128*128+threads-1)/threads, threads>>>(d_w3, d_dw3, learning_rate, 3*3*128*128);
    sgd_update<<<(3*3*128*256+threads-1)/threads, threads>>>(d_w4, d_dw4, learning_rate, 3*3*128*256);
    sgd_update<<<(3*3*256*3+threads-1)/threads, threads>>>(d_w5, d_dw5, learning_rate, 3*3*256*3);

    sgd_update<<<1,256>>>(d_b1, d_db1, learning_rate, 256);
    sgd_update<<<1,128>>>(d_b2, d_db2, learning_rate, 128);
    sgd_update<<<1,128>>>(d_b3, d_db3, learning_rate, 128);
    sgd_update<<<1,256>>>(d_b4, d_db4, learning_rate, 256);
    sgd_update<<<1,32 >>>(d_b5, d_db5, learning_rate, 3);

    // ✅ Final sync to ensure all weight updates complete before returning
    CUDA_CHECK(cudaDeviceSynchronize());
}
void GPUAutoencoder::extractFeatures(float* h_input, float* h_latent) {
    int B = batch_size;
    int threads = 256;

    dim3 block16(16, 16);

    // copy input lên GPU
    CUDA_CHECK(cudaMemcpy(
        d_input, h_input,
        B * 3 * 32 * 32 * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    // ================= ENCODER ONLY =================

    // Conv1 + ReLU
    dim3 grid1(2, 2, B * (256 / 4));
    conv2d_cin3_oc4_relu<<<grid1, block16>>>(
        d_input, d_w1, d_b1, d_o1, B, 32, 32, 256
    );

    // MaxPool1
    int n_pool1 = B * 256 * 16 * 16;
    maxpool2x2_opt<<<(n_pool1 + threads - 1) / threads, threads>>>(
        d_o1, d_o2, B, 256, 32, 32
    );

    // Conv2 + ReLU
    dim3 grid2(1, 1, B * (128 / 8));
    conv2d_multi_oc_relu<8><<<grid2, block16>>>(
        d_o2, d_w2, d_b2, d_o3, B, 256, 16, 16, 128
    );

    // MaxPool2 → LATENT
    int n_pool2 = B * 128 * 8 * 8;
    maxpool2x2_opt<<<(n_pool2 + threads - 1) / threads, threads>>>(
        d_o3, d_o4, B, 128, 16, 16
    );

    // ================= COPY LATENT =================
    CUDA_CHECK(cudaMemcpy(
        h_latent,
        d_o4,
        B * 128 * 8 * 8 * sizeof(float),
        cudaMemcpyDeviceToHost
    ));
}
void GPUAutoencoder::loadWeights(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "Cannot open weights file: " << path << "\n";
        exit(1);
    }

    // ===== allocate host buffers =====
    std::vector<float> h_w1(3*3*3*256);
    std::vector<float> h_w2(3*3*256*128);
    std::vector<float> h_w3(3*3*128*128);
    std::vector<float> h_w4(3*3*128*256);
    std::vector<float> h_w5(3*3*256*3);

    std::vector<float> h_b1(256);
    std::vector<float> h_b2(128);
    std::vector<float> h_b3(128);
    std::vector<float> h_b4(256);
    std::vector<float> h_b5(3);

    // ===== read raw binary =====
    f.read((char*)h_w1.data(), h_w1.size()*sizeof(float));
    f.read((char*)h_w2.data(), h_w2.size()*sizeof(float));
    f.read((char*)h_w3.data(), h_w3.size()*sizeof(float));
    f.read((char*)h_w4.data(), h_w4.size()*sizeof(float));
    f.read((char*)h_w5.data(), h_w5.size()*sizeof(float));

    f.read((char*)h_b1.data(), h_b1.size()*sizeof(float));
    f.read((char*)h_b2.data(), h_b2.size()*sizeof(float));
    f.read((char*)h_b3.data(), h_b3.size()*sizeof(float));
    f.read((char*)h_b4.data(), h_b4.size()*sizeof(float));
    f.read((char*)h_b5.data(), h_b5.size()*sizeof(float));

    f.close();

    // ===== copy to device =====
    cudaMemcpy(d_w1, h_w1.data(), h_w1.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w2, h_w2.data(), h_w2.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w3, h_w3.data(), h_w3.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w4, h_w4.data(), h_w4.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w5, h_w5.data(), h_w5.size()*sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_b1, h_b1.data(), h_b1.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, h_b2.data(), h_b2.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b3, h_b3.data(), h_b3.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b4, h_b4.data(), h_b4.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b5, h_b5.data(), h_b5.size()*sizeof(float), cudaMemcpyHostToDevice);

    std::cout << "[INFO] Weights loaded successfully: " << path << "\n";
}