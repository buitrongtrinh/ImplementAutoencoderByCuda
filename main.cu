#include <iostream>
#include "dataset.cuh"
#include "gpu_autoencoder.cuh"

struct GPUTimer {
    cudaEvent_t start, stop;
    GPUTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    ~GPUTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    void tic() { cudaEventRecord(start); }
    float toc() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

int main() {

    const int BATCH_SIZE = 64;
    const int EPOCHS = 20;
    const float LR = 0.001f;

    CIFAR10Dataset dataset;
    dataset.loadData("cifar-10-batches-bin");

    GPUAutoencoder model(BATCH_SIZE, LR);

    int num_batches = dataset.getNumTrainBatches(BATCH_SIZE);

    GPUTimer timer;

    std::cout << "\n========== GPU TRAINING ==========\n";

    for (int epoch = 0; epoch < EPOCHS; epoch++) {

        dataset.shuffle();
        float epoch_loss = 0.0f;

        timer.tic();

        for (int b = 0; b < num_batches; b++) {

            float* batch =
                dataset.getBatch(b * BATCH_SIZE, BATCH_SIZE, true);

            std::vector<float> output(BATCH_SIZE * 3 * 32 * 32);

            float loss = model.forward(batch, output.data());

            // backward + SGD (bạn đã có test code, tái dùng)
            model.backward();   // nếu bạn gộp backward vào class
                               // hoặc gọi thủ công như test trước

            epoch_loss += loss;

            delete[] batch;

            if (b % 50 == 0) {
                std::cout << "[Epoch " << epoch
                          << "/"<<EPOCHS<<"] Batch " << b
                          << "/" << num_batches
                          << " | Loss = " << loss << "\n";
            }
        }

        float time_ms = timer.toc();
        std::cout << "Epoch " << epoch
                  << " | Avg Loss = " << epoch_loss / num_batches
                  << " | Time = " << time_ms << " ms\n";
    }

    model.saveWeights("autoencoder_gpu.weights");

    std::cout << "========== TRAINING DONE ==========\n";
    return 0;
}
