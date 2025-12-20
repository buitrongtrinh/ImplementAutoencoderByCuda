#include <iostream>
#include "dataset.cuh"
#include "gpu_autoencoder.cuh"
#include <fstream>

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

void logGPUMemory(std::ostream& out, const std::string& tag) {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    float used  = (total_mem - free_mem) / (1024.0f * 1024.0f);
    float total = total_mem / (1024.0f * 1024.0f);

    out << "[GPU MEM][" << tag << "] "
        << used << " / " << total << " MB\n";
}

int main() {

    const int BATCH_SIZE = 64;
    const int EPOCHS = 20;
    const float LR = 0.001f;

    std::cout << "========== GPU AUTOENCODER TRAINING ==========\n";
    std::cout << "Batch size = " << BATCH_SIZE << "\n";
    std::cout << "Epochs     = " << EPOCHS << "\n";
    std::cout << "LR         = " << LR << "\n";

    std::ofstream log("gpu_training_log.txt");
    if (!log.is_open()) {
        std::cerr << "Cannot open log file!\n";
        return 1;
    }

    CIFAR10Dataset dataset;
    dataset.loadData("cifar-10-batches-bin");

    log << "========== GPU AUTOENCODER TRAINING ==========\n";
    log << "Batch size = " << BATCH_SIZE << "\n";
    log << "Epochs     = " << EPOCHS << "\n";
    log << "LR         = " << LR << "\n\n";

    // ===== GPU memory BEFORE model =====
    logGPUMemory(log,        "Before Model Init");

    GPUAutoencoder model(BATCH_SIZE, LR);

    // ===== GPU memory AFTER model init =====
    logGPUMemory(log,        "After Model Init");

    int num_batches = dataset.getNumTrainBatches(BATCH_SIZE);

    GPUTimer timer;
    float total_time_ms = 0.0f;
    std::cout<<"Training ....... \n";
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        std::cout << "[Epoch " << epoch << "]\n";
        log << "[Epoch " << epoch << "]\n";

        dataset.shuffle();
        float epoch_loss = 0.0f;

        timer.tic();

        for (int b = 0; b < num_batches; b++) {

            float* batch =
                dataset.getBatch(b * BATCH_SIZE, BATCH_SIZE, true);

            std::vector<float> output(BATCH_SIZE * 3 * 32 * 32);

            float loss = model.forward(batch, output.data());
            model.backward();

            epoch_loss += loss;
            delete[] batch;

            // peak memory: chỉ log 1 lần
            if (epoch == 0 && b == 0) {
                logGPUMemory(log,        "During Training (1st batch)");
            }
            if (b % 100 == 0) {
                std::cout << "=== Batch " << b << "/" << num_batches
                      << " | Loss = " << loss << "\n";
                log << "=== Batch " << b <<"/" << num_batches
                    << " | Loss = " << loss << "\n";
            }
        }

        float time_ms = timer.toc();
        total_time_ms += time_ms;

        float avg_loss = epoch_loss / num_batches;

        std::cout << "Epoch " << epoch
                  << " | Avg Loss = " << avg_loss
                  << " | Time = " << time_ms << " ms\n";

        log << "Epoch " << epoch
            << " | Avg Loss = " << avg_loss
            << " | Time = " << time_ms << " ms\n";
    }

    logGPUMemory(log,        "After Training");

    std::cout << "Total training time = "
              << total_time_ms << " ms\n";

    log << "\nTotal training time = "
        << total_time_ms << " ms\n";

    model.saveWeights("autoencoder_gpu.weights");

    log << "========== TRAINING DONE ==========\n";
    log.close();

    return 0;
}

