#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <filesystem>

#include "../include/dataset.h"
#include "../include/autoencoder.h"
#include "../include/loss.h"


// Hàm ghi baseline performance
void logBaseline(
    const std::string& outpath,
    int epoch,
    int total_epochs,
    float avg_loss,
    double epoch_time_sec,
    double avg_batch_time_ms,
    double throughput_img_per_sec,
    int batch_size,
    int num_batches
) {
    std::ofstream out(outpath, std::ios::app);

    if (!out.is_open()) {
        std::cerr << "Cannot open: " << outpath << std::endl;
        return;
    }

    out << "========== Epoch " << epoch << "/" << total_epochs << " ==========\n";
    out << "Average Loss: " << avg_loss << "\n";
    out << "Epoch Time:  " << epoch_time_sec << " seconds\n";
    out << "Average Batch Time: " << avg_batch_time_ms << " ms\n";
    out << "Throughput: " << throughput_img_per_sec << " images/second\n";
    out << "Batch Size: " << batch_size << "\n";
    out << "Num Batches: " << num_batches << "\n";
    out << "==========================================\n\n";

    out.close();
}

void train(int BATCH_SIZE, int EPOCHS, float LEARNING_RATE, int LIMIT, const std::string& LOAD_PATH) {
    std::string weightOutPath = "/autoencoder_cpu.weights";
    std::string reportOutPath = "/cpu_training_log.txt";

    std::cout << "Saving weights to: " << weightOutPath << std::endl;
    std::cout << "Saving baseline report to: " << reportOutPath << std::endl;
    
    // load dataset
    CIFAR10Dataset dataset;
    if (!dataset.loadData("../cifar-10-batches-bin")) {
        std::cerr << "Failed to load dataset!" << std::endl;
        return;
    }

    Autoencoder model;
    MSELoss loss_fn;

    // Load pretrained weights
    if (!LOAD_PATH.empty()) {
        std::cout << "\nLoading pretrained weights from: " << LOAD_PATH << std::endl;
        model.loadWeights(LOAD_PATH);
    }
    
    int num_batches;
    int num_data = dataset.getSizeTrain();

    if (LIMIT > 0) {
        num_data = LIMIT;
        num_batches = LIMIT / BATCH_SIZE;
        if (num_batches < 1) num_batches = 1;
    } else {
        num_batches = dataset.getNumTrainBatches(BATCH_SIZE);
    }

    std::cout << "\n========== Training Started with "<<num_data<<" samples ==========" << std::endl;
    std::cout << "Batch size: " << BATCH_SIZE << std::endl;
    std::cout << "Num batches: " << num_batches << std::endl;
    std::cout << "Epochs: " << EPOCHS << std::endl;
    std::cout << "Learning rate: " << LEARNING_RATE << std::endl;
    std::cout << "======================================\n" << std::endl;


    for (int epoch = 0; epoch < EPOCHS; epoch++) {

        auto epoch_start = std::chrono::high_resolution_clock::now();

        dataset.shuffle();
        float total_loss = 0.0f;
        double total_batch_time_ms = 0.0;

        std::cout << "Epoch [" << epoch + 1 << "/" << EPOCHS <<"]\n";

        for (int batch_idx = 0; batch_idx < num_batches; batch_idx++) {

            int start_index = batch_idx * BATCH_SIZE;
            if (LIMIT > 0 && start_index >= LIMIT)
                break;

            auto b0 = std::chrono::high_resolution_clock::now();

            Tensor batch = dataset.getBatch(start_index, BATCH_SIZE, true);
            Tensor output = model.forward(batch);

            float batch_loss = loss_fn.forward(output, batch);
            total_loss += batch_loss;

            Tensor grad = loss_fn.backward(output, batch);
            model.backward(grad, LEARNING_RATE);

            auto b1 = std::chrono::high_resolution_clock::now();
            total_batch_time_ms += std::chrono::duration<double, std::milli>(b1 - b0).count();

            if ((batch_idx + 1) % 100 == 0) {
                std::cout << "  Batch [" << batch_idx + 1 << "/" << num_batches
                          << "] Loss: " << batch_loss << std::endl;
            }
        }

        auto epoch_end = std::chrono::high_resolution_clock::now();
        double epoch_time_sec = std::chrono::duration<double>(epoch_end - epoch_start).count();

        float avg_loss = total_loss / num_batches;
        double avg_batch_time_ms = total_batch_time_ms / num_batches;

        double throughput = (double)(BATCH_SIZE * num_batches) / epoch_time_sec;

        std::cout << "Epoch [" << epoch + 1 << "/" << EPOCHS << "] "
                  << "Loss: " << std::fixed << std::setprecision(6) << avg_loss
                  << " Time: " << epoch_time_sec << "s" << std::endl;

        // ghi baseline
        logBaseline(
            reportOutPath,
            epoch + 1,
            EPOCHS,
            avg_loss,
            epoch_time_sec,
            avg_batch_time_ms,
            throughput,
            BATCH_SIZE,
            num_batches
        );
    }

    // lưu trọng số vào trainX/weights.bin
    model.saveWeights(weightOutPath);

    std::cout << "\n========== Training Completed ==========" << std::endl;
    std::cout << "Weights saved to: " << weightOutPath << std::endl;
}

int main(int argc, char* argv[]) {
    int batch_size = 4;
    int epochs = 1;
    float lr = 0.001f;
    int data_limit = -1;
    std::string load_path = "";

    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--batch" && i + 1 < argc) {
            batch_size = std::atoi(argv[++i]);
        }
        else if (arg == "--epochs" && i + 1 < argc) {
            epochs = std::atoi(argv[++i]);
        }
        else if (arg == "--lr" && i + 1 < argc) {
            lr = std::atof(argv[++i]);
        }
        else if (arg == "--limit" && i + 1 < argc) {
            data_limit = std::atoi(argv[++i]);
        }
        else if (arg == "--load" && i + 1 < argc) {  
            load_path = argv[++i];
        }
    }

    std::cout << "CIFAR-10 Autoencoder - CPU Implementation" << std::endl;
    std::cout << "=========================================\n" << std::endl;

    train(batch_size, epochs, lr, data_limit, load_path);

    return 0;
}
