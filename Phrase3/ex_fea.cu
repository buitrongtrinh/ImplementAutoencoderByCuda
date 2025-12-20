#include <iostream>
#include <fstream>
#include <vector>
#include "dataset.cuh"
#include "gpu_autoencoder.cuh"

int main() {

    const int BATCH_SIZE = 64;

    std::cout << "========== FEATURE EXTRACTION ==========\n";

    CIFAR10Dataset dataset;
    dataset.loadData("../cifar-10-batches-bin");

    GPUAutoencoder model(BATCH_SIZE, 0.0f);
    model.loadWeights("autoencoder_gpu.weights");

    std::ofstream train_feat("train_features.bin", std::ios::binary);
    std::ofstream train_lbl ("train_labels.bin",   std::ios::binary);
    std::ofstream test_feat ("test_features.bin",  std::ios::binary);
    std::ofstream test_lbl  ("test_labels.bin",    std::ios::binary);

    if (!train_feat || !train_lbl || !test_feat || !test_lbl) {
        std::cerr << "Cannot open output files!\n";
        return 1;
    }

    std::vector<float> latent(BATCH_SIZE * 128 * 8 * 8);

    // ================= TRAIN =================
    int train_batches = dataset.getNumTrainBatches(BATCH_SIZE);
    std::cout << "[Extract] TRAIN\n";

    for (int b = 0; b < train_batches; b++) {

        float* batch = dataset.getBatch(b * BATCH_SIZE, BATCH_SIZE, true);
        model.extractFeatures(batch, latent.data());

        train_feat.write(
            (char*)latent.data(),
            latent.size() * sizeof(float)
        );

        for (int i = 0; i < BATCH_SIZE; i++) {
            int lbl = dataset.getTrainLabel(b * BATCH_SIZE + i);
            train_lbl.write((char*)&lbl, sizeof(int));
        }

        delete[] batch;

        if (b % 100 == 0)
            std::cout << "Train batch " << b << "/" << train_batches << "\n";
    }

    // ================= TEST =================
    int test_batches = dataset.getNumTestBatches(BATCH_SIZE);
    std::cout << "[Extract] TEST\n";

    for (int b = 0; b < test_batches; b++) {

        float* batch = dataset.getBatch(b * BATCH_SIZE, BATCH_SIZE, false);
        model.extractFeatures(batch, latent.data());

        test_feat.write(
            (char*)latent.data(),
            latent.size() * sizeof(float)
        );

        for (int i = 0; i < BATCH_SIZE; i++) {
            int lbl = dataset.getTestLabel(b * BATCH_SIZE + i);
            test_lbl.write((char*)&lbl, sizeof(int));
        }

        delete[] batch;

        if (b % 50 == 0)
            std::cout << "Test batch " << b << "/" << test_batches << "\n";
    }

    train_feat.close();
    train_lbl.close();
    test_feat.close();
    test_lbl.close();

    std::cout << "========== EXTRACTION DONE ==========\n";
    return 0;
}
