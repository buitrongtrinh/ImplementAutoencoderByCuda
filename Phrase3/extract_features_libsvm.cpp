#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <limits>
#include <cmath>

#include "dataset.cuh"
#include "gpu_autoencoder.cuh"

// ================= STATS STRUCT =================
struct FeatureStats {
    long long samples = 0;
    long long nonzero = 0;
    double sum = 0.0;
    float minv = std::numeric_limits<float>::max();
    float maxv = std::numeric_limits<float>::lowest();
    long long label_count[10] = {0};
};

// ================= LIBSVM WRITER + STATS =================
void writeBatchLibSVM(
    std::ofstream& out,
    const std::vector<float>& latent,
    const std::vector<int>& labels,
    int batchSize,
    FeatureStats& stats
) {
    const int C = 128;
    const int H = 8;
    const int W = 8;
    const int FEAT = C * H * W; // 8192

    for (int b = 0; b < batchSize; b++) {

        int label = labels[b];
        out << label;
        stats.samples++;
        stats.label_count[label]++;

        int base = b * FEAT;
        int idx = 1;

        for (int i = 0; i < FEAT; i++) {
            float v = latent[base + i];

            stats.minv = std::min(stats.minv, v);
            stats.maxv = std::max(stats.maxv, v);
            stats.sum += v;

            if (v != 0.0f) {
                out << " " << idx << ":" << v;
                stats.nonzero++;
            }
            idx++;
        }
        out << "\n";
    }
}

// ================= PRINT STATS =================
void printStats(const std::string& name, const FeatureStats& s) {
    const int FEAT = 128 * 8 * 8;
    long long total_feat = s.samples * FEAT;

    std::cout << "\n========== STATS: " << name << " ==========\n";
    std::cout << "Samples           : " << s.samples << "\n";
    std::cout << "Feature dim       : " << FEAT << "\n";
    std::cout << "Non-zero / sample : "
              << (double)s.nonzero / s.samples << "\n";
    std::cout << "Sparsity (%)      : "
              << 100.0 * (1.0 - (double)s.nonzero / total_feat) << "\n";
    std::cout << "Feature min       : " << s.minv << "\n";
    std::cout << "Feature max       : " << s.maxv << "\n";
    std::cout << "Feature mean      : "
              << s.sum / total_feat << "\n";

    std::cout << "Label distribution:\n";
    for (int i = 0; i < 10; i++) {
        std::cout << "  " << i << ": " << s.label_count[i] << "\n";
    }
}

// ================= MAIN =================
int main() {

    const int BATCH_SIZE = 64;
    const int FEAT_DIM = 128 * 8 * 8;

    std::cout << "========== FEATURE EXTRACTION (LIBSVM) ==========\n";

    // ================= LOAD DATASET =================
    CIFAR10Dataset dataset;
    dataset.loadData("../cifar-10-batches-bin");

    // ================= LOAD MODEL =================
    GPUAutoencoder model(BATCH_SIZE, 0.0f);
    model.loadWeights("autoencoder_gpu.weights");

    // ================= OUTPUT FILES =================
    std::ofstream train_svm("train.libsvm");
    std::ofstream test_svm("test.libsvm");

    if (!train_svm || !test_svm) {
        std::cerr << "Cannot open LIBSVM output files!\n";
        return 1;
    }

    // ================= BUFFERS =================
    std::vector<float> latent(BATCH_SIZE * FEAT_DIM);
    std::vector<int> labels(BATCH_SIZE);

    FeatureStats train_stats, test_stats;

    // ================= TRAIN =================
    int train_batches = dataset.getNumTrainBatches(BATCH_SIZE);
    std::cout << "[Extract] TRAIN\n";

    for (int b = 0; b < train_batches; b++) {

        float* batch = dataset.getBatch(b * BATCH_SIZE, BATCH_SIZE, true);
        model.extractFeatures(batch, latent.data());

        for (int i = 0; i < BATCH_SIZE; i++)
            labels[i] = dataset.getTrainLabel(b * BATCH_SIZE + i);

        writeBatchLibSVM(
            train_svm, latent, labels, BATCH_SIZE, train_stats
        );

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

        for (int i = 0; i < BATCH_SIZE; i++)
            labels[i] = dataset.getTestLabel(b * BATCH_SIZE + i);

        writeBatchLibSVM(
            test_svm, latent, labels, BATCH_SIZE, test_stats
        );

        delete[] batch;

        if (b % 50 == 0)
            std::cout << "Test batch " << b << "/" << test_batches << "\n";
    }

    train_svm.close();
    test_svm.close();

    // ================= PRINT STATS =================
    printStats("TRAIN", train_stats);
    printStats("TEST", test_stats);

    std::cout << "\n========== EXTRACTION DONE ==========\n";
    std::cout << "Files:\n";
    std::cout << "  - train.libsvm\n";
    std::cout << "  - test.libsvm\n";

    return 0;
}
