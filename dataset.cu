#include "dataset.cuh"
#include <iostream>
#include <fstream>
#include <algorithm>

CIFAR10Dataset::CIFAR10Dataset() : rng(std::random_device{}()) {}

bool CIFAR10Dataset::loadData(const std::string& data_path) {
    std::cout << "Loading CIFAR-10 dataset..." << std::endl;
    
    train_images.reserve(NUM_TRAIN * IMAGE_SIZE);
    train_labels.reserve(NUM_TRAIN);
    
    for (int batch = 1; batch <= 5; batch++) {
        std::string filename = data_path + "/data_batch_" + std::to_string(batch) + ".bin";
        if (!loadBatch(filename, train_images, train_labels)) {
            std::cerr << "Failed to load " << filename << std::endl;
            return false;
        }
    }
    
    std::string test_file = data_path + "/test_batch.bin";
    if (!loadBatch(test_file, test_images, test_labels)) {
        std::cerr << "Failed to load " << test_file << std::endl;
        return false;
    }
    
    std::cout << "Dataset loaded successfully!" << std::endl;
    std::cout << "Training samples: " << train_labels.size() << std::endl;
    std::cout << "Test samples: " << test_labels.size() << std::endl;
    
    shuffle_indices.resize(train_labels.size());
    for (size_t i = 0; i < shuffle_indices.size(); i++) {
        shuffle_indices[i] = i;
    }
    
    return true;
}

bool CIFAR10Dataset::loadBatch(const std::string& filename, 
                               std::vector<float>& images, 
                               std::vector<int>& labels) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    const int record_size = 1 + 3072;
    std::vector<unsigned char> buffer(record_size);
    
    while (file.read(reinterpret_cast<char*>(buffer.data()), record_size)) {
        labels.push_back(buffer[0]);
        
        for (int i = 1; i < record_size; i++) {
            images.push_back(buffer[i] / 255.0f);
        }
    }
    
    file.close();
    return true;
}

void CIFAR10Dataset::shuffle() {
    std::shuffle(shuffle_indices.begin(), shuffle_indices.end(), rng);
}

float* CIFAR10Dataset::getBatch(int start_idx, int batch_size, bool is_train) {
    const auto& images = is_train ? train_images : test_images;
    int num_samples = is_train ? NUM_TRAIN : NUM_TEST;

    int actual_batch = std::min(batch_size, num_samples - start_idx);
    float* batch = new float[actual_batch * IMAGE_SIZE];

    for (int b = 0; b < actual_batch; b++) {
        int idx = is_train ? shuffle_indices[start_idx + b]
                           : start_idx + b;

        // copy 1 image (3072 floats)
        std::copy(
            images.begin() + idx * IMAGE_SIZE,
            images.begin() + (idx + 1) * IMAGE_SIZE,
            batch + b * IMAGE_SIZE
        );
    }

    return batch;
}


int CIFAR10Dataset::getNumTrainBatches(int batch_size) const {
    return (train_labels.size() + batch_size - 1) / batch_size;
}

int CIFAR10Dataset::getNumTestBatches(int batch_size) const {
    return (test_labels.size() + batch_size - 1) / batch_size;
}

int CIFAR10Dataset::getSizeTrain() const{
    return NUM_TRAIN;
}
int CIFAR10Dataset::getSizeTest() const{
    return NUM_TEST;
}