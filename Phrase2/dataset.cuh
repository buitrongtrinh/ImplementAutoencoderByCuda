#pragma once

#include <string>
#include <random>

#define NUM_TRAIN 50000
#define NUM_TEST 10000
#define IMAGE_SIZE 3072

class CIFAR10Dataset {
private:
    std::vector<float> train_images;
    std::vector<float> test_images;
    std::vector<int> train_labels;
    std::vector<int> test_labels;
    
    std::vector<int> shuffle_indices;
    std::mt19937 rng;
    
    bool loadBatch(const std::string& filename, 
                   std::vector<float>& images, 
                   std::vector<int>& labels);
    
public:
    CIFAR10Dataset();
    bool loadData(const std::string& data_path);
    void shuffle();
    float* getBatch(int start_idx, int batch_size, bool is_train = true);
    int getNumTrainBatches(int batch_size) const;
    int getNumTestBatches(int batch_size) const;
    int getSizeTrain() const;
    int getSizeTest() const;
};