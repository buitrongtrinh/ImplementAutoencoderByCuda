#pragma once

#include "../include/tensor.h"
#include <string>
#include <random>

class CIFAR10Dataset {
private:
    std::vector<float> train_images;
    std::vector<float> test_images;
    std::vector<int> train_labels;
    std::vector<int> test_labels;
    
    int num_train = 50000;
    int num_test = 10000;
    int image_size = 3 * 32 * 32;
    
    std::vector<int> shuffle_indices;
    std::mt19937 rng;
    
    bool loadBatch(const std::string& filename, 
                   std::vector<float>& images, 
                   std::vector<int>& labels);
    
public:
    CIFAR10Dataset();
    bool loadData(const std::string& data_path);
    void shuffle();
    Tensor getBatch(int start_idx, int batch_size, bool is_train = true);
    int getNumTrainBatches(int batch_size) const;
    int getNumTestBatches(int batch_size) const;
    int getSizeTrain() const;
    int getSizeTest() const;
};