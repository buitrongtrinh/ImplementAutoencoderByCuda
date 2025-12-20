#include "../include/autoencoder.h"
#include <iostream>
#include <fstream>

Autoencoder::Autoencoder() 
    : conv1(3, 16, 3, 1, 1),
      conv2(16, 8, 3, 1, 1),
      conv3(8, 8, 3, 1, 1),
      conv4(8, 16, 3, 1, 1),
      conv5(16, 3, 3, 1, 1) {
    
    std::cout << "Autoencoder initialized!" << std::endl;
}

Tensor Autoencoder::forward(const Tensor& input) {
    auto x = conv1.forward(input);
    x = relu1.forward(x);
    x = pool.forward(x);
    
    x = conv2.forward(x);
    x = relu2.forward(x);
    
    x = conv3.forward(x);
    x = relu3.forward(x);
    
    x = upsample.forward(x);
    
    x = conv4.forward(x);
    x = relu4.forward(x);
    
    x = conv5.forward(x);
    x = relu5.forward(x);
    
    return x;
}

void Autoencoder::backward(const Tensor& grad_output, float learning_rate) {
    auto grad = relu5.backward(grad_output);
    grad = conv5.backward(grad);
    
    grad = relu4.backward(grad);
    grad = conv4.backward(grad);
    
    grad = upsample.backward(grad);
    
    grad = relu3.backward(grad);
    grad = conv3.backward(grad);
    
    grad = relu2.backward(grad);
    grad = conv2.backward(grad);
    
    grad = pool.backward(grad);
    grad = relu1.backward(grad);
    grad = conv1.backward(grad);
    
    conv1.updateWeights(learning_rate);
    conv2.updateWeights(learning_rate);
    conv3.updateWeights(learning_rate);
    conv4.updateWeights(learning_rate);
    conv5.updateWeights(learning_rate);
}

Tensor Autoencoder::extractFeatures(const Tensor& input) {
    auto x = conv1.forward(input);
    x = relu1.forward(x);
    x = pool.forward(x);
    x = conv2.forward(x);
    x = relu2.forward(x);
    return x;
}

void Autoencoder::saveWeights(const std::string& path) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file to save weights: " << path << std::endl;
        return;
    }

    auto saveTensor = [&](const Tensor& t) {
        file.write(reinterpret_cast<const char*>(t.data.data()),
           t.data.size() * sizeof(float));

    };

    // Save all conv layer weights & biases
    saveTensor(conv1.getWeights());  
    saveTensor(conv1.getBias());

    saveTensor(conv2.getWeights());  
    saveTensor(conv2.getBias());

    saveTensor(conv3.getWeights());  
    saveTensor(conv3.getBias());

    saveTensor(conv4.getWeights());  
    saveTensor(conv4.getBias());

    saveTensor(conv5.getWeights());  
    saveTensor(conv5.getBias());

    file.close();
    std::cout << "Weights saved to: " << path << std::endl;
}

void Autoencoder::loadWeights(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open weight file: " << path << std::endl;
        return;
    }

    auto loadTensor = [&](Tensor& t) {
        file.read(reinterpret_cast<char*>(t.data.data()),
           t.data.size() * sizeof(float));
    };

    // Load all conv layer weights & biases
    loadTensor(conv1.getWeights());  
    loadTensor(conv1.getBias());

    loadTensor(conv2.getWeights());  
    loadTensor(conv2.getBias());

    loadTensor(conv3.getWeights());  
    loadTensor(conv3.getBias());

    loadTensor(conv4.getWeights());  
    loadTensor(conv4.getBias());

    loadTensor(conv5.getWeights());  
    loadTensor(conv5.getBias());

    file.close();
    std::cout << "Weights loaded from: " << path << std::endl;
}
