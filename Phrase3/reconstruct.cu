#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include "dataset.cuh"
#include "gpu_autoencoder.cuh"
#include <sys/stat.h>

void savePPM(const std::string& filename, const float* data, int width, int height) {
    std::ofstream f(filename, std::ios::binary);
    if (!f.is_open()) return;
    
    f << "P6\n" << width << " " << height << "\n255\n";
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < 3; c++) {
                int idx = c * height * width + y * width + x;
                float val = std::max(0.0f, std::min(1.0f, data[idx]));
                unsigned char pixel = static_cast<unsigned char>(val * 255.0f);
                f.write(reinterpret_cast<char*>(&pixel), 1);
            }
        }
    }
    f.close();
}

int main(int argc, char* argv[]) {
    
    const int BATCH_SIZE = 64;
    const int NUM_IMAGES = 10;
    
    std::string weights_path = "autoencoder_gpu.weights";
    std::string data_path = "../cifar-10-batches-bin";
    std::string output_dir = "./output_images";
    
    if (argc > 1) weights_path = argv[1];
    if (argc > 2) data_path = argv[2];
    if (argc > 3) output_dir = argv[3];
    
    mkdir(output_dir.c_str(), 0755);
    
    // Load
    CIFAR10Dataset dataset;
    dataset.loadData(data_path);
    
    GPUAutoencoder model(BATCH_SIZE, 0.0f);
    model.loadWeights(weights_path);
    
    // Forward
    float* batch = dataset.getBatch(0, BATCH_SIZE, false);
    std::vector<float> output(BATCH_SIZE * 3 * 32 * 32);
    float loss = model.forward(batch, output.data());
    
    // Debug output range
    float min_val = output[0], max_val = output[0];
    int neg = 0, over = 0;
    for (int i = 0; i < BATCH_SIZE * 3 * 32 * 32; i++) {
        min_val = std::min(min_val, output[i]);
        max_val = std::max(max_val, output[i]);
        if (output[i] < 0) neg++;
        if (output[i] > 1) over++;
    }
    
    std::cout << "MSE Loss: " << loss << "\n";
    std::cout << "Output range: [" << min_val << ", " << max_val << "]\n";
    std::cout << "Out of [0,1]: " << neg << " negative, " << over << " >1\n";
    
    // Save images
    for (int i = 0; i < NUM_IMAGES; i++) {
        int offset = i * 3 * 32 * 32;
        savePPM(output_dir + "/orig_" + std::to_string(i) + ".ppm", batch + offset, 32, 32);
        savePPM(output_dir + "/recon_" + std::to_string(i) + ".ppm", output.data() + offset, 32, 32);
    }
    
    std::cout << "Saved " << NUM_IMAGES << " image pairs to " << output_dir << "\n";
    
    delete[] batch;
    return 0;
}