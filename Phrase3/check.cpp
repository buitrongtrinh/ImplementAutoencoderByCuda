#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cfloat>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./check_bin_stats file.bin\n";
        return 1;
    }

    const char* filename = argv[1];
    const float eps = 1e-8f;

    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Cannot open file: " << filename << "\n";
        return 1;
    }

    // File size
    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (file_size % sizeof(float) != 0) {
        std::cerr << "Warning: file size not multiple of sizeof(float)\n";
    }

    size_t num_elements = file_size / sizeof(float);
    std::vector<float> data(num_elements);

    if (!file.read(reinterpret_cast<char*>(data.data()), file_size)) {
        std::cerr << "Failed to read file\n";
        return 1;
    }

    // Statistics
    double sum = 0.0;
    double sum_sq = 0.0;
    float min_val = FLT_MAX;
    float max_val = -FLT_MAX;

    size_t zero_count = 0;
    size_t nan_count = 0;
    size_t inf_count = 0;

    for (float v : data) {
        if (std::isnan(v)) {
            nan_count++;
            continue;
        }
        if (std::isinf(v)) {
            inf_count++;
            continue;
        }

        if (std::fabs(v) < eps)
            zero_count++;

        min_val = std::min(min_val, v);
        max_val = std::max(max_val, v);

        sum += v;
        sum_sq += v * v;
    }

    size_t valid_count = num_elements - nan_count - inf_count;
    double mean = (valid_count > 0) ? sum / valid_count : 0.0;
    double var = (valid_count > 0) ? (sum_sq / valid_count - mean * mean) : 0.0;
    double stddev = (var > 0) ? std::sqrt(var) : 0.0;

    // Output
    std::cout << "📄 File: " << filename << "\n";
    std::cout << "🔢 Elements      : " << num_elements << "\n";
    std::cout << "✅ Valid values  : " << valid_count << "\n";
    std::cout << "0️⃣ Near-zero     : " << zero_count
              << " (" << (100.0 * zero_count / num_elements) << "%)\n";
    std::cout << "NaN              : " << nan_count << "\n";
    std::cout << "Inf              : " << inf_count << "\n";
    std::cout << "⬇ Min            : " << min_val << "\n";
    std::cout << "⬆ Max            : " << max_val << "\n";
    std::cout << "📊 Mean           : " << mean << "\n";
    std::cout << "📈 Std            : " << stddev << "\n";

    return 0;
}
