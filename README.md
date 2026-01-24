# Autoencoder-based Unsupervised Feature Learning System

**Đồ án cuối kỳ môn Lập trình Song song (Parallel Programming - CSC14120)**

## 📖 Giới thiệu (Introduction)

Feature engineering là một thách thức cơ bản trong Machine Learning. Dự án này xây dựng một hệ thống học đặc trưng không giám sát (unsupervised feature learning) dựa trên **Convolutional Autoencoder** cho bộ dữ liệu **CIFAR-10**, kết hợp với bộ phân lớp **SVM**.

Mục tiêu chính của đồ án là chứng minh sức mạnh của tính toán song song bằng cách tối ưu hóa tốc độ huấn luyện Autoencoder trên GPU sử dụng **CUDA**, giảm thiểu thời gian huấn luyện từ hàng giờ xuống còn vài giây mà vẫn đảm bảo hiệu năng phân loại thực tế.

## 👥 Thành viên thực hiện (Team Members)

| STT | Họ và Tên | MSSV | Vai trò & Đóng góp |
|:---:|:---|:---:|:---|
| 1 | **Bùi Trọng Trịnh** | 22120390 | Triển khai Phase 3: Code GPU Optimized, Shared Memory, Tối ưu hóa. Viết báo cáo Phase 3. |
| 2 | **Nguyễn Lê Phúc Thắng** | 22120332 | Code CPU Baseline, Data Loader. Viết báo cáo Phase 1, 2. |
| 3 | **Âu Lê Tuấn Nhật** | 22120250 | Tích hợp SVM, Visualization. Tổng hợp báo cáo. |

## ⚙️ Kiến trúc hệ thống (System Pipeline)

Hệ thống hoạt động theo pipeline 2 giai đoạn chính:

1.  **Giai đoạn 1: Unsupervised Feature Learning**
    * Huấn luyện Convolutional Autoencoder để tái tạo ảnh đầu vào mà không cần nhãn.
    * Mục tiêu: Học các đặc trưng (features) đại diện tốt nhất cho dữ liệu ảnh.

2.  **Giai đoạn 2: Supervised Classification**
    * Sử dụng phần **Encoder** (đã huấn luyện ở giai đoạn 1) để trích xuất đặc trưng từ tập dữ liệu.
    * Dùng các đặc trưng này để huấn luyện bộ phân lớp **SVM (Support Vector Machine)** cho bài toán phân loại hình ảnh.

## 📊 Dữ liệu (Dataset)

Dự án sử dụng bộ dữ liệu **CIFAR-10**:
* **Kích thước ảnh:** $32 \times 32$ pixels (RGB).
* **Số lượng:** 60,000 ảnh (50,000 cho tập train, 10,000 cho tập test).
* **Số lớp:** 10 lớp (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).
* **Tiền xử lý:** Chuẩn hóa giá trị pixel về đoạn $[0, 1]$. Layout dữ liệu: Batch - Channel - Height - Width.

## 🚀 Công nghệ & Tối ưu hóa (Technologies & Optimization)

Dự án tập trung vào việc so sánh và tối ưu hóa hiệu năng giữa CPU và GPU:

* **Baseline:** Triển khai thuật toán trên CPU để làm cơ sở so sánh.
* **GPU Acceleration:**
    * Sử dụng **CUDA** để song song hóa quá trình huấn luyện.
    * Tối ưu hóa bộ nhớ với **Shared Memory**.
    * Tối ưu hóa các hyperparameter: Learning rate schedules (cosine annealing, warmup), lựa chọn Optimizer (Adam, AdamW, SGD with momentum).
    * Kỹ thuật Regularization: Dropout, Weight decay.

## 📈 Kết quả thực nghiệm (Experimental Results)

### 1. Hiệu năng & Độ chính xác
* **Tăng tốc (Speedup):** Pipeline đạt được mức tăng tốc đáng kể nhờ GPU acceleration so với phiên bản CPU baseline.
* **Độ chính xác (Accuracy):** Sau khi trích xuất đặc trưng và huấn luyện với SVM, hệ thống đạt độ chính xác **67.69%** trên tập test.

### 2. Trực quan hóa tái tạo ảnh (Reconstruction Visualization)
Dưới đây là kết quả tái tạo ảnh từ tập Test sau khi training:

<div align="center">
  <img width="100%" alt="Reconstruction Results" src="https://github.com/user-attachments/assets/fd06e23c-3cf0-4e14-8932-9d68a17afdeb" />
</div>

**Nhận xét:**
* **Loss:** `0.0660035`
* **Đánh giá:** Autoencoder đã học được các đặc trưng cấu trúc chính của vật thể. Tuy nhiên, với mức loss hiện tại, ảnh tái tạo vẫn xuất hiện hiện tượng lệch màu (color shift) nhẹ so với ảnh gốc. Điều này có thể được cải thiện bằng cách điều chỉnh loss function hoặc tăng cường color augmentation.

## 📝 Kết luận (Conclusion)
Dự án chứng minh tầm quan trọng của phương pháp tối ưu có hệ thống trong Deep Learning và hiệu quả của lập trình song song trong việc xử lý các tác vụ tính toán nặng. Mặc dù độ chính xác chưa đạt mức SOTA, nhưng hệ thống đã giải quyết tốt bài toán về tối ưu hóa thời gian huấn luyện.

## 🔮 Hướng phát triển (Future Work)

Để thu hẹp khoảng cách với các mô hình state-of-the-art (SOTA), nhóm đề xuất các cải tiến trong tương lai:

1.  **Cải thiện kiến trúc:** Sử dụng kiến trúc mạng sâu hơn (Deeper Architectures).
2.  **Chiến lược huấn luyện:** Áp dụng các chiến lược training tốt hơn và tối ưu hóa end-to-end.
3.  **Data Augmentation:** Tăng cường dữ liệu (Color jittering,...) để kỳ vọng cải thiện +5-10% accuracy.
4.  **Transfer Learning:** Pre-train trên dataset lớn hơn (như ImageNet) và fine-tune trên CIFAR-10.
