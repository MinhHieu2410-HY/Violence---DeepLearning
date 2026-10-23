# Phát hiện bạo lực bằng học sâu

Bài tập lớn nghiên cứu và so sánh hai hướng phát hiện dấu hiệu bạo lực trong ảnh: phát hiện trực tiếp bằng mô hình một giai đoạn và phát hiện người kết hợp phân loại hành vi. Dự án có giao diện demo Streamlit để thử nghiệm trên ảnh, video và nguồn camera.

**Tác giả:** Nguyễn Minh Hiếu — 12423049  
**Trường:** Đại học Sư phạm Kỹ thuật Hưng Yên

## 1. Mục tiêu

- Nhận diện hai lớp: `NonViolence` và `Violence`.
- So sánh khả năng định vị đối tượng và nhận diện bạo lực ở mức ảnh.
- Phân tích sự đánh đổi giữa bỏ sót và cảnh báo nhầm.
- Xây dựng demo hỗ trợ quan sát kết quả dự đoán, tập trung hiển thị cảnh báo bạo lực.

## 2. Phương pháp

| Hướng tiếp cận | Mô hình | Cách xử lý |
|---|---|---|
| Một giai đoạn | YOLOv8n | Dự đoán trực tiếp hộp và lớp hành vi |
| Một giai đoạn | EfficientDet-D0 | Dự đoán trực tiếp hộp và lớp hành vi |
| Hai giai đoạn | YOLOv8n + MobileNetV3-Large | Phát hiện người, gom nhóm, cắt vùng và phân loại |
| Hai giai đoạn | YOLOv8n + EfficientNet-B0 | Phát hiện người, gom nhóm, cắt vùng và phân loại |

Các detector trực tiếp được fine-tuning từ trọng số pretrained. Trong hai pipeline, YOLOv8n pretrained được giữ nguyên để phát hiện người; MobileNetV3-Large và EfficientNet-B0 được fine-tuning để phân loại hành vi.

Pipeline hai giai đoạn sử dụng DBSCAN để gom các hộp người, tạo vùng cắt có padding và đưa vào bộ phân loại. Điểm bạo lực của ảnh được lấy từ điểm cao nhất trong các vùng cắt. Ngưỡng kết luận được chọn trên tập validation rồi áp dụng cho tập test.

Mô hình xử lý từng khung hình độc lập; hệ thống chưa học trực tiếp diễn biến chuyển động theo thời gian.

## 3. Dữ liệu và EDA

Bộ dữ liệu gốc: [Violence — Roboflow, phiên bản 2](https://universe.roboflow.com/nguyen-minh-hieu-ev5ll/violence-bblcr/dataset/2).

| Tập dữ liệu | Số ảnh |
|---|---:|
| Train | 26.013 |
| Validation | 1.497 |
| Test | 748 |
| **Tổng** | **28.258** |

Tổng cộng **102.515 hộp gán nhãn**, gồm **70.918 NonViolence** và **31.597 Violence**. Một ảnh có thể chứa hộp của cả hai lớp. Ở mức ảnh, ảnh có ít nhất một hộp Violence được xem là ảnh bạo lực.

Một số kết quả khảo sát:

- Số hộp NonViolence nhiều hơn Violence khoảng **2,24 lần**.
- Trong mẫu EDA 900 ảnh, diện tích hộp trung bình của Violence chiếm khoảng **10,66%** diện tích ảnh, so với **5,59%** của NonViolence.
- Tập test có **577 ảnh bạo lực** và **171 ảnh không bạo lực**; cần đọc Precision, Recall và F1 cùng với Accuracy do phân bố ảnh mất cân bằng.

## 4. Kết quả thực nghiệm

Các số liệu dưới đây thuộc **thực nghiệm gốc trong báo cáo, trước các đợt fine-tuning bổ sung dữ liệu**.

### Định vị và nhận diện đối tượng

| Mô hình | mAP@0.50 (%) | mAP@0.50:0.95 (%) |
|---|---:|---:|
| **YOLOv8n** | **84,64** | **61,34** |
| EfficientDet-D0 | 74,64 | 44,74 |

### Nhận diện bạo lực ở mức ảnh

Đánh giá trên cùng **748 ảnh test**. Precision, Recall và F1 được báo cáo cho lớp Violence. Mỗi phương án sử dụng ngưỡng riêng đã chọn trên validation.

| Phương án | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) |
|---|---:|---:|---:|---:|
| **YOLOv8n** | **85,83** | **88,93** | 93,24 | **91,03** |
| EfficientDet-D0 | 84,36 | 85,28 | 96,36 | 90,48 |
| YOLOv8n + MobileNetV3-Large | 80,21 | 80,34 | 98,44 | 88,47 |
| YOLOv8n + EfficientNet-B0 | 78,74 | 78,87 | **98,96** | 87,78 |

### Bỏ sót và cảnh báo nhầm

| Phương án | Ảnh bạo lực bị bỏ sót / 577 | Ảnh không bạo lực bị báo nhầm / 171 |
|---|---:|---:|
| YOLOv8n | 39 | 67 |
| EfficientDet-D0 | 21 | 96 |
| YOLOv8n + MobileNetV3-Large | 9 | 139 |
| YOLOv8n + EfficientNet-B0 | 6 | 153 |

**Nhận xét:** YOLOv8n đạt F1 cao nhất trong thực nghiệm. Hai pipeline giảm bỏ sót nhưng tăng đáng kể cảnh báo nhầm, nên chưa cải thiện hiệu quả tổng thể. Recall cao không đồng nghĩa với khả năng phân biệt hai lớp tốt hơn.

## 5. Chạy demo Streamlit

Hướng dẫn dưới đây áp dụng khi tệp giao diện của bạn có tên `demo.py`. Cần bổ sung checkpoint và các tệp phụ thuộc đúng với phiên bản mã nguồn được đưa lên repository.

### Tạo môi trường trên Windows

Chạy trong thư mục chứa mã nguồn:

```bat
python -m venv venv
venv\Scripts\activate
```

Nếu repository có `requirements.txt`, cài thư viện bằng:

```bat
python -m pip install -r requirements.txt
```

Nếu chưa có tệp này, cần tạo danh sách thư viện từ các import thực tế trong mã nguồn. Demo YOLO thường sử dụng PyTorch, Ultralytics, OpenCV, Pillow và Streamlit; pipeline hai giai đoạn cần thêm thư viện tương ứng của classifier và DBSCAN.

### Cấu hình trọng số

1. Chuẩn bị checkpoint đã huấn luyện, ví dụ `best.pt` cho YOLOv8n.
2. Chỉnh đường dẫn trọng số trong `demo.py` theo vị trí tệp trên máy.
3. Kiểm tra thứ tự lớp của checkpoint trước khi lọc kết quả Violence.
4. Chạy giao diện:

```bat
python -m streamlit run demo.py
```

Đối với webcam dùng OpenCV, camera được mở trên máy chạy Python. Nếu triển khai ứng dụng trên máy chủ, cách này không tự truy cập webcam của người đang mở trình duyệt.

## 6. Huấn luyện và đánh giá

- Sử dụng học chuyển giao thay vì huấn luyện toàn bộ mô hình từ đầu.
- Detector dùng kích thước đầu vào 640; classifier dùng kích thước 224.
- Với classifier: freeze backbone trong 5 epoch đầu, sau đó unfreeze với learning rate nhỏ để tinh chỉnh các đặc trưng pretrained.
- Chọn checkpoint và ngưỡng trên validation; sử dụng test để đánh giá cuối cùng.
- Báo cáo riêng chỉ số mức box và mức ảnh vì hai phép đo có đơn vị đánh giá khác nhau.

Để tái lập chính xác, cần cung cấp notebook huấn luyện, cấu hình, phiên bản thư viện và checkpoint tương ứng. Các checkpoint fine-tuning bổ sung không được dùng thay cho checkpoint gốc khi tái lập bảng kết quả trên.

## 7. Giới hạn và hướng phát triển

- Dự đoán từng frame có thể nhầm bạo lực với đùa giỡn, ôm nhau hoặc hoạt động thể thao.
- Góc quay, ánh sáng, che khuất và độ phân giải khác dữ liệu huấn luyện có thể ảnh hưởng kết quả.
- Kết quả trên tập test chưa chứng minh khả năng tổng quát hóa sang camera độc lập; kiểm tra trùng tệp không loại trừ các frame gần giống nhau.
- Chưa có kiểm thử tải để khẳng định hiệu năng nhiều camera hoặc FPS realtime trong môi trường triển khai.
- Hướng phát triển: bổ sung dữ liệu phù hợp bối cảnh, giảm cảnh báo nhầm, kết hợp thông tin nhiều frame và đo độ trễ trên phần cứng đích.

## 8. Lưu ý khi đưa mã nguồn lên GitHub

Không commit môi trường `venv`, dữ liệu lớn, khóa API, mật khẩu camera hoặc đường dẫn chứa thông tin truy cập. Checkpoint và dữ liệu cần có hướng dẫn tải riêng nếu không được cung cấp trong repository. Kiểm tra điều kiện sử dụng của dữ liệu trước khi phân phối lại.

Dự án phục vụ học tập và thực nghiệm. Cảnh báo của mô hình cần được con người kiểm tra trước khi kết luận về một sự việc.
