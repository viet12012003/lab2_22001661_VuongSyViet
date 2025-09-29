# Lab 2: Building a Spark NLP Pipeline

## 1. Các bước triển khai

- Đọc dữ liệu từ file `data/c4-train.00000-of-01024-30K.json` (giới hạn 1000 dòng để xử lý nhanh).
- Tiền xử lý văn bản:
  - Tách từ (tokenization) bằng `Tokenizer` hoặc `RegexTokenizer`.
  - Loại bỏ stopwords với `StopWordsRemover`.
- Vector hóa:
  - Sử dụng `HashingTF` để chuyển từ thành vector đặc trưng (TF).
  - Sử dụng `IDF` để tính trọng số nghịch đảo tần suất tài liệu (TF-IDF).
- Thực hiện các pipeline nâng cao:
  - Thay đổi loại tokenizer.
  - Thay đổi kích thước vector hóa (numFeatures).
  - Thêm bước phân loại với `LogisticRegression`.
  - Thử vector hóa bằng `Word2Vec`.
- Ghi log và kết quả ra các file riêng biệt trong thư mục `log/` và `results/`.

## 2. Hướng dẫn chạy code và ghi log

- Cài đặt Java 17+, Scala 2.13, Spark 4.0.1.
- Chạy lệnh sau trong thư mục dự án:
  ```
  sbt run
  ```
- Kết quả sẽ được ghi ra các file:
  - Log: `log/lab17_metrics.log`, `log/lab17_metrics1.log`, ..., `log/lab17_metrics4.log`
  - Output: `results/lab17_pipeline_output.txt`, `results/lab17_pipeline_output1.txt`, ..., `results/lab17_pipeline_output4.txt`
- Có thể xem chi tiết các bước pipeline và thời gian thực thi trong file log.

## 3. Giải thích kết quả thu được

### 3.1. Kết quả pipeline cơ bản

- **File `results/lab17_pipeline_output.txt`** lưu 20 kết quả đầu ra đầu tiên của pipeline xử lý văn bản cơ bản:
  - Mỗi kết quả gồm:
    - Đoạn văn bản gốc (chỉ in ra tối đa 100 ký tự đầu để dễ quan sát).
    - Vector đặc trưng TF-IDF (dạng sparse vector): thể hiện mức độ quan trọng của từng từ trong văn bản so với toàn bộ tập dữ liệu. Vector này có kích thước 20.000 chiều, mỗi chiều là một chỉ số hash của từ.
  - Ý nghĩa: Kết quả cho thấy quá trình chuyển đổi văn bản sang dạng số để phục vụ các bài toán học máy. Các giá trị lớn trong vector thể hiện các từ nổi bật trong văn bản đó.

### 3.2. So sánh các pipeline nâng cao

- **File `results/lab17_pipeline_output1.txt`**:
  - Sử dụng Tokenizer thường (tách từ theo khoảng trắng), không dùng regex. Kết quả cho thấy số lượng token có thể ít hơn, một số dấu câu không được tách riêng, dẫn đến vector TF-IDF có thể khác biệt so với pipeline dùng RegexTokenizer.
- **File `results/lab17_pipeline_output2.txt`**:
  - Sử dụng RegexTokenizer và giảm kích thước vector hóa xuống 1000 chiều (`numFeatures=1000`). Kết quả cho thấy hiện tượng hash collision rõ rệt: nhiều từ khác nhau bị ánh xạ vào cùng một chiều, làm giảm độ phân biệt của vector. Điều này thể hiện rõ trong log với cảnh báo về số lượng từ vựng thực tế lớn hơn số chiều vector.
- **File `results/lab17_pipeline_output3.txt`**:
  - Thêm bước phân loại với `LogisticRegression`. Kết quả mỗi dòng gồm:
    - Văn bản gốc (rút gọn).
    - Nhãn (label) gán ngẫu nhiên (0 hoặc 1).
    - Dự đoán của mô hình (prediction).
    - Xác suất dự đoán (probability vector).
  - Ý nghĩa: Pipeline này minh họa cách kết hợp tiền xử lý văn bản với mô hình phân loại. Dữ liệu đầu vào chưa có nhãn thực nên label được sinh ngẫu nhiên chỉ để minh họa quy trình.
- **File `results/lab17_pipeline_output4.txt`**:
  - Sử dụng vector hóa bằng `Word2Vec`. Kết quả mỗi dòng gồm:
    - Văn bản gốc (rút gọn).
    - Vector embedding (100 chiều) đại diện cho toàn bộ văn bản, được tính trung bình từ embedding của các từ trong văn bản.
  - Ý nghĩa: Word2Vec giúp biểu diễn ngữ nghĩa của văn bản tốt hơn so với TF-IDF, có thể dùng cho các bài toán phân loại, clustering hoặc các tác vụ NLP nâng cao.

### 3.3. Ý nghĩa các file log

- Các file log (`lab17_metrics.log`, `lab17_metrics1.log`, ..., `lab17_metrics4.log`) ghi lại:
  - Thời gian thực thi từng bước (fit, transform).
  - Kích thước từ vựng thực tế sau tiền xử lý.
  - Cảnh báo nếu số chiều vector nhỏ hơn số từ vựng (hash collision).
  - Thông tin về loại tokenizer, vectorizer, classifier sử dụng trong từng pipeline.
  - Đường dẫn file log để tiện tra cứu.

### 3.4. Tổng kết

- Các kết quả cho thấy sự khác biệt rõ rệt giữa các pipeline về cách biểu diễn văn bản, độ chi tiết của vector và khả năng ứng dụng cho các bài toán NLP khác nhau.
- Việc ghi log chi tiết giúp dễ dàng đánh giá hiệu năng, phát hiện vấn đề và so sánh giữa các phương án triển khai.

## 4. Khó khăn gặp phải và cách giải quyết

- Lỗi khi sử dụng Kryo với Java 17+ (InaccessibleObjectException):
  -> Đã khắc phục bằng cách thêm các tùy chọn `--add-opens` vào `build.sbt` để mở quyền truy cập các module Java cần thiết.
- Hash collision khi numFeatures nhỏ hơn số từ vựng: Đã ghi chú rõ trong log và so sánh kết quả giữa các pipeline.
- Lỗi khi ghi file: Đã kiểm tra và đảm bảo tạo thư mục trước khi ghi.

## 5. Tài liệu tham khảo

Tài liệu chính thức Apache Spark MLlib:

- https://spark.apache.org/docs/latest/ml-guide.html
- https://spark.apache.org/docs/latest/ml-features.html
