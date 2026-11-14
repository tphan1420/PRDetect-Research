Kiểm tra tính đúng đắn của quan hệ cú pháp → Class-Preserving Syntax Swap → Class-Conditional Graph MixStyle → Length-Balanced Training.
Cả 4 hướng đều nhằm mục tiêu cải thiện khả năng nhận diện văn bản AI trên dữ liệu khác với dữ liệu huấn luyện (Cross-Dataset), nhưng mỗi hướng giải quyết một vấn đề khác nhau.

## 1. Kiểm tra tính đúng đắn của quan hệ cú pháp

Mục tiêu: Đảm bảo R-GCN đang sử dụng đúng các quan hệ cú pháp mà spaCy cung cấp.

Hiện tại, log AI phát hiện nghi vấn về cách đánh số quan hệ và chiều của cạnh trong đồ thị.

Bạn cần làm gì?

- Kiểm tra `building_graph.py` tạo các cạnh và gán `relation ID` như thế nào.
- Kiểm tra `fwd_mask()` trong `QuangV3.py` có hiểu đúng các ID đó không.
- Kiểm tra Swap Augmentation có thực sự chọn các nút cùng loại quan hệ cú pháp không.

Kết quả mong muốn: Xác nhận mô hình đang xử lý đúng quan hệ cú pháp. Nếu phát hiện lỗi thì sửa và huấn luyện lại.

Đây là bước kiểm tra kỹ thuật, chưa phải một cải tiến nghiên cứu mới.

## 2. Class-Preserving Syntax Swap

Mục tiêu: Cải thiện Swap Augmentation hiện tại.

Hiện tại, mô hình có thể lấy embedding của một từ trong văn bản Human để thay vào embedding của một từ trong văn bản AI.

Bạn sẽ bổ sung một điều kiện: chỉ lấy embedding từ văn bản cùng nhãn.

Swap hiện tại

Graph AI

Graph Human

Có thể trộn khác nhãn

Swap cải tiến

Graph AI

Graph AI khác

Chỉ lấy cùng nhãn

Bạn cần làm gì?

- Sửa hàm `_augment()` để nút nhận và nút cho embedding phải cùng loại quan hệ cú pháp và cùng nhãn AI/Human.
- Giữ nguyên kiến trúc R-GCN, tỷ lệ Swap và các tham số khác.
- Huấn luyện lại và so sánh với V3 hiện tại.

Kết quả mong muốn: Table 3 tăng, trong khi Table 2 không giảm đáng kể.

## 3. Class-Conditional Graph MixStyle

Mục tiêu: Giúp R-GCN không quá phụ thuộc vào phong cách hoặc phân phối đặc trưng của bộ dữ liệu HC3.

Thay vì đổi embedding của từng nút như Swap, MixStyle pha trộn thống kê đặc trưng của hai đồ thị cùng nhãn.

Ví dụ: Hai văn bản AI có phong cách khác nhau, mô hình tạo ra biểu diễn đặc trưng biến đổi từ hai văn bản này.

Bạn cần làm gì?

- Viết thêm một module MixStyle.
- Tính Mean và Standard Deviation của embedding các nút trong từng graph.
- Chọn một graph khác cùng nhãn để pha trộn hai thống kê này.
- Áp dụng biến đổi khi training, tắt khi testing.
- Huấn luyện V3 + MixStyle, giữ các thành phần khác không đổi.

Kết quả mong muốn: Mô hình vẫn nhận diện tốt HC3 nhưng tổng quát hóa tốt hơn sang GPT-3.5-Mixed.

Đây là một giả thuyết nghiên cứu, chưa có gì đảm bảo MixStyle sẽ cải thiện kết quả.

## 4. Length-Balanced Training

Mục tiêu: Giúp mô hình không dựa quá nhiều vào độ dài văn bản để đoán nhãn AI/Human.

Log cho thấy:

| Dataset      | AI (trung bình) | Human (trung bình) |
| ------------ | --------------- | ------------------ |
| HC3 Train    | 180 từ          | 150 từ             |
| GPT-3.5 Test | 504 từ          | 803 từ             |

Nếu mô hình học rằng văn bản dài thường do AI tạo, quy luật đó có thể gây sai khi chuyển sang GPT-3.5.

Bạn cần làm gì?

- Thống kê độ dài văn bản AI và Human trong HC3.
- Chia văn bản thành các nhóm độ dài, ví dụ 0–100, 101–200, 201–300 từ.
- Lấy mẫu hoặc gán trọng số để phân phối độ dài giữa hai nhãn gần nhau hơn.
- Huấn luyện lại V3 với dữ liệu đã cân bằng, không thay đổi kiến trúc.

Kết quả mong muốn: Accuracy cross-dataset tăng, đặc biệt khi gặp văn bản có độ dài khác HC3.

Lưu ý: Nên kiểm tra bằng số nút trong đồ thị và theo dõi số mẫu bị loại khi cân bằng dữ liệu.

## 5. Nên triển khai theo thứ tự nào?

| Thứ tự | Hướng                            | Mức độ thực hiện | Mục đích                    |
| ------ | -------------------------------- | ---------------- | --------------------------- |
| 1      | Kiểm tra quan hệ cú pháp         | Dễ               | Xác minh tính đúng đắn      |
| 2      | Class-Preserving Syntax Swap     | Dễ – Trung bình  | Cải tiến augmentation       |
| 3      | Class-Conditional Graph MixStyle | Trung bình       | Tăng khả năng tổng quát hóa |
| 4      | Length-Balanced Training         | Trung bình       | Giảm phụ thuộc độ dài       |

Điều quan trọng: Không thêm cả ba phương pháp huấn luyện mới cùng lúc.

Hãy thử từng phương pháp độc lập trên V3 gốc, giữ cùng cách đánh giá và random seeds. Như vậy, bạn mới biết chính xác phương pháp nào thực sự có tác dụng.

Nếu mục tiêu là nghiên cứu khoa học, mình khuyên bạn ưu tiên Class-Preserving Syntax Swap và Class-Conditional Graph MixStyle, vì chúng tạo ra những giả thuyết rõ ràng, có thể thiết kế thực nghiệm để kiểm chứng.