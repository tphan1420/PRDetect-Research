# Table 4 — So sánh GCN, RGCN và RGCN + R-Drop trên seed 2025–2026

## 1. Phạm vi và nguồn dữ liệu

Báo cáo này chỉ tổng hợp **hai seed 2025 và 2026** cho ba mô hình đã train trên HC3:

- **GCN**: checkpoint người thực hiện đã train lại bằng code đã chỉnh sửa; không phải trọng số gốc do tác giả bài báo cung cấp.
- **RGCN**: mô hình dùng quan hệ phụ thuộc cú pháp, không sử dụng R-Drop.
- **RGCN + R-Drop**: mô hình RGCN được train với R-Drop; các checkpoint có tên chứa `rdrop_a0.1`.

Nguồn kết quả:

- [Kết quả GCN](../result/table4_hc3_gcn_3seeds.zip).
- [Kết quả RGCN](../result/table4_hc3_rgcn_3seeds.zip).
- [Kết quả RGCN + R-Drop](../result/table4_hc3_rdrop_3seeds.zip).
- [Bài báo PRDetect, Table 4 và mục 5.5](../papers/2025.findings-naacl.464.pdf).

Mỗi mô hình có 10 report được sử dụng: 2 seed × 5 điều kiện. Các report tương ứng có cùng nhãn của 1.000 mẫu, gồm 494 mẫu AI và 506 mẫu human, và cùng ngưỡng phân loại 0.5. Quy ước nhãn là `0 = AI`, `1 = human`; F1 là F1 của lớp human.

Các điều kiện gồm văn bản gốc (Original) và bốn loại nhiễu ở tỷ lệ 25%: Insert, Repeat, Replace và Delete. Dòng `Detect` trong Table 4 được hiểu là Delete theo mô tả ở mục 5.5.

**Baseline dùng để so sánh là GCN người thực hiện đã train lại**, không phải số liệu PRDetect tác giả báo cáo trong paper. Bài báo chỉ được dẫn làm nguồn mô tả benchmark Table 4.

**Giới hạn khi diễn giải:** dữ liệu seed 2024 có trong các ZIP nhưng được loại khỏi phạm vi báo cáo theo yêu cầu. Trước đó, R-Drop giảm điểm nhiễu so với RGCN ở seed 2024; vì vậy không dùng riêng báo cáo hai seed này để tuyên bố cải thiện ổn định trên mọi seed. Các report test cũng không đủ xác minh toàn bộ cấu hình train hoặc chứng minh các cache graph hoàn toàn giống nhau.

## 2. Kết quả chi tiết của GCN

Accuracy và recall được biểu diễn theo phần trăm; AUC và F1 nằm trong khoảng 0–1. Giá trị được làm tròn khi trình bày, không làm tròn trước khi tính trung bình.

| Seed | Điều kiện | Accuracy (%) | AUC | F1 human | Recall AI (%) | Recall human (%) |
|---|---|---:|---:|---:|---:|---:|
| 2025 | Original | 99.00 | 0.9969 | 0.9901 | 98.79 | 99.21 |
| 2025 | Insert 25% | 87.50 | 0.9922 | 0.8899 | 74.90 | 99.80 |
| 2025 | Repeat 25% | 97.40 | 0.9972 | 0.9747 | 95.75 | 99.01 |
| 2025 | Replace 25% | 71.00 | 0.9804 | 0.7769 | 41.50 | 99.80 |
| 2025 | Delete 25% | 71.90 | 0.9674 | 0.7827 | 43.12 | 100.00 |
| 2026 | Original | 97.40 | 0.9960 | 0.9746 | 96.36 | 98.42 |
| 2026 | Insert 25% | 83.30 | 0.9872 | 0.8584 | 66.19 | 100.00 |
| 2026 | Repeat 25% | 94.40 | 0.9945 | 0.9469 | 90.08 | 98.62 |
| 2026 | Replace 25% | 67.80 | 0.9683 | 0.7583 | 35.02 | 99.80 |
| 2026 | Delete 25% | 68.40 | 0.9771 | 0.7620 | 36.03 | 100.00 |

Accuracy trung bình trên bốn tập nhiễu: **81.95% ở seed 2025**, **78.475% ở seed 2026**; trung bình hai seed là **80.2125%**.

## 3. Kết quả chi tiết của RGCN

| Seed | Điều kiện | Accuracy (%) | AUC | F1 human | Recall AI (%) | Recall human (%) |
|---|---|---:|---:|---:|---:|---:|
| 2025 | Original | 98.70 | 0.9977 | 0.9871 | 98.79 | 98.62 |
| 2025 | Insert 25% | 87.80 | 0.9952 | 0.8922 | 75.51 | 99.80 |
| 2025 | Repeat 25% | 96.20 | 0.9972 | 0.9637 | 92.71 | 99.60 |
| 2025 | Replace 25% | 72.20 | 0.9859 | 0.7845 | 43.72 | 100.00 |
| 2025 | Delete 25% | 70.20 | 0.9871 | 0.7725 | 39.68 | 100.00 |
| 2026 | Original | 98.70 | 0.9978 | 0.9872 | 98.18 | 99.21 |
| 2026 | Insert 25% | 89.30 | 0.9957 | 0.9042 | 78.54 | 99.80 |
| 2026 | Repeat 25% | 96.50 | 0.9972 | 0.9664 | 93.32 | 99.60 |
| 2026 | Replace 25% | 74.10 | 0.9857 | 0.7962 | 47.57 | 100.00 |
| 2026 | Delete 25% | 69.00 | 0.9862 | 0.7655 | 37.25 | 100.00 |

Accuracy trung bình trên bốn tập nhiễu: **81.60% ở seed 2025**, **82.225% ở seed 2026**; trung bình hai seed là **81.9125%**.

## 4. Kết quả chi tiết của RGCN + R-Drop

| Seed | Điều kiện | Accuracy (%) | AUC | F1 human | Recall AI (%) | Recall human (%) |
|---|---|---:|---:|---:|---:|---:|
| 2025 | Original | 98.40 | 0.9978 | 0.9841 | 98.99 | 97.83 |
| 2025 | Insert 25% | 91.90 | 0.9954 | 0.9258 | 83.81 | 99.80 |
| 2025 | Repeat 25% | 97.50 | 0.9974 | 0.9757 | 95.75 | 99.21 |
| 2025 | Replace 25% | 78.40 | 0.9864 | 0.8241 | 56.28 | 100.00 |
| 2025 | Delete 25% | 74.40 | 0.9868 | 0.7981 | 48.18 | 100.00 |
| 2026 | Original | 98.60 | 0.9977 | 0.9862 | 98.58 | 98.62 |
| 2026 | Insert 25% | 92.40 | 0.9959 | 0.9300 | 84.82 | 99.80 |
| 2026 | Repeat 25% | 97.30 | 0.9972 | 0.9739 | 95.14 | 99.41 |
| 2026 | Replace 25% | 78.70 | 0.9864 | 0.8261 | 56.88 | 100.00 |
| 2026 | Delete 25% | 70.20 | 0.9871 | 0.7725 | 39.68 | 100.00 |

Accuracy trung bình trên bốn tập nhiễu: **85.55% ở seed 2025**, **84.65% ở seed 2026**; trung bình hai seed là **85.10%**.

## 5. Cách tổng hợp và nhận xét

- Accuracy của mỗi điều kiện trong bảng tổng hợp là trung bình cộng của hai seed: `Acc_mean = (Acc_2025 + Acc_2026) / 2`.
- Average nhiễu là trung bình cộng của Insert, Repeat, Replace và Delete; **không tính Original**. Các tập và seed được cho trọng số bằng nhau, không ensemble dự đoán giữa các checkpoint.
- Chênh lệch được tính bằng **điểm phần trăm**, không phải tỷ lệ tăng tương đối: `Delta = 100 × (accuracy_model_1 − accuracy_model_2)` với accuracy gốc ở thang 0–1.
- Trên hai seed được chọn, GCN → RGCN tăng **1.70 điểm phần trăm** ở Average nhiễu. Tuy nhiên RGCN không tăng ở mọi điều kiện: trung bình Delete giảm từ 70.15% xuống 69.60%.
- Thêm R-Drop vào RGCN tăng **3.1875 điểm phần trăm** ở Average nhiễu. Cả hai seed đều tăng trên cả bốn tập nhiễu so với RGCN; mức tăng trung bình lớn nhất ở Replace (**5.40 điểm phần trăm**).
- Tổng mức tăng Average nhiễu của RGCN + R-Drop so với GCN là **4.8875 điểm phần trăm**. Accuracy Original của R-Drop thấp hơn RGCN **0.20 điểm phần trăm**.
- So với baseline GCN, RGCN + R-Drop tăng accuracy trung bình ở cả năm điều kiện: Original **+0.30**, Insert **+6.75**, Repeat **+1.50**, Replace **+9.15**, Delete **+2.15 điểm phần trăm**. Mức tăng lớn nhất ở Replace.
- Recall human rất cao trên Replace/Delete, trong khi recall AI thấp: nhiều văn bản AI bị nhiễu vẫn bị nhận nhầm thành human. R-Drop cải thiện nhưng chưa giải quyết hết điểm yếu này.

## 6. Bảng tổng hợp cuối — trung bình seed 2025 và 2026

GCN, RGCN và R-Drop dưới đây là kết quả trung bình hai seed. Hai cột chênh lệch dùng đơn vị **điểm phần trăm**. Cột cuối so sánh RGCN + R-Drop với baseline GCN: `Delta = 100 × (Acc_RDrop − Acc_GCN)`, với accuracy gốc ở thang 0–1. Các phép tính dùng số chưa làm tròn.

| Điều kiện | GCN (%) | RGCN (%) | RGCN + R-Drop (%) | Δ R-Drop − RGCN | Δ R-Drop − GCN |
|---|---:|---:|---:|---:|---:|
| Original | 98.20 | 98.70 | 98.50 | −0.20 | +0.30 |
| Insert 25% | 85.40 | 88.55 | 92.15 | +3.60 | +6.75 |
| Repeat 25% | 95.90 | 96.35 | 97.40 | +1.05 | +1.50 |
| Replace 25% | 69.40 | 73.15 | 78.55 | +5.40 | +9.15 |
| Delete 25% | 70.15 | 69.60 | 72.30 | +2.70 | +2.15 |
| **Average 4 tập nhiễu** | **80.21** | **81.91** | **85.10** | **+3.19** | **+4.89** |
