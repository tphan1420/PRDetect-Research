# Tổng hợp Table 3 và Table 4 — GCN, RGCN, RGCN + R-Drop

## 1. Phạm vi, nguồn dữ liệu và cách tính

Báo cáo tổng hợp **hai seed 2025 và 2026**, với ba mô hình train trên HC3:

- **GCN:** checkpoint người thực hiện tự train lại bằng code đã chỉnh sửa, không phải checkpoint hoặc điểm số do tác giả paper cung cấp.
- **RGCN:** checkpoint baseline của thí nghiệm R-Drop, tên `hc3_rdrop_exp_rgcn_model_{seed}.pth`.
- **RGCN + R-Drop:** checkpoint `hc3_rdrop_exp_rdrop_a0.1_rgcn_model_{seed}.pth`, hệ số R-Drop là 0.1.

Nguồn:

- [ZIP kết quả Table 3](../result/table3_hc3_to_gpt35_3models.zip): manifest, 30 report JSON, CSV theo seed và CSV tổng hợp.
- [Báo cáo Table 4 đã tổng hợp](./Table4_GCN_RGCN_RDrop_Seeds2025_2026.md).
- Đối chiếu lại số liệu Table 4 với các ZIP [GCN](../result/table4_hc3_gcn_3seeds.zip), [RGCN](../result/table4_hc3_rgcn_3seeds.zip) và [R-Drop](../result/table4_hc3_rdrop_3seeds.zip); chỉ lấy seed 2025–2026.

Quy ước:

- Ngưỡng phân loại cố định **0.5**, `0 = AI`, `1 = human`; **F1 là F1 của lớp human**, không phải macro-F1.
- Accuracy và recall biểu diễn theo phần trăm; AUC và F1 dùng thang 0–1.
- `Mean ± SD` là trung bình và độ lệch chuẩn **mẫu** qua hai seed (`ddof=1`), không phải khoảng tin cậy.
- Chênh lệch accuracy dùng **điểm phần trăm**: `Δ = 100 × (accuracy_RDrop − accuracy_baseline)`.
- Không ensemble. Mỗi checkpoint được test riêng; các giá trị trung bình được tính từ số liệu chưa làm tròn.
- Mỗi report có 1.000 mẫu. Table 3 có 500 AI + 500 human; Table 4 có 494 AI + 506 human.
- Table 3 chỉ xét **HC3 → GPT-3.5**; chưa có kết quả chiều GPT-3.5 → HC3 trong báo cáo này.

## 2. Table 3 — Cross-dataset HC3 → GPT-3.5

Benchmark gồm GPT-3.5 Original và synonym replacement ở các mức 5%, 10%, 20%, 30%. Manifest ghi batch size test là 8. Đã kiểm tra 30 report có cùng ngưỡng và cùng chuỗi nhãn giữa các mô hình/seed trên từng split.

### 2.1. Accuracy chi tiết theo seed

| Seed | Điều kiện | GCN (%) | RGCN (%) | RGCN + R-Drop (%) | Δ R-Drop − RGCN | Δ R-Drop − GCN |
|---|---|---:|---:|---:|---:|---:|
| 2025 | Original | 69.90 | 62.70 | 68.40 | +5.70 | −1.50 |
| 2025 | Synonym 5% | 69.40 | 61.30 | 68.50 | +7.20 | −0.90 |
| 2025 | Synonym 10% | 69.10 | 60.80 | 67.80 | +7.00 | −1.30 |
| 2025 | Synonym 20% | 67.80 | 59.90 | 67.20 | +7.30 | −0.60 |
| 2025 | Synonym 30% | 66.90 | 59.50 | 66.60 | +7.10 | −0.30 |
| 2026 | Original | 64.90 | 62.50 | 64.50 | +2.00 | −0.40 |
| 2026 | Synonym 5% | 63.80 | 61.80 | 64.40 | +2.60 | +0.60 |
| 2026 | Synonym 10% | 63.60 | 61.30 | 63.90 | +2.60 | +0.30 |
| 2026 | Synonym 20% | 62.50 | 60.60 | 63.30 | +2.70 | +0.80 |
| 2026 | Synonym 30% | 62.20 | 59.90 | 62.60 | +2.70 | +0.40 |

### 2.2. Accuracy trung bình hai seed

| Điều kiện | GCN (%) | RGCN (%) | RGCN + R-Drop (%) | Δ R-Drop − RGCN | Δ R-Drop − GCN |
|---|---:|---:|---:|---:|---:|
| Original | 67.40 ± 3.54 | 62.60 ± 0.14 | 66.45 ± 2.76 | +3.85 | −0.95 |
| Synonym 5% | 66.60 ± 3.96 | 61.55 ± 0.35 | 66.45 ± 2.90 | +4.90 | −0.15 |
| Synonym 10% | 66.35 ± 3.89 | 61.05 ± 0.35 | 65.85 ± 2.76 | +4.80 | −0.50 |
| Synonym 20% | 65.15 ± 3.75 | 60.25 ± 0.49 | 65.25 ± 2.76 | +5.00 | +0.10 |
| Synonym 30% | 64.55 ± 3.32 | 59.70 ± 0.28 | 64.60 ± 2.83 | +4.90 | +0.05 |

### 2.3. AUC và F1 human trung bình hai seed

| Điều kiện | GCN AUC | RGCN AUC | R-Drop AUC | GCN F1 | RGCN F1 | R-Drop F1 |
|---|---:|---:|---:|---:|---:|---:|
| Original | 0.8806 | 0.8803 | 0.8777 | 0.7495 | 0.7240 | 0.7430 |
| Synonym 5% | 0.8805 | 0.8845 | 0.8811 | 0.7456 | 0.7192 | 0.7442 |
| Synonym 10% | 0.8799 | 0.8837 | 0.8805 | 0.7441 | 0.7166 | 0.7407 |
| Synonym 20% | 0.8793 | 0.8826 | 0.8793 | 0.7374 | 0.7125 | 0.7376 |
| Synonym 30% | 0.8762 | 0.8800 | 0.8770 | 0.7340 | 0.7097 | 0.7342 |

### 2.4. Tổng hợp Table 3 và nhận xét

Dòng Average dưới đây lấy trung bình **cả 5 điều kiện, gồm Original**; đây là thống kê bổ sung của báo cáo, không phải một tập test độc lập.

| Phạm vi | GCN Accuracy (%) | RGCN Accuracy (%) | R-Drop Accuracy (%) | Δ R-Drop − RGCN | Δ R-Drop − GCN |
|---|---:|---:|---:|---:|---:|
| Average 5 điều kiện, seed 2025 | 68.62 | 60.84 | 67.70 | +6.86 | −0.92 |
| Average 5 điều kiện, seed 2026 | 63.40 | 61.22 | 63.74 | +2.52 | +0.34 |
| **Average 5 điều kiện, hai seed** | **66.01** | **61.03** | **65.72** | **+4.69** | **−0.29** |

- R-Drop tăng accuracy so với baseline RGCN ở **10/10** cặp seed × điều kiện. Mức tăng trung bình là **4.69 điểm %**.
- Tuy nhiên, so với GCN, R-Drop thấp hơn trung bình **0.29 điểm %**. R-Drop chỉ nhỉnh hơn GCN ở trung bình Synonym 20% và 30%, lần lượt **0.10** và **0.05 điểm %**; hai mức chênh nhỏ này chưa đủ để tuyên bố vượt trội.
- AUC của R-Drop thấp hơn RGCN ở cả 5 điều kiện trung bình. Vì vậy kết quả không chứng minh R-Drop cải thiện mọi khía cạnh: accuracy/F1 ở ngưỡng 0.5 tăng, nhưng khả năng xếp hạng theo AUC không tăng trong thí nghiệm này.
- Trung bình cả 5 điều kiện và hai seed, recall AI là **34.38% / 23.64% / 34.00%**, còn recall human là **97.64% / 98.42% / 97.44%**, tương ứng GCN / RGCN / R-Drop. R-Drop khôi phục phần lớn recall AI bị thấp ở baseline RGCN, nhưng cả ba vẫn bỏ sót nhiều văn bản AI khi chuyển miền.
- RGCN ở đây là **baseline trong thí nghiệm R-Drop**. Không đánh đồng kết quả này với checkpoint RGCN cũ `hc3_rgcn_model_2026.pth` hoặc dùng nó để kết luận mọi phiên bản RGCN đều kém GCN.

## 3. Table 4 — HC3 structural noise 25%

Benchmark gồm HC3 Original và Insert, Repeat, Replace, Delete ở mức 25%. Chỉ dùng seed 2025–2026 từ các gói kết quả có ba seed.

### 3.1. Accuracy chi tiết theo seed

| Seed | Điều kiện | GCN (%) | RGCN (%) | RGCN + R-Drop (%) | Δ R-Drop − RGCN | Δ R-Drop − GCN |
|---|---|---:|---:|---:|---:|---:|
| 2025 | Original | 99.00 | 98.70 | 98.40 | −0.30 | −0.60 |
| 2025 | Insert 25% | 87.50 | 87.80 | 91.90 | +4.10 | +4.40 |
| 2025 | Repeat 25% | 97.40 | 96.20 | 97.50 | +1.30 | +0.10 |
| 2025 | Replace 25% | 71.00 | 72.20 | 78.40 | +6.20 | +7.40 |
| 2025 | Delete 25% | 71.90 | 70.20 | 74.40 | +4.20 | +2.50 |
| 2026 | Original | 97.40 | 98.70 | 98.60 | −0.10 | +1.20 |
| 2026 | Insert 25% | 83.30 | 89.30 | 92.40 | +3.10 | +9.10 |
| 2026 | Repeat 25% | 94.40 | 96.50 | 97.30 | +0.80 | +2.90 |
| 2026 | Replace 25% | 67.80 | 74.10 | 78.70 | +4.60 | +10.90 |
| 2026 | Delete 25% | 68.40 | 69.00 | 70.20 | +1.20 | +1.80 |

### 3.2. Accuracy trung bình hai seed

| Điều kiện | GCN (%) | RGCN (%) | RGCN + R-Drop (%) | Δ R-Drop − RGCN | Δ R-Drop − GCN |
|---|---:|---:|---:|---:|---:|
| Original | 98.20 ± 1.13 | 98.70 ± 0.00 | 98.50 ± 0.14 | −0.20 | +0.30 |
| Insert 25% | 85.40 ± 2.97 | 88.55 ± 1.06 | 92.15 ± 0.35 | +3.60 | +6.75 |
| Repeat 25% | 95.90 ± 2.12 | 96.35 ± 0.21 | 97.40 ± 0.14 | +1.05 | +1.50 |
| Replace 25% | 69.40 ± 2.26 | 73.15 ± 1.34 | 78.55 ± 0.21 | +5.40 | +9.15 |
| Delete 25% | 70.15 ± 2.47 | 69.60 ± 0.85 | 72.30 ± 2.97 | +2.70 | +2.15 |
| **Average 4 tập nhiễu** | **80.21 ± 2.46** | **81.91 ± 0.44** | **85.10 ± 0.64** | **+3.19** | **+4.89** |

Average nhiễu **không tính Original**. SD ở dòng Average được tính từ hai trung bình theo seed, không phải trung bình SD của bốn điều kiện.

### 3.3. AUC và F1 human trung bình hai seed

| Điều kiện | GCN AUC | RGCN AUC | R-Drop AUC | GCN F1 | RGCN F1 | R-Drop F1 |
|---|---:|---:|---:|---:|---:|---:|
| Original | 0.9964 | 0.9977 | 0.9977 | 0.9823 | 0.9872 | 0.9851 |
| Insert 25% | 0.9897 | 0.9955 | 0.9957 | 0.8741 | 0.8982 | 0.9279 |
| Repeat 25% | 0.9958 | 0.9972 | 0.9973 | 0.9608 | 0.9651 | 0.9748 |
| Replace 25% | 0.9744 | 0.9858 | 0.9864 | 0.7676 | 0.7904 | 0.8251 |
| Delete 25% | 0.9723 | 0.9866 | 0.9869 | 0.7724 | 0.7690 | 0.7853 |
| Average 4 tập nhiễu | 0.9830 | 0.9913 | 0.9916 | 0.8437 | 0.8557 | 0.8783 |

### 3.4. Nhận xét Table 4

- R-Drop tăng accuracy so với RGCN và GCN ở **8/8** cặp seed × điều kiện nhiễu. Average nhiễu tăng chính xác **3.1875 điểm %** so với RGCN và **4.8875 điểm %** so với GCN.
- Lợi ích lớn nhất ở Replace: **+5.40 điểm %** so với RGCN và **+9.15 điểm %** so với GCN khi lấy trung bình hai seed.
- Có đánh đổi nhỏ ở HC3 Original: R-Drop thấp hơn RGCN **0.20 điểm %**, dù vẫn cao hơn GCN trung bình **0.30 điểm %**.
- Delete vẫn là điểm yếu của R-Drop (72.30%). Recall AI thấp và recall human gần 100% trên Replace/Delete cho thấy còn nhiều văn bản AI bị nhiễu bị nhận nhầm thành human; AUC cao chưa đủ để giải thích hoặc khắc phục nguyên nhân.
- Seed 2024 không nằm trong phạm vi báo cáo này. Không suy rộng kết luận hai seed thành hiệu quả ổn định trên mọi seed; tài liệu Table 4 nguồn đã ghi nhận kết quả kém hơn RGCN ở seed 2024.

## 4. Kết luận chung

Trong phạm vi checkpoint và hai seed được kiểm tra, **R-Drop là cải tiến có ích so với baseline RGCN của thí nghiệm**: tăng accuracy trên cross-dataset và structural noise. Tuy nhiên, **chưa thể tuyên bố R-Drop vượt GCN trên mọi benchmark**: Table 3 trung bình vẫn thấp hơn GCN một chút, còn Table 4 nhiễu thì vượt rõ hơn.

Các chênh lệch ở đây là mô tả kết quả thực nghiệm, chưa phải kiểm định ý nghĩa thống kê. Không trộn điểm tác giả paper, checkpoint RGCN cũ hoặc V3 của Quang vào các bảng này.

## 5. Bảng tổng hợp cuối — trung bình seed 2025 và 2026

Hai cột cuối so sánh accuracy của R-Drop với baseline RGCN và GCN tự train lại, đơn vị **điểm phần trăm**. Không tính một trung bình chung giữa Table 3 và Table 4 vì chúng đo hai mục tiêu khác nhau.

| Benchmark / điều kiện | GCN (%) | RGCN (%) | RGCN + R-Drop (%) | Δ R-Drop − RGCN | Δ R-Drop − GCN |
|---|---:|---:|---:|---:|---:|
| Table 3 — GPT-3.5 Original | 67.40 | 62.60 | 66.45 | +3.85 | −0.95 |
| Table 3 — Synonym 5% | 66.60 | 61.55 | 66.45 | +4.90 | −0.15 |
| Table 3 — Synonym 10% | 66.35 | 61.05 | 65.85 | +4.80 | −0.50 |
| Table 3 — Synonym 20% | 65.15 | 60.25 | 65.25 | +5.00 | +0.10 |
| Table 3 — Synonym 30% | 64.55 | 59.70 | 64.60 | +4.90 | +0.05 |
| **Table 3 — Average 5 điều kiện** | **66.01** | **61.03** | **65.72** | **+4.69** | **−0.29** |
| Table 4 — HC3 Original | 98.20 | 98.70 | 98.50 | −0.20 | +0.30 |
| Table 4 — Insert 25% | 85.40 | 88.55 | 92.15 | +3.60 | +6.75 |
| Table 4 — Repeat 25% | 95.90 | 96.35 | 97.40 | +1.05 | +1.50 |
| Table 4 — Replace 25% | 69.40 | 73.15 | 78.55 | +5.40 | +9.15 |
| Table 4 — Delete 25% | 70.15 | 69.60 | 72.30 | +2.70 | +2.15 |
| **Table 4 — Average 4 tập nhiễu** | **80.21** | **81.91** | **85.10** | **+3.19** | **+4.89** |
