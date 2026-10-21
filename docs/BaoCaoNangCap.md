# BÁO CÁO TỔNG HỢP NÂNG CẤP PRDETECT: TỪ GCN SANG RGCN

**Phạm vi:** toàn bộ tài liệu trong `docs/` và log trong `result/`

**Đối sánh chính:** `hc3_gcn` và `hc3_rgcn`, huấn luyện trên HC3, seed 2026

**Ngày tổng hợp:** 2026-10-21

---

## Tóm tắt điều hành

PRDetect phát hiện văn bản do LLM sinh bằng cách kết hợp embedding RoBERTa với cây cú pháp phụ thuộc. Mô hình cũ dùng GCN đồng nhất nên mọi cạnh cú pháp đều dùng chung một phép biến đổi; nhãn như `nsubj`, `dobj`, `amod`, `compound` hay `punct` bị san phẳng. Mô hình hiện tại thay GCN bằng RGCN, giữ hướng và hơn 60 loại quan hệ, đồng thời dùng phân rã cơ sở với $B=30$ để kiểm soát số tham số.

Đối chiếu trực tiếp `result/RGCN_test_result.txt` cho thấy 242 dòng hợp lệ tạo thành **120 tập kiểm thử duy nhất**. Log ghi “121 file” do `llama2_gen` xuất hiện hai lần. Theo Accuracy, RGCN tăng trên **105/120 tập (87,5%)**, giảm trên **13/120 (10,83%)** và hòa trên **2/120 (1,67%)**. Mức tăng lớn nhất là **+12,90 điểm phần trăm** ở `hc3_test_insert_50`.

Các kết quả nổi bật gồm HC3 **97,40% → 98,60%**, GPT-3.5-Mixed **64,90% → 72,60%**, dịch ngược GPT-3.5-Mixed **65,00% → 77,20%**, và HC3 50–100 token **88,33% → 98,33%**. Tuy nhiên, RGCN vẫn giảm trên một số tập matched, văn bản rất ngắn, nhiễu xóa/lặp cực đoan; trên LLaMA/Mistral, Accuracy tăng nhưng AUC có thể giảm.

Kết luận: RGCN sửa đúng điểm yếu biểu diễn của GCN và tạo cải thiện rộng, nhưng kết quả hiện tại mới là một lần chạy seed 2026, chưa có khoảng tin cậy, kiểm định thống kê, calibration hoặc benchmark chi phí GCN–RGCN cùng điều kiện.

---

## 1. Mô hình cũ và các hạn chế

### 1.1. Pipeline PRDetect-GCN

1. SpaCy phân tích văn bản thành cây cú pháp phụ thuộc.
2. RoBERTa-base sinh véc-tơ 768 chiều cho token/nút.
3. Cây được chuyển thành ma trận kề nhị phân $A$.
4. GCN hai tầng lan truyền đặc trưng.
5. Global mean pooling tạo biểu diễn toàn văn bản.
6. Lớp tuyến tính và sigmoid dự đoán Human/AI; quy ước dự án là $y=1$ cho Human, $y=0$ cho AI.

Phép cập nhật GCN:

$$
H^{(l+1)}=\sigma\left(\widetilde D^{-\frac12}\widetilde A\widetilde D^{-\frac12}H^{(l)}W^{(l)}\right),
$$

với $\widetilde A=A+I$, $\widetilde D$ là ma trận bậc và $W^{(l)}$ là trọng số dùng chung cho mọi cạnh.

### 1.2. Điểm mạnh vốn có

- Khai thác cấu trúc cú pháp thay vì chỉ dựa vào xác suất token bề mặt.
- Kết hợp ngữ nghĩa RoBERTa với topology của cây phụ thuộc.
- Có tính giải thích qua cấu trúc cây và liên kết giữa các token.
- Tài liệu gốc báo cáo PRDetect kiểm thử trong 2,5 giây trên RTX 4090, so với 28,3 giây của CoCo và 1.298,4 giây của DetectGPT trong thiết lập tương ứng.
- PRDetect nguyên bản bền với thay từ đồng nghĩa; đây là kết quả của toàn hệ thống PRDetect, không phải bằng chứng riêng cho RGCN.

### 1.3. Hạn chế của GCN cũ

**Mất thông tin quan hệ.** Ma trận $A$ chỉ biểu diễn có/không có cạnh. Quan hệ chủ ngữ, tân ngữ, bổ ngữ và dấu câu đều dùng cùng $W^{(l)}$.

**Tổng hợp tĩnh.** Chuẩn hóa theo bậc không đánh giá độ tin cậy của từng láng giềng; cạnh sai do typo, homoglyph, whitespace hoặc parser vẫn truyền nhiễu.

**Tầm nhìn cục bộ.** Hai tầng chỉ bao phủ vùng hai bước nhảy. Tăng độ sâu trực tiếp lại gây over-smoothing.

**Phụ thuộc parser và miền.** Sai số tokenization/parsing truyền vào đồ thị; mô hình huấn luyện trên HC3 dễ suy giảm ở miền, LLM hoặc chiến lược giải mã mới.

**Yếu ở văn bản ngắn.** Đồ thị nhỏ chứa ít tín hiệu. RGCN mới cải thiện mạnh HC3 50–100 token, nhưng GPT-3.5 10–100 token vẫn gần mức ngẫu nhiên.

**Ngưỡng quyết định cố định.** Accuracy/F1 có thể giảm khi phân phối logit dịch chuyển, dù AUC còn tốt.

---

## 2. Các cải tiến đã triển khai

### 2.1. Đồ thị đa quan hệ có hướng

Đồ thị mới là:

$$\mathcal G=(\mathcal V,\mathcal E,\mathcal R),$$

trong đó $\mathcal V$ là các token, $\mathcal E$ chứa cạnh có hướng $(j,r,i)$ và $\mathcal R$ gồm hơn 60 nhãn phụ thuộc SpaCy. So với ma trận nhị phân, biểu diễn mới giữ đầu–cuối, hướng và loại quan hệ.

### 2.2. RGCN

Trạng thái nút $i$ được cập nhật bởi:

$$
h_i^{(l+1)}=\sigma\left(
\sum_{r\in\mathcal R}\sum_{j\in\mathcal N_i^r}
\frac{1}{c_{i,r}}W_r^{(l)}h_j^{(l)}
+W_0^{(l)}h_i^{(l)}
\right).
$$

| Ký hiệu | Ý nghĩa |
|---|---|
| $h_i^{(l)}\in\mathbb R^{d_l}$ | Biểu diễn nút $i$ ở tầng $l$ |
| $\mathcal R$ | Tập loại quan hệ cú pháp |
| $\mathcal N_i^r$ | Láng giềng gửi thông điệp đến $i$ qua quan hệ $r$ |
| $W_r^{(l)}$ | Phép biến đổi riêng cho quan hệ $r$ |
| $W_0^{(l)}$ | Phép biến đổi self-loop/root |
| $c_{i,r}=|\mathcal N_i^r|$ | Chuẩn hóa theo số láng giềng thuộc quan hệ $r$ |
| $\sigma$ | Hàm kích hoạt phi tuyến |

Khác biệt cốt lõi là $W^{(l)}$ duy nhất của GCN được thay bằng họ $\{W_r^{(l)}\}$, cho phép mô hình học tác động riêng của từng quan hệ.

### 2.3. Phân rã cơ sở

Nếu mỗi quan hệ có ma trận độc lập:

$$P_{\mathrm{full}}=|\mathcal R|d_ld_{l+1}.$$

Với $|\mathcal R|\approx60$, $d_l=768$, $d_{l+1}=256$, một tầng cần khoảng **11,80 triệu** tham số quan hệ. RGCN dùng:

$$W_r^{(l)}=\sum_{b=1}^{B}a_{rb}^{(l)}V_b^{(l)},$$

$$P_{\mathrm{basis}}=Bd_ld_{l+1}+|\mathcal R|B.$$

Với $B=30$, ví dụ trên còn khoảng **5,90 triệu**, giảm gần 50%. $V_b$ là cơ sở dùng chung và $a_{rb}$ là hệ số kết hợp. Tác dụng chống overfit là giả thuyết hợp lý nhưng vẫn cần ablation theo $B$.

### 2.4. Thành phần phần mềm được tài liệu ghi nhận

| Thành phần | Vai trò |
|---|---|
| `dep_vocab.py`, `dep_vocab.json` | Ánh xạ nhãn phụ thuộc sang ID |
| `building_graph.py` | Sinh cạnh và `edge_type`, hỗ trợ cạnh hai chiều |
| `model/RGCN.py` | RGCN hai tầng, bốn tầng và cấu hình linh hoạt |
| `rgcn.py` | Huấn luyện, early stopping, TensorBoard, checkpoint |
| `test_rgcn.py`, `test.py` | Đánh giá Accuracy, ROC-AUC, F1 |
| `detect.py` | Suy luận CLI bằng GCN hoặc RGCN |

Đây là trạng thái được mô tả trong tài liệu; báo cáo không thay thế kiểm toán mã nguồn từng hàm.

---

## 3. Thiết lập và cách đọc kết quả

So sánh chính trong `RGCN_test_result.txt` dùng HC3, seed 2026 và hai checkpoint `hc3_gcn`/`hc3_rgcn`. Tài liệu mô tả batch size 32, Adam, learning rate $10^{-4}$ và tối đa 40 epoch.

$$\mathrm{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}$$

$$\mathrm{Precision}=\frac{TP}{TP+FP},\quad \mathrm{Recall}=\frac{TP}{TP+FN}$$

$$F_1=2\frac{\mathrm{Precision}\cdot\mathrm{Recall}}{\mathrm{Precision}+\mathrm{Recall}}.$$

ROC-AUC đo năng lực xếp hạng qua mọi ngưỡng. Do đó Accuracy tăng nhưng AUC giảm nghĩa là điểm vận hành hiện tại tốt hơn, không đồng nghĩa chất lượng xếp hạng tổng thể tốt hơn.

`result/test_result.txt` chứa các lần chạy lịch sử 2024–2025 với nhiều seed, checkpoint và cấu hình. Nó chỉ dùng để đối chiếu bối cảnh, không ghép vào bảng chính vì không bảo đảm cùng điều kiện.

---

## 4. Kết quả tổng hợp trên 120 tập

### 4.1. Toàn cục

| Theo Accuracy | Số tập | Tỷ lệ |
|---|---:|---:|
| RGCN tăng | 105 | 87,50% |
| RGCN giảm | 13 | 10,83% |
| Không đổi | 2 | 1,67% |
| **Tổng** | **120** | **100%** |

### 4.2. Benchmark đại diện

| Tập | GCN Acc | RGCN Acc | $\Delta$ | GCN AUC | RGCN AUC | GCN F1 | RGCN F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `hc3_test` | 97,40% | **98,60%** | **+1,20** | 0,9960 | **0,9979** | 0,9746 | **0,9861** |
| `gpt3.5_mixed_test_split` | 64,90% | **72,60%** | **+7,70** | 0,8614 | **0,8782** | 0,7347 | **0,7794** |
| `mini_raid_test_all` | 68,59% | **70,39%** | **+1,80** | 0,7176 | **0,7310** | 0,7375 | **0,7475** |
| `mini_raid_test_news` | 67,27% | **68,57%** | **+1,30** | 0,7093 | **0,7148** | 0,7483 | **0,7562** |
| `mini_raid_test_abstracts` | **70,47%** | 70,37% | -0,10 | 0,6596 | **0,6709** | **0,7607** | 0,7574 |
| `raid_llama_greedy_test` | 74,40% | **76,80%** | **+2,40** | **0,8875** | 0,8617 | 0,7836 | **0,7945** |
| `raid_llama_sampling_test` | 73,05% | **74,50%** | **+1,45** | **0,8743** | 0,8491 | 0,7748 | **0,7786** |
| `raid_mistral_test` | 78,10% | **78,40%** | **+0,30** | **0,9080** | 0,8788 | **0,8089** | 0,8059 |

RGCN nổi bật nhất trên GPT-3.5-Mixed. Trên RAID/LLaMA/Mistral, Accuracy thường tăng nhưng AUC giảm 0,025–0,029 ở ba tập LLaMA/Mistral; không thể kết luận mô hình vượt GCN trên mọi chỉ số.

### 4.3. Mười mức tăng Accuracy lớn nhất

| Tập | GCN | RGCN | $\Delta$ điểm % |
|---|---:|---:|---:|
| `hc3_test_insert_50` | 64,60% | 77,50% | **+12,90** |
| `gpt3.5_mixed_test_translate` | 65,00% | 77,20% | **+12,20** |
| `hc3_test_replace_25` | 67,80% | 78,80% | **+11,00** |
| `hc3_test_50_100` | 88,33% | 98,33% | **+10,00** |
| `hc3_test_insert_30` | 78,90% | 88,80% | **+9,90** |
| `hc3_test_replace_30` | 61,90% | 71,50% | **+9,60** |
| `hc3_test_insert_70` | 60,00% | 69,00% | **+9,00** |
| `hc3_test_replace_15` | 82,60% | 91,30% | **+8,70** |
| `gpt3.5_mixed_test_adj_10` | 61,40% | 70,10% | **+8,70** |
| `hc3_test_insert_25` | 83,30% | 91,70% | **+8,40** |

Xu hướng: RGCN hưởng lợi khi token bị thay đổi nhưng đồ thị vẫn còn đủ quan hệ cú pháp hữu ích.

### 4.4. Theo độ dài

| HC3 | GCN Acc | RGCN Acc | $\Delta$ |
|---|---:|---:|---:|
| 50–100 | 88,33% | **98,33%** | **+10,00** |
| 100–150 | 97,00% | **98,00%** | **+1,00** |
| 150–200 | **98,75%** | 95,00% | **-3,75** |
| 200–250 | 96,67% | **98,33%** | **+1,66** |
| 250–300 | 98,00% | **100,00%** | **+2,00** |
| >300 | 96,15% | **100,00%** | **+3,85** |

Ngoại lệ 150–200 token giảm 3,75 điểm dù AUC RGCN đạt 1,0, có thể do ngưỡng hoặc tập con nhỏ; cần số mẫu và nhiều seed để xác nhận.

Trên GPT-3.5-Mixed rất ngắn, RGCN chỉ đạt 50,00%, 51,00%, 55,50%, 53,50% và 55,00% ở các khoảng 10–20, 20–40, 40–60, 60–80, 80–100 token. Cải thiện trên HC3 ngắn không khái quát ngay sang miền khác.

### 4.5. Kháng nhiễu và viết lại

| Kịch bản HC3 | GCN Acc | RGCN Acc | $\Delta$ | RGCN F1 |
|---|---:|---:|---:|---:|
| Đồng nghĩa 30% | 96,60% | **98,60%** | **+2,00** | 0,9862 |
| Chèn 15% | 90,00% | **96,60%** | **+6,60** | 0,9674 |
| Chèn 25% | 83,30% | **91,70%** | **+8,40** | 0,9241 |
| Xóa 15% | 83,20% | **90,10%** | **+6,90** | 0,9107 |
| Xóa 25% | 68,40% | **74,00%** | **+5,60** | 0,7956 |
| Thay thế 15% | 82,60% | **91,30%** | **+8,70** | 0,9208 |
| Thay thế 25% | 67,80% | **78,80%** | **+11,00** | 0,8268 |
| Lặp 15% | 95,70% | **98,60%** | **+2,90** | 0,9863 |
| Dịch ngược | 76,60% | **84,80%** | **+8,20** | 0,8671 |
| LLaMA-3 8B viết lại | 87,98% | **91,98%** | **+4,00** | 0,9254 |

Với xóa 70–75%, cả hai mô hình chỉ đạt 50,60%; với lặp 75%, RGCN là 86,50% so với GCN 87,70%. RGCN bền hơn ở nhiễu vừa phải nhưng không thể phục hồi khi cấu trúc bị phá hủy gần hoàn toàn.

### 4.6. Các mức giảm đáng chú ý

| Tập | GCN | RGCN | $\Delta$ điểm % |
|---|---:|---:|---:|
| `matched_gpt3.5_mixed_test_adj_5_synonym_replace` | 64,70% | 59,70% | **-5,00** |
| `matched_gpt3.5_mixed_test_split` | 79,90% | 75,00% | **-4,90** |
| `hc3_test_150_200` | 98,75% | 95,00% | **-3,75** |
| `hc3_test_repeat_75` | 87,70% | 86,50% | -1,20 |
| `hc3human` | 97,92% | 96,84% | -1,08 |
| `gpt3.5_mixed_test_20_40` | 52,00% | 51,00% | -1,00 |

`matched_gpt3.5_mixed_test_split` có AUC tăng 0,8914 → 0,8968 nhưng Accuracy giảm, phù hợp với lệch ngưỡng. Ở tập matched synonym 5%, cả AUC lẫn Accuracy giảm, nên calibration không chắc giải quyết toàn bộ.

---

## 5. Điểm mạnh của mô hình hiện tại

1. **Biểu diễn đúng bản chất dữ liệu hơn:** giữ loại và hướng quan hệ cú pháp.
2. **Cải thiện rộng:** 105/120 tập tăng Accuracy, bao phủ in-domain, domain shift, nhiều LLM và nhiều phép nhiễu.
3. **Bền khi cấu trúc còn bảo toàn:** tăng mạnh với chèn, thay thế, đồng nghĩa, dịch ngược và viết lại.
4. **Khắc phục một phần văn bản ngắn:** HC3 50–100 token tăng 10 điểm phần trăm.
5. **Kiểm soát tham số:** basis decomposition giảm gần một nửa tham số quan hệ trong cấu hình minh họa.
6. **Giữ tính giải thích:** có thể truy về loại cạnh thay vì chỉ embedding chuỗi.
7. **Tương thích pipeline cũ:** encoder, pooling và classifier được giữ, giúp cô lập thay đổi graph encoder.

---

## 6. Điểm yếu và hạn chế hiện tại

1. **Chỉ một seed trong phép so sánh chính:** chưa có mean ± std, confidence interval hay kiểm định paired.
2. **AUC/F1 không luôn tăng:** LLaMA/Mistral cho thấy đánh đổi giữa Accuracy tại ngưỡng hiện tại và năng lực xếp hạng.
3. **Giảm trên tập matched:** mô hình chưa hoàn toàn bất biến theo miền/cách chọn mẫu.
4. **Văn bản GPT-3.5 rất ngắn vẫn yếu:** Accuracy khoảng 50–55,5%.
5. **Không bền trước phá hủy cực đoan:** xóa 70–75% làm Accuracy gần ngẫu nhiên.
6. **Phụ thuộc SpaCy:** lỗi parse làm sai cả cạnh lẫn `edge_type`; khả năng đa ngôn ngữ chưa được chứng minh.
7. **Chưa benchmark chi phí riêng:** RGCN phải lưu `edge_type` và biến đổi theo quan hệ. Số 2.545,3 giây trong `DuAn.md` thuộc CoCo, không phải RGCN; không được dùng để kết luận chi phí RGCN.
8. **$B=30$ chưa có ablation:** chưa biết cấu hình này tối ưu hay mức tăng đến từ nhãn cạnh hay số tham số.
9. **Hai tầng vẫn có tầm nhìn hạn chế:** tăng tầng có rủi ro over-smoothing; GraphNorm/Dirichlet Energy mới là đề xuất.
10. **Thiếu chỉ số vận hành an toàn:** chưa có confusion matrix, calibration error, TPR@1%FPR hay TPR@5%FPR; false positive trên văn bản người là rủi ro quan trọng.
11. **Một số tập chỉ có một lớp:** AUC=`nan`, F1=0 ở `gpt2`, `hc3chatgpt`, `llama2_gen`; Accuracy đơn lẻ có thể gây hiểu nhầm.
12. **Log lịch sử không đồng nhất:** `test_result.txt` trộn nhiều thời điểm, seed và checkpoint, không thể so sánh trực tiếp.

---

## 7. Mức độ bằng chứng của các kết luận

**Được log hỗ trợ:** RGCN tăng Accuracy ở đa số tập; tăng mạnh trên chèn/thay thế và GPT-3.5-Mixed; có trường hợp giảm trên matched, đoạn ngắn, nhiễu cực đoan; AUC không luôn tăng.

**Giải thích hợp lý nhưng chưa chứng minh nhân quả:** nhãn quan hệ giúp học “dấu vân tay cú pháp”; basis decomposition chống overfit; tính bất biến cấu trúc tạo độ bền. Cần ablation để xác nhận.

**Mới là lộ trình, chưa có kết quả triển khai trong log hiện tại:** DeBERTa-v3, GATv2/RGAT, GraphNorm, ràng buộc Dirichlet Energy, contrastive domain generalization, Mixture-of-Experts và Platt/temperature scaling.

---

## 8. Đề xuất cải tiến và thí nghiệm tiếp theo

### Ưu tiên 1 — tăng độ tin cậy

- Chạy ít nhất 5 seed; báo cáo mean ± std và bootstrap 95% CI.
- Giữ cùng split, checkpoint selection, batch size và threshold cho mọi cặp.
- Báo confusion matrix, precision/recall từng lớp, PR-AUC, ECE/Brier, TPR@1%FPR và TPR@5%FPR.
- Tách tập hai lớp và một lớp; không báo AUC khi không xác định.
- Ghi metadata cấu hình và loại bản ghi trùng trong log.

### Ưu tiên 2 — xác định đóng góp RGCN

- Ablation: bỏ/xáo trộn edge labels; có/không hướng; có/không self-loop.
- Quét $B\in\{4,8,16,30,60\}$ và báo tham số, VRAM, thời gian train/inference.
- So sánh 2/4/6 tầng với residual và normalization.
- Phân tích lỗi theo loại quan hệ và độ tin cậy parser.

### Ưu tiên 3 — xử lý điểm yếu

- Calibration threshold trên validation cho các tập AUC tốt nhưng Accuracy giảm.
- Kết hợp graph embedding với embedding `[CLS]` cho văn bản ngắn/cây hỏng.
- Thử RGAT/GATv2 để hạ trọng số cạnh nhiễu; đánh giá riêng homoglyph, typo, whitespace.
- Dùng GraphNorm, residual và theo dõi Dirichlet Energy khi tăng độ sâu.
- Huấn luyện contrastive/domain-generalization trên nhiều miền.
- Chỉ mở rộng đa ngôn ngữ sau khi thay parser và vocabulary quan hệ phù hợp.

---

## 9. Kết luận

Nâng cấp GCN → RGCN có cơ sở toán học rõ ràng và được dữ liệu hiện tại ủng hộ mạnh. RGCN sửa trực tiếp việc mất nhãn quan hệ của mô hình cũ, tăng Accuracy trên 87,5% số tập và cải thiện lớn ở domain shift cùng nhiều tấn công token-level.

Mô hình mạnh nhất khi văn bản còn đủ cấu trúc cú pháp; yếu nhất khi đồ thị quá nhỏ, bị phá hủy cực đoan, parser không đáng tin hoặc phân phối miền làm lệch ngưỡng. Kết quả một seed, AUC giảm ở một số benchmark, thiếu calibration và thiếu benchmark chi phí là các khoảng trống phải giải quyết trước khi tuyên bố ưu thế ổn định trong triển khai thực tế.

---

## 10. Nguồn nội bộ đã sử dụng

- `docs/DuAn.md`: PRDetect gốc, pipeline, kết quả nền, điểm mạnh và hạn chế.
- `docs/Update.md`: phân tích RAID và lộ trình kiến trúc.
- `docs/BaoCao_PhanTich_GCN_sang_RGCN.md`: toán học và kết quả tiêu biểu.
- `docs/BaoCao_SoSanh_HieuNang_RGCN_Full.md`: phân nhóm 120 tập và trường hợp tăng/giảm.
- Bản cũ của `docs/BaoCaoNangCap.md`: thành phần triển khai và vận hành.
- `result/RGCN_test_result.txt`: nguồn chính cho so sánh GCN–RGCN seed 2026.
- `result/test_result.txt`: log lịch sử để đối chiếu bối cảnh, không ghép vào phép so sánh chính.
