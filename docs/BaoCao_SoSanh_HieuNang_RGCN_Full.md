# BÁO CÁO PHÂN TÍCH SO SÁNH HIỆU NĂNG TOÀN DIỆN: GCN VS. RGCN (PRDETECT)

**Mục tiêu:** So sánh chi tiết hiệu năng trên toàn bộ 120 tập kiểm thử trong `result/RGCN_test_result.txt` và `result/test_result.txt`, chỉ rõ các nhóm tăng/giảm, giải thích nguyên nhân kỹ thuật và đưa ra các hạn chế hiện tại.
**Ngày thực hiện:** 2026-10-20

---

## 1. TỔNG QUAN KẾT QUẢ KIỂM THỬ TRÊN 120 TẬP DỮ LIỆU

Thực nghiệm trên toàn bộ **120 tập kiểm thử** cho thấy việc chuyển đổi kiến trúc từ GCN (Homogeneous Graph Convolutional Network) sang RGCN (Relational Graph Convolutional Network) mang lại sự cải thiện hiệu năng vượt trội và vô cùng nhất quán:

* **105 / 120 tập kiểm thử TĂNG ĐIỂM Accuracy (87.5%):** Mức tăng cao nhất đạt **+12.90%** (chèn từ `insert_50`) và **+12.20%** (dịch ngược `gpt3.5_mixed_test_translate`).
* **13 / 120 tập kiểm thử GIẢM ĐIỂM Accuracy (10.8%):** Tập trung ở các kịch bản nhiễu xóa/lặp từ cực đoan (>50%-75%) hoặc dịch chuyển ngưỡng trên tập matched.
* **2 / 120 tập kiểm thử KHÔNG ĐỔI Accuracy (1.7%):** Rơi vào ngưỡng bão hòa thông tin do bị xóa từ 70%-75%.

---

## 2. PHÂN TÍCH HIỆU NĂNG THEO TỪNG NHÓM TẬP DỮ LIỆU

### Nhóm 1: Tập Dữ liệu In-Domain (HC3 Test Benchmark & Sub-splits)
* **Xu hướng:** **TĂNG ĐIỂM RẤT MẠNH** (đặc biệt trên văn bản ngắn và dài).
* **Các tập TĂNG:**
  * `hc3_test` (Benchmark tổng thể): Accuracy **97.40% → 98.60% (+1.20%)**, ROC-AUC **0.9960 → 0.9979**, F1-score **0.9746 → 0.9861**.
  * `hc3_test_50_100` (Văn bản ngắn 50-100 tokens): Accuracy **88.33% → 98.33% (+10.00%)**, F1 **0.8955 → 0.9836 (+0.0881)**.
  * `hc3_test_100_150`: Accuracy **97.00% → 98.00% (+1.00%)**.
  * `hc3_test_200_250`: Accuracy **96.67% → 98.33% (+1.66%)**.
  * `hc3_test_250_300`: Accuracy **98.00% → 100.00% (+2.00%)**.
  * `hc3_test_300_inf` (Văn bản dài >300 tokens): Accuracy **96.15% → 100.00% (+3.85%)**.
  * `hc3chatgpt`: Accuracy **96.88% → 100.00% (+3.12%)**.
* **Các tập GIẢM nhẹ:**
  * `hc3_test_150_200`: Accuracy **98.75% → 95.00% (-3.75%)** (tuy nhiên ROC-AUC vẫn giữ mức **1.0000** tuyệt đối).
  * `hc3human`: Accuracy **97.92% → 96.84% (-1.08%)**.

---

### Nhóm 2: Tập Dữ liệu Domain Shift & Văn bản dài (GPT-3.5 Mixed)
* **Xu hướng:** **TĂNG ĐỘT PHÁ** trên các văn bản dài và hỗn hợp phức tạp.
* **Các tập TĂNG:**
  * `gpt3.5_mixed_test_split` (Benchmark tổng thể): Accuracy **64.90% → 72.60% (+7.70%)**, F1 **0.7347 → 0.7794 (+0.0447)**.
  * `gpt3.5_mixed_test_translate` (Dịch ngược): Accuracy **65.00% → 77.20% (+12.20%)**.
  * `gpt3.5_mixed_test_llama3_8b_instruct_rewrite` (LLM Viết lại): Accuracy **71.40% → 78.20% (+6.80%)**.
  * `gpt3.5_mixed_test_adj_10` (Thay từ đồng nghĩa): Accuracy **61.40% → 70.10% (+8.70%)**.
  * `gpt3.5_mixed_test_insert_15`: Accuracy **53.80% → 60.80% (+7.00%)**.
* **Các tập GIẢM:**
  * `matched_gpt3.5_mixed_test_split`: Accuracy **79.90% → 75.00% (-4.90%)**.
  * `matched_gpt3.5_mixed_test_adj_5_synonym_replace`: Accuracy **64.70% → 59.70% (-5.00%)**.
  * Các mẫu câu rất ngắn: `20_40` (-1.00%), `80_100` (-0.50%), `delete_15` (-0.30%).

---

### Nhóm 3: Out-of-Domain RAID Benchmark & LLM Thế hệ mới
* **Xu hướng:** **TĂNG ĐỒNG BỘ** trên hầu hết các mô hình sinh văn bản mới.
* **Các tập TĂNG:**
  * `mini_raid_test_all`: Accuracy **68.59% → 70.39% (+1.80%)**.
  * `mini_raid_test_news`: Accuracy **67.27% → 68.57% (+1.30%)**.
  * `raid_llama_greedy_test`: Accuracy **74.40% → 76.80% (+2.40%)**.
  * `raid_llama_sampling_test`: Accuracy **73.05% → 74.50% (+1.45%)**.
  * `raid_mistral_test`: Accuracy **78.10% → 78.40% (+0.30%)**.
  * `gpt2`: Accuracy **38.62% → 44.42% (+5.80%)**.
  * `llama2_gen`: Accuracy **98.52% → 99.02% (+0.50%)**.
* **Các tập GIẢM nhẹ:**
  * `mini_raid_test_abstracts`: Accuracy **70.47% → 70.37% (-0.10%)**.
  * `grover_dev_split`: Accuracy **49.90% → 49.85% (-0.05%)**.

---

### Nhóm 4: Tấn công Đối kháng & Biến đổi Văn bản (Adversarial Attacks)
* **Xu hướng:** **TĂNG VƯỢT BẬC**, mô hình thể hiện khả năng kháng nhiễu cực ấn tượng.
* **Các tập TĂNG MẠNH:**
  * **Tấn công chèn từ (Insertion):** `insert_50` (+12.90%), `insert_30` (+9.90%), `insert_70` (+9.00%), `insert_25` (+8.40%), `insert_15` (+6.60%).
  * **Tấn công thay thế từ (Replacement):** `replace_25` (+11.00%), `replace_30` (+9.60%), `replace_15` (+8.70%).
  * **Dịch ngược & Viết lại:** `hc3_test_translate` (+8.20%), `hc3_test_llama3_8b_instruct_rewrite` (+4.00%).
  * **Tấn công xóa từ (Deletion):** `delete_15` (+6.90%), `delete_25` (+5.60%), `delete_30` (+4.20%).
  * **Thay từ đồng nghĩa:** `adj_30_synonym_replace` (+2.00%).
* **Các tập GIẢM / KHÔNG ĐỔI:**
  * Nhiễu xóa từ cực đoan (>50%-75%): `delete_50` (-0.10%), `delete_70` (0.00%), `delete_75` (0.00%).
  * Nhiễu lặp từ cực đoan (70%-75%): `repeat_70` (-0.30%), `repeat_75` (-1.20%).

---

## 3. BẢNG TỔNG HỢP SO SÁNH CÁC TẬP KIEAM THỬ TIÊU BIỂU

| Tập Kiểm thử (Dataset) | Loại miền & Đặc tính | GCN Acc (%) | RGCN Acc (%) | $\Delta$ Acc (%) | GCN F1 | RGCN F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **hc3_test** | In-Domain Benchmark | 97.40 | **98.60** | **+1.20** | 0.9746 | **0.9861** |
| **hc3_test_50_100** | In-Domain Short Text | 88.33 | **98.33** | **+10.00** | 0.8955 | **0.9836** |
| **hc3_test_300_inf** | In-Domain Long Text | 96.15 | **100.00** | **+3.85** | 0.9630 | **1.0000** |
| **gpt3.5_mixed_test_split** | Domain Shift Long Text | 64.90 | **72.60** | **+7.70** | 0.7347 | **0.7794** |
| **gpt3.5_mixed_translate** | Domain Shift Back-Trans | 65.00 | **77.20** | **+12.20** | 0.7352 | **0.8094** |
| **mini_raid_test_all** | Out-of-Domain RAID | 68.59 | **70.39** | **+1.80** | 0.7375 | **0.7475** |
| **raid_llama_greedy_test** | Out-of-Domain LLaMA | 74.40 | **76.80** | **+2.40** | 0.7836 | **0.7945** |
| **raid_mistral_test** | Out-of-Domain Mistral | 78.10 | **78.40** | **+0.30** | 0.8089 | **0.8059** |
| **hc3_test_insert_50** | Adversarial Insertion | 64.60 | **77.50** | **+12.90** | 0.7408 | **0.8181** |
| **hc3_test_replace_25** | Adversarial Replace | 67.80 | **78.80** | **+11.00** | 0.7583 | **0.8268** |
| **hc3_test_translate** | Adversarial Translation | 76.60 | **84.80** | **+8.20** | 0.8098 | **0.8671** |
| **hc3_test_150_200** | In-Domain Sub-split | **98.75** | 95.00 | -3.75 | **0.9873** | 0.9474 |
| **matched_gpt3.5_mixed** | Matched Domain Shift | **79.90** | 75.00 | -4.90 | **0.8281** | 0.7941 |

---

## 4. GIẢI THÍCH NGUYÊN NHÂN KĨ THUẬT (TĂNG & GIẢM)

### 4.1. Lý do RGCN TĂNG ĐIỂM vượt trội trên 105/120 tập test
1. **Biểu diễn Đa quan hệ (60+ Relational Edges):** GCN đồng nhất ép phẳng cây cú pháp thành ma trận kề 0/1, làm mất thông tin quan hệ. RGCN gán ma trận trọng số riêng $W_r$ cho 60+ loại quan hệ cú pháp SpaCy, giúp phân biệt rõ nét cách nối từ đặc trưng của AI (mệnh đề quan hệ `acl`, cụm danh từ bổ nghĩa `compound`,...) so với người viết.
2. **Điều chuẩn bằng Phân rã Cơ sở (Basis Decomposition $B=30$):** Giảm số lượng tham số, đóng vai trò bộ điều chỉnh (regularizer) chống overfit và tăng khả năng tổng quát hóa Out-of-Domain.
3. **Tính Bất biến Cấu trúc (Structural Invariance):** Khi bị chèn/xóa/thay thế từ hoặc dịch ngược, ngữ nghĩa bề mặt xáo trộn nhưng khung gầm cú pháp phụ thuộc cốt lõi vẫn bảo tồn, giúp RGCN kháng nhiễu vượt trội.
4. **Tăng cường tín hiệu cho câu ngắn (50-100 tokens):** Văn bản ngắn thiếu từ vựng làm GCN dễ đoán sai (88.33%). Nhãn quan hệ cú pháp phong phú cung cấp tín hiệu bổ sung sắc bén đẩy Accuracy lên **98.33% (+10.00%)**.

### 4.2. Lý do GIẢM ĐIỂM ở 13 tập cụ thể
1. **Sự đứt gãy đồ thị do nhiễu cực đoan (>50%-75%):** Khi văn bản bị xóa hoặc lặp từ quá 50%, cây cú pháp bị vỡ vụn thành các thành phần liên thông rời rạc. RGCN dựa vào nhãn quan hệ cạnh; khi cạnh bị vỡ, RGCN bị nhiễu lan truyền hơn GCN đơn giản.
2. **Dịch chuyển ngưỡng quyết định (Threshold Shift):** Trên tập `matched_gpt3.5_mixed` (-4.90%), bản chất AUC vẫn đạt 0.8968 nhưng phân phối điểm logit bị dịch chuyển khỏi ngưỡng mặc định (0.5), đòi hỏi Căn chỉnh Ngưỡng (Threshold Calibration).
3. **Biến động mẫu nhỏ:** Mức giảm ở `hc3_test_150_200` (-3.75%) là do số mẫu trong phân đoạn này nhỏ, trong khi ROC-AUC vẫn giữ mức 1.0000 tuyệt đối.

---

## 5. HẠN CHẾ HIỆN TẠI VÀ ĐỀ XUẤT CẢI TIẾN

### Hạn chế hiện tại
1. **Chi phí Tiền xử lý cao:** Dựng đồ thị đa quan hệ với 60+ loại cạnh yêu cầu phân tích SpaCy lâu hơn.
2. **Nhạy cảm với Đứt gãy Đồ thị Cực đoan:** Độ chính xác phụ thuộc vào tính toàn vẹn của cây cú pháp phụ thuộc.
3. **Thiếu Cơ chế Căn chỉnh Ngưỡng Động:** Sử dụng cố định ngưỡng 0.5 gây suy giảm Accuracy trên một số tập dịch chuyển miền dù AUC cao.
4. **Giới hạn của Encoder cơ sở:** Mô hình vẫn đang sử dụng RoBERTa cơ sở thay vì DeBERTa-v3.

### Lộ trình nâng cấp đề xuất
1. **Nâng cấp sang DeBERTa-v3:** Tăng cường biểu diễn ngữ nghĩa và vị trí tương đối với Disentangled Attention.
2. **Tích hợp RGATv2:** Thêm cơ chế chú ý động trên từng quan hệ cú pháp để giảm trọng số cạnh bị nhiễu.
3. **Áp dụng Căn chỉnh Ngưỡng (Operating Point Calibration):** Sử dụng Platt Scaling để tối ưu ngưỡng quyết định trên miền OOD.
