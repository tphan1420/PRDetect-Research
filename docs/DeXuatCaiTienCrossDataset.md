# Đề xuất cải tiến PRDetect cho cross-dataset

## 1. Kết luận chính

RGCN không phải nút thắt duy nhất và cũng không nên là thay đổi tiếp theo duy nhất. Kết quả hiện tại cho thấy ba vấn đề lớn hơn:

1. Protocol huấn luyện hiện tại không còn tương đương với protocol của tác giả.
2. Node embedding RoBERTa đang được tạo bằng cách đưa token SpaCy trực tiếp qua `convert_tokens_to_ids`, không căn chỉnh subword BPE và không dùng attention mask/special token.
3. Đồ thị chỉ chứa cạnh `dependent -> head` trong từng câu; các cây câu trong một tài liệu dài bị rời nhau. Tăng số tầng RGCN không thể truyền thông tin giữa các thành phần không liên thông.

Vì vậy, hướng có triển vọng nhất là **mô hình đa góc nhìn có cổng thích nghi**: nhánh Transformer tuần tự + nhánh đồ thị phân cấp + nhánh phong cách/ký tự, sau đó huấn luyện đa miền bằng supervised contrastive learning và consistency loss.

## 2. Đọc lại kết quả hiện tại

### 2.1. `cross_test.txt` chưa phải kết quả RGCN

Toàn bộ checkpoint trong [`cross_test.txt`](../cross_test.txt) có tên `hc3_gcn_model_<seed>.pth`. Trung bình sáu seed:

| Mức nhiễu | Accuracy trung bình | Min–max | Accuracy tác giả | Chênh lệch |
|---:|---:|---:|---:|---:|
| 0% | 66,52% | 59,70–71,30% | 73,20% | -6,68 điểm % |
| 5% | 65,77% | 59,30–70,50% | 73,00% | -7,23 điểm % |
| 10% | 65,32% | 58,80–69,70% | 72,60% | -7,28 điểm % |
| 20% | 64,13% | 57,90–68,80% | 71,70% | -7,57 điểm % |
| 30% | 63,52% | 57,00–67,70% | 70,90% | -7,38 điểm % |

Khoảng dao động giữa các seed rất lớn. Đây là tín hiệu về protocol tối ưu hóa/checkpoint selection, không chỉ về năng lực kiến trúc.

### 2.2. RGCN thực tế đã gần kết quả cross-dataset của tác giả

[`metrics.json`](../metrics.json) dùng checkpoint `hc3_rgcn_model_2026.pth`. So với Bảng 3 của bài báo:

| Mức nhiễu GPT-3.5-Mixed | RGCN seed 2026 | Tác giả | Chênh lệch |
|---:|---:|---:|---:|
| 0% | 72,60% | 73,20% | -0,60 điểm % |
| 5% | 71,40% | 73,00% | -1,60 điểm % |
| 10% | 70,70% | 72,60% | -1,90 điểm % |
| 30% | 69,70% | 70,90% | -1,20 điểm % |

Do đó, nhận định chính xác hơn là: **GCN tái huấn luyện đang thấp hơn tác giả đáng kể; RGCN một seed đã gần mức tác giả nhưng chưa vượt ổn định**.

### 2.3. Dữ liệu không phải nguyên nhân trực tiếp

SHA-256 của `hc3_train`, `hc3_val`, `hc3_test` và `gpt3.5_mixed_test_split` trong cây hiện tại trùng với bản mã nguồn tác giả ở [`TacGia/PRDetect/`](../TacGia/PRDetect/). Sai khác nhiều khả năng nằm ở preprocessing runtime, training protocol và checkpoint selection.

## 3. Vì sao protocol hiện tại chưa so sánh công bằng với paper

Bản tác giả trong [`TacGia/PRDetect/gcn.py`](../TacGia/PRDetect/gcn.py) cập nhật từng đồ thị một, theo thứ tự cố định, dùng Adam không `weight_decay` và chạy 40 epoch. Biến `tag` được đặt lại mỗi epoch nên early stopping thực tế không hoạt động.

Bản hiện tại trong [`gcn.py`](../gcn.py) và [`rgcn.py`](../rgcn.py) dùng:

- mini-batch 32;
- `shuffle=True`;
- `weight_decay=1e-5`;
- early stopping patience 5;
- checkpoint theo validation accuracy tại ngưỡng 0,5.

Đây đều là thay đổi hợp lý về kỹ thuật, nhưng làm thay đổi quỹ đạo tối ưu hóa. Cần có một chế độ `paper_protocol` để tách hai câu hỏi:

1. Có tái hiện được kết quả tác giả không?
2. Kiến trúc mới có tốt hơn dưới cùng một protocol không?

## 4. Nút thắt biểu diễn cần sửa trước khi đổi GNN

### 4.1. Căn chỉnh SpaCy token với RoBERTa BPE

[`building_graph.py`](../building_graph.py) hiện gọi:

```python
token_ids = tokenizer.convert_tokens_to_ids(chunk)
```

trên danh sách token SpaCy. RoBERTa sử dụng byte-level BPE và mã hóa khoảng trắng trong token; token SpaCy không tương ứng một-một với vocabulary item của RoBERTa. Cách hiện tại cũng không thêm `<s>`, `</s>`, attention mask hoặc xử lý subword.

Thay bằng `RobertaTokenizerFast`/`DebertaV2TokenizerFast` với `is_split_into_words=True`, lấy `word_ids()`, rồi mean-pool các subword thuộc cùng một token SpaCy. Với tài liệu dài, dùng sliding window có overlap và hợp nhất embedding ở vùng chồng lấn.

Đây là ablation cần chạy đầu tiên vì nó sửa chất lượng node feature cho cả GCN, RGCN và mọi kiến trúc graph mới.

### 4.2. Fine-tune encoder có kiểm soát

Pipeline hiện tại tiền tính embedding từ RoBERTa frozen. Có thể triển khai theo ba nấc:

1. Frozen encoder + projection `768 -> 256` + LayerNorm.
2. Fine-tune 2–4 tầng cuối bằng learning rate nhỏ hơn graph head 10 lần.
3. LoRA/adapters nếu VRAM hạn chế.

Không nên thay RoBERTa bằng DeBERTa rồi vẫn dùng cách ánh xạ token sai; hiệu quả của encoder mới sẽ bị triệt tiêu.

## 5. Kiến trúc đề xuất: PRDetect-MV

### 5.1. Nhánh tuần tự

Dùng DeBERTa-v3-base cho văn bản dưới 512 subword. Với tài liệu dài, dùng hierarchical Transformer: encode từng cửa sổ/câu, sau đó attention-pool các vector cửa sổ. Nhánh này giữ thông tin trật tự, discourse và ngữ nghĩa mà cây phụ thuộc không biểu diễn đầy đủ.

Đầu ra: `z_seq`.

### 5.2. Nhánh đồ thị phân cấp cấp tài liệu

Không chỉ đổi `RGCN -> RGAT`. Trước hết cần sửa topology:

- cạnh phụ thuộc xuôi và ngược với vocabulary cố định;
- cạnh tuần tự `next_token`/`prev_token`;
- node câu nối với root của câu;
- một node tài liệu nối với mọi node câu;
- tùy chọn cạnh coreference/entity giữa các câu.

Kiến trúc tối thiểu nên là 2–3 tầng RGATv2 có residual + GraphNorm. Kiến trúc nâng cao có thể dùng Heterogeneous Graph Transformer hoặc GraphGPS, kết hợp message passing cục bộ với global attention.

Đầu ra nên ghép mean-pooling, max-pooling và biểu diễn document node: `z_graph`.

Lợi ích của document node quan trọng hơn việc tăng RGCN từ 2 lên 6 tầng: nó nối trực tiếp các cây câu đang rời nhau và rút ngắn khoảng cách truyền thông tin.

### 5.3. Nhánh phong cách và ký tự

Thêm một nhánh nhẹ để giảm phụ thuộc vào chủ đề và parser:

- phân phối độ dài câu/từ;
- type-token ratio và entropy từ vựng;
- mật độ dấu câu, POS và dependency label;
- tỷ lệ lặp n-gram;
- burstiness của độ dài câu;
- đặc trưng ký tự/byte bằng CharCNN nhỏ để bắt homoglyph, whitespace và typo.

Có thể thay nhóm handcrafted feature bằng style encoder tiền huấn luyện khi có tài nguyên. Nghiên cứu về style embedding cho thấy loại biểu diễn này ít phụ thuộc nội dung hơn semantic embedding và khái quát tốt qua miền/LLM.

Đầu ra: `z_style`.

### 5.4. Gated fusion theo độ tin cậy

Không concat cố định. Dùng một gating network nhận:

- độ dài văn bản;
- tỷ lệ token `<unk>`/subword;
- parser confidence hoặc tỷ lệ cạnh `dep`/`<unk>`;
- số câu, số node và số thành phần liên thông.

Sau đó:

$$
[g_s,g_g,g_t]=\operatorname{softmax}(\operatorname{MLP}(q)),
$$

$$
z=g_s z_{seq}+g_g z_{graph}+g_t z_{style}.
$$

Khi văn bản ngắn hoặc parse lỗi, mô hình tự giảm trọng số graph. Khi văn bản dài có cấu trúc tốt, nhánh graph được dùng nhiều hơn.

## 6. Mục tiêu huấn luyện cho cross-domain

### 6.1. Multi-source balanced sampling

Huấn luyện trên HC3 + GPT-3.5-Mixed + phần train của RAID, cân bằng theo `(label, domain, generator)`. Nếu chỉ huấn luyện HC3 thì không thể kỳ vọng một thay đổi GNN tự giải quyết hoàn toàn domain shift.

Validation nên dùng leave-one-domain-out hoặc leave-one-generator-out, không chọn checkpoint chỉ bằng HC3 validation accuracy.

### 6.2. Supervised contrastive learning xuyên miền

Trong mỗi batch, positive pair là hai mẫu cùng nhãn Human/AI nhưng khác domain/generator; negative pair là khác nhãn. Hàm loss:

$$
\mathcal L=\mathcal L_{CE}+\lambda\mathcal L_{SupCon}.
$$

Cách này ép biểu diễn giữ tín hiệu Human/AI thay vì ghi nhớ chủ đề của HC3 hay định dạng news của GPT-3.5-Mixed.

### 6.3. Perturbation consistency

Tạo hai view của cùng văn bản: clean và synonym/delete/insert/back-translation/rewrite. Tối ưu thêm:

$$
\mathcal L_{cons}=\operatorname{JS}(p(y|x),p(y|\tilde x)).
$$

Augmentation phải áp dụng đối xứng cho cả Human và AI để mô hình không học shortcut “có nhiễu = AI”.

### 6.4. Domain-adversarial head

Thêm gradient reversal head dự đoán domain. Encoder chính học làm tốt nhãn Human/AI nhưng làm kém dự đoán domain. Đây là bước sau SupCon; cần ablation vì loại bỏ quá mạnh thông tin domain có thể làm mất một phần tín hiệu hữu ích.

## 7. Calibration và checkpoint selection

[`metrics.json`](../metrics.json) cho thấy GPT-3.5-Mixed clean có accuracy 0,726 nhưng AUROC 0,8782. Mô hình còn năng lực xếp hạng nhưng ngưỡng 0,5 không ổn định khi đổi miền.

Các thay đổi cần có:

- chọn checkpoint bằng trung bình AUROC/PR-AUC trên nhiều validation domain;
- fit temperature trên validation để hiệu chỉnh xác suất;
- chọn threshold theo mục tiêu cụ thể: balanced accuracy, F1, hoặc `TPR@1%FPR`/`TPR@5%FPR`;
- tuyệt đối không chọn threshold trên test set;
- báo ECE và Brier score bên cạnh Accuracy/AUROC/F1.

Calibration là bước rẻ và cần làm, nhưng không thay thế cải tiến representation vì AUC trên các tập RAID/LLaMA/Mistral vẫn giảm khi chuyển GCN sang RGCN.

## 8. Thứ tự thí nghiệm khuyến nghị

| ID | Thay đổi | Mục tiêu | Độ phức tạp |
|---|---|---|---|
| E0 | Tái hiện `paper_protocol` cho GCN, 5 seed 2021–2025 | Xác định khoảng cách tái hiện thật | Thấp |
| E1 | Sửa BPE–SpaCy alignment, giữ nguyên GCN | Đo đóng góp node feature | Trung bình |
| E2 | RGCN với cạnh ngược cố định + sentence/document node | Sửa topology tài liệu | Trung bình |
| E3 | Hybrid `sequence + graph` với gated fusion | Bù tín hiệu ngoài cú pháp | Trung bình–cao |
| E4 | Thêm style/CharCNN branch | Cross-domain + adversarial noise | Trung bình |
| E5 | Multi-source + SupCon + consistency | Học biểu diễn bất biến miền | Cao |
| E6 | RGATv2/HGT/GraphGPS | Học trọng số cạnh và global attention | Cao |
| E7 | Temperature/threshold calibration | Chọn operating point đúng | Thấp |

Chỉ chuyển sang E6 sau khi E1–E3 đã có kết quả. Nếu E1 hoặc E3 tạo cải thiện lớn, nguyên nhân chính không nằm ở lựa chọn RGCN/RGAT mà ở input representation và thiếu nhánh tuần tự.

## 9. Ma trận ablation tối thiểu

Mọi cấu hình chạy cùng 5 seed, cùng split và cùng budget epoch:

1. GCN paper-faithful.
2. GCN + token alignment đúng.
3. RGCN + token alignment đúng.
4. (3) + inverse edges.
5. (4) + sentence/document node.
6. Sequence-only DeBERTa.
7. Sequence + graph concat.
8. Sequence + graph gated fusion.
9. (8) + style/CharCNN.
10. (9) + SupCon/consistency.

Báo cáo theo từng domain, generator, decoding strategy, độ dài và perturbation; dùng mean ± std, bootstrap 95% CI, AUROC, PR-AUC, F1, balanced accuracy, ECE, Brier và `TPR@1/5%FPR`.

## 10. Các hướng không nên ưu tiên ngay

- **RGCN sâu 4–6 tầng:** không nối được các cây câu rời nhau và tăng nguy cơ over-smoothing.
- **Dirichlet Energy trước khi sửa topology:** giải quyết over-smoothing nhưng không giải quyết token alignment hoặc domain shift.
- **MoE lớn:** chi phí cao và dễ học domain shortcut khi dữ liệu từng miền chưa cân bằng.
- **Chỉ thay RoBERTa bằng DeBERTa:** không đủ nếu vẫn ánh xạ token sai và chỉ dùng graph branch.
- **Chỉ đổi RGCN sang RGAT:** attention trên cùng một graph thiếu cạnh toàn cục vẫn không phục hồi discourse giữa các câu.

## 11. Tài liệu tham khảo trực tiếp

- Li et al. (2025), [PRDetect: Perturbation-Robust LLM-generated Text Detection Based on Syntax Tree](../papers/2025.findings-naacl.464.pdf).
- Dugan et al. (2025), [GenAI Content Detection Task 3: Cross-Domain Machine Generated Text Detection Challenge](https://aclanthology.org/2025.genaidetect-1.45/).
- Li et al. (2024), [MAGE: Machine-generated Text Detection in the Wild](https://aclanthology.org/2024.acl-long.3/).
- Huang et al. (2025), [MAGRET: Machine-generated Text Detection with Rewritten Texts](https://aclanthology.org/2025.coling-main.557/).
- Durandard et al. (2026), [Evaluating Style Embeddings for Machine-Generated Text Detection](https://aclanthology.org/2026.lrec-1.205/).

