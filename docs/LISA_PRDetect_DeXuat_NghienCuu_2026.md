# LISA-PRDetect — Length-Invariant, Structure-Aware PRDetect

> **Cập nhật implementation:** đặc tả thực thi hiện hành là
> [LISA_GCN_Core.md](LISA_GCN_Core.md). LISA-GCN Core bắt buộc dùng GCN hai
> lớp gốc, không dùng RGCN/V3, không dùng perturbation loss, và không đưa
> syntax-aware swap/EMA/R-Drop/CMixStyle vào phiên bản đầu tiên. Các đoạn đề
> xuất lịch sử phía dưới có nhắc RGCN hoặc perturbation được giữ làm bối cảnh,
> không phải hướng dẫn implementation hiện tại.

**Tình trạng:** Đề xuất nghiên cứu, CHƯA được thực nghiệm. Không có đảm bảo vượt PRDetect gốc.
**Mục tiêu:** Train HC3 -> test GPT-3.5-Mixed không dùng nhãn target cho mô hình hoặc chọn hyperparameter; giữ robust với nhiễu HC3 Table 4; kiểm tra thêm domain/generator khác.

## 1. Căn cứ

Paper gốc: Li et al., NAACL Findings 2025, https://aclanthology.org/2025.findings-naacl.464/
- Table 1 (trang 4): HC3 mean text length Human 147.93, AI 178.65; GPT-3.5 Mixed Human 756.55, AI 501.13. **Quan sát tương quan, không phải bằng chứng nhân quả.**
- Table 1: mean node counts HC3 Human 20.23, AI 25.34; GPT Mixed Human 25.08, AI 25.09. Không đồng nhất số node/tree với độ dài toàn document.
- Table 3 (trang 7): HC3->GPT35 Original 73.2, synonyms 5/10/20/30% lần lượt 73.0/72.6/71.7/70.9%.
- Table 4 (trang 7), seed 2024, HC3 Original 98.5, Insert 98.3, Repeat 98.2, Replace 79.8, Delete 74.7%; paper in nhầm "Detect" cho Delete.
- Table 2: HC3 0% 98.78, HC3 30% 98.64; GPT training->test GPT original 96.56, 30% 96.38 (trung bình 5 seeds 2021-2025).
- Appendix C: paper đã làm đánh giá length-matched trong nội miền; do đó đóng góp không được mô tả như lần đầu nhận thấy length bias.

Bối cảnh thử nghiệm V3: RGCN+features+syntax swap; abla2 (tắt handcrafted structural features, giữ swap) 2025/2026 batch=1 đạt 70.8/71.4% nhưng chỉ hai seeds và chưa có Table 4. Các phương pháp subword alignment, CMixStyle, class-swap, soft semantic gates từng suy giảm metric hoặc structural robustness trong lịch sử. Không đề xuất lặp chúng như mới.

## 2. Giả thuyết nghiên cứu

H1: Mô hình hiện tại sử dụng dấu hiệu tương quan với độ dài document, kích thước/cấu trúc cây, chủ đề và nguồn viết khiến HC3 validation cao nhưng cross-domain yếu. Giả thuyết kiểm chứng bằng phân tích mô hình length-only, size-only và predictive score theo từng bucket.
H2: Bộ biểu diễn bằng *phân phối bằng chứng cú pháp cục bộ* từ các cửa sổ cùng độ dài, thay vì graph embedding toàn văn bản, làm giảm lệ thuộc kích thước và thể loại.
H3: Đồng nhất dự đoán qua các góc nhìn độ dài + robust optimization theo nhóm môi trường từ source-only giảm worst-group loss và variance qua seeds, trong khi mild textual perturbations bảo vệ Table 4.

## 3. Kiến trúc LISA-PRDetect

### A. Multi-scale syntactic multi-instance encoder

- Bắt đầu từ văn bản raw, parse spaCy câu + dependency. Ưu tiên giữ nguyên checkpoint RoBERTa frozen/legacy word embeddings của PRDetect/V3 để tránh thất bại A1. Không thêm depth/children/distance handcrafted trong thử nghiệm chính (có ablation).
- Chia văn bản thành các *sentence-aligned* windows mục tiêu 96–160 spaCy tokens (mốc ban đầu: 128). Không cắt cây dependency giữa chừng; nếu câu >160 token, xử lý riêng bằng phân tách có kiểm soát. Ghi số token thực tế mỗi window.
- Mỗi window có graph riêng; encoder RGCN hoặc GCN cho node features -> pool mean (option: std); tạo logit z_j của window.
- Document logit = mean_j(z_j). Không đưa số windows, độ dài văn bản hoặc đặc trưng đếm thô vào classifier. Nếu inference có nhiều windows, lấy toàn bộ windows với stride cố định, không lấy chỉ một window đầu.
- Dùng số windows cố định trong *một optimizer forward* được sampling trong train để giới hạn VRAM; inference xử lý tuần tự các windows và kết hợp bằng mean logits. Thay đổi batch train không được thay đổi inference scores quá sai số số học.

### B. Counterfactual scale consistency

Tạo hai views của một document từ các cửa sổ sentence-aligned khác nhau hoặc độ rộng khác nhau (96,128,160), cùng nhãn document. Tối ưu nhãn document và consistency nhẹ:

    L_ce = BCEWithLogits(z_doc, y)
    L_scale = (sigmoid(z_view_a) - sigmoid(z_view_b))^2

Bắt đầu alpha=0.1 sau warmup 2 epochs. KHÔNG giả định mọi short snippet luôn có tín hiệu nhãn rõ như document; theo dõi F1 theo length bucket, giảm alpha nếu giảm.

### C. Source-only group-robust objective

Gán environment từ HC3 *training only*: (topic/category hoặc cluster theo prompt), (length quantile bin); nếu nhóm đủ lớn thêm nhãn label. Nhóm nhỏ gộp, loại rò rỉ nhãn khi clustering. GroupDRO cập nhật trọng số môi trường dựa trên source minibatch/epoch loss, bổ sung regularization và early stopping như Sagawa et al. ICLR 2020. Một lựa chọn đơn giản ổn định hơn là mean group loss + 0.3 * CVaR(top 25% group losses).

    L_group = weighted group BCE
    L_total = L_group + 0.1 L_scale + 0.1 L_perturb + regularization

Để giữ robustness, xây view từ HC3 *train* với 5–10% synonym/repeat/insert/delete nhẹ (label-preserving sau lọc) và tái parse; không dùng trực tiếp HC3 test nhiễu 25% để huấn luyện; tăng cường online từng batch hoặc cache tuần tự.

Lưu ý: bản đầu có thể chỉ sử dụng L_scale+GroupDRO với raw text; controlled perturbation là component ablation riêng. Không đưa tất cả mới vào một run mà không có ablation sau cùng.

## 4. Luồng xử lý (đề xuất)

    Text -> sentence-aware windowing -> per-window dependency graphs
         -> shared relation-aware encoder -> window embeddings/logits
         -> length-neutral mean-logit aggregation -> document prediction
       + paired length-view consistency (same source document)
       + environment-balanced / worst-group supervised objective
       + optional mild edit consistency from source training text

## 5. Lộ trình tối thiểu

1. **Audit/dry run**: kiểm tra no target leakage, graph cache, source split theo question_id; encode một document ở inference batches 1, 8, 32 và xác nhận số dự đoán giống nhau; test mean logit với duplicate windows gần invariance.
2. **Feasibility trên 2 seeds**: 2025, 2026, batch cố định 1, đối chiếu V3/Abla2; sử dụng validation HC3 clean, validation held-out topic/length và source-generated noise để chọn checkpoint, không dùng GPT labels. Theo dõi AUC, AI Recall, Table4 proxy.
3. **Ablation**: PRDetect GCN reproducible; V3 baseline; size-normalized windows only; +scale-consistency; +group-robust; +mild edit regularizer; hoàn chỉnh. Để tách đóng góp, giữ tất cả yếu tố còn lại giống nhau.
4. **Final**: 5 seeds 2021–2025 để đối chiếu thiết kế paper và/hoặc bộ seeds độc lập preregistered; báo cáo mean±std, 95% bootstrap CI theo văn bản, worst seed, length-stratified results, TPR@5%FPR và Table 2/3/4. Chỉ công bố lợi thế khi protocol cùng split, model selection, test perturbations; Table4 paper 1 seed là giới hạn so sánh.
5. **Generalization ngoài cặp benchmark**: sample test từ RAID/MAGE theo domain và generator mới; không dùng benchmark này để tune nếu trình bày là held-out.

## 6. Điều kiện để nói "thắng tác giả"

- Điểm > PRDetect gốc Table3 tất cả mức 0/5/10/20/30; Table4 Insert/Repeat/Replace/Delete so same test and perturbation; Table2 không sụp. Ưu tiên chênh lệch có khoảng tin cậy và thực nghiệm đa seed; không cherry-pick seeds, epoch, threshold.
- Nếu có khác biệt về dữ liệu, nhãn, source embeddings, graph builder và seed, công bố thành hai protocol và không gọi là fair same-protocol replication.
- Không thể cam kết bất kỳ mức xác suất thành công (>90%) trước thử nghiệm. Đề xuất là research hypothesis có nền tảng logic, không phải kết quả đã được chứng minh.

## 7. Nghiên cứu tham khảo

1. PRDetect, Li et al., NAACL Findings 2025: https://aclanthology.org/2025.findings-naacl.464/
2. DISGEN: Enhancing Size Generalization in GNNs, Huang et al., ICML 2024: https://proceedings.mlr.press/v235/huang24ac.html
3. Distributionally Robust Neural Networks for Group Shifts, Sagawa et al., ICLR 2020: https://mlanthology.org/iclr/2020/sagawa2020iclr-distributionally/
4. How to Generalize Detection of AI-generated Text: Confounding Neurons, Borile & Abrate, EMNLP Findings 2025: https://aclanthology.org/2025.findings-emnlp.1388/
5. Explaining Generalization of AI-Generated Text Detectors Through Linguistic Analysis, Xia et al., EACL 2026: https://aclanthology.org/2026.eacl-long.307/
6. RAID benchmark, Dugan et al., ACL 2024: https://aclanthology.org/2024.acl-long.674/
7. MAGE, Li et al., ACL 2024: https://aclanthology.org/2024.acl-long.3/
