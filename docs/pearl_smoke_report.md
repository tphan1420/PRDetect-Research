# PEARL-Lite — kết quả smoke thực tế

Ngày chạy: 15/11/2026. Phạm vi đã được người dùng mở rộng từ P0–P2 sang end-to-end. Mục tiêu ở lượt này là kiểm chứng thực thi; không phải nghiệm thu hiệu quả nghiên cứu.

## Đã thực thi

Windows / Python 3.13.0 / CPU; torch 2.15.0, PyG 2.8.1, transformers 4.57.6, spaCy 3.8.7, en_core_web_sm 3.8.0, NumPy 2.5.4, sklearn 1.9.1, pytest 9.2.0. Dùng RoBERTa-base thật (hidden 768), lookup/chunking legacy. Warm-start GCN từ `model/hc3_gcn_model_2025.pth`, không chỉnh sửa checkpoint gốc.

- Prepare: 24 HC3 source train documents, 8 source validation documents; 5 edit families × ratio 0.1; 120 train pairs, 40 validation pairs; không có parse failure.
- Source split check toàn bộ 8,000/1,000 rows: quarantine 17 train rows trùng nội dung/normalized content với validation. Dữ liệu baseline giữ nguyên. Có 8 train synonym no-op pairs do lexicon pilot không có từ phù hợp.
- Predictor: 3 epoch; checkpoint chọn bằng source-validation retention log-loss.
- Joint fine-tune: 2 epoch, lr=1e-4, epsilon=0.8, lambda_edge=0.1, batch=4 docs; dùng clean classification + valid masked edge loss, cân bằng theo document. Thời gian gồm tạo cache embeddings và train khoảng 16.93 giây sau khi model đã tải.
- Checkpoint tốt nhất ở epoch 1: validation clean BCE=0.0155153, accuracy=1.0 trên **8 mẫu**; epoch 2 BCE=0.05359. Gradient L1 lớn nhất của predictor trong joint training=0.82024; lưu/load checkpoint thành công.
- Single-text inference đã chạy với `The careful doctor helps people.`; trả về p(Human), q và weights. Không truyền bản clean tham chiếu hay provenance.
- 19 unit/integration tests pass. Integration test riêng dùng synthetic embeddings được ghi rõ trong test; CLI smoke ở trên dùng RoBERTa thật.
- Diagnostics gọi **builder baseline thật** trên 6 văn bản (2–406 nodes): embeddings max absolute difference=0, edge_index exact, epsilon=1 probability difference=0. Tolerance atol=1e-6, rtol=1e-5. Hash 11 baseline code/checkpoint files không đổi.

## Alignment và relation labels

| Partition | Alignment hợp lệ | Stable | Changed | Unknown | Negative / valid |
|---|---:|---:|---:|---:|---:|
| Train | 17,037 / 17,355 = 98.17% | 14,466 | 1,576 | 583 | 9.82% |
| Validation | 8,489 / 8,645 = 98.20% | 7,220 | 806 | 279 | 10.04% |

Counts tính trên các cặp, nên token nguồn được lặp qua edit families. Stable/changed chỉ có nhãn khi hai endpoint map 1:1. Delete endpoint, split/merge và whitespace resegmentation → unknown, không gán âm. Test cover identity, synonym, repeated tokens, insert, delete, split, merge, head change, relation change; case split `doctor→medical worker` bị mask đúng. Không claim đã review thủ công 30–50 corpus pairs trong lượt smoke này. Các thống kê family/relation/class/ratio/length và lý do alignment nằm trong manifest/gzip records, có thể mở lại để kiểm tra sâu.

## Predictor validation nhỏ

8,026 valid arc observations, positive rate=0.899576. Bảng dưới đây là kết quả thật sau **3 epoch smoke**, không phải huấn luyện hội tụ.

| Predictor | AUROC | AUPRC | Brier | ECE |
|---|---:|---:|---:|---:|
| Constant prior | 0.5000 | 0.8996 | 0.09034 | 0.00218 |
| Relation prior | 0.5586 | 0.9147 | 0.09004 | 0.01235 |
| Relation + POS prior | 0.5582 | 0.9139 | 0.09063 | 0.01701 |
| MLP smoke | 0.4633 | 0.8882 | 0.20279 | 0.33467 |

MLP q mean=0.56491, std=0.01238: predictor còn yếu, tập trung quanh hằng số và chưa vượt priors. Không xem lượt này là P2-pass theo tiêu chí nghiên cứu của tài liệu V2. Joint training được chạy theo yêu cầu mới nhằm kiểm chứng end-to-end; không phải bằng chứng chất lượng cho downstream. Không chạy shuffled control, nhiều seed hay ablations; có tùy chọn `shuffle_control` nếu cần sau này. q trên graph edited chỉ là diagnostics, không đánh giá bằng forward labels không hợp lệ và không thêm EditedView.

## Đánh giá từng điều kiện

Đọc file benchmark có sẵn, không dùng generator training thay benchmark. Kiểm tra mỗi file có 1,000 rows và label ordering khớp original. Smoke chọn 2 AI + 2 Human mỗi condition, cùng quy tắc chỉ số/seed. Toàn bộ 15 điều kiện chạy khoảng 52.76 giây. **N=4 mỗi ô: các accuracy sau chỉ xác nhận đường thực thi, không phải điểm Table 2/3/4 đầy đủ hoặc chứng minh cải thiện baseline.**

| Bảng | Domain | Conditions theo thứ tự | Accuracy smoke |
|---|---|---|---|
| Table 2 | HC3 | original, syn5, syn10, syn20, syn30 | 1.00, 1.00, 1.00, 1.00, 1.00 |
| Table 3 | HC3-trained → GPT3.5-Mixed | original, syn5, syn10, syn20, syn30 | 0.75, 0.75, 0.75, 0.50, 0.50 |
| Table 4 | HC3 | original, insert25, repeat25, replace25, delete25 | 1.00, 0.75, 1.00, 0.75, 0.75 |

Raw metrics, prediction probabilities và confusion matrices: `output/pearl/smoke/evaluation.json`. Checkpoint: `output/pearl/smoke/gcn/best.pt`. Edge metrics/history: `output/pearl/smoke/edge/metrics.json`. Train history/provenance: `output/pearl/smoke/gcn/training.json`. Parity: `output/pearl/smoke/parity.json`. Các output nằm trong Git ignore, không copy embeddings vào repo tracking.

## Files và tái lập

Mã mới trong [pearl/](../pearl/): `__init__`, `data`, `config`, `graph`, `edit_provenance`, `alignment`, `relation_retention`, `prepare`, `edge_features`, `edge_predictor`, `metrics`, `train_edge`, `embeddings`, `weighted_gcn`, `train_gcn`, `evaluate`, `infer`, `diagnostics`, `download_roberta` (các module `.py`), cùng README và requirements.

Config mới: [smoke](../configs/pearl_smoke.json), [HC3 full](../configs/pearl_hc3.json). Tests mới: [unit tests](../tests/test_pearl.py), [integration test](../tests/test_pearl_pipeline.py). Tài liệu mới: [audit](pearl_baseline_audit.md), báo cáo này. Chỉ thêm `output/pearl/` vào `.gitignore`, giữ thay đổi `notebooks/kaggle` có sẵn của người dùng.

Lệnh cài đặt, prepare, train, evaluate và infer đầy đủ trong [README](../pearl/README.md). Config smoke đã có output; để tái chạy, copy config sang run mới với các đường dẫn output mới. Full config bỏ giới hạn document và không dùng `--limit` khi evaluate.

## Chưa kiểm chứng

- Full source training, full benchmark, GPT-trained Table 2 và chiều GPT→HC3 chưa chạy. Dữ liệu benchmark có sẵn; giới hạn hiện tại là ngân sách smoke và chưa có baseline checkpoint GPT. Máy chỉ có CPU.
- Parser lịch sử không được pin trong baseline; parity bằng 0 chỉ chứng minh cùng implementation/parser/encoder hiện tại. Không xác minh được cache/parser revision dùng để tạo checkpoint lịch sử.
- Baseline ban đầu có khả năng được train trên các source texts trùng validation; lọc train derivative không thể đảo ngược thông tin trong pretrained baseline weights. Không claim tuyệt đối không leakage cho baseline lịch sử.
- Lexicon synonym và vocabulary random hiện nhỏ; no-op được báo rõ. Không khẳng định bảo toàn ngữ nghĩa, không train classification trên edits.
- Đĩa output lúc kiểm tra ~1.02 GB gồm cả tải model/cache, không tạo graph `.pkl`; chỉ hai cache embeddings sạch. Full prepare giữ metadata trong RAM, cần tăng subset dần theo tài nguyên.

Kết luận kỹ thuật: pipeline end-to-end đã chạy được. Kết luận nghiên cứu: **chưa đủ bằng chứng PEARL tốt hơn baseline; predictor smoke chưa đạt tiêu chí P2 của V2**.
