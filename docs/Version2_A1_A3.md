# PRDetect V2 — A1 và A2

Pipeline hiện tại chỉ gồm:

- A1: căn chỉnh token SpaCy với Transformer subword bằng fast tokenizer, mean-pool subword và sliding window.
- A2: graph word/sentence/document, dependency hai chiều và cạnh tuần tự, RGCN hai tầng với residual + GraphNorm.

Không còn sequence branch, gated fusion, `--mode` hay gate metrics trong pipeline.
Tên file tài liệu được giữ để các liên kết cũ không bị hỏng.

## Mã nguồn

- [building_graph_v2.py](../building_graph_v2.py): xây graph A1+A2, không tạo sequence embedding.
- [relation_vocab_v2.py](../relation_vocab_v2.py): 128 relation ID cố định.
- [v2_data.py](../v2_data.py): đọc graph; bỏ qua trường sequence cũ nếu có.
- [PRDetectV2.py](../model/PRDetectV2.py): graph classifier, quality features tùy chọn.
- [train_v2.py](../train_v2.py): train/validation; mặc định tắt quality features.
- [test_v2.py](../test_v2.py): Accuracy, AUROC, F1, confusion matrix, số dự đoán Human.
- [export_a1_v1.py](../export_a1_v1.py): thay riêng embedding A1 vào cache graph V1.

## Kiến trúc

Word embedding -> RGCN -> GraphNorm + residual + GELU + dropout
-> RGCN -> GraphNorm + residual + GELU + dropout
-> concat(mean pool, max pool, document node) -> projection -> classifier -> logit.

Document node khởi tạo bằng mean sentence embedding; sentence node bằng mean word embedding.
Mặc định `use_quality=False`: quality input không đi vào model.
Để so sánh công bằng khi bật/tắt quality, classifier giữ nguyên kích thước;
vector quality được thay bằng zero khi tắt. Quality encoder vẫn được khởi tạo
để hai lần train cùng seed có cùng khởi tạo, nhưng không được thực thi khi tắt.

## Định dạng và tương thích

Graph schema vẫn là `2.0`; graph cache cũ dùng được vì không thay đổi word
embedding/topology. Các trường thừa cũ không được dùng.
Checkpoint mới có `format_version="2.1-graph"`, không tương thích checkpoint
model cũ. Không chuyển checkpoint cũ bằng cách xóa key hoặc đổi metadata;
phải train lại. Không xóa cache/checkpoint/log cũ.

Tên checkpoint mặc định:

- `model/hc3_prdetect_v2_a1a2_noquality_2026.pth`
- `model/hc3_prdetect_v2_a1a2_quality_2026.pth`

## Chạy cơ bản

```bash
python building_graph_v2.py --file hc3_train hc3_val hc3_test
python train_v2.py --seed 2026 --result-dir result/ablation_a1a2
python test_v2.py --file hc3_test --checkpoint model/hc3_prdetect_v2_a1a2_noquality_2026.pth --output result/ablation_a1a2/hc3_clean.json
```

Không cần build lại graph nếu đã có đầy đủ cache V2.
Xem [hướng dẫn kiểm chứng 1 và 2](./KiemChung_A1_A2.md) để chạy thí nghiệm đối chứng.

## Giới hạn

Encoder vẫn frozen; chưa có huấn luyện với structural noise hay consistency loss.
Điểm clean cao không đảm bảo chống insert/delete/repeat.
Lần sửa này không chạy huấn luyện hoặc đánh giá dataset; kết quả cần được đo trên Kaggle.
