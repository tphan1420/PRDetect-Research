# Kiểm chứng quality features và riêng A1 trên Kaggle

## 0. Phạm vi và điều kiện

Pipeline [V2 hiện tại](./Version2_A1_A3.md) chỉ còn A1+A2. Không dùng `--mode`.
Không xóa cache, checkpoint hoặc log trước đây. Checkpoint graph model mới phải train lại.
Graph V2 cũ vẫn dùng được; trường sequence thừa sẽ bị bỏ qua.

Hai phép kiểm chứng độc lập, không so chéo điểm tuyệt đối để quy kết nguyên nhân:

| Kiểm chứng | Đối chứng | Biến thể | Biến được thay |
|---|---|---|---|
| 1 | A1+A2 có quality | A1+A2 không quality | Quality input |
| 2 | RGCN V1 + embedding cũ | Cùng RGCN V1 + embedding A1 | Chỉ node embedding |

Chạy trong thư mục gốc repository. Dùng cùng seed, split, encoder, cache và
hyperparameter cho hai nhánh của từng phép kiểm chứng. Ví dụ dưới dùng seed 2026;
nên lặp lại cả hai nhánh với seed 2024, 2025, 2026 trước khi kết luận ổn định.
Không chọn checkpoint, threshold hay hyperparameter dựa trên các test sau đây.

### Cache cần có

- `graph_data_v2/`: train, validation và toàn bộ test dưới đây, do builder V2 tạo.
- `graph_data/`: cache V1 gốc tương ứng, đặc biệt cần cho kiểm chứng 2.

Các split tối thiểu:

```text
hc3_train
hc3_val
hc3_test
hc3_test_insert_25
hc3_test_delete_25
hc3_test_repeat_25
gpt3.5_mixed_test_split
gpt3.5_mixed_test_adj_30_synonym_replace
```

Nếu thiếu cache V2, chỉ build các split còn thiếu (không cần build lại toàn bộ):

```bash
python building_graph_v2.py --file hc3_train hc3_val hc3_test hc3_test_insert_25 hc3_test_delete_25 hc3_test_repeat_25 gpt3.5_mixed_test_split gpt3.5_mixed_test_adj_30_synonym_replace
```

Dùng đúng encoder và phiên bản SpaCy/model đã dùng cho cache V1. Tránh build
đè cache V1 bằng môi trường mới: kiểm chứng 2 cần reference cố định.

## 1. Có phải quality features làm model học shortcut?

Huấn luyện lại cả hai cấu hình, không so trực tiếp checkpoint graph-only cũ với model mới:

```bash
python train_v2.py --use-quality --seed 2026 --result-dir result/ablation_a1a2
python train_v2.py --seed 2026 --result-dir result/ablation_a1a2
```

Mọi cấu hình khác giữ nguyên. Khi tắt quality, model không đọc quality input và
không chạy quality encoder; classifier nhận zero vector ở phần đó. Giữ nguyên
chiều classifier và thứ tự khởi tạo tham số để tránh thêm biến gây nhiễu.

Đánh giá cả hai checkpoint trên cùng các split (Bash/Kaggle):

```bash
for variant in quality noquality; do
  for split in hc3_test hc3_test_insert_25 hc3_test_delete_25 hc3_test_repeat_25 gpt3.5_mixed_test_split gpt3.5_mixed_test_adj_30_synonym_replace; do
    python test_v2.py --file "$split" --checkpoint "model/hc3_prdetect_v2_a1a2_${variant}_2026.pth" --output "result/ablation_a1a2/${variant}_${split}_2026.json"
  done
done
```

Đọc kết quả:

- Nếu không quality tăng mạnh Accuracy/AUC trên structural noise: quality có đóng góp tiêu cực.
- Nếu cả hai vẫn gần 50%: quality không phải lời giải thích đầy đủ; xem kiểm chứng 2.
- Nếu Accuracy tăng nhưng AUC tương đương: khác biệt thiên về dịch chuyển score/ngưỡng.
- Không suy ra quan hệ nhân quả chỉ từ một seed hoặc từ việc model có feature độ dài.

JSON bổ sung `predicted_human`, `mean_probability_human`,
`confusion_matrix_ai_human`. Confusion matrix có hàng là nhãn thật, cột là nhãn dự đoán,
theo thứ tự `[AI=0, Human=1]`: `[[AI->AI, AI->Human], [Human->AI, Human->Human]]`.
Nếu `predicted_human` gần 1000 trên bộ test 1000 mẫu thì model vẫn sụp về Human.

## 2. Có phải representation A1 làm mất robustness của V1?

### 2.1. Xuất graph đối chứng: chỉ thay embedding

```bash
python export_a1_v1.py --file hc3_train hc3_val hc3_test hc3_test_insert_25 hc3_test_delete_25 hc3_test_repeat_25 gpt3.5_mixed_test_split gpt3.5_mixed_test_adj_30_synonym_replace
```

Script đọc V2 cache cùng tên và reference V1, ghi `graph_data/a1_<split>.pkl`.
Nó chỉ thay `all_token_embeddings` bằng các word embedding A1; giữ nguyên
`all_edge_index`, `all_edge_type` và `y` của reference V1. Không thêm sentence/document
node, cạnh ngược, cạnh tuần tự, GraphNorm, residual, quality features hoặc classifier mới.

Script dừng nếu lệch số mẫu, source index không đầy đủ, label/order khác, số word
node hoặc embedding dimension khác, hoặc dependency endpoints không khớp reference.
Các kiểm tra này không thay thế provenance của dữ liệu: phải dùng đúng các split gốc
và cùng tokenization/SpaCy model. Cache V1 cũ không lưu text nên không thể kiểm tra
đồng nhất văn bản chỉ bằng pickle. Không sửa/xóa các kiểm tra khi bị báo mismatch.
Đầu ra đã tồn tại sẽ không bị ghi đè trừ khi truyền `--overwrite`.

### 2.2. Huấn luyện bằng đúng script và classifier V1

```bash
python rgcn.py --dataset hc3_v1_control --train_file hc3_train --val_file hc3_val --seed 2026 --num_layers 2 --hidden_dim 256 --output_dim 64 --num_bases 30 --dropout 0.5 --batch_size 32 --lr 0.0001 --weight_decay 0.00001 --epochs 40 --patience 5 --no_test

python rgcn.py --dataset hc3_a1_only --train_file a1_hc3_train --val_file a1_hc3_val --seed 2026 --num_layers 2 --hidden_dim 256 --output_dim 64 --num_bases 30 --dropout 0.5 --batch_size 32 --lr 0.0001 --weight_decay 0.00001 --epochs 40 --patience 5 --no_test
```

Không sửa [RGCN2](../model/RGCN.py) hoặc [rgcn.py](../rgcn.py).
Hai lần dùng cùng loss/optimizer/checkpoint selection của V1. Validation phải dùng
embedding khớp với train; không train A1 rồi validation bằng cache embedding cũ.
Không lấy checkpoint V1 cũ làm đối chứng duy nhất vì training protocol/seed có thể khác.

### 2.3. Đánh giá đúng cache cho mỗi checkpoint

```bash
for split in hc3_test hc3_test_insert_25 hc3_test_delete_25 hc3_test_repeat_25 gpt3.5_mixed_test_split gpt3.5_mixed_test_adj_30_synonym_replace; do
  python test_rgcn.py --dataset hc3_v1_control --seed 2026 --file "$split" --save
  python test_rgcn.py --dataset hc3_a1_only --seed 2026 --file "a1_${split}" --save
done
```

[test_rgcn.py](../test_rgcn.py) ghi cả hai cấu hình vào `result/test_result.txt`,
phân biệt bằng model name `hc3_v1_control_rgcn` và `hc3_a1_only_rgcn`.
Giữ nguyên `dep_vocab.json` giữa export, train và test để V1 sử dụng cùng relation IDs.

Đọc kết quả:

- V1 ổn, A1-only sụp: A1 representation/preprocessing có đóng góp vào suy giảm, không phải do A2/A3.
- V1 và A1-only đều ổn, A1+A2 không quality sụp: cần kiểm chứng A2 và graph head/normalization riêng.
- Cả V1 và A1-only sụp: kiểm tra lại cache, dữ liệu, preprocessing versions và protocol trước khi quy lỗi A1.
- A1-only tốt trên GPT-3.5 nhưng kém structural noise: có trade-off, chưa phải nâng cấp robustness toàn diện.

Phép này đo toàn bộ preprocessing A1 (tokenization/alignment, special tokens/mask,
windowing và encoder eval), không chứng minh riêng BPE alignment là nguyên nhân.

## 3. Phạm vi xác minh local

Không chạy train/evaluation dataset trong lần sửa này. Môi trường local chưa có
Torch/PyG nên forward/backward phải được xác nhận trên Kaggle. Chưa tạo commit.
