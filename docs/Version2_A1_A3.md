# PRDetect Version 2 - triển khai A1 đến A3

## 1. Phạm vi

Nhánh `version2` triển khai ba cải tiến A1–A3, tương ứng với E1–E3 trong [DeXuatCaiTienCrossDataset.md](./DeXuatCaiTienCrossDataset.md):

- **A1:** căn chỉnh token SpaCy với subword của Transformer.
- **A2:** đồ thị phân cấp hai chiều ở cấp tài liệu.
- **A3:** mô hình hybrid kết hợp nhánh sequence và nhánh graph bằng gated fusion.

Pipeline V1 được giữ nguyên. V2 dùng file mã nguồn, thư mục graph và checkpoint riêng để có thể chạy ablation trực tiếp với GCN/RGCN cũ.

Không có huấn luyện hoặc kiểm thử thực nghiệm nào được chạy trong lần triển khai này. Mã đã được kiểm tra tĩnh bằng diagnostics của editor.

## 2. Các file mới

| File | Vai trò |
|---|---|
| [`relation_vocab_v2.py`](../relation_vocab_v2.py) | Relation schema V2 bất biến |
| [`building_graph_v2.py`](../building_graph_v2.py) | Tạo embedding đã căn chỉnh và graph phân cấp |
| [`v2_data.py`](../v2_data.py) | Nạp, kiểm tra schema và chuyển graph sang PyG `Data` |
| [`model/PRDetectV2.py`](../model/PRDetectV2.py) | RGCN hai tầng + sequence branch + gated fusion |
| [`train_v2.py`](../train_v2.py) | Huấn luyện và chọn checkpoint theo validation AUROC |
| [`test_v2.py`](../test_v2.py) | Đánh giá checkpoint V2 trên Kaggle |

Graph V2 được ghi vào `graph_data_v2/`, còn checkpoint mặc định được ghi vào `model/<dataset>_prdetect_v2_<mode>_<seed>.pth`.

## 3. A1 - căn chỉnh SpaCy và Transformer

### 3.1. Vấn đề của V1

V1 lấy danh sách token SpaCy rồi gọi trực tiếp:

```python
tokenizer.convert_tokens_to_ids(spacy_tokens)
```

RoBERTa sử dụng byte-level BPE nên một token SpaCy có thể tương ứng nhiều subword. Token cũng mang thông tin khoảng trắng như ký hiệu `Ġ`. Vì vậy việc tra ID trực tiếp không đảm bảo node embedding thực sự đại diện cho từ tương ứng.

### 3.2. Cách V2 xử lý

[`FrozenTextEncoder.encode_words`](../building_graph_v2.py) dùng fast tokenizer với:

```python
is_split_into_words=True
return_overflowing_tokens=True
return_special_tokens_mask=True
return_attention_mask=True
```

Mỗi cửa sổ dùng `word_ids()` để ánh xạ subword về chỉ số token SpaCy. Các hidden state thuộc cùng một token được mean-pool:

$$
h^{word}_i=\frac{1}{|S_i|}\sum_{j\in S_i}h^{subword}_j.
$$

Các cải tiến liên quan:

- dùng `<s>`/`</s>` đúng theo tokenizer;
- truyền `attention_mask` vào encoder;
- hỗ trợ văn bản dài bằng sliding window mặc định 512 subword, overlap 64;
- hợp nhất embedding ở vùng overlap bằng trung bình;
- đo `unk_subword_ratio` và `missing_word_ratio` để gating network biết độ tin cậy của representation.

Encoder mặc định vẫn là `roberta-base` để so sánh công bằng với V1. Có thể truyền `--encoder microsoft/deberta-v3-base` khi build lại toàn bộ train/validation/test graph. Không được trộn graph tạo bởi hai encoder khác nhau; loader và checkpoint có kiểm tra metadata này.

### 3.3. Sequence embedding

Ngoài word embedding, mỗi cửa sổ tạo một vector bằng mean pooling trên các content subword, loại padding và special token. Vector tài liệu là trung bình của các cửa sổ:

$$
z_{seq}=\frac{1}{K}\sum_{k=1}^{K}\operatorname{MeanPool}(H_k).
$$

Encoder được freeze trong giai đoạn build graph. Vì vậy huấn luyện V2 không cần giữ Transformer trong VRAM và phù hợp hơn với Kaggle.

## 4. A2 - đồ thị phân cấp hai chiều

### 4.1. Relation schema cố định

[`relation_vocab_v2.py`](../relation_vocab_v2.py) định nghĩa 128 relation ID cố định:

- 60 dependency relation thuận;
- 60 dependency relation nghịch;
- 8 relation cấu trúc cấp tài liệu.

Các relation cấu trúc là:

```text
next_token, prev_token,
word_to_sentence, sentence_to_word,
next_sentence, prev_sentence,
sentence_to_document, document_to_sentence
```

V2 không thêm relation động khi xử lý validation/test. Thứ tự label được lưu trong từng graph pickle và checkpoint; [`load_payload`](../v2_data.py) từ chối graph có schema khác.

### 4.2. Các loại node

Mỗi tài liệu gồm:

- **word node:** embedding đã căn chỉnh ở A1;
- **sentence node:** mean embedding của các word node trong câu;
- **document node:** mean embedding của các sentence node; sequence embedding được giữ riêng cho nhánh tuần tự để hai nhánh không dùng trùng document vector đầu vào.

Nếu tài liệu có $N$ từ và $M$ câu thì graph có $N+M+1$ node.

### 4.3. Cạnh dependency hai chiều

V1 chỉ có cạnh `dependent -> head`. V2 giữ cạnh này và thêm cạnh nghịch `head -> dependent` với relation ID riêng. Vì PyG truyền message từ `edge_index[0]` sang `edge_index[1]`, thay đổi này cho phép cả head và dependent nhận ngữ cảnh cú pháp.

### 4.4. Kết nối cấp tài liệu

Các word node nối hai chiều với sentence node. Mọi sentence node nối hai chiều với document node. Các token liền kề trong một câu và các câu liền kề cũng được nối hai chiều.

Nhờ đó, các cây dependency vốn rời nhau trở thành một graph liên thông. Với hai tầng RGCN:

- tầng 1 truyền thông tin word → sentence và sentence → document;
- tầng 2 cho phép document context quay lại sentence/word đồng thời tạo document representation toàn cục.

Điều này giải quyết ngữ cảnh tài liệu mà không cần tăng RGCN lên 4–6 tầng.

### 4.5. Quality features

Mỗi graph lưu năm đặc trưng:

1. `log(1 + số từ)`;
2. `log(1 + số câu)`;
3. tỷ lệ unknown subword;
4. tỷ lệ word không nhận được subword embedding;
5. tỷ lệ dependency label ngoài schema.

Các đặc trưng này không trực tiếp quyết định nhãn; chúng cung cấp tín hiệu độ tin cậy cho gated fusion.

## 5. A3 - hybrid sequence và graph

Kiến trúc trong [`PRDetectV2`](../model/PRDetectV2.py) gồm hai nhánh dùng chung representation Transformer đã tính ở A1.

### 5.1. Graph branch

Graph branch dùng hai `RGCNConv`:

```text
input_dim → hidden_dim → graph_dim
```

Mỗi tầng có:

- basis decomposition;
- root weight/self transformation;
- residual projection;
- GraphNorm theo từng tài liệu;
- GELU và dropout.

Sau tầng cuối, graph representation ghép ba phép pooling:

$$
z_g=\operatorname{MLP}([
\operatorname{MeanPool}(H),
\operatorname{MaxPool}(H),
h_{document}
]).
$$

Document node được kiểm tra nghiêm ngặt: mỗi graph phải có đúng một node loại `document`.

### 5.2. Sequence branch

Sequence embedding được chiếu về cùng `fusion_dim` với graph vector:

$$
z_s=\operatorname{MLP}(z_{seq}).
$$

Nhánh này giữ thông tin tuần tự/ngữ nghĩa mà dependency graph không thể biểu diễn đầy đủ, đặc biệt khi văn bản ngắn hoặc parser không ổn định.

### 5.3. Gated fusion

Quality features được mã hóa thành $z_q$. Gating network nhận cả hai representation và quality vector:

$$
[g_g,g_s]=\operatorname{softmax}
(\operatorname{MLP}([z_g,z_s,z_q])).
$$

Vector hợp nhất:

$$
z=g_gz_g+g_sz_s.
$$

Classifier nhận $[z,z_q]$ và trả về một logit Human/AI. Training dùng `BCEWithLogitsLoss`, ổn định số học hơn `sigmoid + BCELoss` của V1.

[`test_v2.py`](../test_v2.py) báo thêm `mean_graph_gate` và `mean_sequence_gate` để quan sát nhánh nào được sử dụng nhiều hơn trên từng benchmark.

## 6. Định dạng graph và checkpoint

### 6.1. Graph pickle

Payload V2 lưu:

```text
schema_version
encoder_name
encoder_hidden_size
num_relations
relation_labels
node_type_labels
graphs
y
source_indices
failures
source
```

Mỗi phần tử trong `graphs` gồm:

```text
x
edge_index
edge_type
node_type
sequence_embedding
quality_features
```

[`validate_graph`](../building_graph_v2.py) kiểm tra shape, phạm vi node/relation và số document node trước khi ghi pickle.

### 6.2. Checkpoint

Checkpoint không chỉ chứa `state_dict` mà còn chứa:

```text
format_version
model_state
model_config
training_config
graph_metadata
epoch
validation_metrics
```

Nhờ đó [`test_v2.py`](../test_v2.py) khởi tạo đúng kiến trúc và từ chối graph dùng encoder/schema khác.

## 7. Cách chạy trên Kaggle

### 7.1. Build HC3 graph

```bash
python building_graph_v2.py \
  --file hc3_train hc3_val hc3_test \
  --encoder roberta-base
```

Nếu muốn dùng DeBERTa:

```bash
python building_graph_v2.py \
  --file hc3_train hc3_val hc3_test \
  --encoder microsoft/deberta-v3-base
```

Phải build lại tất cả graph cần dùng bằng cùng `--encoder`, `--max-length` và `--stride`.

### 7.2. Huấn luyện

```bash
python train_v2.py \
  --dataset hc3 \
  --train-file hc3_train \
  --val-file hc3_val \
  --mode hybrid \
  --seed 2026 \
  --epochs 40 \
  --batch-size 16
```

Checkpoint mặc định:

```text
model/hc3_prdetect_v2_hybrid_2026.pth
```

Checkpoint được chọn theo validation AUROC. Nếu validation chỉ chứa một lớp, script dùng Accuracy làm fallback.

### 7.3. Build và đánh giá cross-dataset

```bash
python building_graph_v2.py \
  --file data/original_text/benchmarks/gpt35_mixed/gpt3.5_mixed_test_split.json \
  --encoder roberta-base

python test_v2.py \
  --file gpt3.5_mixed_test_split \
  --checkpoint model/hc3_prdetect_v2_hybrid_2026.pth \
  --output result/v2_gpt35_clean.json
```

Ví dụ synonym 30%:

```bash
python building_graph_v2.py \
  --file data/perturbed_text/gpt35_mixed/lexical_synonym/gpt3.5_mixed_test_adj_30_synonym_replace.json \
  --encoder roberta-base

python test_v2.py \
  --file gpt3.5_mixed_test_adj_30_synonym_replace \
  --checkpoint model/hc3_prdetect_v2_hybrid_2026.pth \
  --output result/v2_gpt35_synonym_30.json
```

### 7.4. Chạy ablation

Ba chế độ dùng cùng graph, quality features và training protocol:

```bash
python train_v2.py --mode sequence_only --result-dir result/ablation
python train_v2.py --mode graph_only --result-dir result/ablation
python train_v2.py --mode hybrid --result-dir result/ablation
```

Checkpoint và training summary có tên chứa mode nên không ghi đè nhau. Dùng
`test_v2.py` với checkpoint tương ứng; mode được đọc trực tiếp từ checkpoint.

## 8. Các đảm bảo tương thích

- Không thay đổi [`building_graph.py`](../building_graph.py), [`rgcn.py`](../rgcn.py), [`test_rgcn.py`](../test_rgcn.py) hoặc checkpoint V1.
- Graph V1 tiếp tục nằm trong `graph_data/`; graph V2 nằm trong `graph_data_v2/`.
- V2 dùng cùng quy ước label của dự án: `Human=1`, `AI=0`.
- Dataset có tên stem trùng nhau sẽ không được chọn ngầm; [`resolve_source`](../building_graph_v2.py) yêu cầu đường dẫn đầy đủ để tránh build nhầm split.
- Graph lỗi được ghi trong trường `failures`; `source_indices` giữ ánh xạ về bản ghi gốc.

## 9. Giới hạn của phiên bản này

- Transformer encoder được freeze và chạy ở bước preprocessing; chưa fine-tune end-to-end.
- Chưa triển khai A4 style/CharCNN, supervised contrastive learning hoặc consistency loss.
- Chưa calibration threshold; `test_v2.py` mặc định dùng 0,5 nhưng hỗ trợ `--threshold`.
- Graph pickle lớn hơn V1 vì chứa word, sentence, document và sequence embedding.
- Kết quả thực tế vẫn cần được đánh giá nhiều seed trên Kaggle; file [`datasets/cross_test.txt`](../datasets/cross_test.txt) chỉ là baseline GCN, không phải kết quả V2.
