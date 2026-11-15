# HƯỚNG DẪN IMPLEMENT PRISM-LITE TRÊN GCN GỐC CỦA PRDETECT

**Loại tài liệu:** Đặc tả kỹ thuật + kế hoạch triển khai + kiểm thử + thực nghiệm có kiểm soát  
**Trạng thái:** Thiết kế đề xuất; chưa phải mã đã được kiểm chứng trên repository  
**Mục đích:** Cung cấp yêu cầu đủ cụ thể để Codex hoặc một lập trình viên hiện thực hóa PRISM-Lite mà không vô tình thay đổi baseline PRDetect.

## 1. Mục tiêu nghiên cứu và phạm vi

PRISM-Lite kiểm tra hai giả thuyết:

- **H1 — Structural Fallback Edges:** Các cạnh liên kết token liên tiếp trong cùng câu có thể tạo đường truyền dự phòng khi dependency parse bị thay đổi do Insert/Repeat/Replace/Delete.
- **H2 — Learned Edge Reliability:** Cơ chế attention và gate học được có thể điều chỉnh đóng góp của dependency/fallback edges, hạn chế tác động của các messages kém hữu ích khi graph bị nhiễu.

Đây là giả thuyết, **không phải kết quả đã được chứng minh**. Ưu tiên giữ nguyên độ chính xác HC3 clean và synonym (Table 2), cải thiện Table 4, đồng thời đo Table 3 để phát hiện trade-off. PRISM-Lite không tự có cơ chế domain-generalization mạnh như LISA.

**Nguồn thiết kế:** Báo cáo `deep-research-report.md`, phần *Typed Structural Graph* và *Confidence-Weighted Relation-Aware GNN*. Bản Lite cố ý bỏ phần lớn mô-đun của PRISM-full.

## 2. Những phần giữ nguyên tuyệt đối

1. Dữ liệu và train/validation/test splits của GCN gốc.
2. Quy trình spaCy: model, tokenization, sentence/dependency parsing và cách chia chunk, nếu có.
3. **Frozen RoBERTa node embeddings và ánh xạ token/node đúng của baseline**. Không chuyển sang aligned BPE embeddings từ V2/LISA; không tự ý thay feature extractor.
4. Quy ước nhãn: `Human = 1`, `AI = 0`.
5. Graph/document pooling và classification head của baseline, trong phạm vi tương thích kích thước.
6. Định nghĩa metrics, threshold 0.5, và preprocessing/evaluation scripts.
7. GCN gốc (B0) và các code V1/V3/LISA: không chỉnh sửa hoặc ghi đè.

**Phải audit repository trước:** Chưa có mã nguồn GCN đầy đủ trong tài liệu này, nên các file và tham số thực tế phải được xác minh trong repo. Không giả định baseline chắc chắn là PyG, có bi-directional edges, có CLS pooling hoặc có graph embedding theo đúng một văn bản.

## 3. Kiến trúc PRISM-Lite

```text
Raw document
    │
    ▼
[Giữ nguyên preprocessing PRDetect]
spaCy parse ───────────── Frozen RoBERTa node embeddings
    │                                   │
    ▼                                   │
Dependency edges                          │
    │                                   │
    ├── (F1/P1) Add in-sentence sequential fallback edges
    │                                   │
    └──────────── edge_index + edge_type ┘
                          │
          ┌───────────────┴───────────────┐
          │                               │
        R0/F1                           E1/P1
     2-layer GCN                 2-layer sparse edge-attention
     with residual                 + edge gate + residual
          │                               │
          └───────────────┬───────────────┘
                          ▼
                 Original graph pooling
                          ▼
                Original BCE classifier
                          ▼
                    Human / AI
```

**Bốn thành phần của P1 (Full Lite):**
- GCN-compatible node features + dependency edges kế thừa PRDetect.
- Sequential fallback edges **chỉ bên trong mỗi câu spaCy**.
- Two-layer learned edge attention + reliability gate.
- Residual/skip connections và pooling/classifier tương thích baseline.

**Không thuộc Lite:** RGCN/typed dependency labels, POS/depth embedding mới, semantic stream, Scale Consistency, GroupDRO, CERE, CR-MUSE, contrastive, DANN, EMA/SWA, perturbation-aware training.

## 4. Hợp đồng dữ liệu graph

Mỗi graph (hoặc từng graph/chunk nếu baseline tổ chức như vậy) cần:

| Trường | Kiểu/shape | Quy tắc |
|---|---|---|
| `x` | `float [N, D]` | Features **y nguyên từ baseline** |
| `edge_index` | `long [2, E]` | Hàng 0: source `j`; hàng 1: destination `i` theo quy ước implement |
| `edge_type` | `long [E]` | `0 = dependency`, `1 = fallback`; không cần kiểu self-loop nếu dùng skip |
| `sentence_id` | `long [N]` hoặc metadata tương đương | Thu được bằng ánh xạ spaCy-node đáng tin cậy |
| `batch` | `long [N]` khi mini-batch | Không được nối cạnh giữa các graph của hai documents |
| `label` | scalar | `Human=1`, `AI=0` |

### 4.1. Luật sinh fallback edges

Cho mỗi câu spaCy với dãy **node indices tương ứng đúng thứ tự token** `[v0, v1, ..., vk]`, bổ sung hai cạnh có hướng `(v_i, v_{i+1})` và `(v_{i+1}, v_i)`.

- Không nối từ token cuối câu sang token đầu câu tiếp theo.
- Không nối qua boundaries của graph/chunk hoặc hai documents.
- Không thêm cạnh cho cặp đã có trong dependency edges; **dependency luôn ưu tiên**.
- Giữ directed-edge convention của baseline: nếu baseline dùng adjacency vô hướng/đối xứng, đối xứng dependency theo đúng cách baseline đang dùng. Không tự sửa baseline để đồng bộ với P1.
- Không thêm self-loop vào graph input cho lớp attention; self-information được xử lý bởi `W_skip`. Với biến thể R0/F1 dùng GCN gốc, tuân thủ hành vi self-loop vốn có của GCN.
- Deduplicate trên **cặp có hướng** `(src, dst)`, không chỉ trên cặp vô hướng; `edge_type` phải đồng bộ với `edge_index` sau sắp xếp/deduplicate.
- Đồ thị rỗng/câu một token: để layer skip xử lý; không tạo cạnh giả.

**Chú ý quan trọng về cache:** Nếu cache legacy không chứa sentence boundaries/node-token mapping, **không suy ra câu từ index**. Cần dựng metadata từ chính pipeline spaCy baseline, hoặc dựng lại graph/metadata mà không thay đổi phương pháp tạo `x`. Ghi version/hash preprocessing và hash config vào cache riêng `prism_lite`.

### 4.2. Pseudocode edge construction (minh họa, cần nối với repo)

```python
DEP, FALL = 0, 1

def add_sequential_fallback(dep_edges, sentence_node_lists):
    # dep_edges: iterable[(src, dst)] đã theo convention baseline
    # sentence_node_lists: list[list[node_id]] lấy từ metadata spaCy chuẩn
    result = {(int(u), int(v)): DEP for u, v in dep_edges if u != v}
    for nodes in sentence_node_lists:
        for u, v in zip(nodes, nodes[1:]):
            if u == v:
                continue
            result.setdefault((int(u), int(v)), FALL)
            result.setdefault((int(v), int(u)), FALL)
    # sort giúp tái lập kết quả giữa seed/run
    pairs = sorted(result)
    edge_index = torch.tensor(pairs, dtype=torch.long).T.contiguous()
    edge_type = torch.tensor([result[e] for e in pairs], dtype=torch.long)
    return edge_index, edge_type
```

Phải xử lý riêng trường hợp không có cạnh để `edge_index` luôn có shape `[2, 0]`. Đoạn code trên chỉ là pseudocode; `sentence_node_lists` **không được** sinh bằng cách giả định spaCy token index trùng RoBERTa subword index.

## 5. Hai encoder trong ablation

### 5.1. GCN + residual (R0/F1)

Giữ GCNConv/GCN implementation baseline, thêm skip/projection nếu kích thước thay đổi:

\[
H^{(l+1)}=\mathrm{LN}\bigl(W_{skip}H^{(l)}+\mathrm{GCNLayer}(H^{(l)},E)\bigr).
\]

Giữ activation/dropout theo baseline, xác định chính xác vị trí trước/sau LayerNorm rồi khóa xuyên suốt R0/F1. Với F1, khác biệt duy nhất so R0 là có thêm fallback edges.

### 5.2. Attention + learned gate + residual (E1/P1)

Với cạnh từ `j` sang `i`, đặt loại cạnh `t_ij ∈ {DEP, FALL}`:

\[
e_{ij}=Emb(t_{ij}),\quad
s_{ij}=MLP_{att}([h_i;h_j;e_{ij}]),\quad
c_{ij}=\sigma(MLP_{gate}([h_i;h_j;e_{ij}])).
\]

Chọn **thiết kế kỹ thuật mặc định của PRISM-Lite hướng dẫn này**:

\[
\alpha_{ij}=\operatorname{softmax}_{j\in\mathcal N(i)}(s_{ij}),\qquad
m_i=\sum_{j\in\mathcal N(i)}\alpha_{ij}\,c_{ij}\,W_vh_j.
\]

\[
h_i'=\mathrm{LN}(W_{skip}h_i+m_i).
\]

**Lý do tách gate khỏi softmax:** Bản PRISM-full đề xuất `softmax(s + log(c))`, nhưng khi một node chỉ có **một incoming edge**, trọng số của edge đó vẫn bằng 1 cho dù `c` gần 0. Đưa `c` vào message giúp giảm biên độ truyền từ cạnh đó. Đây là **điều chỉnh kỹ thuật có chủ đích so với công thức PRISM-full**, chưa được benchmark; không tuyên bố là cải tiến đã chứng minh. Nhược điểm cần theo dõi là `c` có thể co về gần 0 và triệt tiêu graph signals.

Không gọi `c_ij` là *parser confidence*: nó chỉ là hệ số hữu ích/tin cậy **học bởi classifier**, chưa được hiệu chuẩn với accuracy của spaCy parser.

### 5.3. PyG implementation sketch (không phải code chạy độc lập)

```python
# x: [N, D]; edge_index: [2, E]; edge_type: [E]
# projection dimensions, dropout, activation, norm tuân thủ config.
src, dst = edge_index
emb = self.edge_embedding(edge_type)          # [E, d_edge]
x_i, x_j = x[dst], x[src]
edge_features = torch.cat([x_i, x_j, emb], dim=-1)
scores = self.att_mlp(edge_features).squeeze(-1)  # [E]
gates = torch.sigmoid(self.gate_mlp(edge_features)).squeeze(-1)  # [E]
alpha = pyg_softmax(scores, dst, num_nodes=x.size(0))
messages = (alpha * gates).unsqueeze(-1) * self.value_proj(x_j)
aggregated = scatter(messages, dst, dim=0, dim_size=x.size(0), reduce='sum')
out = self.norm(self.skip_proj(x) + aggregated)
```

- Nếu E = 0, tạo `aggregated = zeros_like(skip_proj(x))` và vẫn forward thành công.
- Lặp **hai lớp**, dùng output lớp trước làm input lớp sau.
- Đảm bảo `torch_geometric.utils.softmax` nhóm theo **destination**; đừng dùng softmax toàn bộ tensor cạnh.
- Ở biến thể E1/P1, một attention head để giảm phức tạp; không cần relation matrices theo nhãn phụ thuộc.
- `edge_type` vẫn được embedding dù chỉ có dependency edges (E1).
- Theo dõi `mean(gate)` theo DEP/FALL; kiểm tra gradient của cả attention MLP và gate MLP.
- Nếu gate collapse gần 0, xem đây là *failure mode*; không âm thầm chỉnh ngưỡng theo target test.

## 6. Ablation cần hỗ trợ trong một training pipeline

| ID | Backbone | Residual | Fallback | Edge attention/gate | Mục đích |
|---|---|---:|---:|---:|---|
| `B0` | GCN gốc, không sửa | Theo gốc | Không | Không | Mốc chuẩn |
| `R0` | GCN | Có | Không | Không | Tách ảnh hưởng residual |
| `F1` | GCN | Có | Có | Không | Tác động fallback |
| `E1` | Edge-attention GNN | Có | Không | Có | Tác động attention/gate |
| `P1` | Edge-attention GNN | Có | Có | Có | PRISM-Lite đầy đủ |

`R0→F1` và `E1→P1` là so sánh tác động fallback; `R0→E1` và `F1→P1` kiểm tra thay encoder (attention/gate). Cần ghi chú rằng thay GCN bằng attention/gate thay cả phép message normalization, vì vậy không thể quy toàn bộ chênh lệch riêng cho gate nếu chưa có ablation `attention-only`.

**Ablation tùy chọn nếu P1 có tín hiệu tốt:** `P1_without_gate` (attention-only) để chứng minh gate bổ sung đóng góp so với attention thông thường. Không cần chạy ngay.

## 7. Cấu trúc file gợi ý (phải điều chỉnh theo repo)

```text
prism_lite/
  __init__.py
  graph_builder.py        # metadata + fallback edges + cache validation
  layers.py               # EdgeAttentionGateLayer
  model.py                # R0/F1/E1/P1 variants, original pooling
  train.py                # train script mới, không overwrite baseline
  infer.py                # checkpoint inference
  evaluate.py             # adapter sang eval Tables 2/3/4 sẵn có
  config.py               # train/eval settings, graph/preprocessing version
  tests/
    test_graph_edges.py
    test_layer.py
    test_train_smoke.py
  README.md
```

Ưu tiên tái sử dụng module sẵn có và import baseline **read-only** thay vì copy toàn bộ code. Không tạo nhiều file nếu repo đã có tổ chức tốt hơn. Không gọi `GCN2.py` là đúng đường dẫn đến khi đã audit (tên này từng xuất hiện trong tài liệu LISA nhưng cần xác nhận trong repo thực tế).

## 8. Quy trình triển khai từng bước

### Bước 0 — Repository audit

- Liệt kê chính xác các file baseline graph builder, model, train, inference, evaluation.
- Chạy smoke baseline; lưu hash của preprocessing/config, seed, dataset split.
- Kiểm tra node ↔ spaCy token alignment, chunk boundary, edge directionality, self-loops, graph pooling, sigmoid/BCELoss hay logit/BCEWithLogitsLoss.
- Không thay đổi baseline trong audit.

### Bước 1 — Graph builder

- Bổ sung `sentence_node_lists` và graph fields từ source pipeline hiện tại.
- Implement 2 edge modes: `dependency_only`, `dependency_plus_sequential`.
- Unit test trên graph nhỏ gồm ít nhất 2 câu, graph một node, graph có dependency trùng fallback và batch đa graph.
- In thống kê số node/cạnh trước–sau, độ tăng cạnh và % overlap.

### Bước 2 — Residual control

- Viết `R0` và `F1` sử dụng cùng 2-layer GCN, cùng residual, pooling/classifier.
- Kiểm tra `R0` có thể học, loss hữu hạn, evaluation đúng nhãn.
- Không kết luận chỉ từ một epoch.

### Bước 3 — Attention/gate

- Implement `EdgeAttentionGateLayer`, sau đó `E1` và `P1`.
- Chạy forward/backward toy graph và kiểm tra gate gradient không bằng 0 khi có cạnh.
- Verify attention sum ~1 cho destination có incoming edges; gate không bị nhầm vào attention sum.
- Đảm bảo inference và save/load reproducibility.

### Bước 4 — Train entrypoint

- Seed, splits, optimizer, scheduler, batch size, early stopping, checkpoint selection **bám sát GCN baseline**.
- Dùng cùng validation protocol cho mọi variant; nếu dùng `BCEWithLogitsLoss` cho R0/F1/E1/P1 trong khi baseline dùng `BCELoss(sigmoid)`, ghi rõ hai phép là tương đương về mục tiêu nhưng khác ổn định số học.
- Tối đa 25 epoch/patience 5 là cấu hình pilot nếu baseline không quy định; **để so sánh chính thức, khớp budget và checkpoint rule giữa các variant**.
- Lưu `model_state`, optimizer/scheduler state, seed, config, node feature protocol, sentence boundary protocol, edge version, metrics và preprocessing hash.

### Bước 5 — Evaluation có kiểm soát

- **Table 2:** HC3 train→HC3 test, original + synonyms 5/10/20/30%; GPT in-domain cần checkpoint train riêng trên GPT dataset.
- **Table 3:** HC3 train→GPT test và, khi có checkpoint, GPT train→HC3 test, original + synonyms 5/10/20/30%.
- **Table 4:** HC3 original + Insert/Repeat/Replace/Delete 25%.
- Báo cáo Accuracy, ROC-AUC, Macro-F1, Recall AI/Human, confusion matrix, sample IDs; cùng preprocessing và seed khi paired compare.
- Ghi kết quả nguồn HC3 validation trước khi xem target test. Không chọn mô hình hoặc threshold dựa trên nhãn GPT test hay Table 4 test.

### Bước 6 — Research decision

- Nếu F1 không tốt hơn R0 trên source validation/validation perturbations, chưa có lý do triển khai gate phức tạp chỉ dựa vào giả thuyết.
- Nếu P1 hơn E1 nhưng kém R0 về clean nhiều, đó là trade-off; không tuyên bố cải thiện toàn diện.
- Chỉ chạy 5 seed khi có bằng chứng pilot và protocol đã khóa.

## 9. Kiểm thử bắt buộc (acceptance checklist)

### Graph

- [ ] `x` của baseline và PRISM có cùng `N, D`, cùng giá trị và cùng thứ tự node ở cùng đầu vào.
- [ ] Giữ đầy đủ dependency edges sau xử lý; những cạnh không liên quan đến chiều có cùng convention baseline.
- [ ] Các cạnh fallback đi **hai chiều**, nối token kề nhau **cùng câu** và không vượt document/chunk.
- [ ] Không duplicate; dependency edges ưu tiên nếu overlap; không có self-loops ngoài convention.
- [ ] `edge_type.shape[0] == edge_index.shape[1]`; mọi index nằm trong `[0, N)`.
- [ ] Graph một node/không có cạnh hoạt động.
- [ ] Batched graph không có cạnh xuyên graph; metadata sentence không bị lẫn.

### GNN

- [ ] Forward output shape đúng số nodes, số documents và classifier labels.
- [ ] Attention sum bằng 1 theo từng node đích có incoming edges.
- [ ] Gate nằm trong `(0,1)`; gradients đến gate/attention/value/skip parameters.
- [ ] Gate có tác dụng giảm message ngay cả khi node có một incoming edge.
- [ ] Finite forward/backward: không NaN/Inf; graph có edge 0 không crash.
- [ ] Save/load checkpoint cho cùng prediction trong `eval()`.

### Research/evaluation

- [ ] B0 nguyên trạng; không làm thay đổi artifacts gốc.
- [ ] R0/F1/E1/P1 cùng splits, epochs budget, source-validation checkpoint logic.
- [ ] Tính đúng labels: `Human=1`, `AI=0`; recall AI xác định đúng class 0.
- [ ] Có bảng per-seed, không chỉ mean; không tune trên target test.
- [ ] Log train/val loss, HC3 source-val accuracy, Recall AI/Human, LR, gate stats, runtime/VRAM.

## 10. Rủi ro thường gặp và cách xử lý

| Rủi ro | Triệu chứng | Hướng kiểm tra |
|---|---|---|
| Fallback nối sai ranh giới câu | Table 2/4 giảm bất thường | In trace node IDs + sentence spans + edge pairs |
| Không khớp RoBERTa/spaCy index | Node/edge semantics sai | Kiểm tra mapping ở input thật, không chỉ shape |
| Duplicated self-loops | Thay normalization GCN | Audit `GCNConv(add_self_loops, normalize)` và graph input |
| Gate bị vô hiệu bởi singleton softmax | Gate đổi mà output không đổi | Dùng gate ở message như Mục 5.2 |
| Gate collapse gần 0 | Graph contribution biến mất | Theo dõi gate per-type và ablation no-gate |
| Fallback edges chi phối parsing edges | Clean accuracy giảm | Thống kê gate/messages theo edge_type |
| Cached graph sai version | Kết quả giữa seed/run không nhất quán | Cache key có parser, tokenizer, edge scheme, features |
| Nhầm nhãn AI/Human | Recall bất thường | Confusion matrix với label convention cụ thể |
| Overfit Table 4 sau nhiều thử nghiệm | Claim không còn độc lập | Lựa chọn bằng source validation; holdout mới khi cần |
| E1/P1 tăng RAM/latency | Kaggle OOM | Sparse `edge_index`, không dense N×N, giảm batch |

## 11. Gợi ý CLI và file bàn giao

Tên lệnh sau **chỉ là giao diện mong muốn**; Codex phải hiện thực hóa/điều chỉnh với repo, không được tuyên bố chạy được nếu chưa chạy:

```bash
python -m prism_lite.train --variant R0 --seed 2025 --dataset hc3
python -m prism_lite.train --variant F1 --seed 2025 --dataset hc3
python -m prism_lite.train --variant E1 --seed 2025 --dataset hc3
python -m prism_lite.train --variant P1 --seed 2025 --dataset hc3

python -m pytest prism_lite/tests -q
python -m prism_lite.evaluate --checkpoint <CHECKPOINT> --suite hc3_clean
```

Yêu cầu Codex bàn giao: danh sách file tạo/sửa; model flow; graph schema; lệnh train/infer/eval thực tế; log unit tests và smoke test; số tham số, runtime/VRAM; bảng kết quả pilot **nếu đã thực sự chạy**; README ghi rõ những module **chưa implement**.

## 12. Tiêu chí tiếp tục nghiên cứu

Không có ngưỡng Accuracy được bảo đảm. Mốc tham khảo theo paper PRDetect:

| Bảng | Mốc giấy gốc | Điều cần đo |
|---|---|---|
| Table 2 HC3 Original | 98.78% | Có duy trì ~98–99% không? |
| Table 3 HC3→GPT Original / Syn 30% | 73.2% / 70.9% | Có cross-domain gain mà không đổi clean quá nhiều? |
| Table 4 Insert / Repeat | 98.3% / 98.2% | Có bảo toàn robustness không? |
| Table 4 Replace / Delete | 79.8% / 74.7% | Edge modifications có giúp trên structural corruption không? |

Phân biệt số paper (nhiều seed hoặc cấu hình công bố) với paired GCN nhóm tự train. Pilot 1–2 seed là exploratory; kết quả cuối cần các seeds đã cố định và kiểm tra thống kê phù hợp. Nếu PRISM-Lite chỉ cải thiện Table 4 nhưng thua LISA ở Table 3, ghi nhận đây là hai cơ chế khác nhau thay vì gộp thành một kết luận thắng toàn diện.

## 13. Các điểm cần Codex xác minh trước khi code

1. GCN baseline trong repo là PyG hay manual adjacency-based implementation?
2. `x` thực sự tương ứng spaCy token, chunk token hay feature unit nào? Có sentence IDs không?
3. Dependency adjacency có hướng, hai hướng hay chuẩn hóa đặc biệt?
4. Original graph pooling/classifier nhận node outputs theo shape nào?
5. Tập structural perturbation được generate ở pipeline nào và có reparsing mỗi mẫu không?
6. Cache invalidation/versioning có đảm bảo khi thêm fallback edges không?
7. B0 train bằng BCELoss + sigmoid hay BCEWithLogitsLoss? Phải dùng cùng convention evaluation.

**Không tự suy đoán các thông tin này.** Nếu repo không cung cấp, Codex phải ghi rõ hạn chế và thực hiện kiểm tra đầu vào nhỏ trước khi train.

---

### Tài liệu tham khảo

- Xiang Li et al., *PRDetect: Perturbation-Robust LLM-generated Text Detection Based on Syntax Tree*, Findings of NAACL 2025. https://aclanthology.org/2025.findings-naacl.464/
- File nghiên cứu được cung cấp: `deep-research-report.md`, các mục về typed structural graph và learned edge reliability.
- PyTorch Geometric `GCNConv`: https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.nn.conv.GCNConv.html
- PyTorch Geometric sparse group softmax: https://pytorch-geometric.readthedocs.io/en/stable/modules/utils.html

> **Lời dặn cuối cho Codex:** Audit repo → triển khai graph metadata và fallback → tests → R0/F1 → attention E1/P1 → smoke test → hướng dẫn lệnh chạy. Không tự chạy full benchmark; không thay đổi GCN gốc, V1, V3, LISA; không tune trên GPT-3.5 test.
