# HƯỚNG DẪN TRIỂN KHAI PEARL-GCN CHO PRDETECT

**PEARL — Perturbation-Aligned Edge Reliability Learning**  
**Đích triển khai:** GCN gốc của PRDetect (không phải V1/V3-RGCN hoặc LISA)  
**Trạng thái:** Tài liệu thiết kế/triển khai và kiểm chứng; **chưa có kết quả PEARL thực nghiệm**  
**Tài liệu nguồn:** `PEARL_CoSoLyLuan_ThietKe_Va_KeHoachThucNghiem(1).md` (đề xuất nghiên cứu ngày 14/11/2026)

> **Lưu ý quan trọng:** Đường dẫn file, tên script và lệnh CLI ở đây là **giao diện đích được đề xuất**, không phải xác nhận rằng repository của bạn đã có các file đó. Agent triển khai **phải audit repository thật trước**, điều chỉnh API tương thích, không tự giả định `GCNConv`, graph data format hoặc đường dẫn datasets.

## 0. Mục tiêu và giới hạn

### 0.1. Mục tiêu khoa học

Kiểm tra giả thuyết: **nhãn về khả năng *giữ nguyên quan hệ cú pháp* trước/sau một phép edit, tạo từ token provenance trên dữ liệu nguồn train, có giúp GCN đưa ra dự đoán bền vững hơn hay không?**

Mô hình phải học một dự đoán **single-view** `q(e,G)` và dùng nó để cân lại cạnh của dependency graph. Tại inference **không cần văn bản gốc, edit log, nhãn retention, nhãn lớp hay dữ liệu target**.

Các mục tiêu benchmark, chưa được bảo đảm:

- **Table 2:** duy trì HC3 ~98–99% và các mức synonym; đánh giá GPT-3.5 in-domain bằng checkpoint train GPT-3.5 riêng.
- **Table 3:** kiểm tra khả năng HC3→GPT-3.5-Mixed, cả Original và synonym 5/10/20/30%; báo cả chiều ngược khi có checkpoint.
- **Table 4:** đặc biệt quan tâm Replace/Delete 25%, nhưng phải giữ Insert/Repeat và HC3 Original.

**Không chọn mô hình theo GPT-3.5-Mixed target test hoặc Table 4 test.** Chúng là kết quả đánh giá sau khi đã khóa hyperparameters từ source validation.

### 0.2. Những gì cố ý KHÔNG làm trong bản đầu

- Không đổi frozen RoBERTa node embeddings, tokenizer, spaCy parser, hai tầng GCN, pooling, classifier hoặc nhãn `Human=1, AI=0` của baseline.
- Không thay thế GCN bằng RGCN/GAT; không thêm fallback edges, structural features, LISA windows, attention, gated fusion, EMA, SWA, GroupDRO, CMixStyle, R-Drop, domain-adversarial learning.
- Không thay đổi thuật toán sinh bản nhiễu dùng ở test hay sử dụng nhãn target để chọn thiết kế.
- Không kết hợp các phép regularization mới trước khi **PEARL-Lite** có lợi ích riêng.

### 0.3. Định nghĩa lõi

Từ graph gốc `G`, edit theo phân phối `Q` thành graph `G'`. Với cạnh gốc `e=(u,v,r)` và ánh xạ token đáng tin `m`, nhãn:

\[
 s_e = \mathbf{1}\{(m(u),m(v),r)\in E'\},
\]

**chỉ khi** hai đầu mút đều tồn tại và căn chỉnh được duy nhất. Nếu không: `unknown` / `valid_mask=False`; không đặt bằng 0.

Bộ dự đoán: `q_phi(e,G) = sigmoid(MLP(edge_features(e,G)))`. Gán trọng số truyền tin:

\[
 w_e = \varepsilon + (1-\varepsilon)q_\phi(e,G),\quad 0 < \varepsilon \le 1.
\]

Đưa `w_e` vào toán tử GCN theo **cùng chiều cạnh, cách đối xứng hóa, tự vòng, normalization** như baseline. Mặc định thăm dò `epsilon=0.8`, không xem là giá trị tối ưu. `epsilon=1` **phải tái lập chính xác GCN gốc tại cùng checkpoint** (trong sai số số học).

Loss đầu tiên:

\[
 L_{PEARL}=L_{cls}+\lambda_eL_{edge},
\]

với `L_cls` trên **clean** và `L_edge` chỉ trên các cặp/cạnh train có nhãn hợp lệ. Nếu thêm BCE trên augmented text trong biến thể sau, tất cả baseline đối chứng phải có **classification exposure** giống hệt.

---

## 1. Kiến trúc và phân chia train/inference

```text
TRAIN SOURCE ONLY

  x (clean) ----------------> spaCy parse G ----> frozen RoBERTa features X
      |                               |
      v                               |
  edit with provenance log            |
      |                               |
      v                               |
  x' (edited) -----> spaCy parse G'    |
      |                |              |
      +----- token/span alignment ----+
                       |
                       v
              edge retention labels (1/0/unknown)
                       |
        current-graph edge features phi(e,G)
                       |
                       v
               q_phi(e,G) predictor
                       |
           weighted adjacency (same baseline GCN)
                       |
                original 2-layer GCN
                       |
              original readout/classifier
                       |
                 BCE_class + lambda*BCE_edge

INFERENCE SINGLE VIEW

  text -> spaCy -> graph + RoBERTa X -> edge features -> q_phi
       -> weighted adjacency -> 2-layer GCN -> original head
       -> sigmoid -> Human/AI (threshold 0.5)
```

**Không nhầm lẫn hai khái niệm:** `q_phi` ước lượng *xác suất cạnh hiện tại được giữ nguyên dưới phân phối edit Q*, **không phải** xác suất parser đoán đúng hoặc xác suất rằng cạnh đã bị thay đổi. Cạnh mới tạo ra bởi một bản Replace/Delete tại test có thể vẫn nhận `q` cao.

---

## 2. Giai đoạn P0 — Audit và khóa baseline (bắt buộc)

### 2.1. Kiểm tra repository thật

Agent phải đọc code thật và ghi nhận trong `docs/pearl_baseline_audit.md`:

1. Model GCN gốc: tên lớp, đầu vào `X/A/edge_index`, số layer, activation, dropout, readout, head; đang trả **probability** hay **logit**? (Một bản GCN2 cũ trả sigmoid và train bằng `BCELoss`.)
2. spaCy tokenizer/parse, số node mỗi graph, phân đoạn câu/chunk, embedding RoBERTa và quy tắc word–subword legacy.
3. Cách mã hóa cạnh: directed/undirected, đảo chiều head→child hay child→head, duplicate edges, self-loops, adjacency normalization, caching normalization.
4. Format cache `.pkl`, dữ liệu raw, index node, train/val/test split, scripts Table 2/3/4, label convention, seed, optimizer, scheduler, max epochs, batch size, checkpoint rule.
5. Code đã thực hiện GraphConv bằng `GCNConv` hay dense/sparse matrix multiplication? Phải chọn **một** giải pháp truyền `edge_weight` phù hợp API thật; không rewrite cả graph builder nếu không cần.
6. Repo có metadata cho `token.idx`, `token.head.i`, `token.dep_`, `token.pos_`, `token.sent` hay không? Nếu thiếu, xuất **sidecar metadata** gắn với graph đã sinh từ raw text. Không suy ra `dep`/POS từ adjacency nhị phân.

### 2.2. Khóa các bất biến

- Baseline được đối chiếu bằng code, seed và checkpoint version; **không ghi đè** model gốc.
- Không đổi graph topology hoặc RoBERTa embedding của B0 để tiện triển khai PEARL.
- Lưu hash split, sha256 input dataset, phiên bản spaCy, RoBERTa, graph builder, code commit.
- Nếu cần sửa bug graph builder ở bản legacy: **tái train cả baseline và PEARL** sau khi sửa; không dùng checkpoint cũ với mapping mới.

**Điều kiện vượt P0:** Có thể chạy một lượt inference GCN baseline từ cùng cache; đã xác định rõ normalization/edge orientation.

---

## 3. Giai đoạn P1 — Tạo paired edit data và nhãn retention

### 3.1. Phân tách dữ liệu trước augmentation

- Tạo/khóa train/validation/test theo **document ID** trước khi sinh bất kỳ edit nào.
- Chỉ sinh cặp edit từ **train** để học `q` và tùy chọn từ **source validation** để kiểm tra `q` mà không cập nhật trọng số.
- Seed riêng cho edit generator; không tạo bản sao của cùng document ở các split khác nhau.
- Ưu tiên một cặp `clean + một edit` cho mỗi document/iteration, tránh tăng đột biến cache.

### 3.2. Các edit dùng cho pilot

Bắt đầu với `synonym` có provenance rõ và `replace` mức nhẹ. Sau khi căn chỉnh đã đúng, mở rộng sang `insert`, `repeat`, `delete`. Không cần chạy tất cả 25% corruption ngay từ pilot. Tỷ lệ/loại edit là định nghĩa của `Q`, phải được lưu chính xác và tách khỏi benchmark generator.

- **Synonym:** giữ ID nguồn của token thay bằng từ tương ứng **chỉ nếu** tokenization vẫn căn chỉnh duy nhất.
- **Insert:** token mới nhận ID mới; ID token cũ không thay đổi.
- **Repeat:** bản sao token nhận ID mới; không đoán matching bằng vị trí hoặc ký tự thuần túy.
- **Replace:** nếu một token thành nhiều spaCy tokens hoặc ngược lại, đánh dấu `unknown` khi không có ánh xạ 1–1 chắc chắn.
- **Delete:** token bị xóa không còn mapping; các cạnh liên quan được loại khỏi edge BCE, không gán nhãn changed bằng suy đoán.

### 3.3. Provenance schema đề xuất

Một record `pair_meta` chứa tối thiểu:

```json
{
  "doc_id": "source_train_doc_id",
  "split": "train",
  "edit_seed": 2025,
  "edit_family": "synonym",
  "edit_ratio": 0.05,
  "generator_version": "edit-v1",
  "original_text_hash": "sha256...",
  "edited_text_hash": "sha256...",
  "token_alignment": [
    {"src_node": 0, "dst_node": 0, "status": "unique"},
    {"src_node": 1, "dst_node": null, "status": "unknown"}
  ],
  "edge_labels": [
    {"src_edge_id": 0, "stable": 1, "valid": true},
    {"src_edge_id": 1, "stable": null, "valid": false}
  ]
}
```

Các hash/ID ở trên là **ví dụ định dạng**, không phải dữ liệu có thật. Thực tế nên lưu dạng nén hoặc JSONL/Parquet và dùng ID/index trỏ vào clean graph cache, tránh sao chép embedding RoBERTa lớn trong mỗi pair.

### 3.4. Quy tắc alignment chính xác

1. Khi áp dụng edit, lưu `token_uid` của token gốc, token mới, span ký tự **trước/sau**, và loại thao tác.
2. Chạy **cùng pipeline spaCy** trên hai text; chuyển các UID/span về spaCy token indices.
3. Chỉ xác nhận mapping `src_node -> dst_node` nếu duy nhất và nhất quán; khi spaCy split/merge, resegment hoặc có nhiều mapping có thể đúng thì `unknown`.
4. Không giả định node indices bảo toàn qua insert/delete. Không dùng `SequenceMatcher` hay `editdistance` thuần ký tự làm ground truth khi có repeated tokens.
5. Với cạnh có hai đầu mút `u`,`v` được map, kiểm tra `(m(u),m(v),dep_label)` trong graph mới. Muốn giám sát cả topology-only cần **tách task** khỏi full relation retention thay vì trộn nhãn.
6. Không coi việc `r` đổi là lỗi parser; `changed` chỉ có nghĩa **cặp head/dependent/relation không được giữ nguyên**.

### 3.5. Kiểm tra P1 trước khi train

Báo cáo theo edit family và theo nhãn Human/AI:

- Số document/cặp thành công/thất bại.
- Tỷ lệ `unique` vs `unknown` cho token và cạnh; tỷ lệ `stable`/`changed` chỉ trong cạnh hợp lệ.
- Thống kê theo dependency label, POS và độ dài.
- Kiểm tra thủ công **ít nhất 30–50 cặp đa dạng**: synonym, insert, repeat, replace, delete; ưu tiên lỗi split/merge, từ lặp, dấu câu.
- Kiểm tra automated trên cặp **identity edit**: mọi cạnh hợp lệ phải `stable=1`.
- Nếu tỷ lệ `unknown` quá cao hoặc negative labels thiếu nghiêm trọng, dừng và báo rõ; không tự gán unknown=0 để đẹp metric.

**Điều kiện vượt P1:** Sinh được nhãn cạnh đúng, audit thủ công đạt chất lượng chấp nhận được, có đủ cả positive/negative cho source validation.

---

## 4. Giai đoạn P2 — Train edge-retention predictor *trước GCN*

### 4.1. Feature input chỉ dùng thông tin có tại inference

Một schema đầu tiên, nhẹ:

```text
edge_features(e,G) = concat(
    Emb(POS_source),
    Emb(POS_target),
    Emb(dependency_relation),
    Bucket(|token_index_source-token_index_target|),
    LocalDegree(source), LocalDegree(target),
    (optional) edge_direction
)
```

**Không đưa vào predictor:** edit type, edit ratio, token UID, nguyên văn bản gốc nếu đang dự đoán trên graph edited, nhãn stable/changed, document label hoặc source dataset ID. Nếu dùng length, phải xem đó là một shortcut risk và có ablation.

Nếu graph là vô hướng nhưng metadata có directed dependency, dự đoán `q` **một lần theo arc gốc**, sau đó copy weight cho hai cạnh đối xứng; không tạo hai prediction khác nhau cho cùng một arc chỉ vì chiều biểu diễn.

### 4.2. Predictor đề xuất

```python
class EdgeRetentionPredictor(nn.Module):
    # Pseudocode: sizes/embeddings depend on audited graph schema.
    def __init__(self, feature_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, edge_features):
        return self.mlp(edge_features).squeeze(-1)  # RAW logits
```

Train với `BCEWithLogitsLoss` có valid mask; `q=sigmoid(logits)` chỉ khi cần xác suất. Nếu dùng class reweighting, ghi rõ tần suất và chính sách weights; **calibration phải đánh giá trên xác suất chưa được chọn threshold theo test**.

**Không để một document dài có hàng nghìn edges áp đảo loss chỉ do số lượng cạnh.** Ưu tiên loss mean trong từng document rồi mean theo document (hoặc kiểm soát sampling/weighting tương đương và ghi rõ).

### 4.3. Baseline cho edge predictor

- `constant_prior`: xác suất stable chung của source train.
- `relation_prior`: prior theo dependency label; thêm phiên bản `relation+POS` với Laplace smoothing để giảm overfit nhóm hiếm.
- `PEARL_MLP`: cùng train/validation pairs.
- `shuffled_labels`: negative control, đảo nhãn trong source train theo cách khóa seed và giữ nhãn validation nguyên.

Metrics: edge AUROC, average precision/AUPRC (kèm prevalence), Brier, ECE, log loss, F1 (threshold được chọn trên **source validation**, nếu báo), theo relation và edit family. Bootstrap theo document, **không coi các edges của cùng văn bản là độc lập**.

**Điều kiện vượt P2:** MLP vượt các prior đơn giản một cách có ý nghĩa thực tế trên source validation, có phân biệt được stable/changed, không suy biến `q≈1` cho mọi cạnh. Nếu không, **dừng ý tưởng PEARL có giám sát** trước khi train nhiều checkpoint GCN.

---

## 5. Giai đoạn P3 — Tích hợp trọng số cạnh vào GCN gốc

### 5.1. Nguyên tắc bảo toàn baseline

- Giữ nguyên node `X`, số layer, head, pooling và cách đọc output.
- Không sửa file GCN baseline đang dùng để báo cáo; tạo lớp wrapper/layer mới và training entrypoint riêng.
- Không đổi thứ tự nodes hoặc loại/cơ chế adjacency trong lúc thêm q.
- Chỉ thêm `edge_weight` cho **các cạnh đã tồn tại**. PEARL bản đầu **không tạo thêm fallback edge**.

### 5.2. Công thức

Với những dependency arcs có metadata hợp lệ:

\[
q_e=\sigma(a_e),\qquad w_e=\varepsilon+(1-\varepsilon)q_e.
\]

Nếu baseline đối xứng hóa edges, dùng cùng `w_e` cho hai hướng (không tính nhầm cạnh trùng); self-loop có weight 1 như baseline.

\[
\tilde A=A_w+I,\quad \tilde D_{ii}=\sum_j\tilde A_{ij},\quad S_\phi=\tilde D^{-1/2}\tilde A\tilde D^{-1/2}.
\]

**Không** chỉ `edge_weight * message` sau khi GCN đã chuẩn hóa theo adjacency cũ rồi gọi là cùng một toán tử: chuẩn hóa có trọng số phải khớp định nghĩa đã chọn. Nếu baseline vốn có normalization khác, implement theo baseline trước và tài liệu hóa khác biệt; không được im lặng thay đổi cả normalization.

### 5.3. Nếu baseline dùng PyTorch Geometric `GCNConv`

Ví dụ API, **chỉ áp dụng nếu đã xác nhận sử dụng PyG GCNConv**:

```python
q = torch.sigmoid(edge_predictor(edge_features))
w_arc = epsilon + (1.0 - epsilon) * q
edge_weight = expand_arc_weights_to_graph_edges(w_arc, arc_to_edge_map)

# Edge weights phải còn trong autograd; KHÔNG .detach().
h = conv1(x, edge_index, edge_weight=edge_weight)
h = activation(h)
h = conv2(h, edge_index, edge_weight=edge_weight)
```

Bắt buộc kiểm tra: `cached=False` nếu normalization phụ thuộc trainable edge weights; không cache adjacency đã normalized ngoài graph; tránh thêm self-loop hai lần; map weights tới đúng thứ tự `edge_index` sau batching và sau coalesce.

Nếu baseline là dense adjacency hoặc custom GCN, agent phải viết weighted normalization tương ứng, dùng phép toán PyTorch khả vi để gradient từ `L_cls` truyền về `q`.

### 5.4. Huấn luyện hai stage, không làm mất đóng góp gốc

**Stage A — Train predictor:** như P2; lưu `q` checkpoint. Chưa chạy full Table 2/3/4.

**Stage B — Warm-start GCN:** nạp checkpoint GCN cùng seed/split; khởi tạo `q` từ stage A, bật `epsilon` tương đối cao (thăm dò 0.8); ban đầu có thể freeze GCN vài bước/epoch để kiểm tra loss và gradient.

**Stage C — Joint fine-tune:** cập nhật GCN và q với

```python
classification_loss = BCE_same_as_baseline(clean_logits, labels)
edge_loss = masked_document_balanced_edge_bce(
    edge_logits, retention_targets, valid_edge_mask, document_ids
)
loss = classification_loss + lambda_edge * edge_loss
```

Pilot tham số từ tài liệu nghiên cứu (chưa được chứng minh): `lambda_edge ∈ {0.05,0.1,0.2}`, `epsilon ∈ {0.6,0.8,1.0}`. Để tránh grid sweep tốn GPU, bắt đầu **một cấu hình** `lambda_edge=0.1, epsilon=0.8`, chỉ mở rộng nếu source-validation ủng hộ.

**Không ép BCE nhãn Human/AI lên các perturbations khó, không chắc bảo toàn nhãn.** Nếu classifier chỉ thấy clean, B0/B1 phải được so sánh có classification exposure tương đương.

### 5.5. Điều kiện đặc biệt cần thử

- Với **epsilon=1**, model GCN phải cho logits và embeddings giống baseline tại cùng weights, `eval()` mode, trên nhiều graph (bao gồm isolate/short graph) trong sai số số học.
- Cần có gradient khác 0 từ `L_edge` tới predictor khi có valid edges; khi epsilon < 1, thử gradient từ `L_cls` qua adjacency tới predictor. `epsilon=1` cố ý cắt đường gradient classification → q.
- Khi q đồng nhất, PEARL vẫn có thể thay normalization so với baseline (trừ epsilon=1). Không được diễn giải cải thiện do weight scaling mặc định là học reliability tốt.
- Cần ghi phân bố `q`, mean/std của w theo relation, và norm gradient riêng GCN/q.

---

## 6. Thứ tự ablation (tiết kiệm GPU nhưng đủ kiểm chứng)

| ID | Variant | Cùng classification exposure? | Vai trò |
|---|---|---|---|
| **B0** | GCN gốc (clean-only) | Clean | Mốc tái lập |
| **B1** | GCN + edit pairs được nạp/tiền xử lý nhưng `q=1`, classifier vẫn clean-only | Clean | **Đối chứng exposure cho PEARL clean-only** |
| **B1-aug** | GCN + BCE trên những edits đủ điều kiện | Clean + safe edited | Đối chứng lợi ích của augmentation; chỉ so trực tiếp với PEARL nếu PEARL cũng có cùng BCE augmented |
| **B-prior** | GCN + edge weights từ relation/POS retention prior | Clean | Một prior đơn giản có đủ không? |
| **B-gate** | GCN + unguided learned edge gate, không retention labels | Clean | Mức lợi ích từ weighting không có supervision |
| **B-shuffle** | GCN + q predictor train với shuffled retention labels | Clean | Nhãn alignment thật có thêm thông tin không? |
| **PEARL** | GCN + supervised q + weighted adjacency | Clean | Mô hình chính |
| **PEARL-eps1** | Mô hình PEARL, `epsilon=1` | Clean | Sanity: cùng operator GCN gốc |

**Ưu tiên chạy:** B0 → P2 edge predictor → B-prior/B-gate → PEARL, sau đó B-shuffle khi có tín hiệu tốt. Không cần chạy mọi biến thể từ ngày đầu.

**Ràng buộc công bằng:** same dataset/split, seed, optimizer budget, preprocessing, checkpoint selection; các biến thể có BCE trên augmented phải có cùng loại/tỷ lệ edits. Fine-tune từ checkpoint so với train from scratch là một confound khác, nên đối chứng cùng phương thức khởi tạo.

Không lựa chọn variant bằng Accuracy trên GPT-3.5 target test. Nếu mục tiêu là cơ chế robust trên source, xây **source-validation clean + perturbations** được khóa trước, không trùng test.

---

## 7. Train / validation / benchmark protocol

### 7.1. Validation và early stopping

- Mặc định pilot: `seed=2025`; sau khi smoke ổn chạy thêm `2026` để kiểm tra seed sensitivity.
- Chọn epoch theo **source-validation rule đã chốt**. Có thể dùng primary clean BCE và guardrails synonym/structural trên **source-validation** (không dùng official test).
- Không cần cố định số epoch bằng paper nếu training curves còn thay đổi: đặt `max_epochs` và `patience` cố định cho **mọi paired variants**; ghi thêm tổng optimizer steps để đối chiếu batch size.
- Threshold 0.5 theo label Human=1, AI=0; output head có thể là probability hoặc logits, tránh sigmoid hai lần.

### 7.2. Evaluation cuối

| Bảng | Điều kiện | Cần chạy |
|---|---|---|
| **Table 2** | HC3 Original, synonym 5/10/20/30 | PEARL checkpoint train HC3 |
| **Table 2** | GPT3.5-Mixed Original, synonym 5/10/20/30 | **Checkpoint train GPT3.5 riêng** |
| **Table 3** | HC3 → GPT3.5-Mixed, Original, synonym 5/10/20/30 | Checkpoint HC3, zero target-label tuning |
| **Table 3** | GPT3.5-Mixed → HC3 tương ứng | Checkpoint GPT train riêng |
| **Table 4** | HC3 Original, Insert/Repeat/Replace/Delete 25% | Checkpoint HC3 |

Metrics: Accuracy ở threshold 0.5, ROC-AUC, macro-F1, AI recall, Human recall, confusion matrix, calibration/logit distributions, paired sample ID, mean±SD đa seed. Table 4 noise average chỉ tính 4 loại nhiễu, không bao gồm Original. So với GCN **tự train paired** là kiểm định chính; số paper chỉ là mốc công bố theo protocol khác.

**Nhớ rằng Table 3 đã được nhóm xem nhiều lần trong lịch sử các thí nghiệm**: kết quả từ cùng target test hiện mang tính exploratory; muốn claim mạnh cần xác nhận trên held-out dataset/benchmark khác chưa được chọn theo test.

### 7.3. Phân tích cơ chế

- q accuracy không đủ: AUROC/AUPRC/Brier/ECE cho edge retention, theo edit type, relation, POS, class và source-only length bin.
- Kiểm tra `q` trên **graph edited** ở source validation, không chỉ clean, vì inference Table 4 là single-view edited graph.
- Đo tỷ lệ token/arc còn căn chỉnh; tỷ lệ stable/changed/unknown.
- Đo sự biến thiên normalized adjacency `S(G)` vs `S(G')` **sau alignment**, node embedding drift, logit shift, prediction flip-rate, theo loại nhiễu.
- Đo edge importance đối với classification (ví dụ leave-one-arc-out/edge-masking trên source validation với sampling nhỏ), kiểm tra liệu stable edges có thật sự là discriminative edges.
- Kiểm tra `q` có suy biến thành relation-frequency prior hoặc length/topic shortcut không.

---

## 8. Unit tests và integration tests bắt buộc

1. **Identity edit:** tất cả token và dependency arcs giữ nguyên → nhãn stable=1 với mask hợp lệ.
2. **Delete:** cạnh có token mất → unknown/invalid, không ép changed=0.
3. **Insert/repeat:** UID mới không làm lệch mapping token gốc; ambiguous alignments được mask.
4. **Head/relation change:** hai đầu mút còn tồn tại nhưng head hoặc relation khác → changed=0.
5. **Token split/merge:** không gán nhãn xác định khi thiếu mapping 1–1.
6. **No cross-document leak:** pairs/train/val/test không chia sẻ `doc_id` nguồn; `batch` graph không lẫn edge indices.
7. **No provenance at inference:** model chạy chỉ với text, parser graph, node X, edge metadata.
8. **Baseline parity:** epsilon=1, cùng checkpoint, `eval()` → outputs khớp baseline (định trước atol/rtol).
9. **Weighted normalization:** degree/self-loops/duplicate edges/direction khớp baseline và đúng công thức; không NaN/Inf trên tiny/isolated graphs.
10. **Autograd:** ở epsilon<1 `L_cls.backward()` có gradient tới q; `L_edge` gradient tới predictor; không `.detach()` sai.
11. **Save/load parity:** checkpoint sau load cho cùng output.
12. **Predictor sanity:** trên synthetic edges với nhãn phân biệt được, P2 có thể overfit dataset nhỏ; trên shuffled labels không xuất hiện gain giả tạo trên held-out validation.
13. **Class labels:** Human=1, AI=0; AI recall/human recall không bị đảo.
14. **Fixed exposure:** BCE augmented được áp cho PEARL và baseline cùng tập edit nếu lựa chọn dùng augmentation.
15. **Cache versioning:** metadata và graph cache hash mismatch bị từ chối thay vì tái sử dụng âm thầm.

**Không chấp nhận triển khai** chỉ vì một smoke test train loss giảm; cần cả test toán tử và nhãn retention.

---

## 9. Đề xuất tổ chức file và CLI

**Agent phải adapt theo repository thực tế; những path sau không khẳng định đã tồn tại.** Giữ PEARL trong namespace riêng để không ảnh hưởng GCN/V1/V3/LISA:

```text
pearl/
    __init__.py
    edit_provenance.py      # UID, span, edit log, deterministic edit
    alignment.py            # source ↔ edited spaCy tokens; unknown mask
    edge_labels.py          # stable / changed / unknown
    features.py             # inference-available edge features
    predictor.py            # edge-retention MLP
    weighted_gcn.py         # adapter preserve GCN core
    pairs_dataset.py        # lazy pairs, doc-balanced edge loss
    train_edge.py           # P2 only
    train_gcn.py            # P3 joint / fine-tune
    infer.py                # single-view inference
    evaluate.py             # reuse original benchmark protocol
    README.md

tests/test_pearl_alignment.py
tests/test_pearl_labels.py
tests/test_pearl_weighted_gcn.py
tests/test_pearl_training.py
```

**CLI mẫu mục tiêu** (điều chỉnh tên/args sau audit):

```bash
# P1 — kiểm tra nhãn/căn chỉnh trên SOURCE TRAIN
python -m pearl.edit_provenance --split train --seed 2025 --audit-only

# P2 — chỉ train edge predictor và đánh giá trên SOURCE VAL
python -m pearl.train_edge --seed 2025 --epochs 10 --output runs/pearl/edge_2025

# P3 — fine-tune với PEARL; không chạy benchmark đầy đủ
python -m pearl.train_gcn --seed 2025 --epsilon 0.8 --lambda-edge 0.1 \
  --output runs/pearl/gcn_2025

# Unit tests trước khi train
python -m pytest tests/test_pearl_* -q

# Chỉ sau khi khóa cấu hình mới chạy final evaluation
python -m pearl.evaluate --checkpoint runs/pearl/gcn_2025/best.pt \
  --benchmark hc3 --conditions original,syn5,syn10,syn20,syn30
```

`--epochs 10` chỉ là demo CLI cho stage P2; agent phải chọn cấu hình hợp lý từ source validation và tài nguyên. Không giả định sẵn tokenizer metadata hay script eval.

---

## 10. Tiết kiệm GPU/disk Kaggle

- Ngân sách khoảng **20 GB disk**, tối đa **4 graph caches `.pkl`** cùng lúc theo hiện trạng nhóm.
- P1/P2 chỉ cần **clean graph + sidecar metadata + edge labels**; tránh lưu RoBERTa embeddings cho mọi edited view.
- Có thể sinh edit pairs/lấy spaCy parse theo từng family; cache ngắn hạn chỉ khi cần, xóa tuần tự sau khi có metadata đảm bảo tái lập.
- Dùng một small held-out source validation để kiểm tra P2; không phải cứ đạt edge AUC cao là chạy full GCN.
- Ghi riêng CPU preprocessing/edit-align time, GPU train time, raw-text inference latency, cached-graph inference latency, peak VRAM và peak disk.
- Không tạo file cache cho mỗi seed nếu có thể reuse deterministic raw graph và chỉ thay split/augmentation provenance có hash rõ ràng.

---

## 11. Tiêu chí GO/NO-GO trước khi mở rộng

| Stage | GO khi | NO-GO khi |
|---|---|---|
| P0 | Tái lập baseline và parity epsilon=1 | Không rõ normalization, mapping, label convention |
| P1 | Token provenance đúng; valid edges đủ đa dạng | Nhiều `unknown`, nhãn sai do split/merge/repetition |
| P2 | q vượt constant/relation prior trên held-out source và không suy biến | q gần hằng số hoặc chỉ lặp thống kê relation |
| P3 | GCN gradient ổn định, source clean không giảm mạnh so B0/B-prior | Classification collapse, q triệt tiêu cạnh hữu ích |
| Benchmark | Lợi ích paired đáng tin cậy, Table 2/4 guardrails không sụp | Chỉ hơn do augmentation, thay normalization, test tuning hoặc một seed đơn |

Ngưỡng screening có thể khóa trước theo khả năng chấp nhận của nhóm, ví dụ **source clean giảm ≤0.5 điểm %** và **source structural validation không suy giảm**, nhưng các ngưỡng này là chính sách nghiên cứu, **không phải kỳ vọng kết quả**. Table 3 chỉ được đánh giá khi variant đã khóa.

Nếu P2 không vượt relation prior: **dừng PEARL** hoặc báo cáo kết quả âm; không thêm nhiều kỹ thuật khác để che thất bại của giả thuyết chính.

---

## 12. Prompt gọn dành cho Codex

> Đọc toàn bộ `Huong_dan_implement_PEARL_GCN.md` và tài liệu nghiên cứu gốc `PEARL_CoSoLyLuan_ThietKe_Va_KeHoachThucNghiem(1).md`. Audit repository PRDetect thực tế trước, sau đó implement PEARL **theo thứ tự P0 → P1 → P2 → P3**. Ưu tiên giao P0/P1/P2 hoạt động đầy đủ với tests và báo cáo retention labels/predictor trước khi train GCN lớn. Giữ nguyên code/checkpoint GCN gốc, V1/V3, LISA và PRISM; không thay embedding, readout, architecture hoặc test protocol. Nhãn edges phải tạo bằng token provenance thực, không đoán theo vị trí sau insert/delete, và phải mask unknown. Predictor chỉ nhận features có sẵn khi inference, không nhận edit log/nhãn; epsilon=1 phải match baseline. Nếu P2 không vượt constant/relation prior ở source validation, báo kết quả âm và dừng, không cố train full. Nếu P2 đạt, implement weighted GCN có autograd/normalization đúng, unit tests, checkpoint, source-validation, sau đó mới chuẩn bị official Table 2/3/4. Cung cấp files changed, kết quả tests, hướng dẫn chạy và giới hạn còn tồn tại. Không chạy full multi-seed benchmark ở nhiệm vụ implement đầu tiên.

---

## 13. Nguồn tham khảo và ranh giới claim

- **Nguồn đề xuất PEARL:** `PEARL_CoSoLyLuan_ThietKe_Va_KeHoachThucNghiem(1).md`, đặc biệt mục 4–6, 9–11.
- **Paper PRDetect:** Li và cộng sự, *PRDetect: Perturbation-Robust LLM-generated Text Detection Based on Syntax Tree*, Findings of NAACL 2025: <https://aclanthology.org/2025.findings-naacl.464/>.
- Các cách học trọng số/attention cạnh, graph robustness và consistency learning **đã có prior art**. PEARL chỉ có **ứng viên đóng góp** ở *token-provenance-aligned edge-retention supervision* chuyển sang *single-view edge weighting* cho AI-text detection.
- Không được viết “PEARL đã tăng Table 2/3/4” hoặc “đã chứng minh edge accuracy” nếu chưa train và đánh giá.

**Kết quả mong đợi của lần triển khai đầu:** P0 audit + P1 nhãn retention được kiểm chứng + P2 edge predictor và controls. **P3 weighted GCN** chỉ đáng chạy GPU lớn sau khi P2 thuyết phục.
