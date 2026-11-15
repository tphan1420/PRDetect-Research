# HƯỚNG DẪN TRIỂN KHAI PEARL-GCN CHO PRDETECT — BẢN V2 (ĐÃ HIỆU CHỈNH)

**PEARL — Perturbation-Aligned Edge Reliability Learning**  
**Backbone chính:** GCN gốc của PRDetect; RGCN V1 chỉ là kiểm tra khả năng chuyển giao sau này.  
**Trạng thái:** Đề cương triển khai và kiểm chứng; **chưa có kết quả thực nghiệm chứng minh PEARL thắng Table 2/3/4**.  
**Căn cứ trực tiếp:** `Huong_dan_implement_PEARL_GCN(1).md` (bản trước), `PEARL_CoSoLyLuan_ThietKe_Va_KeHoachThucNghiem.md`, bài PRDetect (NAACL Findings 2025), và nhật ký lịch sử cải tiến PRDetect.  
**Mục đích:** Hướng dẫn agent/Codex implement **đúng một giả thuyết khoa học**, có paired controls, bảo toàn baseline, ưu tiên tính khả thi trên Kaggle ~20 GB.

> **Phân biệt quan trọng:** Đây là **tài liệu chỉ dẫn triển khai**, không xác nhận những module, API, script hay đường dẫn bên dưới đã tồn tại trong repository. Trước khi sửa mã, agent phải đọc code và dữ liệu thực tế. **Không lấy số tác giả công bố thay cho baseline GCN nhóm tự train.**

## 0. Mục tiêu, phạm vi và quyết định thiết kế

### 0.1. Câu hỏi khoa học

**RQ:** Nếu dùng các cặp văn bản gốc–chỉnh sửa để giám sát *khả năng một dependency arc duy trì đúng hai đầu mút, hướng và nhãn quan hệ*, liệu một predictor chỉ nhận **một graph tại inference** có thể học trọng số cạnh giúp GCN phân loại Human/AI ổn định hơn không?

Điều cần chứng minh **không phải** “GCN có edge weights”, vì edge weighting/attention đã có từ trước. Ứng viên đóng góp là **edit-provenance-aligned dependency-relation retention supervision → single-view reliability predictor → weighted message passing**.

Bốn giả thuyết kiểm chứng được:

- **H1 (tín hiệu):** Retention không chỉ là một hằng số hoặc lookup theo loại dependency; predictor có khả năng phân biệt `stable/changed` trên **source validation**.
- **H2 (tác động):** Trọng số học từ retention làm giảm prediction flip/logit drift hoặc tăng source robustness so với *same-budget controls*; cạnh có stability cao phải còn mang tín hiệu phân loại.
- **H3 (benchmark):** PEARL giúp Table 2/3/4, đặc biệt HC3→GPT3.5 trong Table 3 **mà không làm sụp Table 4**; đây là giả thuyết chưa được xác nhận và có thể bị bác bỏ.
- **H4 (quy công):** Hiệu quả phải vượt điều có thể giải thích bằng fine-tuning thêm, weighted adjacency hằng số, unguided gate hoặc augmentation.

### 0.2. Phiên bản nào được triển khai?

| Phiên bản | Mô tả | Trạng thái |
|---|---|---|
| **PEARL-Lite / Main** | Edge-retention trên cặp `clean → one edit`; predictor một-view; weighted GCN; classification BCE trên clean | **Bắt buộc đầu tiên** |
| **PEARL-EditedView** | Bổ sung retention trên `edited → re-edited` để predictor thấy graph đã bị sửa | **Chỉ ablation nếu có dấu hiệu clean-to-edited shift** |
| **PEARL-Survival** | Học mục tiêu riêng về việc arc còn tồn tại sau edit, kể cả mất token | **Nghiên cứu phụ; không trộn với retention** |
| **PEARL+AugCls / +Anchor** | Thêm BCE cho edits bảo toàn nhãn hoặc representation consistency | **Hoãn; chỉ thử sau khi Lite cho tín hiệu** |

**Không** đổi frozen RoBERTa/embedding alignment, spaCy parser, graph topology, số tầng GCN, readout, classifier, nhãn lớp, hoặc thuật toán test. Không kéo những cải tiến V1–V3/PRISM/LISA/R-Drop/CMixStyle vào bản đầu.

### 0.3. Quy ước dữ liệu

- `Human = 1`, `AI = 0` **theo baseline paper/code sau khi audit**; nếu repository có quy ước khác phải ghi rõ và nhất quán end-to-end, không đổi âm thầm.
- `G = (V,E)` là dependency graph của **view hiện tại**; `e=(child, head, relation)` là một **directed spaCy dependency arc**, trừ ROOT/self-loop không có head độc lập.
- `T ~ Q` là một phép edit **được sinh trên source train** theo một phân phối cố định có lưu metadata. `G⁺ = Parse(T(x))`.
- `m` là ánh xạ *duy nhất và xác thực được* giữa node ở view trước và node ở view sau. Không coi vị trí token là UID bất biến.
- `q_φ(e,G)` là **ước lượng xác suất relation của cạnh hiện tại được giữ sau một edit tương lai từ Q**, **không** phải độ đúng của parser, không phải xác suất cạnh vốn đã bị thao tác, không phải edge classification importance.

### 0.4. Định nghĩa mục tiêu chính (chỉ relation retention)

Với `e=(u,v,r) ∈ E(G)`, hai đầu mút phải còn tồn tại và có ánh xạ duy nhất `m(u),m(v)`. Định nghĩa

\[
s_e^{\mathrm{rel}} = \mathbf{1}\!\left[(m(u),m(v),r)\in E(G^+)\right],\qquad
M_e=\mathbf{1}[m(u),m(v)\text{ xác định duy nhất}].
\]

- `M_e=1, s_e=1`: **stable**, hai token, head relation và dependency label được giữ.
- `M_e=1, s_e=0`: **changed**, hai token vẫn tồn tại nhưng arc đầy đủ không còn như cũ.
- `M_e=0`: **unknown**; bỏ khỏi `L_edge`, **không** gán thành 0.
- Nếu cả hai đầu mút còn nhưng `head` thay đổi sang token khác, arc cũ nhận 0; nếu `relation` đổi cũng nhận 0.
- Với Delete làm mất một đầu mút, không có nhãn cho **relation retention có điều kiện cả hai đầu mút còn tồn tại**. Đừng báo rằng mô hình đã học “robust với deletion” chỉ vì mask các cạnh này.

Predictor:

\[
q_\phi(e,G)=\sigma\!\left(f_\phi(a_e(G))\right),\quad
w_e=\epsilon+(1-\epsilon)q_\phi(e,G),\quad\epsilon\in(0,1].
\]

Mô hình chính dùng

\[
L_{\mathrm{Lite}}=L_{\mathrm{cls}}(\text{clean})+\lambda_e L_{\mathrm{rel}}(\text{source train edit pairs}).
\]

Khởi điểm *để thử*, **không phải kết quả tối ưu**: `epsilon = 0.8`, `lambda_edge = 0.1`, MLP hidden 64. `epsilon=1` là **sanity control**, không phải candidate PEARL.

---

## 1. Pipeline chuẩn: cái gì giữ, cái gì thay

### 1.1. Original PRDetect

```text
raw text
  -> spaCy tokenization / dependency parse
  -> RoBERTa node embeddings
  -> binary adjacency + baseline normalization
  -> original 2-layer GCN
  -> original readout/classifier
  -> Human/AI

TRAIN: original classification BCE
```

### 1.2. PEARL-Lite train

```text
                     SOURCE TRAIN ONLY

             x (clean) -> spaCy -> G, node X
                |                   |
                |                   +--> edge features a_e(G)
                |                                     |
                v                                     v
           Edit T(x), log              q_phi(e,G): single-view MLP
                |                                     |
                v                                     +--> arc weights
            x' -> Parse G'                            |
                |                                     v
                +--> provenance alignment     weighted-normalized adjacency
                            |                          |
                            v                          v
                   s_rel, valid_mask           same original 2-layer GCN
                            |                          |
                            v                          v
                      L_edge                   original head -> L_cls(clean)
                            \__________________________/
                                        |
                           optimizer: L_cls + λ L_edge
```

**Quan trọng:** graph `G'` ở phiên bản Lite **chỉ cần parse và alignment** để tạo nhãn; không cần chạy RoBERTa hay classification head trên `x'`. Điều này tiết kiệm CPU/GPU/disk và cô lập tác động của edge supervision.

### 1.3. PEARL-Lite inference

```text
single text -> spaCy -> graph G and frozen node X
            -> edge metadata -> q_phi(e,G)
            -> weighted adjacency -> same 2-layer GCN
            -> original head -> Human/AI
```

**Không** cần cặp văn bản, edit generator, provenance, `G'`, target data hoặc nhãn tại inference.

### 1.4. PEARL-EditedView (chỉ nếu được kích hoạt)

```text
x (clean) --T1--> x' (edited) --T2--> x'' (re-edited)
                   |                          |
                   G' --------align---------- G''
                   |
       q_phi(e,G') supervised by retention(G'->G'')
```

- Có **hai observation-level training rows riêng**: `q(e,G)` học từ `G→G'` và `q(e,G')` học từ `G'→G''`.
- Không được dùng retention `G→G'` làm nhãn cho cạnh của `G'`, vì nó không trả lời câu hỏi cạnh `G'` **sẽ còn tồn tại sau một edit mới** hay không.
- `T2` chỉ lấy từ source-train generator; không dùng nhãn Human/AI của `x'`/`x''` trong classifier. Do đó không cần giả định edits mạnh giữ nguyên nhãn phân loại.
- Thêm đối chứng `PEARL-Lite` với *cùng optimizer steps*; ghi overhead bổ sung.

---

## 2. P0 — Audit repository và khóa baseline

### 2.1. Trước khi viết code mới

Agent phải tạo `docs/pearl_baseline_audit.md` ghi ít nhất:

1. **Code thực tế**: class GCN, số layer, dropout/activation, readout, classifier trả `logits` hay `probabilities`, loss tương ứng (`BCEWithLogitsLoss` hoặc `BCELoss`).
2. **Đồ thị thực tế**: spaCy version/model, phân đoạn câu/chunk; node count, index, hướng cạnh `child→head` hay ngược lại, symmetrization, duplicates, self-loops, sparse/dense representation, normalization.
3. **Node embeddings**: RoBERTa checkpoint/hash, token/subword alignment legacy, chunking, dtype và cache; **không thay A1/V2 aligned embedding** trong thí nghiệm PEARL chính.
4. **Data protocol**: nguồn HC3, GPT3.5-Mixed, document ID, train/validation/test split, seeds, perturbation generator cho Table 2 và Table 4, Table 3 hai chiều, label mapping.
5. **Training**: epochs, optimizer/scheduler, microbatch, checkpoint selection, steps/seed, state_dict và checkpoint path.
6. **Graph metadata**: có hay không `token.idx`, `head.i`, `dep_`, `pos_`, char offsets, sentence/chunk ID và mapping directed arc ↔ tensor edge. Nếu thiếu, tạo **sidecar từ raw text bằng cùng parser**, không suy ra relation từ adjacency nhị phân.
7. **Integrity**: SHA256 raw datasets, split index, graph cache, parser version, code commit, checkpoint; các graph trong cache có đồng nhất với text input không.

### 2.2. Điều kiện bảo toàn baseline (không thương lượng)

- Không chỉnh code/chkpt gốc, V1, V3, PRISM, LISA. PEARL ở namespace riêng.
- **Không thay** embedding, graph topology, readout/head/threshold hoặc normalization khi so paired. Nếu phát hiện bug buộc phải sửa, **tái train cả baseline và PEARL dưới pipeline đã sửa**; báo các bảng cũ chỉ để tham khảo.
- Thêm sidecar phải giữ **đúng thứ tự node/edge** của tensor baseline. Nếu không bảo đảm, dừng.
- `epsilon=1` trên **cùng GCN weights** và `eval()` phải cho logits/embeddings bằng baseline trong tolerance (`atol/rtol` ghi rõ; mặc định mục tiêu `1e-6/1e-5` với FP32). Test nhiều graph, có graph ngắn/isolated.
- Nếu GCN gốc dùng adjacency không đối xứng, không đối xứng hóa chỉ vì thuận tiện cho `q`.

**Đầu ra P0:** audit file, bảng parity, baseline metrics từ checkpoint gốc, và lệnh tái lập. **NO-GO** nếu không rõ hướng/normalization/mapping.

---

## 3. P1 — Sinh edits, căn chỉnh token và nhãn cạnh

### 3.1. Split trước, edit sau

- Khóa splits theo document ID **trước mọi augmentation**. Các bản sửa của cùng document phải ở cùng split.
- `train`: sinh pairs dùng để tối ưu `q`. `source validation`: sinh pairs với seed độc lập, dùng đánh giá `q`/early stopping, **không cập nhật weights**.
- `target test`, `official Table 2/3/4 test`: **không** tạo edge labels để train/tune/chọn epoch/threshold.
- Đưa pair generator vào config có version và hash, ghi phân phối `Q` gồm family, ratio, rule chọn từ, sampling seed. Không sử dụng output của benchmark test generator làm training labels.

### 3.2. Lộ trình edit khả thi

| Giai đoạn | Edit | Vai trò | Cảnh báo |
|---|---|---|---|
| Pilot 1 | Identity, synonym 5–10% | Unit tests, giữ chủ thể câu, positive labels | Có thể gần như toàn `stable=1` |
| Pilot 2 | Replace nhẹ 5–10%, Insert/Repeat nhẹ | Tạo negative labels có ý nghĩa; kiểm tra UID/parse changes | Không được ép split/merge thành 1:1 |
| Pilot 3 | Delete nhẹ 5–10% | Stress test mapping và nguồn âm tính nếu hai endpoint còn | Deleted endpoints phải masked cho *relation* |
| Optional | Nhiễu mạnh hơn/khác family | Kiểm định độ bền và generalization | Phải khóa trên source validation, không nhìn target test |

Cần đặc biệt chú ý: với `synonym` đa số arc vẫn giữ nguyên; với `delete` đa số arc mất endpoint bị **mask**. Nếu negative rate trên edges hợp lệ rất thấp, supervised retention classification có thể **không khả thi**. Phải kiểm tra định lượng thay vì tự gán thêm nhãn 0 sai để cân bằng.

### 3.3. Provenance làm sao cho đúng

1. Khi sinh edit, quản lý UID ở mức token/lexical span trước edit; với `insert/repeat`, token mới được UID riêng; với `delete`, UID biến mất.
2. Lưu spans trước và sau theo `char_start,char_end`, original/source UID, `edit_op`. Sau edit chạy **cùng spaCy pipeline**; map UID/span sang spaCy indices.
3. Chỉ chấp nhận source token ↔ edited token **1:1 duy nhất** với span/provenance nhất quán. Nếu tokenization split/merge/resegmentation hoặc lặp gây ambiguity: `unknown`.
4. Không suy ra mapping bằng cùng `token.i`, chỉ số vị trí hoặc `SequenceMatcher`/Levenshtein thuần túy; với `repeat`, clone có UID khác source.
5. Label trên directed spaCy arc `e=(child,head,relation)`. Khi source arc có cả hai endpoint xác định, kiểm tra **đúng triple** trong parsed edited graph. `relation` đổi thì negative; arcs thuộc ROOT hoặc parser pseudo-node xử lý theo rule được ghi trong audit.
6. Khi GCN dùng hai chiều/symmetric graph, **nội dung supervision vẫn gắn vào một directed source dependency arc**; sau đó `arc_to_edge_map` nhân weight sang tensor-edge theo quy tắc đã audit.
7. Với edit xóa một endpoint: `relation_valid=False`. **Không** gọi đây là changed relation. Nếu cần xác suất tồn tại, định nghĩa nhãn mới ở mục 9, không trộn với `relation`.

### 3.4. Record tối thiểu

```json
{
  "source_doc_id": "doc-42",
  "split": "train",
  "view_id": "clean",
  "next_view_id": "edit-1",
  "parser_version": "record-exact-version",
  "generator_version": "edit-v1",
  "edit_family": "replace",
  "edit_ratio": 0.10,
  "edit_seed": 2025,
  "source_text_hash": "sha256-of-actual-text",
  "next_text_hash": "sha256-of-actual-text",
  "alignment": [{"src_node": 4, "dst_node": 5, "unique": true}],
  "arc_labels": [
    {"src_arc_id": 7, "relation_retained": 1, "relation_valid": true},
    {"src_arc_id": 8, "relation_retained": null, "relation_valid": false}
  ]
}
```

Ví dụ **schema**, không phải kết quả thật. Với file thực tế ưu tiên Parquet/JSONL nén, chỉ lưu metadata và labels; không copy RoBERTa embeddings sang từng pair.

### 3.5. QA và thống kê bắt buộc

- Theo `edit_family`, `ratio`, `Human/AI`, `length bin`: số pairs thành công, tỷ lệ token unique/unknown, cạnh valid/unknown, stable/changed trong valid.
- Identity phải cho toàn `valid=true, stable=1` trên các arcs hợp lệ; nếu không, mapping/parser sai.
- Với một phép đổi head/relation có hai endpoint giữ: phải có negative đúng.
- Với Delete mất endpoint: `valid=false`, không phát sinh negative relation giả.
- Audit thủ công 30–50 pairs đa dạng; ghi lỗi split/merge, punctuation, repeated words, chunk boundaries.
- **Báo riêng** tỷ lệ negative trong valid edges; không đánh đồng edge accuracy cao với predictor tốt.
- Nếu không đủ nhãn âm, đề xuất sửa `Q` **chỉ trên source train/source validation** và ghi như một phiên bản dataset mới; không suy ra kết quả trước.

**Đầu ra P1:** sidecar parser metadata + deterministic pair records + statistics + unit tests. **NO-GO** nếu provenance không đáng tin hoặc label distribution gần đơn lớp đến mức không thể so sánh predictor.

---

## 4. P2 — Huấn luyện và kiểm chứng Edge Reliability Predictor

### 4.1. Feature schema: chỉ dùng thông tin của graph hiện tại

```text
edge_features(e,G) = concat(
  Emb(POS_child), Emb(POS_head),
  Emb(dep_relation),
  Bucket(abs(child_index - head_index)),
  child_degree, head_degree,
  optional_direction_from_parser
)
```

- Predictor **không nhận** edit type/ratio/seed, provenance UID, nhãn stable/changed, nhãn Human/AI, source dataset ID hay view trước đó.
- Không thêm RoBERTa node feature hoặc semantic gate trong Lite; nếu muốn thử phải làm ablation riêng vì lịch sử V2 cho thấy thay embedding dễ làm Table 4 sụp.
- `dep_relation` và POS lấy từ spaCy metadata, **không** lấy qua giả định `r < R/2` từ lịch sử V3.
- Cùng một directed dependency arc dùng **một prediction q**. Nếu graph tensor có chiều đối xứng, copy q sang các tensor edges phù hợp.

### 4.2. Predictor và loss

```python
class EdgeRetentionPredictor(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, edge_features):
        return self.mlp(edge_features).squeeze(-1)  # logits, no sigmoid here
```

Với document `d`, các arcs hợp lệ `E_d^v` và nhãn `s_e`, định nghĩa:

\[
\ell_{\mathrm{rel}}(d)=\frac{1}{|E_d^v|}\sum_{e\in E_d^v}
\operatorname{BCEWithLogits}(f_\phi(a_e),s_e),
\quad L_{\mathrm{rel}}=\frac{1}{|D^v|}\sum_{d\in D^v}\ell_{\mathrm{rel}}(d).
\]

`D^v` **chỉ bao gồm documents có ít nhất một valid arc**. Đây là **document-balanced loss**; không để document dài áp đảo. Nếu đổi class weighting/sampling do âm tính hiếm, **báo rõ policy và đánh giá calibration trên phân phối validation nguyên thủy**; không tự nhận sigmoid sau weighted BCE là probability đã calibrated.

### 4.3. Predictor controls và metrics

| Predictor | Vai trò |
|---|---|
| `constant_prior` | Xác suất giữ relation chung từ source-train |
| `relation_prior` | Prior theo dependency label, Laplace smoothing |
| `relation+POS_prior` | Control mạnh hơn cho predictor dùng POS |
| `MLP` | Chính PEARL |
| `shuffled_label_MLP` | Negative control: xáo nhãn train, giữ validation thật |

Báo **AUROC** (nếu cả hai lớp có mặt), **AUPRC + prevalence**, **log-loss**, **Brier**, **ECE**, confusion theo family, ratio, relation và độ dài. Bootstrap **theo document**, không bootstrap arcs độc lập.

**Quan trọng về GO/NO-GO:**

- **P2-pass** khi MLP thực sự dự đoán retention tốt hơn *constant* và có tín hiệu khác *relation(+POS) prior*, không suy biến gần hằng số. Nếu chỉ ngang prior, **không thể claim đã học reliability theo ngữ cảnh**.
- **P2-pass chưa có nghĩa classifier sẽ tốt hơn.** Phải kiểm tra P3, classification importance, và source robustness.
- Nếu negative hợp lệ hiếm, AUC/precision không ổn định, hoặc nhãn edit không đúng, **NO-GO** trước GPU training. Không cố ép label 0 cho deletion để tăng metric.

### 4.4. Kiểm tra phân phối trên edited graph **trước** khi chạy extension

Ở source validation, tính `q` cả cho `clean G` và `edited G'` (graph `G'` không cần nhãn forward-retention để tính q):

- Phân bố `q`, entropy/variance theo view, relation, class và độ dài.
- Tỷ lệ q gần 1, gần 0 và sự dịch chuyển phân phối `q(G)` so `q(G')`.
- High-q tại cạnh mới sinh bởi Insert/Replace có bất thường không? **Cạnh mới không có cùng forward retention label từ `G→G'`**; không dùng nhãn đó để tuyên bố predictor sai.
- Nếu `q(G')` rất lệch khỏi clean và source structural validation yếu, **xét P2-B (EditedView)**, không tự động chuyển thẳng sang hai-step training.

### 4.5. P2-B — EditedView extension (chỉ khi cần)

Nếu P2-A cho kết quả tốt trên clean nhưng graph edited làm lộ domain shift lớn:

1. Dùng **chính source train documents**: `G0=Parse(x)` → edit `T1` → `G1=Parse(x')` → edit `T2` → `G2=Parse(x'')`.
2. `q(e,G0)` nhận label retention của `G0→G1`; **`q(e,G1)` chỉ nhận label của `G1→G2`**.
3. `T2` có edit log/UID mới gắn với **current view `x'`**; UID nguồn cũ không tự động có ý nghĩa cho bước hai nếu mất hoặc split/merge.
4. Train predictor trên **mixture clean-origin/edited-origin**, mỗi type cân bằng theo document/view để không nhân đôi đóng góp của một document dài.
5. Không dùng classification label trên edited view, không dùng target data, không cần edited RoBERTa embedding nếu chỉ train `q`.
6. So sánh **P2-A vs P2-B** trên cùng source-validation, rồi mới xem downstream. `P2-B` là **biến thể**, không được lặng lẽ coi là PEARL-Lite.

Rủi ro: sau hai edits, ngữ nghĩa có thể bị phá hỏng, nhưng vì chỉ học *dependency retention* (không học class label trên views đó), không mặc định gây sai nhãn Human/AI. Tuy vậy, `Q2` có thể tạo phân phối phi thực tế; cần lưu tỷ lệ và audit readability.

---

## 5. P3 — Weighted GCN và quy trình huấn luyện công bằng

### 5.1. Weighted operator

Với arc có metadata hợp lệ:

\[
q_e=\sigma(f_\phi(a_e)),\quad w_e=\epsilon+(1-\epsilon)q_e.
\]

Ánh xạ trọng số này sang **đúng những tensor edges đã có trong baseline**. Không thêm/cắt cạnh, không sửa node order. Với graph symmetric có cặp edges cho một dependency arc thì cả hai hướng nhận cùng weight; self-loops giữ như baseline, thường weight 1.

Nếu baseline đúng kiểu symmetric normalized GCN:

\[
A_w(i,j)=w_{ij}\mathbf{1}[(i,j)\in E_{\mathrm{baseline}}],\quad
\widetilde A=A_w+I,\quad
\widetilde D_{ii}=\sum_j\widetilde A_{ij},\quad
S_\phi=\widetilde D^{-1/2}\widetilde A\widetilde D^{-1/2}.
\]

\[
H^{(l+1)}=\sigma(S_\phi H^{(l)}W^{(l)}).
\]

Nếu baseline dùng operator khác, **sửa phương trình/implementation theo thực tế**, không lén đổi baseline sang symmetric normalization. Với `epsilon=1`, mọi cạnh gốc có weight 1 nên **operator phải chính xác bằng baseline** tại cùng checkpoint.

### 5.2. Ví dụ adapter (chỉ nếu baseline dùng PyG GCNConv)

```python
arc_logits = edge_predictor(arc_features)           # gradients retained
q_arc = torch.sigmoid(arc_logits)
w_arc = epsilon + (1.0 - epsilon) * q_arc
edge_weight = expand_to_original_edges(w_arc, arc_to_edge_index_map)

h = conv1(x, edge_index, edge_weight=edge_weight)
h = activation(h)
h = conv2(h, edge_index, edge_weight=edge_weight)
logits_or_proba = original_readout_and_head(h, batch)
```

- `cached=False`; không cache normalized adjacency khi q còn học.
- Không `.detach()` weights, không tạo duplicate self-loops; kiểm tra `coalesce`/edge order/batching, **kể cả khi `arc_to_edge_index_map` có indexing không liên tiếp**.
- Không dùng cùng lúc manual normalization và `GCNConv(normalize=True)` khiến chuẩn hóa hai lần. Nếu tự chuẩn hóa thì gọi layer với option phù hợp; phải đạt `epsilon=1 parity`.
- Nếu graph custom/dense/sparse, viết một **adapter nhỏ có autograd**, không rewrite toàn pipeline.

### 5.3. Lịch train khả thi, tránh thêm yếu tố gây nhiễu

**Stage 0 — B0 Baseline:** load/retrain GCN gốc cùng seed, khóa checkpoint và hash. Ghi original train steps/metrics.

**Stage A — P2 Predictor:** train q từ edge labels bằng **source train**, checkpoint theo source-validation edge metrics. Chưa dùng label của target.

**Stage B — PEARL fine-tune:** nạp **cùng GCN checkpoint** từ Stage 0, nạp q từ Stage A, train joint trên **clean classification + edge retention**; khởi điểm `epsilon=0.8`, `lambda_edge=0.1`. Không cần freeze GCN vài epoch như một policy mặc định; chỉ làm freeze với cùng control/schedule nếu audit cho thấy cần để ổn định. Freeze ngẫu hứng làm paired comparison không công bằng.

**Stage B-control — GCN-FT:** nạp **cùng GCN checkpoint** và fine-tune thêm **cùng số optimizer steps, LR/scheduler, batches, checkpoint selection rule** nhưng chỉ clean BCE, `q=1` (không module q tham gia classification). Đây là **control bắt buộc**, bởi PEARL nhận thêm training budget sau baseline.

**Không dùng** edited class BCE trong PEARL-Lite; nếu mở rộng thêm, phải có `GCN-FT+AugCls` đối chứng sử dụng chính xác same eligible edits, label policy và steps.

Một lịch chạy thăm dò:

```text
seed 2025:
  (1) audit + baseline checkpoint
  (2) source-train paired edits -> P2 + edge controls
  (3) GCN-FT matched control
  (4) PEARL-Lite matched fine-tune
  (5) source validation: clean + synonym + structural
  (6) khóa config trước một lượt official benchmark

Nếu stage (5) khả quan: seed 2026, sau đó mở rộng 5 seeds.
```

### 5.4. Một lựa chọn hyperparameter nhỏ, có khóa trước

- Bắt đầu đúng **một setting**: `epsilon=0.8`, `lambda_edge=0.1`, `hidden_dim=64`, dropout/GCN params **giống baseline**.
- Nếu source validation không đạt và cơ chế có tín hiệu, có thể thử **tối đa một sweep nhỏ được định nghĩa trước**: `epsilon ∈ {0.6,0.8}`, `lambda_edge ∈ {0.05,0.1,0.2}`. Mỗi paired control sử dụng budget so sánh phù hợp.
- `epsilon=1` **chỉ parity/sanity**, không dùng nó làm PEARL chính.
- Quy tắc chọn checkpoint **phải cố định trên source validation**, ví dụ primary `clean BCE`, guardrails source `synonym/structural` theo quy tắc được đăng ký trước; không chọn theo official benchmark. Không đổi rule giữa các variants.
- Threshold classification 0.5 theo baseline; nếu baseline xuất logits thì 0.5 áp lên `sigmoid(logits)` tương đương logit≥0. Nếu baseline đã sigmoid, không sigmoid lần nữa.

### 5.5. Guardrails / unit checks trước khi benchmark

- `epsilon=1`: cùng model weights, graph và `eval()` → logit/readout parity.
- Gradients: `L_edge` phải về q; với `epsilon<1`, `L_cls` phải có đường gradient qua weighted adjacency về q.
- Constant q test: có thể làm thay đổi normalization dù q bằng nhau; phải có **uniform-weight control**, không quy lợi ích đó cho edge supervision.
- Theo dõi q mean/std/p5/p95 theo relation, class, clean/edited; train gradients GCN/q; guard against q collapse, NaN/Inf, duplicate-edge mismatches.

---

## 6. Ablation matrix — so sánh đúng bản chất đóng góp

### 6.1. Bộ controls cốt lõi

| ID | Mô hình | Cùng GCN khởi tạo và fine-tune steps? | Mục đích |
|---|---|---|---|
| **B0** | GCN gốc, checkpoint không fine-tune | Mốc ban đầu | Tham khảo khả năng reproduce |
| **B-FT** | GCN gốc fine-tune clean BCE | **Có, với PEARL** | **Bắt buộc**: kiểm soát thêm training |
| **B-uniform** | GCN-FT + constant edge weight `w=c`, cùng normalize | **Có** | Phân biệt learned weighting với thay scale adjacency |
| **B-prior** | Weighted GCN với relation(+POS) retention priors nguồn | **Có** | Đơn giản hơn q-MLP có đủ không? |
| **B-gate** | Learned q, chỉ tối ưu class BCE, không edge supervision | **Có** | Tách lợi ích từ learned gate với retention labels |
| **PEARL-Lite** | Supervised q + weighted GCN + clean BCE | **Có** | Phương pháp chính |
| **B-shuffle** | PEARL với edge labels xáo (giữ class proportions) | **Có** | Negative control cho giá trị alignment labels |
| **PEARL-eps1** | `epsilon=1`, same GCN weights | Sanity | Khóa operator về GCN gốc |

**Độ ưu tiên khi tài nguyên hạn chế:** B0 parity → P2 priors → **B-FT & PEARL-Lite** → B-uniform/B-prior → B-gate/B-shuffle. B-shuffle chỉ chạy đầy đủ nếu có gain cần xác nhận.

- **Tất cả** downstream paired variants phải dùng **cùng source split, GCN initialization, train mini-batch order, optimizer budget và checkpoint selection rule**. q-only pretraining được báo riêng chi phí; riêng việc PEARL có auxiliary supervision là bản chất phương pháp và phải có controls prior/shuffled.
- Đối chứng B-gate có thể cần cơ chế khởi tạo q công bằng để không bị bất lợi bởi q random; ghi rõ. Không dùng số V1/V3 lịch sử làm paired control.
- Nếu PEARL có BCE trên edited texts ở một nghiên cứu tiếp theo, B-FT+AugCls phải có *cùng classification exposure*.

### 6.2. Bộ ablation chỉ khi có lý do

| ID | Kích hoạt khi | Câu hỏi |
|---|---|---|
| `PEARL-EditedView` | P2 tốt clean, q trên edited shifted mạnh | Edited-origin supervision có cứu Table 4 không? |
| `PEARL-Survival` | Relation labels quá nhiều unknown dưới Delete | Separate survival prediction có hữu ích không? |
| `PEARL+AugCls` | Muốn tận dụng edits đủ điều kiện giữ nhãn | Gain do augmentation hay edge supervision? |
| `PEARL→RGCN V1` | PEARL tốt trên GCN 2 seeds | Khả năng chuyển kiến trúc, có paired V1 controls? |

**Không** mở đồng thời các extensions để “cứu bảng điểm”. Nếu Lite thất bại, trước tiên báo kết quả và chẩn đoán nguyên nhân.

---

## 7. Validation, ba benchmark và tiêu chí đánh giá

### 7.1. Source-only model selection

- Huấn luyện HC3: chọn `q`, `epsilon`, `λ`, checkpoint và classifier threshold **chỉ từ HC3 train/validation** theo rule khóa trước. Chạy HC3→GPT3.5 test **một lần sau khi khóa**; không lấy GPT labels để điều chỉnh.
- Huấn luyện GPT3.5: tương tự, dùng GPT3.5 train/validation, rồi kiểm tra GPT3.5→HC3.
- `source validation` có clean + perturbations **tự sinh từ source-validation docs** với fixed seeds, tách khỏi official evaluation.
- Table 3 target đã được nhóm xem nhiều trong lịch sử. Vì vậy mọi quyết định tiếp tục được chọn dựa trên Table 3 test đều làm mất tính độc lập của test; khi viết paper cần additional held-out validation/benchmark chưa tham gia vòng ý tưởng.

### 7.2. Protocol Table 2, Table 3, Table 4

| Bảng | Train checkpoint | Test conditions | Mục tiêu |
|---|---|---|---|
| **Table 2 / HC3** | HC3-trained | Original, synonyms 5/10/20/30% | Giữ clean, cải thiện hoặc bảo toàn synonym |
| **Table 2 / GPT3.5** | GPT-trained riêng | Original, synonyms 5/10/20/30% | Đo in-domain GPT |
| **Table 3 / HC3→GPT** | HC3-trained | GPT Original + synonyms 5/10/20/30% | **Ưu tiên OOD**, không target tuning |
| **Table 3 / GPT→HC3** | GPT-trained | HC3 Original + synonyms 5/10/20/30% | OOD chiều ngược |
| **Table 4 / HC3** | HC3-trained | Original, Insert/Repeat/Replace/Delete 25% | Chú ý Replace/Delete, nhưng không giảm Insert/Repeat |

**Bẫy nhãn trong paper:** Table 4 ghi một row là `Detect` nhưng mô tả prose là `Delete`. Cần đối chiếu **official code/generator thực tế** để ghi đúng phép nhiễu. Không tự sửa benchmark mà thiếu chứng cứ.

Số trong paper chỉ là **tham chiếu công bố**, không phải ngưỡng hoàn toàn tương đương: PRDetect Table 2 HC3 Original khoảng 98.78%; Table 3 HC3→GPT Original 73.2%; Table 4 Replace 79.8% và row `Detect` 74.7%. Với paired research, đối chứng chủ yếu là **GCN nhóm tự train cùng split, seed, checkpoints, generator và budgets**.

### 7.3. Metrics bắt buộc

- Main: Accuracy @ threshold 0.5, **macro-F1**, AUROC, AI recall, Human recall, confusion matrix; nếu benchmark paper báo binary F1 theo lớp, báo **thêm F1 tương ứng**.
- Edge: AUROC/AUPRC/prevalence, Brier/ECE, negative/valid counts, q distributions theo edit family/view.
- Robustness: Δ so clean, error increase, prediction flip-rate trên *paired text edits*, worst condition, Table 3 mean và Table 4 noise-only mean (không cộng Original vào Table 4 average).
- Báo **mỗi ô** Table 2/3/4, không chỉ average; paired Δ với cùng sample IDs và seeds; mean ± SD trên 5 seeds, bootstrap **theo document** cho uncertainty, không coi 25 noise views của cùng document là độc lập.
- Thời gian: source P1/P2 preprocessing, P3 GPU training, online inference từ raw text, inference từ cached graphs, peak disk/VRAM.

### 7.4. Stability ≠ classification importance: phép kiểm định cơ chế bắt buộc

**Không được giả định:** `high q → important for Human/AI classification`.

Trên **source validation**, đo *classification importance* của một tập con cạnh bằng cách che/tắt **một arc tại một thời điểm** và đo thay đổi classification loss/logit (với cùng normalization, re-compute degrees); group theo quantile q. Không dùng target test labels cho bước này.

Tạo bảng 2×2 hoặc continuous scatter:

| | Classification importance cao | Importance thấp |
|---|---|---|
| **Retention cao** | Cạnh vừa ổn định vừa hữu ích: tín hiệu PEARL kỳ vọng | Có thể là edge ổn định nhưng vô ích |
| **Retention thấp** | **Rủi ro PEARL:** down-weight cạnh hữu ích | Edge không ổn định và ít hữu ích |

Đo Spearman giữa q và importance, **report cả phân bố không mong đợi**. Đánh giá drift `S(G)↔S(G')` chỉ trên aligned subgraph và classification logit drift trên cả document. Tránh diễn giải sai: mức drift thấp chưa chứng minh accuracy tăng.

### 7.5. Tiêu chí quyết định nội bộ (không phải kết quả)

- **GO từ P0:** baseline parity đầy đủ.
- **GO từ P1:** provenance đúng; nhãn stable/changed đủ đa dạng cho source validation.
- **GO từ P2:** signal so priors hiện hữu, không q collapse, q on edited không bất thường nghiêm trọng; nếu thiếu thì dừng hoặc chạy đúng một diagnostic EditedView.
- **GO từ P3:** PEARL vs B-FT có source clean trong guardrail (ví dụ giảm tối đa 0.5 điểm %, **chỉ là quy tắc nhóm**) và source-structural validation không giảm; không chọn theo target.
- **GO công bố:** hiệu quả paired nhất quán; so prior/gate/augmentation fair; 5 seeds; bổ sung kiểm chứng dataset độc lập; các hạn chế OOD/weakness với Delete được báo trung thực.
- **NO-GO:** improvement chỉ do fine-tune, gán nhãn sai, q chỉ là prior, clean collapse, hoặc trade-off Table 4 khó chấp nhận.

---

## 8. Unit tests / integration tests: điều kiện nghiệm thu

1. `identity_edit`: tất cả arcs hợp lệ có `stable=1`.
2. `head_change` và `relation_change`: hai endpoint map duy nhất nhưng triple khác → `stable=0`.
3. `delete_endpoint`: relation label `unknown`, `valid=False`, không âm giả.
4. `repeat/insert`: UID mới không làm lệch UID gốc; split/merge token phải unknown.
5. `no_test_leak`: không có doc ID hoặc parent ID từ source test/target test trong train pairs.
6. `schema_parity`: metadata arc maps chính xác tới adjacency tensor indices, kể cả graph symmetric, root, self-loop, duplicate.
7. `epsilon1_parity`: same checkpoint+eval mode, layers/head outputs khớp baseline.
8. `weighted_norm`: recompute weighted degree đúng, không double norm/double self-loop, không cache normalized adjacency có q trainable.
9. `autograd`: `L_cls` gradient tới q khi `epsilon<1`; `L_edge` gradient tới q, không detach nhầm.
10. `view_training`: `q(G')` labels **chỉ** từ `G'→G''`, không dùng nhãn `G→G'`.
11. `serialization`: load/save checkpoint gồm GCN + predictor, feature schema, parser version; inference output parity sau reload.
12. `class_labels`: Human/AI convention và threshold không lật.
13. `matched_steps`: B-FT và PEARL có cùng GCN init, optimizer steps, batch sampler, scheduler; report thay đổi nếu không thể match chính xác.
14. `single_view_infer`: không yêu cầu edit log, origin text hoặc nhãn lớp.
15. `edge_predictor_sanity`: predictor overfit synthetic separable data; trên shuffled labels không tự nhiên outperform validation prior.
16. `no_silent_cache`: cache hash/version mismatches phải fail loud, không reuse.

**Chỉ `train_loss` giảm không đủ điều kiện nghiệm thu.** Agent phải báo danh sách tests pass/fail và hạn chế rõ ràng.

---

## 9. Các hướng mở rộng và điều cần tránh

### 9.1. Relation-retention vs edge-survival: tuyệt đối không trộn label

Định nghĩa mới **nếu thực sự nghiên cứu Delete**:

\[
s_e^{\mathrm{surv}}=\mathbf{1}[\text{arc }e\text{ vẫn tồn tại dưới mapping phù hợp sau edit}],
\]

có thể cho `0` khi endpoint bị xóa **và provenance chắc chắn**. Nhưng `survival` thay đổi cả event space và xác suất mục tiêu; nó không còn cùng đại lượng với `s_rel` vốn **điều kiện cả hai endpoint còn**.

- Nếu muốn thêm `survival`, dùng **một predictor head riêng** hoặc một thí nghiệm tách biệt; không gán Delete endpoint=0 vào `L_rel`.
- Nếu T chọn xóa ngẫu nhiên theo token, survival có thể phụ thuộc mạnh vào độ dài/degree và chẳng liên quan đến nhãn Human/AI; không mặc nhiên tin sẽ tăng Delete.
- Trên test Delete, edges đã mất **không thể nhận w và không thể được phục hồi** bằng PEARL. Chỉ những cạnh còn lại được reweight.
- Phải có separate ablation `relation only` vs `survival only` vs `both` nếu nhắm phát triển thành paper; **không** thuộc PEARL-Lite.

### 9.2. Không tăng module khi thiếu bằng chứng

- Không auto-add CERE, CR-MUSE, R-Drop, EMA mới, GAT/RGCN, hierarchical graph, semantic gate, auxiliary sentence loss.
- Không âm thầm đổi lexical embedding/subword alignment: lịch sử V2 cho thấy có thể nâng Table 3 nhưng phá Table 4.
- Không dùng target-domain labels để chọn hyperparameter hoặc chỉnh threshold; Table 3 đã bị nhóm quan sát nhiều lần, nên claim mới cần độc lập hơn.
- **Không coi q là parser confidence** hoặc edge label là nhãn “quan hệ đúng/sai”. PEARL chỉ học *retention under training edit distribution Q*.

---

## 10. Tổ chức mã và CLI tham khảo

> Đường dẫn dưới đây là **đề xuất**, không xác nhận repository hiện có những tệp này. Agent phải audit để thích nghi API.

```text
pearl/
  __init__.py
  edit_provenance.py      # deterministic UID/span edit logs
  alignment.py            # reliable one-to-one token mapping
  relation_retention.py   # stable/changed/unknown labels
  edge_features.py        # inference-available features
  edge_predictor.py       # q logits
  weighted_gcn.py         # preserve baseline op, w and normalization
  train_edge.py           # P2-A predictor and priors
  train_gcn.py            # B-FT + PEARL matched fine-tune
  evaluate.py             # reuse official metrics and conditions
  diagnostics.py          # source view shift + edge importance
  edited_view.py          # optional P2-B extension (G→G'→G'')
  README.md

tests/
  test_pearl_alignment.py
  test_pearl_labels.py
  test_pearl_weighted_gcn.py
  test_pearl_training.py
  test_pearl_edited_view.py
```

CLI **mẫu sau khi agent implement**:

```bash
# P0: audit baseline and epsilon=1 operator parity
python -m pearl.diagnostics --stage p0 --seed 2025

# P1: generate edit-provenance and retention labels (source only)
python -m pearl.edit_provenance --split train --seed 2025 --audit-only

# P2: predictor plus constant/relation/POS priors
python -m pearl.train_edge --seed 2025 --output runs/pearl/p2_seed2025

# P3-control: same initial GCN checkpoint, matched fine-tuning
python -m pearl.train_gcn --variant baseline_ft --seed 2025 \
  --output runs/pearl/bft_seed2025

# P3-main: PEARL-Lite
python -m pearl.train_gcn --variant pearl_lite --seed 2025 \
  --epsilon 0.8 --lambda-edge 0.1 \
  --output runs/pearl/lite_seed2025

# run tests before benchmarks
python -m pytest tests/test_pearl_* -q

# final evaluation only after source-only configuration lock
python -m pearl.evaluate --checkpoint runs/pearl/lite_seed2025/best.pt \
  --benchmark hc3 --conditions original,syn5,syn10,syn20,syn30
```

**Bắt buộc báo lại:** file changed, tests pass/fail, q metrics, baseline parity, GCN-FT/PEARL paired source val, peak disk/VRAM và những file/code chưa thể audit. **Không tự xưng đã chạy CLI mẫu nếu chưa tạo được lệnh thực.**

---

## 11. Quản lý Kaggle ~20 GB

- Không lưu toàn bộ edit variants dưới dạng full RoBERTa embedding. **P1/P2 chỉ cần parser metadata + UID + `s_rel/M` + graph clean dùng chung**.
- Dùng stream/generate per document hoặc one-family-at-a-time; sidecar lưu compressed Parquet/JSONL, checksum và parser version. Nếu re-parse quá tốn CPU, cache **một** edited parse nhỏ, xóa sau khi đã tạo labels.
- Chỉ giữ **tối đa ~4 graph `.pkl` lớn cùng lúc**, theo hạn chế nhóm; đánh giá Table 2→3→4 tuần tự, xóa cached graphs sau khi lưu metrics/checksums.
- Khi thử P2-B, cần `G'`, `G''` **parse structures**, chưa cần tạo full embeddings cho chúng nếu classifier clean-only. Giới hạn docs và tỷ lệ edit để giảm thời gian.
- Reuse deterministic splits/raw graph giữa seeds; source edit seeds độc lập nhưng kiểm soát được. Theo dõi preprocess seconds, training seconds, inference seconds, peak VRAM/disk.
- Chạy pilot seed 2025 → paired replicate seed 2026 → chỉ mở 5 seeds khi source-validation cho tín hiệu.

---

## 12. Checklist giao việc cho Codex (dùng nguyên văn nếu cần)

> **Nhiệm vụ:** Đọc đầy đủ file `Huong_dan_implement_PEARL_GCN_v2.md`, file nghiên cứu `PEARL_CoSoLyLuan_ThietKe_Va_KeHoachThucNghiem.md` và audit repository PRDetect thật. Implement **PEARL-Lite trên GCN gốc** theo thứ tự **P0→P1→P2→P3**, nhưng **giai đoạn giao việc đầu chỉ cần P0/P1/P2 + unit tests + source-validation edge predictor report**. Không tự giả định file/class/API đã tồn tại. Giữ baseline GCN, V1/V3, PRISM/LISA và test generator nguyên trạng. Ghi directed arc metadata, đảm bảo provenance 1:1 thật, gán `s_rel` 1/0 chỉ khi cả hai endpoint có mapping xác thực; `unknown` không gán 0. Predictor input chỉ gồm features tại inference, không có edit labels/provenance. So constant/relation(+POS) priors và báo q calibration trên clean/edited source validation. **Không chạy P3 lớn nếu nhãn không đủ chất lượng hoặc predictor không có tín hiệu.** Nếu tới P3, phải thêm `B-FT` fine-tune từ **cùng checkpoint, cùng optimizer steps/selection rule** để kiểm soát extra training, và test `epsilon=1` parity + weighted normalization + autograd. Không thêm EditedView/Survival/anchor/augmentation/other backbones vào main path nếu chưa có bằng chứng. Báo các files changed, cách chạy thực tế, tests pass/fail, kết quả source validation và giới hạn còn tồn tại. Không chạy full official benchmark ở nhiệm vụ implement đầu tiên.

---

## 13. Tính mới, nguồn và cách diễn giải khoa học

### 13.1. Ranh giới claim

**Đã biết:** PRDetect dùng cú pháp dependency graph/GCN; GAT và GNNGuard đã nghiên cứu edge weighting; perturbation/augmentation và consistency learning không phải đóng góp mới tự thân.  
**Ứng viên đóng góp:** Nhãn `retention` được tạo bằng **edit provenance + dependency-arc matching**, huấn luyện **single-view predictor** rồi truyền sang GCN như trọng số cạnh cho MGT detection.  
**Chưa chứng minh:** Sự độc đáo tuyệt đối trong literature 2023–2026, khả năng thắng tất cả các ô Table 2/3/4, tính ổn định trên các nguồn/phân phối edit khác, hoặc kết quả causal về phân loại.

Nếu P2 không vượt priors hoặc P3 không vượt matched B-FT/B-prior, đó là **kết quả âm**, không thay đổi diễn giải thành “PEARL đã cải thiện robustness”.

### 13.2. Tham khảo

- Li et al. (2025), [PRDetect: Perturbation-Robust LLM-generated Text Detection Based on Syntax Tree](https://aclanthology.org/2025.findings-naacl.464/), *Findings of NAACL 2025*. File PDF nhóm cung cấp.
- `Context_LichSuCaiTien_PRDetect(1).md`: lịch sử cải tiến nội bộ, ghi rõ kết quả âm và các vấn đề baseline/structural mapping.
- `PEARL_CoSoLyLuan_ThietKe_Va_KeHoachThucNghiem.md`: phân tích nền tảng, giới hạn và triển khai ý tưởng PEARL (tài liệu đề xuất, không phải paper đã được bình duyệt).
- Kipf & Welling (2017), *Semi-Supervised Classification with Graph Convolutional Networks*, ICLR 2017.
- Veličković et al. (2018), *Graph Attention Networks*, ICLR 2018.
- Zhang & Zitnik (2020), *GNNGuard: Defending Graph Neural Networks against Adversarial Attacks*, NeurIPS 2020.
- Jin et al. (2020), *Graph Structure Learning for Robust Graph Neural Networks*, KDD 2020.
- Xie et al. (2020), *Unsupervised Data Augmentation for Consistency Training*, NeurIPS 2020.

### 13.3. Kế hoạch viết paper nếu kết quả dương

1. **Finding (phải đo):** Retention khác nhau đáng kể giữa quan hệ/ngữ cảnh và có liên hệ với class-relevant edge importance.
2. **Method:** Provenance-based retention labels → single-view reliability → weighted-GCN; training không cần target labels.
3. **Mechanism:** Matched baselines + q calibration + adjacency/logit drift + class importance; giải thích cả trường hợp q cao nhưng cạnh vô ích.
4. **Results:** Paired Table 2/3/4, 5 seeds, interval và held-out confirmation, time/memory cost.
5. **Limitations:** Không khôi phục deleted token; không đảm bảo cải thiện cross-domain; train edit distribution Q có thể khác phân phối thực; prior art chưa khảo sát hệ thống hoàn chỉnh.

---

## 14. Nhật ký thay đổi từ bản hướng dẫn ban đầu

| Thay đổi | Vì sao sửa | Ảnh hưởng đến triển khai |
|---|---|---|
| Khóa PEARL-Lite là main path | Giữ đúng **một** giả thuyết nghiên cứu | Giảm module, giảm confounds |
| Bổ sung **B-FT** matched fine-tuning | Không quy nhầm lợi ích của extra training cho PEARL | **Bắt buộc** ở P3 |
| Bổ sung **B-uniform** | Constant q vẫn đổi normalization ở epsilon<1 | Kiểm tra lợi ích của q học thật |
| Clarify relation retention vs survival | Delete endpoints không phải negative relation | Tránh nhãn supervision sai |
| Bổ sung EditedView theo cơ chế **G'→G''** | Forward labels `G→G'` không giám sát được cạnh của G' | Chỉ ablation có điều kiện |
| Đưa edge-importance check thành tiêu chí cơ chế | Stable edge không đồng nghĩa discriminative edge | Giảm nguy cơ claim causal sai |
| Chốt protocol source-only | Tránh target-test tuning trên Table 3 | Tăng độ tin cậy đánh giá |
| Bổ sung q on edited/source-val + GO/NO-GO | Phát hiện clean-to-edited shift trước chạy GPU lớn | Tiết kiệm Kaggle |
| Làm rõ baseline parity và classifier outputs | Tránh đổi operator/đảo nhãn/sigmoid hai lần | Ngăn bug khó phát hiện |
| Giữ file path/CLI là giao diện dự kiến | Chưa có repository thật để map code | Agent phải audit trước |

**Quyết định kỹ thuật cuối:** **triển khai PEARL-Lite từ GCN gốc trước**, dùng `relation retention` có mask, không thay graph topology/embedding, có `B-FT` và `B-uniform` đối chứng. **Không đưa EditedView hay Survival vào main run nếu chưa xuất hiện bằng chứng cụ thể về failure mode.**
