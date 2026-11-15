# PEARL: Perturbation-Aligned Edge Reliability Learning

## Cơ sở lý luận, thiết kế thuật toán và kế hoạch kiểm chứng trên PRDetect

**Phiên bản:** Bản đề xuất nghiên cứu, 14/11/2026  
**Đối tượng:** Phát hiện văn bản Human/AI từ dependency graph; áp dụng trước hết cho GCN gốc PRDetect, sau đó kiểm tra trên RGCN V1.  
**Mục tiêu:** Hướng tới cải thiện Table 2, Table 3, Table 4 của PRDetect, đặc biệt chiều HC3 → GPT3.5-Mixed, trong khi không làm sụp robustness với Insert/Repeat/Delete.  
**Trạng thái:** **ĐỀ XUẤT CHƯA THỰC NGHIỆM.** Không có kết quả Accuracy/F1/AUC nào trong tài liệu là kết quả đã chạy PEARL. Thuật ngữ *PEARL* là tên làm việc của đề xuất, không được khẳng định là một công trình độc lập đã công bố hoặc một thuật toán có tính mới đã được xác nhận.

> **Luận điểm trung tâm:** PRDetect tận dụng cấu trúc cú pháp để chống nhiễu, nhưng hiện trao ảnh hưởng cho các cạnh của dependency graph gần như đồng đều. Khi văn bản bị chỉnh sửa, độ ổn định của từng quan hệ cú pháp không đồng nhất. Nếu ta học được **xác suất một cạnh giữ được mối quan hệ cú pháp khi văn bản bị chỉnh sửa** từ các cặp văn bản căn chỉnh ở *training*, ta có thể ưu tiên luồng thông tin đi qua quan hệ ổn định trong *message passing* ở *inference*. Đây là **giả thuyết kiểm chứng**, không phải định lý rằng mọi nhiễu hoặc mọi domain shift sẽ được khắc phục.

---

## Mục lục

1. [Vấn đề và câu hỏi nghiên cứu](#1-vấn-đề-và-câu-hỏi-nghiên-cứu)
2. [Dữ liệu gốc và chẩn đoán điểm yếu](#2-dữ-liệu-gốc-và-chẩn-đoán-điểm-yếu)
3. [Các nền tảng khoa học](#3-các-nền-tảng-khoa-học)
4. [Định nghĩa chính thức PEARL](#4-định-nghĩa-chính-thức-pearl)
5. [Mô hình GCN + PEARL](#5-mô-hình-gcn--pearl)
6. [Các hàm loss và lịch huấn luyện](#6-các-hàm-loss-và-lịch-huấn-luyện)
7. [Lập luận toán học về ổn định](#7-lập-luận-toán-học-về-ổn-định)
8. [Mục tiêu trên ba benchmark](#8-mục-tiêu-trên-ba-benchmark)
9. [Ablation và protocol đánh giá](#9-ablation-và-protocol-đánh-giá)
10. [Pseudocode và kiến trúc triển khai](#10-pseudocode-và-kiến-trúc-triển-khai)
11. [Rủi ro, giả thuyết có thể bác bỏ và tiêu chí dừng](#11-rủi-ro-giả-thuyết-có-thể-bác-bỏ-và-tiêu-chí-dừng)
12. [Tính mới và khả năng phát triển thành bài báo](#12-tính-mới-và-khả-năng-phát-triển-thành-bài-báo)
13. [Tài liệu tham khảo đã đối chiếu](#13-tài-liệu-tham-khảo-đã-đối-chiếu)

---

## 1. Vấn đề và câu hỏi nghiên cứu

### 1.1. Mô hình xuất phát

PRDetect ([R1], Mục 3, trang 8307–8308) xây dựng dependency graph bằng spaCy, khởi tạo biểu diễn node từ RoBERTa, học biểu diễn với **GCN hai lớp** và huấn luyện bằng **Binary Cross-Entropy**. Cạnh trong công thức GCN truyền tải quan hệ tồn tại/không tồn tại, không sử dụng trực tiếp nhãn quan hệ cú pháp trong phép nhân ma trận. Bài báo dùng synonym replacement bằng WordNet chủ yếu ở bước **kiểm thử**, không phải một chương trình học độ tin cậy cạnh.

Với node embedding ban đầu $X\in\mathbb{R}^{n\times d}$ và ma trận kề $A$, GCN cổ điển dùng:

$$
\hat A=A+I,\qquad S=\hat D^{-1/2}\hat A\hat D^{-1/2},
$$

$$
H^{(1)}=\sigma(SXW_0),\qquad H^{(2)}=\sigma(SH^{(1)}W_1).
$$

Khi biến đổi từ ngữ làm parser đổi quan hệ, cả $A$ và thông tin được truyền dọc theo các cạnh đều có thể thay đổi. Điểm quan trọng: **sự thay đổi này không đồng nghĩa mọi cạnh thay đổi đều sai**. Một đoạn chỉnh sửa hợp lệ có thể tạo quan hệ mới đúng và hữu ích. Do đó, PEARL không tự nhận là bộ *sửa lỗi parser*.

### 1.2. Câu hỏi khoa học chính

> **RQ:** Liệu việc dự đoán và sử dụng *edit-conditional syntactic edge stability* — độ ổn định của quan hệ cú pháp được học từ các cặp văn bản gốc–đã chỉnh sửa — có tăng robustness và khả năng chuyển miền cho bộ phát hiện Human/AI tốt hơn GCN, attention và consistency thông thường, trong khi vẫn giữ clean accuracy?

Các câu hỏi phụ:

- **RQ1 — Measurability:** Có thể tạo nhãn ổn định cạnh đáng tin cậy từ căn chỉnh hai bản văn mà **không dùng nhãn test** hay không?
- **RQ2 — Mechanism:** Các cạnh được dự đoán ổn định có thực sự làm giảm biến thiên biểu diễn khi perturbation xảy ra không?
- **RQ3 — Discriminativeness:** Những cạnh này có giữ được khả năng phân biệt Human/AI không, hay chỉ *ổn định nhưng vô dụng*?
- **RQ4 — Generalization:** Độ ổn định trước word edit có chuyển thành lợi ích trong HC3 → GPT3.5-Mixed không, dù hai domain có quy tắc khác nhau?
- **RQ5 — Robustness trade-off:** PEARL có tạo ra hiện tượng giống V2 (cross-domain tăng nhưng Insert/Delete giảm mạnh) không?

### 1.3. Một đóng góp, không ghép nhiều module thành nhiều đóng góp

Đóng góp đề xuất là **một cơ chế có hai phần liên kết bắt buộc**:

1. Sinh *supervision* ổn định quan hệ cú pháp bằng căn chỉnh trước–sau chỉnh sửa trên dữ liệu **train**.
2. Sử dụng bộ dự đoán này để cấp trọng số cho cạnh trong GCN/RGCN khi inference chỉ có **một** văn bản.

Đây là một khối *perturbation-aligned edge reliability* thống nhất, chứ không phải “attention + đổi loss + augmentation” được coi là ba đóng góp khác nhau.

---

## 2. Dữ liệu gốc và chẩn đoán điểm yếu

### 2.1. Table 2 — In-domain và synonym

| Dataset / PRDetect paper | 0% | 5% | 10% | 20% | 30% |
|---|---:|---:|---:|---:|---:|
| HC3 | 98.78% | 98.78% | 98.72% | 98.74% | 98.64% |
| GPT3.5-Mixed | 96.56% | 96.30% | 96.32% | 96.56% | 96.38% |

Nguồn: [R1], Table 2, trang 8310. Đây là **số tác giả**, không phải checkpoint GCN nhóm tự train. HC3 đã gần bão hòa; một thay đổi làm tăng cross-domain nhưng giảm đáng kể HC3 clean khó được xem là chiến thắng đồng thời.

### 2.2. Table 3 — Cross-dataset

| Train → Test / PRDetect paper | Original | Syn 5% | Syn 10% | Syn 20% | Syn 30% |
|---|---:|---:|---:|---:|---:|
| HC3 → GPT3.5-Mixed | 73.2% | 73.0% | 72.6% | 71.7% | 70.9% |
| GPT3.5-Mixed → HC3 | 87.7% | 87.8% | 87.3% | 86.9% | 87.0% |

Nguồn: [R1], Table 3, trang 8311. Hiệu năng bất đối xứng theo chiều chuyển miền.

Table 1 của paper còn cho thấy độ dài văn bản trung bình **HC3 Human 147.93, AI 178.65** nhưng **GPT3.5-Mixed Human 756.55, AI 501.13**; chênh lệch đảo chiều. Đây là bằng chứng **có thể tồn tại shortcut theo đặc trưng miền/độ dài**, không phải chứng minh PEARL sẽ giải quyết được shortcut đó. Phụ lục Table 8 báo cáo mô hình vẫn đạt accuracy cao khi matching độ dài, nên không thể quy mọi vấn đề cho độ dài.

### 2.3. Table 4 — Nhiễu cấu trúc 25% trên HC3

| Điều kiện / PRDetect paper | Accuracy |
|---|---:|
| Original | 98.50% |
| Insert | 98.30% |
| Repeat | 98.20% |
| Replace | 79.80% |
| Delete* | 74.70% |
| **Mean bốn nhiễu** | **87.75%** |

Nguồn: [R1], Table 4, trang 8311. *Lưu ý bản in ghi `Detect` ở hàng cuối, trong khi phần 5.5 mô tả thao tác `Delete`; phải đối chiếu scripts gốc để tái lập chính xác.*

**Từ dữ liệu có thể kết luận:** Replace/Delete yếu hơn Insert/Repeat. **Chưa thể kết luận:** các lỗi Replace/Delete chủ yếu do sai cấu trúc graph, hay do mất mát biểu diễn từ RoBERTa, thiếu thông tin, lệch xác suất đầu ra hoặc kết hợp các nguyên nhân. PEARL chỉ nhắm trực tiếp vào nhánh *edge-mediated message passing*.

### 2.4. Kết quả âm và hạn chế từ lịch sử nhóm

Nguồn: [R2] `Context_LichSuCaiTien_PRDetect(1).md`, cập nhật 12/11/2026.

- **RGCN V1** checkpoint cũ có cải thiện rõ, nhưng **không tái lập ổn định** khi train lại. Không lấy số cũ làm bằng chứng nhân quả.
- **V2 aligned RoBERTa BPE**: cross GPT clean đạt **86.90%** ở một ablation nhưng HC3 Insert **51.70%**, Delete **51.60%**, Repeat **69.60%**; tránh thay pipeline embedding ngay vòng đầu.
- **V2 hybrid sequence gate** có xu hướng gần 1.0 cho nhánh sequence; không làm lại hybrid sequence–graph như đề xuất mới.
- **V3 CMixStyle** giảm Table 3 **0.68 điểm phần trăm** và Table 4 **0.70 điểm phần trăm** trong cặp thí nghiệm được báo cáo.
- **V3 same-label syntax swap** giảm Table 3 **1.42 điểm phần trăm**; **semantic/structural soft gates** giảm **1.58 điểm phần trăm**; không lặp hai cơ chế này mà coi là novelty.
- **V3** cross-domain có AUC khoảng **91.5%** nhưng recall AI khoảng **34.72%**, human khoảng **98.46%** trong một cấu hình; chỉ sửa threshold không đủ để đảm bảo không làm hỏng các tập khác.

**Kết luận thiết kế:** Đưa PEARL lên **GCN gốc** trước nhằm tránh đồng thời thay embedding, topology, pooling, scheduler. Sau đó chạy trên **RGCN V1 được retrain paired** để kiểm tra tính linh hoạt của ý tưởng.

---

## 3. Các nền tảng khoa học

### 3.1. Graph neural networks nhạy với biến đổi cạnh

GNN dùng adjacency để điều khiển đường truyền thông tin. Chỉ cần thay $A$ thành $A'$ thì toán tử lan truyền $S$ thay thành $S'$; biểu diễn cuối thường đổi ngay cả khi $X$ giữ nguyên. Pro-GNN [R5] và GNNGuard [R4] đều nghiên cứu cấu trúc bị nhiễu và khai thác ý tưởng cải thiện cấu trúc/độ tin cậy cạnh để tăng robustness. Điều này tạo **cơ sở hợp lý** cho can thiệp tại cạnh nhưng **không chứng minh PEARL là mới**.

### 3.2. Không thể gọi edge weighting thông thường là đóng góp mới

GAT [R3] đã học trọng số các node hàng xóm. GNNGuard [R4] đã ước lượng tầm quan trọng cạnh và giảm tác hại của các quan hệ không phù hợp. RGCN [R6] đã phân biệt loại quan hệ. Vì vậy:

- **Không mới:** thêm một MLP dự đoán $q_{ij}$ rồi nhân adjacency.
- **Không mới:** thêm attention vào dependency graph.
- **Không mới:** dùng consistency loss giữa hai augmentation.
- **Ứng viên mới cần chứng minh:** xây dựng *edge-retention supervision* từ **token-level provenance alignment trên hai văn bản thật sự được chỉnh sửa**, rồi chuyển học vấn đề này sang **single-view edge reliability** dùng trong phát hiện Human/AI, có cơ chế kiểm tra riêng so với GAT/GNNGuard và các baseline robustness.

Tính mới vẫn phải qua rà soát đầy đủ các công trình gần chủ đề, đặc biệt ở 2025–2026; nghiên cứu tài liệu ở đây không phải bảo chứng về prior art.

### 3.3. Consistency learning là cơ sở phụ trợ, không phải novelty

Unsupervised Data Augmentation (UDA, [R7]) thể hiện ý tưởng khuyến khích dự đoán ổn định dưới phép augment hợp lệ. Nhưng PEARL khác ở **đơn vị giám sát**: UDA chủ yếu nhằm bất biến dự đoán, còn PEARL lấy nhãn **độ ổn định từng quan hệ cú pháp** để học cách định tuyến thông tin qua graph.

Có thể thêm consistency loss tùy chọn sau khi module chính đã chứng minh hiệu quả, nhưng **không bắt buộc** cho PEARL phiên bản tối giản; nên ablate độc lập để không che nguyên nhân.

### 3.4. Adversarial training không còn là khoảng trống trống rỗng

GREATER [R8] (ACL 2025) đã nghiên cứu adversarial training cho phát hiện văn bản tạo bởi máy. Vì vậy PEARL cần được so sánh với ít nhất một baseline **GCN + cùng augmentation**, chứ không được quy mọi lợi ích do train trên dữ liệu nhiễu thành công trạng của module mới.

### 3.5. Giả thuyết thống kê có điều kiện

Đặt $T\sim\mathcal{Q}$ là phép chỉnh sửa có kiểm soát trên **train** và $e$ là một cạnh phụ thuộc cú pháp. Định nghĩa:

$$
q^\star(e,G) = \Pr_{T\sim\mathcal{Q}}\left[\operatorname{retain}(e;G,G_T)=1 \mid \phi(e,G)\right].
$$

PEARL xấp xỉ $q^\star$ bằng $q_\phi(e,G)$ từ đặc trưng **chỉ có ở inference**. Trực giác: nếu một quan hệ thường biến mất dưới các chỉnh sửa vô hại, thông điệp đi qua nó có thể ít bền vững hơn.

**Điều kiện cần nhưng chưa đủ:** Các quan hệ có độ ổn định cao phải vẫn mang thông tin phân biệt nhãn Human/AI. Nếu chỉ ổn định mà không phân biệt, reweighting có thể làm giảm accuracy.

---

## 4. Định nghĩa chính thức PEARL

### 4.1. Đối tượng dữ liệu

Đối với mỗi mẫu train $(x,y)$:

- $G=(V,E)$ là dependency graph của $x$;
- $x'=T(x)$ là biến thể được tạo **chỉ từ tập train** bằng một phép edit có nhật ký thao tác;
- $G'=(V',E')$ là parse của $x'$;
- $m:V\supseteq U\to V'$ là ánh xạ căn chỉnh các token còn tồn tại. **Không dùng vị trí token sau khi xóa/chèn làm ID token bất biến**.

Mỗi cạnh gốc gồm $(u,v,r)$: token phụ thuộc $u$, head $v$, nhãn dependency relation $r$. Nhãn `r` được dùng để tạo supervision, dù backbone GCN ban đầu không có weight matrix riêng cho từng relation.

### 4.2. Thu thập token provenance

**Ưu tiên:** tạo perturbation trong code có log thao tác và giữ `token_uid` ổn định xuyên suốt hai bản văn. Ví dụ:

```text
Origin:   The[1] careful[2] doctor[3] examined[4] the[5] patient[6]
Synonym:  The[1] cautious[2] doctor[3] examined[4] the[5] patient[6]
Insert:   The[1] very[new-7] careful[2] doctor[3] examined[4] ...
Delete:   The[1] doctor[3] examined[4] the[5] patient[6]
```

Nếu spaCy và generator sử dụng tokenization khác nhau, cần ánh xạ `token_uid → span ký tự → spaCy token(s)`; không tự động giả định 1:1. Những trường hợp **split/merge token**, ambiguous many-to-many hoặc parser re-segmentation phải đánh dấu `unknown` để không tạo nhãn giám sát sai. Không dùng edit-distance theo chữ đơn thuần khi có từ lặp.

### 4.3. Nhãn ổn định cạnh

Với cạnh $e=(u,v,r)\in E$ có cả hai đầu mút căn chỉnh được:

$$
s_e = \mathbb{1}\big[\big(m(u),m(v),r\big)\in E'\big].
$$

Nếu $u$ hoặc $v$ **bị xóa** hoặc không căn chỉnh được chắc chắn: đặt `mask_e=0`, không ép $s_e=0$ bởi vì đây có thể là lỗi thiếu thông tin thay vì cạnh không bền. Để huấn luyện trên graph nhiễu, có thể gán *weak negatives* cho cạnh mới có một đầu mút được chèn, nhưng đây là **biến thể thử nghiệm** với weight thấp, không được tự gán là ground truth đúng/sai parser.

Tương tự với cạnh của $G'$ được ánh xạ ngược về $G$, có thể tạo nhãn hai chiều cho các đầu mút còn theo dõi được. Sử dụng supervision hai chiều chỉ khi nhãn/căn chỉnh chắc chắn.

**Thực tế phải phân biệt ba trạng thái:**

| Trạng thái | Ý nghĩa | Đưa vào $L_{edge}$? |
|---|---|---|
| `stable` | Cặp đầu mút, hướng và relation được giữ | Có, nhãn 1 |
| `changed` | Hai token tồn tại nhưng head hoặc relation thay đổi | Có, nhãn 0 |
| `unknown` | Token mất, nhiều-đối-một, khó căn chỉnh | **Không** |

Nếu hướng cạnh không được lưu đầy đủ trong graph builder GCN gốc, cần lưu metadata phụ thuộc từ spaCy **bên cạnh** adjacency gốc; không thay cấu trúc truyền thông tin của baseline mà không có paired control.

### 4.4. Tính toán reliability từ *single view*

Cho đặc trưng của cạnh $e=(u,v,r)$:

$$
a_e = [\operatorname{Emb}(\mathrm{POS}_u)\|\operatorname{Emb}(\mathrm{POS}_v)\|\operatorname{Emb}(r)\|\operatorname{bucket}(\Delta\mathrm{position})\|\operatorname{local\_topology}(e)].
$$

Có thể bổ sung phép chiếu nhỏ của node feature **đóng băng** từ RoBERTa cho biến thể nâng cao, nhưng phiên bản đầu tiên chỉ dùng POS/dependency/distance/local degree để không lặp rủi ro V2 với alignment embedding. Feature được dùng tại inference **không chứa**: edit type, `token_uid`, nhãn `stable/changed`, nhãn lớp, source dataset hoặc thông tin bản gốc.

$$
q_\phi(e,G)=\operatorname{sigmoid}(\operatorname{MLP}_\phi(a_e))\in(0,1).
$$

**Ý nghĩa:** $q$ ước lượng tần suất một quan hệ tương tự được bảo toàn dưới phân phối biến đổi $\mathcal{Q}$. Nó **không** có khả năng phát hiện chắc chắn rằng một cạnh ở test “đã bị sửa”. Nếu $\mathcal{Q}$ khác nhau mạnh giữa train và test, $q$ có thể sai lệch.

### 4.5. Vấn đề chuyển từ relation-directed sang GCN

Để giữ backbone chuẩn, ta sử dụng `q` như *scalar edge multiplier*, không đổi tensor feature $X$ hay số tầng. Với GCN cần kiểm tra thực thi gốc dùng directed adjacency hay undirected/symmetrized edges. Trọng số cạnh phải được chèn theo **cùng hướng và cách chuẩn hóa** của baseline; nếu muốn chuyển từ directed sang symmetric, đó là một ablation riêng.

Không tùy tiện suy luận ID quan hệ từ điều kiện `r < R/2`: lịch sử V3 ghi rõ lỗi giả định forward/reverse relation ID khi R=51. Khi cần relation typed, lấy nhãn dependency và hướng từ graph-builder metadata, không từ heuristic chia đôi ID.

---

## 5. Mô hình GCN + PEARL

### 5.1. Kiến trúc tối giản

```text
TRAIN
  Raw text x ------------------------------┐
       |                                   |
       v                                   v
  spaCy parse G                    Perturb with edit log
       |                                   |
       |                                   v
       |                                x' -> parse G'
       |                                   |
       +---- Token provenance alignment --+
                         |
                         v
               Edge retention labels s_e
                         |
                         v
         Reliability predictor q_phi(edge, G)
                         |
                         v
             Weighted adjacency S_phi(G)
                         |
                         v
               PRDetect 2-layer GCN
                         |
                         v
             Original classifier head
                         |
                   BCE + Edge BCE

INFERENCE
  One text -> spaCy graph -> q_phi from current graph
           -> weighted adjacency -> same GCN -> Human/AI
```

### 5.2. Công thức weighted adjacency

Đặt $\epsilon\in(0,1]$ là hệ số giữ thông điệp nền. Với cạnh có thật của graph:

$$
w_{uv}=\epsilon+(1-\epsilon)q_\phi(u,v,G).
$$

Nếu GCN dùng đồ thị vô hướng, hợp nhất trọng số ở hai chiều theo một quy tắc nhất quán, ví dụ $w_{uv}=w_{vu}$ từ cùng một dependency arc. Tạo adjacency có trọng số $A_w$, bổ sung self-loop trọng số 1 và chuẩn hóa:

$$
\tilde A = A_w+I,\quad \tilde D_{ii}=\sum_j\tilde A_{ij},\quad S_\phi=\tilde D^{-1/2}\tilde A\tilde D^{-1/2}.
$$

$$
H^{(1)}_\phi=\sigma(S_\phi XW_0),\qquad
H^{(2)}_\phi=\sigma(S_\phi H^{(1)}_\phi W_1).
$$

Readout và classification head ban đầu **giữ nguyên chính xác theo code baseline** trong paired experiments. Trọng số $w\in[\epsilon,1]$ tránh triệt tiêu đột ngột các cạnh quan trọng và bảo toàn một phần inductive bias của PRDetect.

**Quan trọng:** $\epsilon=1$ cho $w=1$ với mọi cạnh — đây là một điểm kiểm tra sanity cho việc PEARL **phải** tái hiện toán tử baseline đến sai số số học khi các phần khác giống nhau. Nếu không tái hiện, có bug ở preprocessing, graph direction, normalization, dropout hoặc checkpoint.

### 5.3. Với RGCN V1

Giữ metadata quan hệ $r$ và chỉ thêm scalar reliability vào từng cạnh:

$$
h_i^{(l+1)}=\sigma\left(W_0^{(l)}h_i^{(l)} +\sum_{r\in\mathcal R}\sum_{j\in\mathcal N_r(i)}\frac{w_{ij}^{(r)}}{Z_{i,r}}W_r^{(l)}h_j^{(l)}\right).
$$

Cần định nghĩa $Z_{i,r}$ **cùng kiểu chuẩn hóa** của V1 đối chứng (count hoặc sum weight theo relation). Nếu đổi normalization đồng thời sẽ có confound. Lúc đầu cho PEARL-GCN chứng minh lợi ích độc lập; khi chuyển sang V1, train paired lại từ đầu.

### 5.4. Hai biến thể để làm ablation khoa học

- **PEARL-Lite:** chỉ có edge stability supervision, $q$, weighted adjacency; giữ BCE giống baseline. Đây là **đề xuất chính**.
- **PEARL + anchor consistency (mở rộng tùy chọn):** thêm bất biến biểu diễn cho node còn căn chỉnh và thuộc subgraph ổn định. Chỉ làm **sau** khi chứng minh PEARL-Lite có hiệu quả; không được che đi việc PEARL-Lite có thể không tăng Table 3.

---

## 6. Các hàm loss và lịch huấn luyện

### 6.1. Classification loss

Theo [R1], label **1 = Human**, **0 = AI**. Với đầu ra $p_\theta(x)$:

$$
L_{cls}(x,y)= -y\log p_\theta(x)-(1-y)\log(1-p_\theta(x)).
$$

Vòng đầu ưu tiên BCE trên **clean** để đánh giá riêng giá trị của edge supervision. Có thể thêm BCE trên **synonym** sau khi xác nhận ý nghĩa và nhãn Human/AI được giữ. Với các phép random Insert/Delete 25%, không mặc nhiên gán nhãn chính xác theo ý nghĩa và training class loss vì có thể phá hỏng semantics; có thể dùng chúng chỉ tạo các nhãn stability và kiểm tra độ nhạy.

### 6.2. Edge retention BCE

Với tập cạnh có nhãn hợp lệ $\mathcal E_{valid}$:

$$
L_{edge}=\frac{1}{\sum_{e\in\mathcal E_{valid}}\alpha_e}
\sum_{e\in\mathcal E_{valid}}\alpha_e\Big[-s_e\log q_e-(1-s_e)\log(1-q_e)\Big].
$$

$\alpha_e$ có thể cân bằng hai lớp stable/changed trong **train**. Không tính trên test. Nếu hầu hết cạnh đều stable, một mạng đoán $q\approx 1$ có thể có accuracy edge cao nhưng không mang thông tin; vì thế báo cáo **AUROC/AUPRC/F1 edge-level, ECE/Brier, entropy và phân bố q theo type**, không chỉ edge accuracy.

### 6.3. Tổng loss tối giản

$$
\boxed{L_{PEARL}=L_{cls}+\lambda_e L_{edge}.}
$$

Khuyến nghị chưa đo: thử $\lambda_e\in\{0.05,0.1,0.2\}$ trên validation nguồn; $\epsilon\in\{0.6,0.8,1.0\}$. $\epsilon=1$ là kiểm tra/đối chứng, không phải ứng viên kỳ vọng tốt nhất. Nếu có regularizer khác, phải được bổ sung dưới nhãn *experimental variant*, không tự động nhập vào main contribution.

### 6.4. Lịch huấn luyện giảm rủi ro bão hòa clean

**Stage A — reproduction (bắt buộc):** train/reproduce GCN hiện tại với seed 2025/2026, lưu graph builder version, hashes, preprocessing, metrics.

**Stage B — warm-start reliability:** dùng cặp train để huấn luyện $q_\phi$ dự đoán retention; backbone classifier vẫn giữ baseline train source, hoặc freeze GCN trong vài epoch đầu. Kiểm tra chất lượng edge label/prediction trước khi cho q tác động lên GCN.

**Stage C — joint fine-tune:** từ checkpoint source tương ứng, đặt $\epsilon$ cao ban đầu (ví dụ 0.8), bật $q$ và tinh chỉnh GCN/head với $L_{cls}+\lambda_eL_{edge}$. Có thể không fine-tune encoder RoBERTa nếu baseline dùng frozen embeddings. Khóa hoặc kiểm soát random number generator để ablation paired.

**Stage D — selection:** chọn epoch/hyperparameter bằng **source validation** (clean + source-generated perturbations), hoặc OOD-validation từ những **source domains được xác định từ trước** nếu protocol cho phép. **Không dùng GPT3.5 test labels để chọn mô hình train trên HC3.** Nếu dùng GPT labels/tuning, đó là *target adaptation*, không còn cùng protocol Table 3.

**Stage E — evaluation:** freeze model và threshold 0.5; chạy Table 2/3/4 tuần tự trên test.

### 6.5. Kiểm soát perturbation

- Bảo đảm **train/test example separation trước khi tạo augmented text** để tránh biến thể của cùng một bản văn lọt sang hai phía.
- Synonym WordNet trong Table 2 của paper chọn adjective và lấy synonym đầu; muốn so sánh trực tiếp cần tái hiện chính xác code và randomization nếu khả thi.
- PEARL không yêu cầu text được paraphrase by LLM; dùng phương pháp sinh **deterministic có edit log** giúp kiểm chứng căn chỉnh dễ hơn.
- Để chứng minh giá trị module, baseline `GCN + identical augmentation` phải nhìn thấy **cùng cặp training**; chỉ khác việc dùng nhãn retention và weighted adjacency.
- Nếu chỉnh sửa làm thay đổi ý nghĩa nghiêm trọng, không áp BCE trên nhãn gốc trong nhánh đó; đây là ranh giới giữa robustness benchmark và giả định bảo toàn nhãn.

---

## 7. Lập luận toán học về ổn định

### 7.1. Tại sao thay cạnh có thể làm đầu ra đổi

Xét GCN ReLU hai lớp không bias để có lập luận đơn giản:

$$
H_1=\sigma(SXW_0),\quad H_2=\sigma(SH_1W_1),
$$

$$
H'_1=\sigma(S'X'W_0),\quad H'_2=\sigma(S'H'_1W_1).
$$

Giả sử:

1. $S,S'$ là toán tử **đối xứng chuẩn hóa**, không âm, có self-loop nên $\|S\|_2,\|S'\|_2\le1$.
2. Hai đồ thị đã **căn chỉnh node**, số node giống nhau hoặc được nhúng về một không gian node chung, để so $S$ với $S'$, $X$ với $X'$.
3. Activation $\sigma$ 1-Lipschitz (ReLU), dùng cùng weights.
4. Gọi $\delta_S=\|S-S'\|_2$ và $\delta_X=\|X-X'\|_F$.

Khi đó:

$$
\|H_1-H_1'\|_F
\le \|W_0\|_2\left(\delta_S\|X\|_F+\delta_X\right).
$$

Vì $\|H_1\|_F\le\|X\|_F\|W_0\|_2$:

$$
\boxed{\|H_2-H_2'\|_F\le
\|W_0\|_2\|W_1\|_2\left(2\delta_S\|X\|_F+\delta_X\right).}
$$

Đây là **bất đẳng thức suy ra từ tính Lipschitz và bất đẳng thức tam giác** trong các giả thiết đã nêu, không phải định lý mới của PEARL. Nó cho thấy giảm chênh lệch **toán tử truyền thông tin** có thể giảm một *upper bound* của độ lệch biểu diễn, nếu chênh lệch embedding không chi phối.

**Giới hạn quan trọng:**

- Với Insert/Delete, hai graph khác số node và readout cũng thay đổi. Phải căn chỉnh/pad trên token còn lại; có thể xuất hiện một **readout discrepancy term** khác. Không được viện bất đẳng thức trên như chứng minh đầy đủ cho Delete.
- Nếu baseline sử dụng directed/không đối xứng, $\|S\|_2\le1$ **không tự động đúng**. Khi ấy dùng các hằng số $\|S\|_2,\|S'\|_2$ thực tế, không áp nguyên văn công thức tối giản.
- Normalization trên adjacency có trọng số phụ thuộc bậc; giảm một số cạnh **không bảo đảm** $\delta_S$ giảm. Cần **đo** độ chênh $\|S_\phi(G)-P^TS_\phi(G')P\|$ trên các token được căn chỉnh.
- Quan trọng hơn, giảm chênh representation không suy ra tăng classification accuracy. Nghiên cứu phải chứng minh đồng thời **stability** và **discriminativeness**.

### 7.2. Ý tưởng cần được kiểm định, không được tự phong là theorem

Ta mong muốn (nhưng chưa chứng minh):

$$
\mathbb E_T\left[\operatorname{dist}\left(f_{\theta,\phi}(G),f_{\theta,\phi}(G_T)\right)\right]
< \mathbb E_T\left[\operatorname{dist}\left(f_{\theta,0}(G),f_{\theta,0}(G_T)\right)\right],
$$

trong khi classification risk trên nguồn **không tăng** và target risk **không tăng**.

Một phản ví dụ dễ thấy: nếu quan hệ cú pháp quan trọng nhất để phân biệt Human/AI lại thường thay đổi dưới synonym editing, PEARL có thể làm giảm trọng số quan hệ đó, làm **sụt** Table 2 và Table 3 dù graph embeddings ổn định hơn.

### 7.3. Tại sao Table 3 khó hơn Table 4 về mặt lý luận

Giả sử $\mathcal D_S$ là domain nguồn HC3 và $\mathcal D_T$ là GPT3.5-Mixed. Bộ học cạnh được huấn luyện theo:

$$
q_{S}(e)=\Pr_{T\sim\mathcal Q_S}(\operatorname{retain}(e)=1\mid\phi(e,G)),\qquad G\sim\mathcal D_S.
$$

Ở miền đích, $\Pr(\operatorname{retain}(e)=1\mid\phi(e,G),G\sim\mathcal D_T)$ có thể **khác**. PEARL không có nguồn lý luận nào bảo đảm $q_S$ chuyển thành $q_T$. Do đó phải kiểm tra **shift của độ ổn định theo relation/POS**, các phân tầng length/domain/topic và AI/Human recall.

### 7.4. Phân tích mechanistic nên đưa vào paper

- Edge-retention AUROC/AUPRC trên *source held-out*.
- Reliability calibration: Brier score/ECE cho edge stability probability.
- Breakdown $\Pr(s=1\mid r)$ theo `nsubj`, `amod`, `advmod`, `obj`, punctuation...
- Edge reliability trên AI vs Human, theo domain/topic/length bin; xem q có trở thành **shortcut mới** hay không.
- Đo $\|S-S'\|$, $\|h-h'\|$, logit shift và flip rate của classifier, trước/sau PEARL.
- Label-preserving probe trên tập synonym đã đánh giá chất lượng, song song với benchmark random 25%.

---

## 8. Mục tiêu trên ba benchmark

| Bảng | Cơ chế PEARL dự kiến | Kỳ vọng định tính (**chưa đo**) | Rủi ro |
|---|---|---|---|
| **Table 2** | Downweight một số cạnh cú pháp nhạy với synonym; giữ $\epsilon$ nền | Accuracy sạch và synonym **không suy giảm đáng kể**; có khả năng tăng robustness | GCN vốn rất tốt; downweight cạnh phân biệt làm giảm clean accuracy |
| **Table 3** | Ưu tiên quan hệ ổn định và tránh một số local syntactic shortcuts | **Có thể** cải thiện HC3 → GPT3.5 nếu stability transfer được | Không có bất biến domain từ edge stability; khả năng cải thiện **không chắc chắn** |
| **Table 4** | Giảm lan truyền qua những quan hệ nhạy với Replace/Delete | Nhắm vào điểm yếu Replace/Delete, giữ Insert/Repeat | Delete mất token thật; parser hallucination; inserted-edge q dự đoán sai |

**Thứ tự ưu tiên nghiên cứu:** (1) giữ Table 4 không sụp, (2) tăng HC3 → GPT3.5, (3) giữ Table 2, (4) tăng chiều GPT3.5 → HC3. **Chiến thắng từng ô** là đích rất khó trên Table 2 vì số gốc đã gần 99%; không được thay điều kiện thành “thắng trung bình” sau khi nhìn test nếu tuyên bố ban đầu là thắng mọi ô.

**Hai chuẩn đối chiếu phải ghi tách:**

1. Paired **GCN nhóm tự train**, cùng seed/split/augmentation, là baseline chính để xác lập hiệu quả phương pháp.
2. **PRDetect paper** [R1] là mốc lịch sử; so với số công bố phải thận trọng vì implementation và preprocessing có thể khác. Nếu nhắm vượt số paper, cần tái lập protocol sát nhất có thể.

---

## 9. Ablation và protocol đánh giá

### 9.1. Matrix bắt buộc

| ID | Variant | Augmentation | Reliability supervision | Weighted adjacency | Mục đích |
|---|---|---|---|---|---|
| B0 | GCN nguyên bản nhóm | Không hoặc theo reproduction | Không | Không | Baseline chính |
| B1 | GCN + identical edit augmentation | **Cùng** cặp PEARL | Không | Không | Loại trừ lợi ích do augmentation |
| B2 | GCN + random edge dropout | Cùng cặp | Không | Ngẫu nhiên | Loại trừ regularization thông thường |
| B3 | GCN + GAT-like edge weights | Cùng cặp | Không | Có | Loại trừ lợi ích attention chung |
| B4 | GCN + uninformative/permuted edge labels | Cùng cặp | Nhãn hoán vị train | Có | Negative control: supervision có thông tin thật? |
| B5 | **GCN + PEARL-Lite** | Cùng cặp | **Có** | **Có** | Đóng góp mục tiêu |
| B6 | GCN + oracle-retained edges (phân tích) | Dùng cặp có provenance | Oracle chỉ ở analysis | Có | Ước lượng trần ổn định, **không phải phương pháp triển khai thực tế** |
| B7 | RGCN V1 retrain ± PEARL | Cùng cặp | Khác nhau | Khác nhau | Kiểm chứng backbone portability |

B6 không được xuất hiện như kết quả inference thực tế vì oracle đòi văn bản gốc để biết cạnh nào đã giữ. Nên thêm **PEARL-constant / $\epsilon=1$** kiểm tra tính tương đương baseline.

### 9.2. Chỉ số bắt buộc

- Accuracy **threshold 0.5**, F1, ROC-AUC, precision, recall Human và recall AI.
- Từng ô Table 2/3/4, không chỉ average.
- Table 3 average = `Original + Syn5 + Syn10 + Syn20 + Syn30` / 5.
- Table 4 noise average = `Insert + Repeat + Replace + Delete` / 4, **không gồm Original**.
- Robustness gap: $\Delta_{perturb}=\operatorname{Acc}_{clean}-\operatorname{Acc}_{perturb}$, tách từng loại.
- Worst-condition accuracy, variation across seeds.
- Edge-level retention AUROC/AUPRC/ECE và chi phí preprocessing/training/test.

### 9.3. Statistical protocol

- **Vòng sàng lọc:** seed 2025, 2026 theo nhật ký nhóm; paired same seed, same baseline cache, same split.
- **Vòng xác nhận:** 5 seeds nhất quán (ví dụ 2021–2025 hoặc một bộ khóa trước); không cherry-pick seed 2026 cũ.
- Báo mean ± std; báo paired differences và bootstrap 95% CI ở mức **document**, tránh giả độc lập giữa 5 perturbed variants từ cùng một document.
- Dùng **validation** để chọn epoch/hyperparams, **test** chỉ mở ở lần chốt. Khi exploration đã xem test nhiều lần, có thể bổ sung held-out evaluation mới để giảm nguy cơ overfitting theo quá trình nghiên cứu.
- Nên bổ sung source/length/topic-matched controlled subsets với cùng code selection cho cả đối chứng.

### 9.4. Tiêu chí stop/go đề xuất (quy tắc nội bộ, không phải benchmark kết quả)

**GO nếu:** có lợi ích paired đáng tin cậy so với B1/B3 trên Table 3 hoặc Replace/Delete, không làm giảm đáng kể HC3 clean, và không làm đổ Insert/Repeat; tín hiệu retention có calibration và AUC thật trên source validation.

**NO-GO nếu:** chỉ hơn B0 nhưng không hơn B1, hoặc có lợi ích chỉ trên seed đơn, hoặc q suy biến thành hằng số, hoặc Table 4 lặp lại sự sụp 50–70% của V2, hoặc tăng Table 3 chỉ nhờ chỉnh threshold bằng target-test.

Tốt nhất **khóa ngưỡng định lượng trước khi chạy** dựa trên chi phí sai số mà nhóm chấp nhận, ví dụ mục tiêu screening +2 điểm phần trăm Table 3 average cùng seed, clean giảm tối đa 0.5 điểm phần trăm, Table 4 average không giảm; đây là **điều kiện dừng định trước, không phải dự báo kết quả**.

---

## 10. Pseudocode và kiến trúc triển khai

### 10.1. Pseudocode tạo nhãn retention

```python
# Training data only. Pseudocode, NOT production-ready.
for (text, y, doc_id) in train_set:
    G = graph_builder_original(text)       # x, POS/dep metadata, RoBERTa node embeddings
    edit = sample_train_only_edit(text)     # returns (perturbed_text, edit_log, token_uid_map)
    Gp = graph_builder_original(edit.text)

    # Explicit provenance. Unknown ambiguous alignments are excluded.
    match = align_spacy_tokens_via_spans_and_edit_log(G, Gp, edit)
    labels = []
    for (u, v, r) in G.typed_dependency_arcs:
        if not (match.is_unique(u) and match.is_unique(v)):
            continue                        # UNKNOWN, do not force negative
        up, vp = match[u], match[v]
        s = int((up, vp, r) in Gp.typed_dependency_arcs)
        labels.append((edge_id(u, v), s))

    save_pair_meta(doc_id, G, Gp, edit.info, labels)
```

### 10.2. Pseudocode PEARL-GCN

```python
class PEARL_GCN:
    def __init__(self, original_gcn, edge_encoder, epsilon=0.8):
        self.gcn = original_gcn
        self.edge_predictor = edge_encoder  # small MLP
        self.epsilon = epsilon

    def predict(self, graph):
        edge_feats = inference_available_edge_features(graph)  # NEVER edit provenance
        q = sigmoid(self.edge_predictor(edge_feats))
        weight = self.epsilon + (1.0 - self.epsilon) * q
        S = weighted_normalized_adjacency(
            graph.edge_index, weight, graph.num_nodes,
            graph_direction_and_loops_from_original_builder=True,
        )
        logits = self.gcn.forward_with_propagator(graph.x, S)
        return logits, q

for epoch in range(num_epochs):
    for pair in train_pair_loader:
        logits, q = model.predict(pair.original_graph)
        loss_cls = BCEWithLogitsLoss(logits, pair.y)
        loss_edge = masked_edge_bce(q, pair.retention_labels,
                                    pair.valid_alignment_mask)
        loss = loss_cls + lambda_edge * loss_edge
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

**Lưu ý:** Pseudocode chỉ diễn giải thuật toán. Muốn implement thực sự phải tuân thủ API graph object và `edge_index`, `batch`, readout, optimizer của repo GCN nhóm; không thể thay thế trực tiếp một dòng `S` nếu pipeline đang dùng sparse adjacency theo format khác. Classifier baseline có thể trả xác suất; khi đó cần nhất quán BCE/BCEWithLogits.

### 10.3. Checkpoint / cache reproducibility

Mỗi checkpoint lưu:

```yaml
model: PEARL_GCN
base_model: PRDetect_GCN_local_retrained
seed: 2025
split_hash: <hash>
train_pair_generator_version: <commit>
parser_name_and_version: <string>
roberta_model_and_version: <string>
graph_builder_version: <commit>
edge_feature_schema_version: v1
edge_label_schema_version: v1
epsilon: 0.8
lambda_edge: 0.1
threshold: 0.5
validation_selection_rule: <fixed rule>
```

### 10.4. Khả thi trên Kaggle khoảng 20 GB

- Không lưu tất cả embedding/graph của 5 mức synonym × cả train/valid/test đồng thời.
- Chỉ lưu **train original graph** và **pair metadata + sparse retention labels**, hoặc sinh từng epoch; nếu tốn parser thì precompute **1–2 perturbation types** theo luồng.
- Test tuần tự Table 2 → Table 3 → Table 4, xóa graph `.pkl` trung gian khi đã trích xuất các metrics/checksums.
- Theo [R2], ưu tiên **≤4 graph `.pkl` cùng lúc**, giữ log và JSON metrics thay vì giữ nhiều graph lớn.
- Không có model generation bổ sung ở inference; overhead chính là edge-feature extraction + MLP (thường nhỏ hơn dependency parsing/RoBERTa, nhưng **chưa đo**).

---

## 11. Rủi ro, giả thuyết có thể bác bỏ và tiêu chí dừng

| Rủi ro | Dấu hiệu kiểm chứng | Giải pháp kiểm tra, không bảo đảm khắc phục |
|---|---|---|
| Mô hình học `q≈1` cho tất cả cạnh | q variance gần 0; edge AUPRC không hơn prevalence | Cân bằng BCE, phân tích per-type; kiểm tra distribution s |
| Cạnh ổn định không phân biệt Human/AI | Edge AUC tốt nhưng classifier tệ | Ablate giữ/loại top-q; đo mutual signal trên validation |
| Edit distribution không chuyển miền | Per-relation retention drift giữa held-out source subsets | Stress test trên perturbation khác; không claim Table 3 |
| Inserted edge nhìn “hợp lý” đối với q | False-high-q tại các cạnh mới | Thử weak negatives từ inserted edges; ablate và cân nhắc bias |
| Delete xóa bằng chứng phân loại | Noise 25% làm mất token then chốt | Không kỳ vọng khôi phục được thông tin; dùng conditional error analysis |
| Làm sai graph/embedding legacy | $\epsilon=1$ khác baseline | Dừng và sửa graph builder, normalization, batching |
| $q$ mang shortcut từ length/topic | performance drift theo bins | Probe conditional on length/domain, không train target-test |
| Hơn nhờ train augmentation, không phải PEARL | B5 ~ B1 | Không công nhận đóng góp dựa trên PEARL |
| Novelty bị trùng prior art | Paper khác có edit-aligned edge retention supervision | Định vị מחדש hoặc bỏ ý tưởng, không đổi tên để nhận novelty |

**Giả thuyết có thể bị bác bỏ rõ ràng:**

- **H1:** B5 có edge stability AUROC/AUPRC vượt constant và shuffled-label controls.
- **H2:** B5 có representation/logit flip rate thấp hơn B1/B3 dưới perturbations và giữ source accuracy.
- **H3:** B5 tăng paired Table 3 (đặc biệt HC3 → GPT3.5) **và** không giảm Table 4 trên các seeds có đối chứng.
- **H4:** Hiệu quả không chỉ đến từ train augmentation hoặc thay đổi normalization.

Bác bỏ H1/H2 khiến lý giải cơ chế suy yếu; bác bỏ H3 khiến PEARL **không đạt mục tiêu người dùng đặt ra**, dù có thể còn đóng góp phân tích hạn chế của GCN.

---

## 12. Tính mới và khả năng phát triển thành bài báo

### 12.1. Phân ranh kế thừa và phần đề xuất

| Ý tưởng | Trạng thái |
|---|---|
| Dependency graph + GCN để detect AI text | **Có từ PRDetect [R1]** |
| Relation-dependent weights | **Có từ RGCN [R6]** |
| Attention / learned edge weight | **Có từ GAT [R3]** |
| Edge denoising / pruning cho robust GNN | **Có từ GNNGuard [R4], Pro-GNN [R5]** |
| Train consistency dưới augmentation | **Có từ UDA [R7]** |
| Adversarial training để detect MGT | **Đã có GREATER [R8]** |
| **Căn chỉnh token provenance để tạo edge-retention supervision cho cú pháp trước–sau edit rồi distill thành single-view reliability cho AI-text detection** | **Ứng viên đóng góp của PEARL; phải kiểm tra prior art đầy đủ + ablation** |

Đề xuất hiện tại **chưa đủ** để tuyên bố “đóng góp khoa học độc lập được chấp nhận” hoặc “đã chứng minh vượt PRDetect”. Nó **đủ để triển khai một nghiên cứu có thể phản biện**: có câu hỏi, giả thuyết, đơn vị can thiệp, control, loss, cách đo cơ chế và tiêu chí bác bỏ.

### 12.2. Cách kể câu chuyện paper nếu kết quả dương

**Problem:** Biểu diễn graph cú pháp có thể đổi mạnh sau lexical edits mặc dù cú pháp toàn cục còn nhiều thông tin hữu ích.  
**Finding (phải đo):** Syntactic dependency relations có *retention rates* khác nhau theo loại quan hệ/ngữ cảnh, và những khác biệt này có liên hệ đến stability của graph detector.  
**Method:** Một single-view edge reliability predictor được huấn luyện từ edit-aligned paired syntax trees, dùng để reweight GCN.  
**Mechanistic evidence:** Giảm representation perturbation/flip rate và/hoặc cải thiện retention calibration; hiệu quả còn khi kiểm soát augmentation.  
**Benchmarks:** Kết quả paired Table 2/3/4 và external perturbations nếu có.

**Tên paper khả dĩ (chưa dùng làm tên công bố):** *Learning Edit-Stable Dependency Relations for Robust and Generalizable Machine-Generated Text Detection.*

### 12.3. Quyết định nghiên cứu được khuyến nghị

1. Chọn **GCN gốc nhóm tự train** làm backbone đầu tiên, không cần V3.
2. Viết **data audit** cho token provenance/graph matching trước; nếu không có label retention đáng tin cậy, dừng.
3. Chạy B1 vs B5 cùng seed và cache; chưa cần các biến thể phức tạp.
4. Nếu B5 ổn, chạy B3 (GAT-like) và B4 (shuffled labels) để xác nhận novelty-related mechanism.
5. Kiểm chứng Table 3 và 4 theo từng ô, không dùng average che các tổn thất.
6. Sau đó mới chuyển sang RGCN V1 và mở rộng kiểm chứng trên 5 seeds.

---

## 13. Tài liệu tham khảo đã đối chiếu

### Nguồn trực tiếp do nhóm cung cấp

- **[R1]** Li, X., Yin, Z., Tan, H., Jing, S., Su, D., Cheng, Y., Shen, H., & Sun, F. (2025). **PRDetect: Perturbation-Robust LLM-generated Text Detection Based on Syntax Tree**. *Findings of NAACL 2025*, 8305–8316. [ACL Anthology](https://aclanthology.org/2025.findings-naacl.464/). Tệp: `2025.findings-naacl.464(2).pdf`. Dữ liệu Table 1–4 và phương pháp lấy từ PDF này.
- **[R2]** **Context nghiên cứu: lịch sử cải tiến PRDetect** (12/11/2026). Tệp: `Context_LichSuCaiTien_PRDetect(1).md`. Nhật ký nội bộ; các số thực nghiệm và nhận xét về V1/V2/V3 được lấy từ đây, **không phải bằng chứng bình duyệt**.

### Công trình nền tảng và prior art

- **[R3]** Veličković, P., et al. (2018). **Graph Attention Networks**. *ICLR 2018*. [Thông tin công trình](https://ai.meta.com/research/publications/graph-attention-networks/). Cơ sở phân biệt PEARL với attention đã có.
- **[R4]** Zhang, X., & Zitnik, M. (2020). **GNNGuard: Defending Graph Neural Networks against Adversarial Attacks**. *NeurIPS 2020*. [Proceedings](https://papers.nips.cc/paper_files/paper/2020/hash/690d83983a63aa1818423fd6edd3bfdb-Abstract.html). Robust edge weighting/pruning có trước PEARL.
- **[R5]** Jin, W., et al. (2020). **Graph Structure Learning for Robust Graph Neural Networks (Pro-GNN)**. *KDD 2020*. [KDD](https://www.kdd.org/kdd2020/accepted-papers/view/graph-structure-learning-for-robust-graph-neural-networks.html). Học cấu trúc bền vững và đối chứng phương pháp.
- **[R6]** Schlichtkrull, M. S., et al. (2018). **Modeling Relational Data with Graph Convolutional Networks**. *ESWC 2018*. [Microsoft Research](https://www.microsoft.com/en-us/research/publication/modeling-relational-data-with-graph-convolutional-networks/). Cơ sở RGCN V1.
- **[R7]** Xie, Q., Dai, Z., Hovy, E., Luong, T., & Le, Q. V. (2020). **Unsupervised Data Augmentation for Consistency Training**. *NeurIPS 2020*. [Proceedings](https://papers.neurips.cc/paper/2020/hash/44feb0096faa8326192570788b38c1d1-Abstract.html). Cơ sở tính bất biến trước augmentation.
- **[R8]** Li, Y., Zhang, Z., Li, C., Shen, C., & Liu, X. (2025). **Iron Sharpens Iron: Defending Against Attacks in Machine-Generated Text Detection with Adversarial Training**. *ACL 2025*. [ACL Anthology](https://aclanthology.org/2025.acl-long.155/). Đối chứng mạnh về adversarial training cho phát hiện MGT.
- **[R9]** Liu, X., Zhang, Z., Wang, Y., Pu, H., Lan, Y., & Shen, C. (2023). **CoCo: Coherence-Enhanced Machine-Generated Text Detection Under Low Resource With Contrastive Learning**. *EMNLP 2023*. [ACL Anthology](https://aclanthology.org/2023.emnlp-main.1005/). Cơ sở graph-based MGT detection.
- **[R10]** Guo, B., et al. (2023). **How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection**. *arXiv:2301.07597*. [arXiv](https://arxiv.org/abs/2301.07597). Dataset HC3.

**Lưu ý phạm vi khảo sát:** Đây là danh sách công trình **đã đối chiếu để xây dựng lập luận**, không phải systematic literature review toàn bộ 2023–2026. Khi chuẩn bị nộp bài, phải tìm tiếp những công trình gần hơn về *perturbation-aware structure alignment*, *dependency-arc stability prediction* và *edge reliability learning*, kể cả workshop/preprint và các công trình xuất bản sau thời điểm bản đề xuất này được viết.

---

## Tuyên bố cuối cùng

PEARL là **một chương trình nghiên cứu kiểm chứng được**, không phải một kết quả đã chứng minh. Giá trị khoa học tiềm năng không nằm ở phép nhân trọng số cạnh, mà nằm ở **cách học tín hiệu retention của dependency relation từ văn bản trước–sau chỉnh sửa và dùng nó trong phát hiện AI-text chỉ với một input tại inference**. Đóng góp chỉ đủ mạnh để viết bài báo sau khi (i) chứng minh cơ chế, (ii) phân biệt được khỏi prior art và augmentation/attention baselines, (iii) cải thiện paired benchmark một cách ổn định và (iv) xử lý trung thực các kết quả âm.
