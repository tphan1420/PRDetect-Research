# Báo cáo nghiên cứu: Kiến trúc phát hiện LLM-generated text bền vững hơn PRDetect dưới perturbation và domain shift

## Phạm vi, kết luận chính và nguyên tắc thiết kế

Bài báo **PRDetect: Perturbation-Robust LLM-generated Text Detection Based on Syntax Tree** đề xuất một pipeline tương đối tối giản: văn bản được phân tích bằng spaCy thành dependency structures; mỗi token trở thành một node được khởi tạo bằng RoBERTa embedding; cạnh được biểu diễn bằng adjacency nhị phân \(A_{ij}\in\{0,1\}\); sau đó một **2-layer vanilla GCN** và classifier được huấn luyện bằng binary cross-entropy. Điểm mạnh cốt lõi của PRDetect là khai thác các đặc trưng cấu trúc syntax mà tác giả quan sát thấy ổn định đáng kể dưới adjective synonym substitution. fileciteturn0file0 citeturn398977search0turn514094view0

Tuy nhiên, chính ba lựa chọn đơn giản đó—**raw contextual node embeddings, relation-agnostic binary graph, và fixed-depth GCN/readout**—tạo ra ba điểm nghẽn tương ứng với Table 2, Table 3 và Table 4. Table 2 cho thấy PRDetect gần như không suy giảm khi thay 5–30% adjective bằng WordNet synonym, nhưng clean accuracy vẫn thấp hơn CoCo: 0.9878 so với 0.9981 trên HC3, và 0.9656 so với 1.0000 trên GPT3.5-Mixed. Table 3 cho thấy transfer bất đối xứng mạnh: HC3\(\rightarrow\)GPT3.5-Mixed chỉ 73.2% clean và 70.9% ở 30%, trong khi GPT3.5-Mixed\(\rightarrow\)HC3 đạt 87.7% clean. Table 4 tiếp tục cho thấy robustness của PRDetect mang tính **perturbation-specific**: Insert 0.983 và Repeat 0.982 gần như không ảnh hưởng, nhưng Replace chỉ 0.798 và hàng “Detect”—gần như chắc chắn là typo của **Delete** theo mô tả Section 5.5—chỉ còn 0.747. citeturn337880view0turn337880view1

Kiến trúc đề xuất ở đây, gọi là **PRISM-Detect — Perturbation-Robust Invariant Syntax–Semantic Multi-view Detector**, gồm bốn module được thiết kế như một hệ thống duy nhất: **typed structural graph**, **confidence-weighted relation-aware dual-stream encoder**, **perturbation/domain-invariant training**, và **length-invariant hierarchical readout**. Mục tiêu không phải thêm nhiều “trick”, mà là thay đúng những inductive bias đang giới hạn PRDetect. Đây là một proposal có thể kiểm chứng thực nghiệm; **không thể khẳng định trước rằng nó sẽ vượt PRDetect ở mọi ô của Tables 2–4 cho tới khi chạy thí nghiệm đầy đủ**.

## Root-cause diagnosis của PRDetect

### Table 2: robustness cao nhưng clean ceiling thấp

Ở Table 2, PRDetect đạt 0.9878/0.9878/0.9872/0.9874/0.9864 trên HC3 và 0.9656/0.9630/0.9632/0.9656/0.9638 trên GPT3.5-Mixed khi perturbation tăng từ 0% lên 30%. Đây là bằng chứng mạnh rằng **adjective synonym substitution do tác giả chọn không làm thay đổi mạnh các summary statistics của syntax tree mà GCN đang sử dụng**. Appendix Table 7 củng cố nhận định này: trên HC3, average depth, number of nodes và root height gần như không đổi từ 0% tới 30%. citeturn337880view0turn337880view3

Nhưng stability không đồng nghĩa với representational completeness. PRDetect bỏ hoàn toàn dependency relation type khi biến parse thành \(A_{ij}\): một `nsubj`, `obj`, `amod`, `advcl` hay `acl` đều trở thành cùng một giá trị 1. Như vậy, hai cây có topology gần giống nhau nhưng chức năng ngữ pháp khác biệt có thể bị GCN biểu diễn gần nhau. Đồng thời, RoBERTa node embedding chứa nhiều thông tin lexical/semantic mạnh nhưng không được decomposition thành “content” và “style”. PRDetect do đó đứng giữa hai cực: topology quá thô, còn node feature lại quá giàu semantics. Cấu hình này giải thích hợp lý việc mô hình rất ổn định với synonym perturbation nhưng chưa tận dụng đầy đủ signal ở clean text. Đây là **chẩn đoán kiến trúc**, không phải causal result đã được tác giả trực tiếp chứng minh. citeturn514094view0

CoCo có lợi thế clean-text khác bản chất: nó xây dựng graph từ entity/coherence interactions giữa các câu và dùng contrastive learning, tức có thể tận dụng discourse-level regularities mà một token dependency GCN cục bộ không trực tiếp mô hình hóa. Chính bài CoCo mô tả explicit coherence graph và contrastive objective là hai thành phần trung tâm của detector. citeturn398977search1turn398977search4 Vì vậy, khoảng cách clean 0.9878→0.9981 và 0.9656→1.0000 không nên được diễn giải đơn giản là “GCN kém hơn”; nguyên nhân hợp lý hơn là **PRDetect đang tối ưu invariance trên một structural representation có entropy thấp**, nên mất một phần discriminative signal mà CoCo giữ lại.

### Table 3: domain shift, lexical leakage và length mismatch

Table 1 của PRDetect cho thấy khác biệt rất lớn về document length: HC3 trung bình khoảng 147.93 token cho human và 178.65 cho machine, trong khi GPT3.5-Mixed lần lượt khoảng 756.55 và 501.13 token. Nhưng “Number of Nodes” được báo cáo chỉ quanh 20–25. Điều này cần diễn giải thận trọng: paper nói long inputs được segment/chunk trước khi dependency parsing, nhưng không mô tả đủ chi tiết để đồng nhất “number of nodes” trong Table 1 với tổng số token của toàn document. Vì vậy, phát biểu “GPT3.5 graph có 4–5 lần nhiều node hơn HC3” **không được dữ liệu Table 1 hỗ trợ trực tiếp**; điều được xác nhận là document-length distribution khác mạnh và short-text performance suy giảm rõ trong Table 5. fileciteturn0file0 citeturn337880view2

Nguyên nhân transfer HC3\(\rightarrow\)GPT3.5-Mixed rơi xuống 73.2% có thể phân rã thành ba loại shift. Thứ nhất là **topic/genre shift**: HC3 là Q&A đa miền, còn GPT3.5-Mixed là news; chính tác giả cũng liên hệ cross-dataset degradation với khác biệt syntactic style giữa question-answer responses và news structure. citeturn337880view1 Thứ hai là **lexical leakage**: RoBERTa embeddings có thể mã hóa topic, register, named entities và generator-specific lexical regularities; vì GCN nhận trực tiếp các embeddings này, classifier không bị buộc phải dựa trên domain-agnostic syntax. Đây là giả thuyết hợp lý nhưng PRDetect không cung cấp probing experiment để chứng minh. Thứ ba là **length/composition shift**: một document dài cung cấp nhiều sentence-level evidence hơn, làm distribution của pooled graph representation khác đáng kể với một short Q&A.

Table 8 đặc biệt quan trọng: khi tác giả chọn các cặp có similar length distributions, PRDetect đạt 0.9338→0.9022 trên HC3 và khoảng 0.954–0.956 trên GPT3.5-Mixed. Điều này cho thấy length mismatch thực sự là một confound đáng kể, nhưng không chứng minh rằng length là nguyên nhân duy nhất của cross-domain failure. citeturn337880view3

Về **over-smoothing**, cần tránh quy kết quá mạnh. Vanilla GCN về lý thuyết thực hiện Laplacian-like smoothing, và nghiên cứu về GNN cho thấy repeated message passing có thể khiến node representations trở nên khó phân biệt khi số layer tăng. citeturn733665search0turn733665search8 Tuy nhiên PRDetect chỉ dùng **2 layers**, nên không có đủ bằng chứng để nói over-smoothing là nguyên nhân trực tiếp của mức 73.2%. Vấn đề hợp lý hơn là **fixed receptive field + degree-normalized aggregation không phân biệt relation**, cộng với distribution shift của sentence/graph composition; over-smoothing chỉ nên được xem là một sensitivity hypothesis cần đo bằng pairwise embedding distance hoặc Dirichlet energy, không phải kết luận đã được chứng minh.

Sự bất đối xứng GPT3.5-Mixed\(\rightarrow\)HC3 = 87.7% so với HC3\(\rightarrow\)GPT3.5-Mixed = 73.2% cũng phù hợp với giả thuyết **coverage asymmetry**: huấn luyện trên documents dài có thể quan sát nhiều local syntactic motifs hơn mỗi sample, trong khi huấn luyện trên short Q&A rồi chuyển sang long-form news tạo ra extrapolation mạnh hơn. Paper ghi nhận chính hiện tượng “model trained on longer texts” chuyển sang short texts tốt hơn chiều ngược lại, nhưng đồng thời thừa nhận nguyên nhân cụ thể vẫn chưa được xác lập. citeturn337880view1turn337880view2

### Table 4: Replace/Delete phá representation mạnh hơn Insert/Repeat

Insert và Repeat có thể tạo thêm local material mà không loại bỏ các dependency cues vốn đã tồn tại. Ngược lại, Replace và đặc biệt Delete thay đổi identity của head/dependent hoặc loại bỏ chính node giữ vai trò structural bridge, nên hậu quả không chỉ là “mất một word embedding” mà là **parse topology drift**.

Điểm cần chỉnh so với cách diễn đạt phổ biến là: nếu văn bản sau perturbation được **re-parse bằng spaCy**, deletion không nhất thiết “cắt cây thành hai component” theo nghĩa graph-theoretic. spaCy `DependencyParser` dự đoán lại `Token.head`, `Token.dep` và cả sentence boundaries; các token còn lại có thể được reattached thành một parse hợp lệ mới. citeturn846428search0turn846428search15 Do đó, failure thực tế có khả năng đến từ **structural reassignment**: head thay đổi, dependency labels thay đổi, root có thể đổi, sentence segmentation có thể drift, và topology mới không còn tương ứng với syntax style của văn bản gốc.

Nếu pipeline của PRDetect biểu diễn các sentence parses như một forest hoặc các chunk-graphs riêng biệt, perturbation cũng có thể làm thay đổi connectivity giữa các local components; nhưng paper không cung cấp đủ implementation detail để khẳng định “Delete tạo disconnected subtrees” trong mọi trường hợp. Điều chắc chắn hơn là adjacency nhị phân

\[
A_{ij}=
\begin{cases}
1, & s_i.\mathrm{head}=s_j,\\
0, & \mathrm{otherwise}
\end{cases}
\]

không có khái niệm **edge confidence**, **relation identity**, hay “edge này vừa bị parser thay đổi mạnh sau perturbation”. Vì vậy một erroneous/reassigned dependency edge được đối xử giống hệt một stable edge. citeturn514094view0 Nghiên cứu dependency-GNN trước đây cũng đã chỉ ra rằng automatically generated dependency trees chứa noise và attention/pruning trên dependency edges có thể giúp giảm tác động của parse errors. citeturn733665search3 GraphMerge cho thấy việc đưa thêm connectivity từ nhiều dependency hypotheses có thể tăng robustness trước parser errors mà không cần tăng sâu GNN. citeturn733665search5

Đây là lý do Table 4 hợp lý về mặt cơ chế: PRDetect không thực sự “robust với perturbation” nói chung; nó robust với những perturbation **preserve coarse syntactic topology**. Replace 0.798 và Delete 0.747 phơi bày đúng giới hạn đó. citeturn337880view1

## Kiến trúc thống nhất PRISM-Detect

### Typed Structural Graph với subword–word alignment

Đầu vào document \(x=(w_1,\ldots,w_n)\) được đưa qua cùng family RoBERTa để bảo đảm comparison công bằng. Với word \(w_i\) tương ứng tập subword \(B_i\), thay vì lấy một representation không được mô tả rõ alignment, ta dùng:

\[
e_i^{\mathrm{sem}}
=
\mathrm{LN}
\left(
\frac{1}{|B_i|}
\sum_{b\in B_i}H_b^{R}
\right).
\]

Sau đó node feature được factorize:

\[
h_i^{(0)}
=
W_e e_i^{\mathrm{sem}}
+
E_{\mathrm{POS}}(p_i)
+
E_{\mathrm{depth}}(d_i)
+
E_{\mathrm{sent}}(s_i)
+
E_{\mathrm{relpos}}(r_i).
\]

Ở đây \(p_i\) là POS, \(d_i\) là syntactic depth đã bucket hóa, \(s_i\) là sentence-position bucket, và \(r_i\) là normalized relative position trong câu. Các features này đều có dimension nhỏ, thí dụ 16–32, để không làm tăng đáng kể memory.

Mỗi dependency arc được giữ cả direction và relation:

\[
e_{ij}=
[\,
E_{\mathrm{dep}}(\rho_{ij});
E_{\mathrm{dir}}(\delta_{ij});
\phi(|d_i-d_j|);
\phi(|i-j|)
\,].
\]

Như vậy `nsubj`, `obj`, `amod`, `advcl` không còn bị collapse thành một cạnh đồng nhất. Đây chính là loại multi-relational structure mà R-GCN được thiết kế để xử lý; R-GCN dùng relation-conditioned message transformations thay vì một \(W\) chung cho mọi cạnh. citeturn733665academia9turn733665search1

Để chống Delete/Replace mà không dựng dense graph, PRISM-Detect bổ sung một tập nhỏ **structural fallback edges**: bidirectional sequential edge \(i\leftrightarrow i+1\), sentence-root↔sentence-root giữa hai câu kề nhau, và optional grandparent edge. Mỗi edge type có embedding riêng; dependency edges vẫn là loại ưu tiên. Complexity giữ ở \(O(|V|+|E|)\), bởi số fallback edges tuyến tính theo số token chứ không phải \(O(n^2)\).

Tác động dự kiến lên Table 2 là tăng clean separability vì giữ được grammatical function signal; lên Table 3 là giảm phụ thuộc vào lexical semantics vì POS/depth/relation cung cấp domain-stable axes; lên Table 4 là tạo redundant sparse paths khi dependency arc bị thay đổi. Việc sử dụng relation labels và noise-aware dependency modeling đã có tiền lệ thành công trong syntax-based NLP, dù chưa được kiểm chứng trên chính benchmark PRDetect. citeturn733665search3turn733665search6

### Confidence-Weighted Relation-Aware GNN và dual-stream residual fusion

Thay vanilla GCN bằng hai layer **Relation-Aware Edge-Denoising Attention**, nhưng vẫn giữ depth bằng 2 để không mất lợi thế latency.

Với edge \(j\rightarrow i\):

\[
q_i=W_q h_i,\qquad
k_{j,r}=W_k h_j+U_r e_{ij},
\]

\[
a_{ij}
=
\frac{
q_i^\top k_{j,r}
}{
\sqrt d
},
\]

và một learned structural reliability gate:

\[
c_{ij}
=
\sigma
\left(
w_c^\top
[
h_i;h_j;e_{ij};
|h_i-h_j|
]
\right).
\]

Attention cuối:

\[
\alpha_{ij}
=
\mathrm{softmax}_{j\in\mathcal N(i)}
\left(
a_{ij}+\log(c_{ij}+\epsilon)
\right),
\]

\[
m_i
=
\sum_{j\in\mathcal N(i)}
\alpha_{ij}
\left(
W_v h_j + V_r e_{ij}
\right),
\]

\[
h_i'
=
\mathrm{LN}
\left(
h_i
+
\gamma_i m_i
\right).
\]

\(c_{ij}\) **không được gọi là parser confidence của spaCy**, vì spaCy API thông thường không cung cấp một calibrated probability cho từng dependency arc; nó là learned edge-reliability score của detector. spaCy chỉ gán `Token.head` và `Token.dep` sau parsing. citeturn846428search0

Để không phụ thuộc hoàn toàn vào graph, PRISM-Detect duy trì song song một semantic stream:

\[
z_{\mathrm{seq}}
=
\mathrm{NormPool}
(
e_1^{\mathrm{sem}},
\dots,e_n^{\mathrm{sem}}
),
\]

và syntax stream \(z_{\mathrm{syn}}\). Fusion dùng gated residual:

\[
g
=
\sigma
\left(
W_g
[
z_{\mathrm{syn}};
z_{\mathrm{seq}}
]
\right),
\]

\[
z
=
g\odot z_{\mathrm{syn}}
+
(1-g)\odot z_{\mathrm{seq}}.
\]

Ý nghĩa quan trọng là **không buộc model chọn syntax hoặc semantics**. Khi parse ổn định như synonym perturbation của Table 2, gate có thể dựa nhiều vào syntax; khi Delete/Replace gây parse drift, sequence residual vẫn giữ lexical/contextual evidence. Điều này trực tiếp giải quyết failure mode của Table 4.

Edge attention thay vì hard binary adjacency cũng phù hợp với kết quả A-GCN rằng automatically parsed dependencies chứa noise và các edges không nên đóng góp ngang nhau. citeturn733665search3 Đồng thời 2-layer residual architecture tránh nhu cầu stack sâu, nên vấn đề over-smoothing được giảm ngay từ thiết kế thay vì phải chữa bằng một GNN rất sâu. Literature về over-smoothing cho thấy repeated graph convolution mới là nguồn chính khiến representations hội tụ; vì vậy giữ encoder nông là lựa chọn an toàn hơn cho bài toán này. citeturn733665search0turn733665search8

Về computation, relation-specific matrices không nên được triển khai một matrix đầy đủ cho mỗi dependency label. Sử dụng basis decomposition:

\[
W_r=\sum_{b=1}^{B}a_{rb}V_b,
\qquad B\ll |\mathcal R|,
\]

giữ parameter count gần vanilla GCN. Với \(B=4\) hoặc \(8\), complexity message passing vẫn tuyến tính theo số cạnh.

### Perturbation-Invariant, Domain-Generalizable Training

Classifier loss cơ bản vẫn là supervised BCE:

\[
\mathcal L_{\mathrm{cls}}
=
-\frac1N
\sum_i
[
y_i\log p_i
+
(1-y_i)\log(1-p_i)
].
\]

Điểm thay đổi là mỗi sample \(x\) sinh hai views: \(x^{(a)}\) mild và \(x^{(h)}\) hard. Mild gồm adjective synonym, edge dropout nhỏ và POS-preserving substitution; hard tập trung vào **Delete và Replace**, vì Table 4 chứng minh đây là worst-case operations. Augmentation cần constraint về label validity: không perturb quá ngưỡng khiến nguồn tác giả không còn định nghĩa hợp lý; chính PRDetect cũng cảnh báo rằng perturbation quá mạnh làm label khó xác định. citeturn514094view0

Representation invariance dùng supervised contrastive/InfoNCE:

\[
\mathcal L_{\mathrm{con}}
=
-\frac{1}{N}
\sum_i
\log
\frac{
\exp(
\mathrm{sim}(z_i,z_i^+)/\tau
)
}{
\sum_{k\neq i}
\exp(
\mathrm{sim}(z_i,z_k)/\tau
)
}.
\]

Prediction consistency thêm symmetric KL hoặc Jensen–Shannon:

\[
\mathcal L_{\mathrm{inv}}
=
\frac12
D_{\mathrm{KL}}
(p(x)\|p(x'))
+
\frac12
D_{\mathrm{KL}}
(p(x')\|p(x)).
\]

Ý tưởng này phù hợp với kết quả rộng hơn trong AIGT detection: SCRN dùng siamese calibration giữa noisy views và báo cáo robustness lẫn cross-domain generalization tốt hơn các baselines trong setting của họ. citeturn398977search2 Graph contrastive learning cũng cung cấp tiền lệ cho việc tạo augmented graph views và tối đa hóa agreement giữa representations. citeturn398977academia5

Để xử lý Table 3 mà không “nhìn” vào target test set, ta tạo **pseudo-domains chỉ từ training data**: topic cluster, source/site nếu metadata tồn tại, length bucket và lexical-cluster ID. Một gradient reversal domain classifier nhận \(z_{\mathrm{style}}\):

\[
\mathcal L_{\mathrm{dom}}
=
-\sum_i\log P(d_i|z_i),
\]

nhưng encoder nhận gradient dấu ngược:

\[
\theta_E
\leftarrow
\theta_E
-
\eta
\left(
\nabla\mathcal L_{\mathrm{cls}}
-
\lambda_d\nabla\mathcal L_{\mathrm{dom}}
\right).
\]

Domain-Adversarial Neural Networks formalize đúng nguyên lý này: học representation vẫn discriminative cho task nhưng kém discriminative cho domain. citeturn733665search2 Quan trọng về fairness: trong benchmark HC3\(\rightarrow\)GPT3.5-Mixed, **không dùng unlabeled target-test GPT3.5-Mixed để adversarial adaptation** trong main result. Nếu muốn chạy unsupervised target adaptation, phải báo cáo thành một setting riêng.

Để tách syntactic style khỏi topical content, representation được split:

\[
z=[z_s;z_c],
\]

với classification ưu tiên \(z_s\), còn \(z_c\) reconstruct/predict masked lexical properties. Ta thêm cross-covariance penalty:

\[
\mathcal L_{\mathrm{decorr}}
=
\left\|
\frac1B
Z_s^\top Z_c
\right\|_F^2.
\]

Total training objective:

\[
\boxed{
\mathcal L
=
\mathcal L_{\mathrm{cls}}
+
\lambda_c\mathcal L_{\mathrm{con}}
+
\lambda_i\mathcal L_{\mathrm{inv}}
+
\lambda_d\mathcal L_{\mathrm{dom}}
+
\lambda_o\mathcal L_{\mathrm{decorr}}
}
\]

với curriculum nên bắt đầu \(\lambda_c,\lambda_i,\lambda_d\) thấp trong 1–2 epochs rồi tăng tuyến tính để tránh invariance loss triệt tiêu discriminative signal.

Table 2 được hưởng lợi từ supervised CE + richer clean features; Table 3 từ domain confusion và style/content disentanglement; Table 4 từ hard Delete/Replace views và consistency training. Các nghiên cứu stress-testing AIGT detectors cho thấy robustness thường không chuyển tự động giữa các attack families, nên training phải chứa nhiều dạng corruption thay vì tối ưu chỉ synonym replacement. citeturn398977search3

### Length-Invariant Hierarchical Readout

Một global sum hoặc uncalibrated attention pooling dễ làm magnitude và evidence count phụ thuộc document length. PRISM-Detect thay bằng **sentence-normalized hierarchical pooling**.

Với sentence \(s\) có \(n_s\) nodes:

\[
u_s
=
\frac{
\sum_{i\in s}
\exp(a_i)\,\tilde h_i
}{
\sum_{i\in s}\exp(a_i)
}.
\]

Sau đó mỗi \(u_s\) được LayerNorm:

\[
\bar u_s=\mathrm{LN}(u_s).
\]

Document representation là normalized first/second moments thay vì sum:

\[
\mu
=
\frac1S\sum_{s=1}^{S}\bar u_s,
\]

\[
\sigma
=
\sqrt{
\frac1S
\sum_{s=1}^{S}
(\bar u_s-\mu)^2+\epsilon
}.
\]

Cuối cùng:

\[
z_{\mathrm{doc}}
=
W_h[\mu;\sigma;u_{\mathrm{CLS}}],
\]

trong đó không đưa raw document length vào classifier.

Thêm **length-crop consistency**: lấy hai crops cùng document với số câu khác nhau nhưng đủ semantic coverage,

\[
\mathcal L_{\mathrm{len}}
=
1-
\cos
(
z_{\mathrm{doc}}^{(c_1)},
z_{\mathrm{doc}}^{(c_2)}
).
\]

Loss cuối có thêm \(+\lambda_l\mathcal L_{\mathrm{len}}\).

Module này nhắm trực tiếp vào Table 3. PRDetect tự báo cáo rằng khi model huấn luyện trên long-form GPT3.5-Mixed được test ở các đoạn chỉ 10–100 tokens, accuracy giảm xuống khoảng 0.585–0.69, và tác giả xem length variation là limitation thực tế. citeturn337880view2 Hierarchical normalization không xóa information về distribution của syntax motifs; nó chỉ ngăn **số lượng câu/node** trở thành shortcut tuyến tính.

Nó cũng giúp Table 2 vì clean và perturbed versions có thể có số token hơi khác; Table 4 đặc biệt hưởng lợi ở Delete/Insert vì representation không scale trực tiếp theo node count.

## Kỳ vọng tác động lên Tables 2–4 và trade-off tính toán

Với Table 2, mục tiêu không phải vượt CoCo 1.0000—điều đó là bất khả về mặt metric khi baseline đã đạt ceiling—mà là **vượt PRDetect tại tất cả perturbation ratios đồng thời giảm clean gap**. Typed dependency relations làm representation giàu hơn, trong khi invariance loss buộc richer representation không trở nên dễ vỡ trước synonym substitution. Vì PRDetect đã quanh 0.986–0.988 trên HC3, biên cải thiện thực tế rất nhỏ và cần paired significance testing; một gain 0.2–0.5 percentage point ở đây đã có thể meaningful hơn một gain lớn trên Table 4.

Với Table 3, đây là nơi kiến trúc có dư địa lớn nhất. Domain-adversarial nuisance removal xử lý lexical/topic leakage; hierarchical readout giảm length shortcut; relation-aware syntax giữ structural signal có khả năng transfer tốt hơn raw RoBERTa semantics. Gain lớn hơn ở HC3\(\rightarrow\)GPT3.5-Mixed được kỳ vọng vì baseline hiện chỉ 73.2%; chiều ngược lại đã 87.7% nên ceiling cao hơn.

Với Table 4, hard Delete/Replace augmentation cộng fallback edges và edge denoising nhắm đúng failure mode. Insert/Repeat phải được coi là **non-regression constraints**: một cải tiến chỉ đưa Replace/Delete lên cao nhưng làm Insert/Repeat tụt đáng kể không nên được xem là vượt PRDetect toàn diện.

Về efficiency, Table 6 báo PRDetect trên GPT3.5-Mixed, RTX 4090, 15 epochs: preprocessing 1754.5 s, training 629.7 s, testing 2.5 s; CoCo lần lượt 2545.3/1655.0/28.3 s. citeturn337880view2 Đây là baseline cần giữ.

PRISM-Detect không cần parser thứ hai trong default configuration; POS, dependency type và depth đều lấy từ **cùng một spaCy parse**, nên preprocessing asymptotically không thay đổi. RoBERTa cũng đã có trong PRDetect node encoding. Extra work chủ yếu là embedding lookup và \(O(|E|d)\) relation-aware attention. Nếu RoBERTa encoder frozen/cached và GNN vẫn 2 layers, **online graph-classifier latency có thể giữ gần cùng order of magnitude với PRDetect**, nhưng không nên tuyên bố 2.5 s trước benchmark.

Cần tách rõ hai loại latency. **Offline preprocessing latency** gồm spaCy parsing + RoBERTa feature extraction + graph serialization. **Online detector latency** chỉ được báo 2.5 s nếu graph/features đã cache giống setup Table 6; nếu production pipeline nhận raw text, phải báo thêm raw-text end-to-end latency. So sánh khác định nghĩa này sẽ không công bằng.

## Giao thức thực nghiệm đủ mạnh để chứng minh “outperform PRDetect”

Main replication trước hết phải tái tạo nguyên trạng PRDetect với five seeds 2021–2025, cùng split, same spaCy/RoBERTa version và cùng 15 epochs. Table 2 của paper chỉ báo mean±SD cho PRDetect, trong khi Table 4 chỉ dùng seed 2024; do đó Table 4 cần được chạy lại đủ 5 seeds trước khi so architecture mới. citeturn337880view0turn337880view1

Bộ đánh giá nên bao phủ toàn bộ clean và perturbation ratios 0/5/10/20/30% trên cả HC3 và GPT3.5-Mixed; cả hai transfer directions HC3\(\rightarrow\)GPT3.5-Mixed và GPT3.5-Mixed\(\rightarrow\)HC3; và Insert/Repeat/Replace/Delete 25% trên HC3. Accuracy phải được bổ sung **macro-F1, AUROC, balanced accuracy và calibration error**, bởi một detector có thể giữ accuracy nhưng đổi false-positive distribution.

Đối với Table 4, mỗi perturbed sample phải được paired với source sample và log các structural drift statistics:

\[
\Delta_E=
1-\frac{|E(x)\cap E(x')|}{|E(x)\cup E(x')|},
\]

\[
\Delta_{\mathrm{depth}}
=
|\bar d(x)-\bar d(x')|,
\]

cùng root-change rate, sentence-boundary-change rate và dependency-label-change rate. Vì spaCy reparses đầu vào hoàn chỉnh, phép đo này mới kiểm nghiệm trực tiếp giả thuyết “Delete/Replace gây parse drift”; không nên suy ra từ accuracy alone. spaCy thực sự dự đoán đồng thời heads, dependency labels và sentence boundaries, nên những drift metrics này có ý nghĩa cơ chế. citeturn846428search0turn846428search5

Để kiểm soát **length leakage**, huấn luyện một probe chỉ nhận \(n_{\text{tokens}},n_{\text{sentences}},n_{\text{nodes}}\). Sau đó đánh giá detector trên length-matched subsets giống tinh thần Appendix Table 8. Nếu PRISM-Detect vẫn cải thiện sau matching, gain không thể quy hoàn toàn cho length shortcut. PRDetect đã cho thấy accuracy vẫn trên 0.90 trong length-matched setting, nên đây là control bắt buộc chứ không phải phụ. citeturn337880view3

Để kiểm soát **topic leakage**, xây split theo topic/source thay vì random document split, đặc biệt với HC3 domains và 17 sources của GPT3.5-Mixed. Để kiểm soát **generator leakage**, nếu metadata cho phép, generator family phải được held out; nếu không, report limitation thay vì diễn giải cross-dataset result thành universal generator generalization.

Với perturbation, cần đánh giá **label validity**. Random 25% deletion có thể biến văn bản thành fragment không tự nhiên; synonym replacement trong PRDetect cũng làm readability giảm khoảng 8.3–8.5% theo GPT-4o-mini assessment của tác giả. citeturn337880view3 Vì vậy nên report song song robustness trên “all perturbations” và subset vượt một fixed readability/semantic-preservation threshold; threshold phải định trước, không tune theo detector.

Significance testing nên dùng **paired bootstrap 10,000 resamples** trên cùng test instances, báo 95% CI cho \(\Delta\)Accuracy và \(\Delta\)macro-F1, đồng thời báo mean±SD qua 5 seeds. Với nhiều ô ở Table 2–4, có thể dùng Holm correction cho family-wise comparisons. Điều kiện mạnh để tuyên bố “outperform across all benchmarks” là lower bound của 95% CI cho paired difference \(>0\) ở từng benchmark family; nếu một vài ô chỉ tie statistically, claim phải hạ xuống “non-inferior”.

## Ablation matrix và các gain kỳ vọng cần được xem là giả thuyết

Các con số dưới đây là **hypothetical target ranges**, dùng để định power/acceptance criteria; chúng không phải kết quả đã quan sát.

| Ablation | Cơ chế bị bỏ | Metric/setting nhạy nhất | Gain kỳ vọng giả thuyết nếu thêm lại | Giải thích kiểm chứng |
|---|---|---|---:|---|
| PRISM-full \(-\) typed relations/POS/depth | Quay gần binary graph của PRDetect | Table 2 clean; Table 3 cross-domain | +0.2–0.6 pp clean; +2–5 pp transfer | Kiểm tra liệu grammatical-function signal thực sự bổ sung discriminability |
| PRISM-full \(-\) edge-denoising/fallback edges | Chỉ dependency arcs cứng | Table 4 Replace/Delete | +5–12 pp Replace; +8–15 pp Delete | Nếu gain biến mất ở Delete, giả thuyết parse-topology robustness được hỗ trợ |
| PRISM-full \(-\) semantic residual stream | Syntax-only | Table 4 Delete/Replace; Table 2 clean | +2–6 pp hard perturbation | Đo liệu semantic fallback cứu trường hợp parse drift hay không |
| PRISM-full \(-\) contrastive + consistency hard augmentation | Chỉ BCE | Table 4; Table 2 20–30% | +4–10 pp Replace/Delete; <1 pp synonym | Xác định robustness đến từ architecture hay training invariance |
| PRISM-full \(-\) domain adversarial/decorr | Giữ topical semantics | Table 3 | +4–10 pp HC3→Mixed; +2–5 pp reverse | Dùng domain-probe accuracy để kiểm tra representation leakage |
| PRISM-full \(-\) hierarchical length-invariant readout | Global pooling | Table 3 + length-stratified test | +3–7 pp HC3→Mixed | Gain phải đi kèm giảm correlation giữa confidence và document length |
| PRISM-full | Toàn bộ hệ thống | Table 2/3/4 | Mục tiêu: >PRDetect từng setting, không regression Insert/Repeat | Chỉ chấp nhận claim khi paired CI hỗ trợ |

Một target hợp lý cho Table 3 là đưa HC3\(\rightarrow\)GPT3.5-Mixed từ 73.2% lên vùng **low/mid-80s** clean và giữ degradation từ 0→30% dưới khoảng 2 percentage points; chiều GPT3.5-Mixed\(\rightarrow\)HC3 nên hướng tới low-90s. Với Table 4, mục tiêu nghiên cứu thực tế hơn là **Replace >0.88–0.92 và Delete >0.85–0.90**, đồng thời Insert/Repeat không thấp hơn khoảng 0.98. Đây chỉ là acceptance targets để đánh giá sức mạnh của proposal, không phải predicted results có độ tin cậy thống kê.

Điểm quyết định của proposal là các module có causal story thống nhất: **typed graph làm syntax representation giàu hơn; edge-denoising và semantic residual làm representation không gãy khi parse drift; contrastive/domain-invariant objectives buộc detector bỏ lexical/topic shortcuts; hierarchical readout loại length shortcut**. Nếu ablation cho thấy một module không tạo gain ở failure mode mà nó được thiết kế để xử lý, module đó nên bị loại thay vì giữ vì complexity.

Về chuẩn công bố ACL/EMNLP/NAACL, contribution mạnh nhất sẽ không chỉ là “accuracy cao hơn PRDetect”, mà là chứng minh ba mệnh đề bằng empirical intervention: **(i)** relation-aware structure nâng clean ceiling mà không phá synonym robustness; **(ii)** domain/length invariance giải thích và thu hẹp cross-dataset asymmetry; **(iii)** learned edge reliability + hard perturbation training làm Replace/Delete robustness correlate trực tiếp với giảm parse-drift sensitivity. Khi ba mệnh đề này cùng được xác nhận bằng 5-seed replication, paired bootstrap CI, length/topic controls và efficiency accounting tương thích Table 6, PRISM-Detect mới có cơ sở thuyết phục để được xem là một successor thực nghiệm của PRDetect thay vì một kiến trúc phức tạp hơn nhưng thiếu causal evidence. citeturn398977search2turn398977search3turn733665search2turn733665search3