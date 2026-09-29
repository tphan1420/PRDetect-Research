# Báo cáo Nâng cấp Kiến trúc Mô hình PRDetect: Chuyển đổi Hình thái Đồ thị và Tích chập Đa Quan hệ (RGCN)

> **Mục tiêu:** Hiện thực hóa Mục 3 trong tài liệu `update.md`, giải quyết hiện tượng suy giảm hiệu suất của mô hình PRDetect trên tập benchmark RAID thông qua việc nâng cấp biểu diễn cây cú pháp từ đồ thị đồng nhất sang đồ thị đa quan hệ (Multi-Relational Heterogeneous Graph).

---

## 1. Bối cảnh & Cơ sở Toán học của Sự Cải tiến

### 1.1. Giới hạn của Kiến trúc GCN Đồng nhất Hiện tại
Trong mô hình PRDetect nguyên bản (Findings of NAACL 2025), cây cú pháp phụ thuộc (Dependency Syntax Tree) được biểu diễn bằng một đồ thị đồng nhất đơn giản với ma trận kề nhị phân $A \in \{0, 1\}^{N \times N}$, trong đó $A_{ij} = 1$ nếu có liên kết phụ thuộc giữa từ $i$ và từ $j$.

Quy tắc lan truyền thông điệp của mạng GCN tiêu chuẩn là:

$$H^{(l+1)} = \sigma\left(\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}H^{(l)}W^{(l)}\right)$$

Trong đó $\tilde{A} = A + I_N$ (đồ thị có self-loop) và $\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$.

> [!WARNING]
> **Điểm nghẽn toán học:** GCN truyền thống sử dụng **duy nhất một ma trận trọng số $W^{(l)}$** cho tất cả các cạnh. Điều này san phẳng toàn bộ tính chất ngữ pháp: liên kết giữa động từ và chủ ngữ (`nsubj`) bị đối xử hoàn toàn bình đẳng với liên kết giữa danh từ và mạo từ (`det`) hay dấu câu (`punct`). Khi đối đầu với 11 mô hình LLM và các kỹ thuật giải mã (Sampling, Repetition Penalty) trong tập dữ liệu RAID, mô hình mất đi khả năng nhận diện các dấu vân tay cú pháp đặc thù.

---

### 1.2. Kiến trúc Đề xuất: Mạng Tích chập Đồ thị Quan hệ (RGCN)
Nhằm giữ trọn vẹn thông tin ngữ pháp, đồ thị cú pháp được mô hình hóa thành **đồ thị đa quan hệ có hướng** $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathcal{R})$, trong đó mỗi cạnh $(j, r, i) \in \mathcal{E}$ mang nhãn quan hệ cú pháp $r \in \mathcal{R}$.

#### A. Công thức cập nhật trạng thái ẩn (Message Passing)
Trạng thái ẩn của node $i$ tại tầng $l+1$ được định nghĩa theo công thức:

$$h_{i}^{(l+1)} = \sigma\left(\sum_{r\in\mathcal{R}}\sum_{j\in\mathcal{N}_{i}^{r}}\frac{1}{c_{i,r}}W_{r}^{(l)}h_{j}^{(l)} + W_{0}^{(l)}h_{i}^{(l)}\right)$$

- $\mathcal{R}$: Không gian gồm **60+ kiểu quan hệ cú pháp phụ thuộc** (`nsubj`, `dobj`, `amod`, `prep`, `compound`, `ROOT`,...).
- $\mathcal{N}_{i}^{r} = \{j \in \mathcal{V} \mid (j, i) \in \mathcal{E}_{r}\}$: Tập hợp các node lân cận nối tới node $i$ thông qua quan hệ $r$.
- $W_{r}^{(l)} \in \mathbb{R}^{d_{l+1} \times d_l}$: Ma trận trọng số biến đổi riêng biệt dành cho quan hệ $r$.
- $W_{0}^{(l)} \in \mathbb{R}^{d_{l+1} \times d_l}$: Trọng số tự lặp riêng biệt (Root / Self-loop weight), giúp duy trì biểu diễn nội tại của node trung tâm $i$.
- $c_{i,r}$: Hằng số chuẩn hóa cấu trúc theo bậc vào của quan hệ:
  $$c_{i,r} = |\mathcal{N}_{i}^{r}|$$
  giúp triệt tiêu hiện tượng mất cân bằng gradient khi một số quan hệ xuất hiện với tần suất quá cao.

---

### 1.3. Điều chuẩn Toán học: Phân rã Cơ sở (Basis-Decomposition)

Nếu phân bổ ma trận độc lập $W_r$ cho từng quan hệ, số lượng tham số cần tối ưu ở mỗi tầng là:

$$\text{Params}_{\text{full}} = |\mathcal{R}| \times d_l \times d_{l+1}$$

Với $|\mathcal{R}| \approx 60$, $d_l = 768$, $d_{l+1} = 256$, một tầng tích chập đòi hỏi tới **$\sim 11.8$ triệu tham số**, tiềm ẩn rủi ro quá khớp (overfitting) rất lớn.

Để giải quyết vấn đề này, cơ chế **Phân rã cơ sở (Basis-Decomposition)** được áp dụng: Mỗi ma trận $W_r^{(l)}$ được phân rã thành tổ hợp tuyến tính của $B$ ma trận cơ sở dùng chung $V_b^{(l)}$:

$$W_{r}^{(l)} = \sum_{b=1}^{B} a_{rb}^{(l)} V_{b}^{(l)}$$

- $V_{b}^{(l)} \in \mathbb{R}^{d_{l+1} \times d_l}$: Các ma trận cơ sở độc lập được chia sẻ giữa tất cả các quan hệ.
- $a_{rb}^{(l)} \in \mathbb{R}$: Hệ số chú ý/trọng số kết hợp của quan hệ $r$ trên cơ sở thứ $b$.
- Số lượng tham số giảm xuống chỉ còn:
  $$\text{Params}_{\text{basis}} = B \times d_l \times d_{l+1} + |\mathcal{R}| \times B$$
  Với $B = 30$, lượng tham số **giảm gần 50%**, đồng thời ép các quan hệ ngữ pháp có tính chất tương đồng (như nhóm bổ nghĩa `amod`, `advmod`, `npadvmod`) chia sẻ tri thức biểu diễn, giúp mô hình tổng quát hóa vượt trội trên các miền văn bản mới.

---

## 2. Bảng Tổng hợp các Thành phần Đã Triển khai

| STT | Thành phần / File | Vai trò & Đóng góp Kỹ thuật |
| :---: | :--- | :--- |
| 1 | [`dep_vocab.py`](file:///d:/Tailieu/HK7/HT%20Th%C3%B4ng%20minh/PRDetect-Research/dep_vocab.py) & [`dep_vocab.json`](file:///d:/Tailieu/HK7/HT%20Th%C3%B4ng%20minh/PRDetect-Research/dep_vocab.json) | Xây dựng bảng tra cứu chuẩn hóa 60+ quan hệ cú pháp phụ thuộc của SpaCy; hỗ trợ mã hóa/giải mã nhất quán xuyên suốt pipeline. |
| 2 | [`building_graph.py`](file:///d:/Tailieu/HK7/HT%20Th%C3%B4ng%20minh/PRDetect-Research/building_graph.py) | Nâng cấp hàm `build_graph()` để trích xuất cả liên kết cạnh và nhãn `edge_type`; hỗ trợ tùy chọn cạnh đảo hai chiều (`--bidirectional`); tự động lưu trữ tensor đa quan hệ vào `.pkl`. |
| 3 | [`model/RGCN.py`](file:///d:/Tailieu/HK7/HT%20Th%C3%B4ng%20minh/PRDetect-Research/model/RGCN.py) | Định nghĩa các lớp mạng `RGCN2` (2 tầng), `RGCN4` (4 tầng), và `RelationalGCN` (tùy biến số tầng, pooling). Tích hợp sẵn cơ chế fallback thuần PyTorch nếu môi trường chưa cài PyG. |
| 4 | [`rgcn.py`](file:///d:/Tailieu/HK7/HT%20Th%C3%B4ng%20minh/PRDetect-Research/rgcn.py) | Script huấn luyện RGCN chuyên dụng; hỗ trợ Early Stopping, ghi log TensorBoard, lưu checkpoint tốt nhất và tự động đánh giá tập kiểm thử. |
| 5 | [`test_rgcn.py`](file:///d:/Tailieu/HK7/HT%20Th%C3%B4ng%20minh/PRDetect-Research/test_rgcn.py) & [`test.py`](file:///d:/Tailieu/HK7/HT%20Th%C3%B4ng%20minh/PRDetect-Research/test.py) | Công cụ đánh giá hiệu năng mô hình với các chỉ số Accuracy, ROC-AUC, F1-score; `test.py` hỗ trợ cờ thống nhất `--model_type {gcn, rgcn}`. |
| 6 | [`detect.py`](file:///d:/Tailieu/HK7/HT%20Th%C3%B4ng%20minh/PRDetect-Research/detect.py) | Công cụ suy luận CLI phân loại văn bản tùy ý bằng RGCN hoặc GCN; khắc phục lỗi cấu trúc JSON đầu vào trong bản cũ. |
| 7 | [`README.md`](file:///d:/Tailieu/HK7/HT%20Th%C3%B4ng%20minh/PRDetect-Research/README.md) & [`DuAn.md`](file:///d:/Tailieu/HK7/HT%20Th%C3%B4ng%20minh/PRDetect-Research/DuAn.md) | Cập nhật toàn diện tài liệu kỹ thuật, sơ đồ kiến trúc và hướng dẫn thực thi theo chuẩn nghiên cứu khoa học. |

---

## 3. Lệnh Thực thi Nhanh (Cheatsheet)

```bash
# 1. Trích xuất đồ thị đa quan hệ (kèm nhãn edge_type)
python building_graph.py --file raid_mistral_test

# 2. Huấn luyện mô hình RGCN (2 tầng, basis decomposition B=30)
python rgcn.py --dataset hc3 --seed 2024 --epochs 40 --lr 0.0001 --num_layers 2 --num_bases 30

# 3. Kiểm thử đánh giá hiệu năng RGCN
python test_rgcn.py --file raid_mistral_test --dataset hc3 --seed 2024 -s

# 4. Suy luận trực tiếp trên đoạn văn bản mới
python detect.py --text "Artificial intelligence has fundamentally changed modern text analysis." --dataset hc3 --seed 2024 --model_type rgcn
```

---

## 4. Quản lý Phiên bản trên Git

- **Nhánh mới:** `feat/rgcn-multi-relational`
- **Mã Commit:** `70e50c4`
- **Tình trạng kiểm thử:** Đã vượt qua toàn bộ các bài kiểm tra tích hợp (End-to-End Test) từ khâu nạp dữ liệu, lan truyền tiến/lùi (Forward/Backward), huấn luyện, tối ưu gradient, lưu checkpoint đến suy luận kiểm thử.
