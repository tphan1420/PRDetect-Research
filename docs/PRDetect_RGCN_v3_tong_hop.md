# PRDetect + R-GCN v3 — Giai đoạn 3 (HC3, Table 2)

> Tổng hợp từ notebook `gpt3-5__2_.ipynb` thành một file Markdown duy nhất (giữ nguyên nội dung markdown và mã nguồn của từng ô).

## Mục lục nhanh

- Tổng quan & cải tiến mới
- Ô 1: Cài đặt môi trường, clone repo
- Ô 2: Dựng đồ thị HC3
- Ô 3: Ghi file `rgcn_plus.py` (mô hình + huấn luyện)
- Mục 1: Kiểm thử đơn vị
- Mục 2: Sweep 1 seed (V0–V5)
- Mục 3: Chạy chính thức 5 seed + bảng tổng hợp
- Mục 4: (Tùy chọn) Bảng 4 của bài báo

---

# PRDetect + R-GCN v3 — **Giai đoạn 3 (HC3, Table 2): huấn luyện bền với nhiễu**

**Hiện trạng (C2_wide, 5 seed) so với bài báo:** 0%: 0.9896 (+0.0018) · 5%: 0.9874 (−0.0004) · 10%: 0.9862 (−0.0010) · 20%: 0.9856 (−0.0018) · 30%: 0.9832 (−0.0032).
Acc sạch đã bão hòa (1 mẫu test = 0.001; CI 95% ±0.0067), nên **chỗ còn lại là độ bền khi nhiễu tăng**: v3 mất 0.0064 từ 0%→30%, bài báo chỉ mất 0.0014.

**Cải tiến mới (đều ở mức đặc trưng, KHÔNG dùng các file `*_adj_*_synonym_replace` để huấn luyện/chọn mô hình):**
1. **Nhiễu thay từ mức nút** (`--aug swap|mask|both`): mỗi đồ thị chọn ngẫu nhiên tới `aug_max` nút; *swap* thay vector bằng vector của nút khác **cùng nhãn quan hệ phụ thuộc** (mô phỏng thay từ mà giữ vai trò cú pháp), *mask* thay bằng vector học được. Cấu trúc cây giữ nguyên.
2. **Mất mát nhất quán** (`--cons`): KL đối xứng giữa dự đoán trên bản sạch và bản nhiễu (kiểu R-Drop) → ép mô hình bất biến với thay từ.
3. **Lịch học ngắn** (`epochs=10`): ở giai đoạn 1 val-loss luôn tốt nhất ở epoch 5–10 khi LR cosine còn cao.
4. **Chọn checkpoint theo `val_loss + val_loss_proxy_nhiễu`** (proxy = nhiễu *swap* áp lên tập val, seed cố định). Không nhìn test để chọn.

**Mục tiêu thực tế:** thắng bài báo ở cả 5 cột (cần 30%: ≥ 0.9864, tức +0.003 so với hiện tại). Tăng thêm 0.01 tuyệt đối ở cột 0% là không khả thi (0.9896→0.9996 ≈ 0–1 lỗi trên 1000 mẫu).

```python
import os, glob, re, subprocess, sys, time, json, pickle
os.chdir('/kaggle/working')
if not os.path.exists('PRDetect-Research'):
    !git clone --depth 1 https://github.com/tphan1420/PRDetect-Research.git
os.chdir('/kaggle/working/PRDetect-Research')
!pip install -q torch_geometric
!python -m spacy download en_core_web_sm
!df -h /kaggle/working | tail -1
```

**Kết quả:**

```text
Cloning into 'PRDetect-Research'...
remote: Enumerating objects: 377, done.[K
remote: Counting objects: 100% (377/377), done.[K
remote: Compressing objects: 100% (265/265), done.[K
remote: Total 377 (delta 91), reused 334 (delta 84), pack-reused 0 (from 0)[K
Receiving objects: 100% (377/377), 131.28 MiB | 26.28 MiB/s, done.
Resolving deltas: 100% (91/91), done.
Updating files: 100% (263/263), done.
Collecting en-core-web-sm==3.8.0
  Downloading en_core_web_sm-3.8.0-py3-none-any.whl (12.8 MB)
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m12.8/12.8 MB[0m [31m27.0 MB/s[0m  [33m0:00:00[0m
[?25h[38;5;2m✔ Download and installation successful[0m
You can now load the package via spacy.load('en_core_web_sm')
[38;5;3m⚠ Restart to reload dependencies[0m
If you are in a Jupyter or Colab notebook, you may need to restart Python in
order to load all the package's dependencies. You can do this by selecting the
'Restart kernel' or 'Restart runtime' option.
/dev/loop2       20G  886M   19G   5% /kaggle/working
```

```python
names = {}
for p in glob.glob('data/**/*.json', recursive=True):
    b = os.path.basename(p)[:-5]
    if b.startswith('hc3'): names[b] = p
TRAIN, VAL, TEST = 'hc3_train', 'hc3_val', 'hc3_test'
TESTS = [TEST] + [f'hc3_test_adj_{r}_synonym_replace' for r in (5, 10, 20, 30)]
assert all(n in names for n in [TRAIN, VAL] + TESTS), 'thiếu file dữ liệu HC3'
DATA = f'--train_file {TRAIN} --val_file {VAL} --test_files {" ".join(TESTS)}'
os.makedirs('graph_data', exist_ok=True)
todo = [n for n in [TRAIN, VAL] + TESTS if not os.path.exists(f'graph_data/{n}.pkl')]
print('Cần dựng:', todo)
if todo: subprocess.run('TQDM_MININTERVAL=30 python building_graph.py --file ' + ' '.join(todo), shell=True)
!ls graph_data | head -20; df -h /kaggle/working | tail -1
```

**Kết quả:**

```text
Cần dựng: ['hc3_train', 'hc3_val', 'hc3_test', 'hc3_test_adj_5_synonym_replace', 'hc3_test_adj_10_synonym_replace', 'hc3_test_adj_20_synonym_replace', 'hc3_test_adj_30_synonym_replace']
```

**Kết quả:**

```text
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
```

**Kết quả:**

```text

[Processing] Đang xử lý xây dựng đồ thị cho: hc3_train ...
-> Nạp văn bản từ: data/original_text/benchmarks/hc3/hc3_train.json
-> Đã đọc 8000 mẫu văn bản từ file hc3_train
--> Đang nạp mô hình SpaCy (en_core_web_sm) và RoBERTa trên thiết bị: cuda ...
```

**Kết quả:**

```text
Loading weights: 100%|██████████| 197/197 [00:00<00:00, 2541.63it/s]
[transformers] [1mRobertaModel LOAD REPORT[0m from: roberta-base
Key                       | Status     | 
--------------------------+------------+-
lm_head.layer_norm.weight | UNEXPECTED | 
lm_head.layer_norm.bias   | UNEXPECTED | 
lm_head.bias              | UNEXPECTED | 
lm_head.dense.weight      | UNEXPECTED | 
lm_head.dense.bias        | UNEXPECTED | 
pooler.dense.weight       | MISSING    | 
pooler.dense.bias         | MISSING    | 

Notes:
- UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.
- MISSING:	those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
Building Graph: 100%|██████████| 8000/8000 [05:23<00:00, 24.72it/s]
```

**Kết quả:**

```text
Elapsed time: 323.62 seconds | Tổng mẫu hợp lệ: 8000
[OK] Đã lưu dữ liệu đồ thị thành công tại: ./graph_data/hc3_train.pkl (CÓ quan hệ đa loại (RGCN))

[Processing] Đang xử lý xây dựng đồ thị cho: hc3_val ...
-> Nạp văn bản từ: data/original_text/benchmarks/hc3/hc3_val.json
-> Đã đọc 1000 mẫu văn bản từ file hc3_val
```

**Kết quả:**

```text
Building Graph: 100%|██████████| 1000/1000 [00:40<00:00, 24.42it/s]
```

**Kết quả:**

```text
Elapsed time: 40.96 seconds | Tổng mẫu hợp lệ: 1000
[OK] Đã lưu dữ liệu đồ thị thành công tại: ./graph_data/hc3_val.pkl (CÓ quan hệ đa loại (RGCN))

[Processing] Đang xử lý xây dựng đồ thị cho: hc3_test ...
-> Nạp văn bản từ: data/original_text/benchmarks/hc3/hc3_test.json
-> Đã đọc 1000 mẫu văn bản từ file hc3_test
```

**Kết quả:**

```text
Building Graph: 100%|██████████| 1000/1000 [00:42<00:00, 23.58it/s]
```

**Kết quả:**

```text
Elapsed time: 42.41 seconds | Tổng mẫu hợp lệ: 1000
[OK] Đã lưu dữ liệu đồ thị thành công tại: ./graph_data/hc3_test.pkl (CÓ quan hệ đa loại (RGCN))

[Processing] Đang xử lý xây dựng đồ thị cho: hc3_test_adj_5_synonym_replace ...
-> Nạp văn bản từ: data/perturbed_text/hc3/lexical_synonym/hc3_test_adj_5_synonym_replace.json
-> Đã đọc 1000 mẫu văn bản từ file hc3_test_adj_5_synonym_replace
```

**Kết quả:**

```text
Building Graph: 100%|██████████| 1000/1000 [00:42<00:00, 23.70it/s]
```

**Kết quả:**

```text
Elapsed time: 42.20 seconds | Tổng mẫu hợp lệ: 1000
[OK] Đã lưu dữ liệu đồ thị thành công tại: ./graph_data/hc3_test_adj_5_synonym_replace.pkl (CÓ quan hệ đa loại (RGCN))

[Processing] Đang xử lý xây dựng đồ thị cho: hc3_test_adj_10_synonym_replace ...
-> Nạp văn bản từ: data/perturbed_text/hc3/lexical_synonym/hc3_test_adj_10_synonym_replace.json
-> Đã đọc 1000 mẫu văn bản từ file hc3_test_adj_10_synonym_replace
```

**Kết quả:**

```text
Building Graph: 100%|██████████| 1000/1000 [00:39<00:00, 25.23it/s]
Building Graph:   0%|          | 0/1000 [00:00<?, ?it/s]
```

**Kết quả:**

```text
Elapsed time: 39.65 seconds | Tổng mẫu hợp lệ: 1000
[OK] Đã lưu dữ liệu đồ thị thành công tại: ./graph_data/hc3_test_adj_10_synonym_replace.pkl (CÓ quan hệ đa loại (RGCN))

[Processing] Đang xử lý xây dựng đồ thị cho: hc3_test_adj_20_synonym_replace ...
-> Nạp văn bản từ: data/perturbed_text/hc3/lexical_synonym/hc3_test_adj_20_synonym_replace.json
-> Đã đọc 1000 mẫu văn bản từ file hc3_test_adj_20_synonym_replace
```

**Kết quả:**

```text
Building Graph: 100%|██████████| 1000/1000 [00:40<00:00, 24.92it/s]
```

**Kết quả:**

```text
Elapsed time: 40.13 seconds | Tổng mẫu hợp lệ: 1000
[OK] Đã lưu dữ liệu đồ thị thành công tại: ./graph_data/hc3_test_adj_20_synonym_replace.pkl (CÓ quan hệ đa loại (RGCN))

[Processing] Đang xử lý xây dựng đồ thị cho: hc3_test_adj_30_synonym_replace ...
-> Nạp văn bản từ: data/perturbed_text/hc3/lexical_synonym/hc3_test_adj_30_synonym_replace.json
-> Đã đọc 1000 mẫu văn bản từ file hc3_test_adj_30_synonym_replace
```

**Kết quả:**

```text
Building Graph: 100%|██████████| 1000/1000 [00:39<00:00, 25.10it/s]
```

**Kết quả:**

```text
Elapsed time: 39.85 seconds | Tổng mẫu hợp lệ: 1000
[OK] Đã lưu dữ liệu đồ thị thành công tại: ./graph_data/hc3_test_adj_30_synonym_replace.pkl (CÓ quan hệ đa loại (RGCN))
hc3_test_adj_10_synonym_replace.pkl
hc3_test_adj_20_synonym_replace.pkl
hc3_test_adj_30_synonym_replace.pkl
hc3_test_adj_5_synonym_replace.pkl
hc3_test.pkl
hc3_train.pkl
hc3_val.pkl
/dev/loop2       20G  8.2G   12G  42% /kaggle/working
```

```python
%%writefile rgcn_plus.py
# -*- coding: utf-8 -*-
"""
rgcn_plus.py - R-GCN cải tiến cho PRDetect (thuần PyTorch, không cần torch_geometric).

Đọc lại các file graph_data/*.pkl do building_graph.py của bạn tạo (KHÔNG sửa tiền xử lý).
Cải tiến so với bản R-GCN hiện tại (xem notebook để biết lý do):
  1. Lớp R-GCN tự cài đặt (basis decomposition, chuẩn hóa 1/c_{i,r}) -> nhanh hơn RGCNConv lặp 60 quan hệ.
  2. Chiếu 768->d trước, residual + LayerNorm, dropout đầu vào, DropEdge (trước chuẩn hóa như bài R-GCN).
  3. Đặc trưng cấu trúc cây tính từ edge_index: độ sâu, số con, khoảng cách phụ thuộc (+ thống kê mức đồ thị).
  4. Readout mean+max+attention (thay cho trung bình logit từng nút).
  5. AdamW + warmup/cosine + grad clip + EMA trọng số; chọn checkpoint theo val loss (không theo val acc).
  6. Ngưỡng quyết định chọn trên VAL (không dùng test); lưu xác suất để ensemble nhiều seed.
"""
import argparse, os, sys, time, json, math, pickle, copy, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------------------------
# 1. Đọc pkl (tự dò tên khóa vì không biết chính xác tên khóa trong pkl RGCN của bạn)
# ----------------------------------------------------------------------------------------------
def _load_pickle(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def _pick(keys, must, avoid=()):
    for k in keys:
        kl = k.lower()
        if any(m in kl for m in must) and not any(a in kl for a in avoid):
            return k
    return None


def read_raw(path):
    """Trả về (emb_list, ei_list, et_list, y_array) dạng numpy/torch theo từng đồ thị."""
    d = _load_pickle(path)
    if isinstance(d, (list, tuple)) and len(d) > 0 and hasattr(d[0], "x"):
        emb = [g.x for g in d]
        ei = [g.edge_index for g in d]
        et = [g.edge_type for g in d]
        y = [g.y for g in d]
    elif isinstance(d, dict):
        keys = list(d.keys())
        ke = _pick(keys, ["embed"]) or _pick(keys, ["feat", "x"])
        ki = _pick(keys, ["edge_index"]) or _pick(keys, ["edge"], avoid=["type", "rel", "attr"])
        kt = _pick(keys, ["edge_type", "etype", "rel"])
        ky = "y" if "y" in d else _pick(keys, ["label"])
        miss = [n for n, k in (("embedding", ke), ("edge_index", ki), ("edge_type", kt), ("label", ky)) if k is None]
        if miss:
            raise KeyError(f"Không tìm thấy khóa {miss} trong {path}. Các khóa có: {keys}")
        emb, ei, et, y = d[ke], d[ki], d[kt], d[ky]
    else:
        raise TypeError(f"Định dạng pkl không nhận diện được: {type(d)}")
    y = np.array([float(torch.as_tensor(v).reshape(-1)[0]) for v in y], dtype=np.float32)
    return list(emb), list(ei), list(et), y


# ----------------------------------------------------------------------------------------------
# 2. Đặc trưng cấu trúc cây phụ thuộc (tính từ edge_index + edge_type)
# ----------------------------------------------------------------------------------------------
MAXD = 64
N_GF = 8


def fwd_mask(et, R, mode):
    half = R // 2
    if mode == "low":
        return et < half
    return (et >= half) & (et < 2 * half)


def parent_array(n, src, dst, et, R, mode):
    m = fwd_mask(et, R, mode) & (src != dst)
    par = -np.ones(n, dtype=np.int64)
    par[src[m]] = dst[m]
    return par, int(m.sum()), np.unique(src[m]).size


def detect_mode(ei_list, et_list, R, n_probe=300):
    """Chiều 'dependent->head' là nhóm quan hệ thấp hay cao? Chọn nhóm có mỗi nút <=1 cạnh ra."""
    best = None
    for mode in ("low", "high"):
        ok = tot = 0
        for ei, et in list(zip(ei_list, et_list))[:n_probe]:
            ei = np.asarray(ei); et = np.asarray(et)
            if ei.shape[1] == 0:
                continue
            n = int(ei.max()) + 1
            _, ne, nu = parent_array(n, ei[0], ei[1], et, R, mode)
            tot += 1
            ok += int(ne == nu and ne > 0)
        frac = ok / max(tot, 1)
        if best is None or frac > best[1]:
            best = (mode, frac)
    return best


def struct_features(n, src, dst, et, R, mode):
    par, _, _ = parent_array(n, src, dst, et, R, mode)
    ptr = np.arange(n)
    depth = np.zeros(n, dtype=np.int64)
    for _ in range(MAXD):
        nxt = par[ptr]
        alive = nxt >= 0
        if not alive.any():
            break
        depth += alive
        ptr = np.where(alive, nxt, ptr)
    depth = np.minimum(depth, MAXD - 1)
    haspar = par >= 0
    nchild = np.bincount(par[haspar], minlength=n) if haspar.any() else np.zeros(n, dtype=np.int64)
    dist = np.where(haspar, np.abs(np.arange(n) - np.where(haspar, par, 0)), 0)
    dbucket = np.minimum(np.floor(np.log2(dist + 1)).astype(np.int64), 7)
    st = np.stack([np.minimum(depth, 31), np.minimum(nchild, 11), dbucket], 1).astype(np.int8)
    roots = ~haspar
    n_roots = max(int(roots.sum()), 1)
    h = np.zeros(n, dtype=np.int64)
    np.maximum.at(h, ptr, depth)
    heights = h[roots] if roots.any() else np.array([0])
    nonleaf = nchild > 0
    gf = np.array([
        math.log1p(n), math.log1p(n_roots), n / n_roots, depth.mean(), depth.max(),
        heights.mean(), np.log1p(dist).mean(), nchild[nonleaf].mean() if nonleaf.any() else 0.0,
    ], dtype=np.float32)
    return st, gf


# ----------------------------------------------------------------------------------------------
# 3. Bộ dữ liệu: gộp tất cả đồ thị thành tensor lớn, tự ghép batch (không cần PyG DataLoader)
# ----------------------------------------------------------------------------------------------
class GraphSet:
    def __init__(self, path, R=None, mode=None, store_dev="cpu", verbose=True):
        t0 = time.time()
        emb, ei, et, y = read_raw(path)
        G = len(y)
        if R is None:
            R = int(max(int(np.asarray(t).max()) for t in et if np.asarray(t).size > 0)) + 1
        if mode is None:
            mode, frac = detect_mode(ei, et, R)
            if verbose:
                print(f"   [struct] chiều dependent->head = nhóm quan hệ '{mode}' (tỉ lệ đồ thị hợp lệ {frac:.3f})")
            self.mode_frac = frac
        self.R, self.mode = R, mode
        nn_, ec = [], []
        X, SRC, DST, REL, ST, GF = [], [], [], [], [], []
        for i in range(G):
            x = emb[i]
            x = torch.as_tensor(x).to(torch.float16) if not torch.is_tensor(x) else x.to(torch.float16)
            emb[i] = None  # giải phóng bản float32 sớm
            n = x.shape[0]
            e = np.asarray(ei[i]).reshape(2, -1).astype(np.int64)
            t = np.asarray(et[i]).reshape(-1).astype(np.int64)
            if t.size and t.max() >= R:
                t = np.minimum(t, R - 1)
            st, gf = struct_features(n, e[0], e[1], t, R, mode)
            X.append(x); nn_.append(n); ec.append(e.shape[1])
            SRC.append(torch.from_numpy(e[0])); DST.append(torch.from_numpy(e[1])); REL.append(torch.from_numpy(t))
            ST.append(torch.from_numpy(st)); GF.append(gf)
        self.nn = torch.tensor(nn_, dtype=torch.long)
        self.ec = torch.tensor(ec, dtype=torch.long)
        self.no = torch.cat([torch.zeros(1, dtype=torch.long), self.nn.cumsum(0)])
        self.eo = torch.cat([torch.zeros(1, dtype=torch.long), self.ec.cumsum(0)])
        self.X = torch.cat(X); del X
        self.SRC, self.DST, self.REL = torch.cat(SRC), torch.cat(DST), torch.cat(REL)
        self.ST = torch.cat(ST).long()
        self.GF = torch.from_numpy(np.stack(GF))
        self.y = torch.from_numpy(y)
        self.G = G
        self.to(store_dev)
        if verbose:
            print(f"   [data] {os.path.basename(path)}: {G} đồ thị, {self.X.shape[0]} nút, R={R}, "
                  f"y mean={float(self.y.mean()):.3f}, {time.time()-t0:.0f}s")

    def to(self, dev):
        for k in ("nn", "ec", "no", "eo", "X", "SRC", "DST", "REL", "ST", "GF", "y"):
            setattr(self, k, getattr(self, k).to(dev))
        self.dev = dev
        return self

    def batch(self, idx, dev, gmean=None, gstd=None):
        idx = idx.to(self.dev)
        ln, le = self.nn[idx], self.ec[idx]
        B = idx.numel()
        cs_n = torch.cumsum(ln, 0) - ln
        cs_e = torch.cumsum(le, 0) - le
        node_ids = torch.arange(int(ln.sum()), device=self.dev) - torch.repeat_interleave(cs_n, ln) + torch.repeat_interleave(self.no[idx], ln)
        edge_ids = torch.arange(int(le.sum()), device=self.dev) - torch.repeat_interleave(cs_e, le) + torch.repeat_interleave(self.eo[idx], le)
        off = torch.repeat_interleave(cs_n, le)
        gf = self.GF[idx]
        if gmean is not None:
            gf = ((gf - gmean.to(self.dev)) / gstd.to(self.dev)).clamp(-5, 5)
        return dict(
            x=self.X[node_ids].to(dev).float(), st=self.ST[node_ids].to(dev),
            src=(self.SRC[edge_ids] + off).to(dev), dst=(self.DST[edge_ids] + off).to(dev), rel=self.REL[edge_ids].to(dev),
            bidx=torch.repeat_interleave(torch.arange(B, device=self.dev), ln).to(dev), B=B,
            gf=gf.to(dev), y=self.y[idx].to(dev),
        )


# ----------------------------------------------------------------------------------------------
# 4. Mô hình
# ----------------------------------------------------------------------------------------------
class RGCNLayer(nn.Module):
    """h_i' = W0 h_i + sum_r sum_{j in N_i^r} (1/c_{i,r}) W_r h_j ,  W_r = sum_b a_rb V_b  (Schlichtkrull 2018, Eq.2-3)."""
    def __init__(self, d, R, nb):
        super().__init__()
        self.d, self.R, self.nb = d, R, nb
        self.V = nn.Parameter(torch.empty(nb, d, d))
        self.comp = nn.Parameter(torch.empty(R, nb))
        self.W0 = nn.Linear(d, d)
        for b in range(nb):
            nn.init.xavier_uniform_(self.V.data[b])
        nn.init.normal_(self.comp, std=1.0 / math.sqrt(nb))

    def forward(self, x, src, dst, rel, norm):
        N, d, nb = x.size(0), self.d, self.nb
        H = (x @ self.V.permute(1, 0, 2).reshape(d, nb * d)).view(N, nb, d)
        coef = self.comp[rel] * norm.unsqueeze(1)            # [E, nb]
        msg = torch.einsum("ebd,eb->ed", H[src], coef.to(H.dtype)).float()
        agg = torch.zeros(N, d, device=x.device, dtype=torch.float32).index_add_(0, dst, msg)
        return self.W0(x).float() + agg


def seg_softmax(s, bidx, B):
    m = torch.full((B,), -1e30, device=s.device, dtype=s.dtype).scatter_reduce(0, bidx, s, "amax", include_self=True)
    e = torch.exp(s - m[bidx])
    den = torch.zeros(B, device=s.device, dtype=s.dtype).index_add_(0, bidx, e)
    return e / (den[bidx] + 1e-9)


class Net(nn.Module):
    def __init__(self, R, in_dim=768, d=256, nb=8, L=2, p=0.3, p_in=0.1, edge_drop=0.0,
                 use_struct=True, use_gfeat=False, readout="pool3", aug="none", aug_max=0.35, mode="low"):
        super().__init__()
        self.R, self.d, self.edge_drop = R, d, edge_drop
        self.aug, self.aug_max, self.mode = aug, aug_max, mode
        if aug in ("mask", "both"):
            self.mask_emb = nn.Parameter(torch.randn(in_dim) * 0.02)
        self.use_struct, self.use_gfeat, self.readout = use_struct, use_gfeat, readout
        self.drop_in = nn.Dropout(p_in)
        self.proj = nn.Linear(in_dim, d)
        self.ln0 = nn.LayerNorm(d)
        if use_struct:
            self.e_dep, self.e_chi, self.e_dis = nn.Embedding(32, 16), nn.Embedding(12, 8), nn.Embedding(8, 8)
            self.sproj = nn.Linear(32, d)
        self.layers = nn.ModuleList([RGCNLayer(d, R, nb) for _ in range(L)])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(L)])
        self.drop = nn.Dropout(p)
        if readout == "meanlogit":      # đúng kiểu PRDetect gốc: điểm từng nút rồi lấy trung bình
            self.head = nn.Linear(d, 1)
        else:
            self.gate = nn.Linear(d, 1)
            ind = 3 * d + (N_GF if use_gfeat else 0)
            self.mlp = nn.Sequential(nn.Linear(ind, d), nn.GELU(), nn.Dropout(p), nn.Linear(d, 1))

    def _augment(self, x, src, dst, rel, bidx, B, mode, gen):
        """Nhiễu mức đặc trưng mô phỏng thay từ: mỗi đồ thị chọn tỉ lệ rho~U(0,aug_max) nút;
        'swap' = thay vector bằng vector của một nút khác CÙNG nhãn quan hệ phụ thuộc (giữ vai trò cú pháp),
        'mask' = thay bằng vector mask học được. Cấu trúc cây không đổi."""
        N, dev = x.size(0), x.device
        half = self.R // 2
        node_rel = torch.full((N,), half, device=dev, dtype=torch.long)
        if self.mode == "low":
            m = (rel < half) & (src != dst)
            r = rel
        else:
            m = (rel >= half) & (rel < 2 * half) & (src != dst)
            r = rel - half
        node_rel[src[m]] = r[m]
        rho = torch.rand(B, device=dev, generator=gen) * self.aug_max
        sel = torch.rand(N, device=dev, generator=gen) < rho[bidx]
        coin = torch.rand(N, device=dev, generator=gen) < 0.5
        if mode == "mask":
            use_mask = sel
        elif mode == "both":
            use_mask = sel & coin
        else:
            use_mask = torch.zeros_like(sel)
        use_swap = sel & ~use_mask
        key = node_rel.float() * 2.0 + torch.rand(N, device=dev, generator=gen)
        order = torch.argsort(key)
        nxt = order.roll(-1)
        same = node_rel[order] == node_rel[nxt]
        donor = torch.empty_like(order)
        donor[order] = torch.where(same, nxt, order)
        x_new = torch.where(use_swap.unsqueeze(1), x[donor], x)
        if mode in ("mask", "both"):
            x_new = torch.where(use_mask.unsqueeze(1), self.mask_emb.to(x.dtype).unsqueeze(0).expand_as(x), x_new)
        return x_new

    def forward(self, b, aug_mode=None, gen=None):
        x, src, dst, rel, bidx, B = b["x"], b["src"], b["dst"], b["rel"], b["bidx"], b["B"]
        N = x.size(0)
        if aug_mode and aug_mode != "none":
            x = self._augment(x, src, dst, rel, bidx, B, aug_mode, gen)
        if self.training and self.edge_drop > 0 and src.numel() > 0:
            keep = torch.rand(src.numel(), device=src.device) >= self.edge_drop
            src, dst, rel = src[keep], dst[keep], rel[keep]
        key = dst * self.R + rel
        cnt = torch.bincount(key, minlength=N * self.R).float()
        norm = 1.0 / cnt[key].clamp(min=1.0)
        h = self.ln0(self.proj(self.drop_in(x)))
        if self.use_struct:
            st = b["st"]
            s = torch.cat([self.e_dep(st[:, 0]), self.e_chi(st[:, 1]), self.e_dis(st[:, 2])], 1)
            h = h + self.sproj(s)
        h = F.gelu(h)
        for lyr, ln in zip(self.layers, self.norms):
            h = h + self.drop(F.gelu(ln(lyr(h, src, dst, rel, norm))))
        if self.readout == "meanlogit":
            sc = self.head(h).squeeze(-1).float()
            cn = torch.zeros(B, device=h.device).index_add_(0, bidx, torch.ones_like(sc))
            return torch.zeros(B, device=h.device).index_add_(0, bidx, sc) / cn.clamp(min=1.0)
        h = h.float()
        cn = torch.zeros(B, device=h.device).index_add_(0, bidx, torch.ones(N, device=h.device)).clamp(min=1.0)
        mean = torch.zeros(B, h.size(1), device=h.device).index_add_(0, bidx, h) / cn.unsqueeze(1)
        mx = torch.full((B, h.size(1)), -1e30, device=h.device).scatter_reduce(
            0, bidx.unsqueeze(1).expand_as(h), h, "amax", include_self=True)
        w = seg_softmax(self.gate(h).squeeze(-1), bidx, B)
        att = torch.zeros(B, h.size(1), device=h.device).index_add_(0, bidx, h * w.unsqueeze(1))
        z = [mean, mx, att] + ([b["gf"]] if self.use_gfeat else [])
        return self.mlp(torch.cat(z, 1)).squeeze(-1)


# ----------------------------------------------------------------------------------------------
# 5. Huấn luyện / đánh giá
# ----------------------------------------------------------------------------------------------
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def predict(model, ds, dev, gm, gs, bs=64, amp=False, aug_mode=None):
    model.eval()
    gen = None
    if aug_mode:
        gen = torch.Generator(device=dev)
        gen.manual_seed(123)
    out = []
    for i in range(0, ds.G, bs):
        idx = torch.arange(i, min(i + bs, ds.G))
        b = ds.batch(idx, dev, gm, gs)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp and dev.type == "cuda"):
            out.append(torch.sigmoid(model(b, aug_mode=aug_mode, gen=gen).float()).cpu())
    return torch.cat(out).numpy()


def metrics(p, y, thr=0.5):
    from sklearn.metrics import roc_auc_score, f1_score, roc_curve
    y = np.asarray(y).astype(int)
    pred = (p >= thr).astype(int)
    acc = float((pred == y).mean())
    f1 = float(f1_score(y, pred))
    auc = float(roc_auc_score(y, p))
    fpr, tpr, _ = roc_curve(y, p)
    tpr1 = float(np.interp(0.01, fpr, tpr))
    return dict(acc=acc, f1=f1, auc=auc, tpr1=tpr1)


def pick_threshold(p, y):
    y = np.asarray(y).astype(int)
    ts = np.linspace(0.05, 0.95, 91)
    accs = np.array([((p >= t).astype(int) == y).mean() for t in ts])
    best = ts[accs >= accs.max() - 1e-9]
    return float(np.median(best))


def sym_kl(p, q, eps=1e-6):
    p, q = p.clamp(eps, 1 - eps), q.clamp(eps, 1 - eps)
    kl = lambda a, b: a * torch.log(a / b) + (1 - a) * torch.log((1 - a) / (1 - b))
    return 0.5 * (kl(p, q) + kl(q, p)).mean()


def val_loss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def train_one(args, seed, tr, va, tests, dev, R):
    set_seed(seed)
    gm = tr.GF.float().mean(0).cpu()
    gs = tr.GF.float().std(0).cpu().clamp(min=1e-6)
    model = Net(R, d=args.hidden, nb=args.num_bases, L=args.layers, p=args.dropout, p_in=args.in_dropout,
                edge_drop=args.edge_drop, use_struct=bool(args.struct), use_gfeat=bool(args.gfeat),
                readout=args.readout, aug=args.aug, aug_max=args.aug_max, mode=tr.mode).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    ema = copy.deepcopy(model) if args.ema else None
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    spe = math.ceil(tr.G / args.batch_size)
    total, warm = spe * args.epochs, max(1, int(0.05 * spe * args.epochs))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))) * 0.95 + 0.05)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp) and dev.type == "cuda")
    ytr = tr.y
    print(f"\n=== seed {seed} | tham số: {nparam/1e6:.2f}M | {spe} bước/epoch | tag={args.tag} ===")
    best = dict(score=1e9, ep=-1, state=None)
    bad, t0, step = 0, time.time(), 0
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(tr.G)
        tl = 0.0
        for i in range(0, tr.G, args.batch_size):
            b = tr.batch(perm[i:i + args.batch_size], dev, gm, gs)
            y = b["y"]
            if args.label_smooth > 0:
                y = y * (1 - args.label_smooth) + 0.5 * args.label_smooth
            with torch.autocast("cuda", dtype=torch.float16, enabled=bool(args.amp) and dev.type == "cuda"):
                logit = model(b)
                logit_a = model(b, aug_mode=args.aug) if args.aug != "none" else None
            loss = F.binary_cross_entropy_with_logits(logit.float(), y)
            if logit_a is not None:
                loss_a = F.binary_cross_entropy_with_logits(logit_a.float(), y)
                loss = 0.5 * (loss + loss_a)
                if args.cons > 0:
                    loss = loss + args.cons * sym_kl(torch.sigmoid(logit.float()), torch.sigmoid(logit_a.float()))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); step += 1
            if ema is not None:
                dcy = min(args.ema_decay, (1 + step) / (10 + step))
                with torch.no_grad():
                    for pe, pm in zip(ema.parameters(), model.parameters()):
                        pe.mul_(dcy).add_(pm.detach(), alpha=1 - dcy)
            tl += loss.item() * b["B"]
        ev = ema if ema is not None else model
        pv = predict(ev, va, dev, gm, gs, amp=bool(args.amp))
        yv = va.y.cpu().numpy()
        m = metrics(pv, yv)
        vl = val_loss(pv, yv)
        pva = predict(ev, va, dev, gm, gs, amp=bool(args.amp), aug_mode="swap")
        vla = val_loss(pva, yv)
        score = {"loss": vl, "lossaug": vl + vla, "auc": -m["auc"], "acc": -m["acc"]}[args.select]
        flag = ""
        if score < best["score"] - 1e-6:
            best.update(score=score, ep=ep, state=copy.deepcopy(ev.state_dict()))
            bad, flag = 0, "  <- tốt nhất"
        else:
            bad += 1
        print(f"[Ep {ep:02d}] train_loss {tl/tr.G:.4f} | val_loss {vl:.4f} acc {m['acc']:.4f} auc {m['auc']:.4f} | proxy-nhiễu loss {vla:.4f} | "
              f"{time.time()-t0:.0f}s{flag}")
        if bad >= args.patience:
            print(f"[dừng sớm] không cải thiện sau {args.patience} epoch")
            break
    final = ema if ema is not None else model
    final.load_state_dict(best["state"])
    pv = predict(final, va, dev, gm, gs, amp=bool(args.amp))
    yv = va.y.cpu().numpy()
    thr = pick_threshold(pv, yv)
    print(f"-> epoch tốt nhất {best['ep']} | val acc@0.5 {metrics(pv, yv)['acc']:.4f} | ngưỡng chọn trên val = {thr:.2f} "
          f"(val acc@thr {metrics(pv, yv, thr)['acc']:.4f}) | train {time.time()-t0:.0f}s")
    pva = predict(final, va, dev, gm, gs, amp=bool(args.amp), aug_mode="swap")
    res = dict(seed=seed, thr=thr, best_ep=best["ep"], params=nparam, train_s=time.time() - t0,
               val_probs=pv, val_y=yv, tests={}, val_loss=val_loss(pv, yv), val_aug_loss=val_loss(pva, yv),
               val_acc_aug=metrics(pva, yv)["acc"])
    for name, ds in tests.items():
        t1 = time.time()
        p = predict(final, ds, dev, gm, gs, amp=bool(args.amp))
        y = ds.y.cpu().numpy()
        res["tests"][name] = dict(p=p, y=y, n=ds.nn.cpu().numpy(), test_s=time.time() - t1)
        m5, mt = metrics(p, y, 0.5), metrics(p, y, thr)
        print(f"   test {name}: acc@0.5 {m5['acc']:.4f} | acc@thr {mt['acc']:.4f} | auc {m5['auc']:.4f} | f1 {m5['f1']:.4f} | tpr@1%fpr {m5['tpr1']:.3f}")
    os.makedirs("model_plus", exist_ok=True)
    torch.save(dict(state=best["state"], args=vars(args), R=R, gm=gm, gs=gs, thr=thr),
               f"model_plus/{args.tag}_seed{seed}.pt")
    return res


def diagnose(args):
    print("=== CHẨN ĐOÁN file pkl:", args.train_file, "===")
    path = f"graph_data/{args.train_file}.pkl"
    d = _load_pickle(path)
    if isinstance(d, dict):
        for k, v in d.items():
            try:
                print(f"  khóa '{k}': {type(v).__name__}, len={len(v)}")
            except Exception:
                print(f"  khóa '{k}': {type(v).__name__}")
    emb, ei, et, y = read_raw(path)
    R = int(max(int(np.asarray(t).max()) for t in et[:2000] if np.asarray(t).size)) + 1
    print(f"  số đồ thị {len(y)} | tỉ lệ nhãn 1 = {y.mean():.3f} | R (max edge_type+1 trên 2000 mẫu đầu) = {R}")
    mode, frac = detect_mode(ei, et, R)
    print(f"  chiều dependent->head: nhóm '{mode}', {frac*100:.1f}% đồ thị khớp tính chất cây (mỗi nút <=1 cạnh ra)")
    if frac < 0.9:
        print("  !! Không xác định chắc được chiều cạnh -> chạy với --struct 0 (bỏ đặc trưng cấu trúc), hoặc gửi mình đoạn mã tạo cạnh.")
    ur, mf = [], []
    for x in emb[:200]:
        x = torch.as_tensor(x).float()
        u = torch.unique(x, dim=0)
        ur.append(u.shape[0] / x.shape[0])
        _, inv, cnts = torch.unique(x, dim=0, return_inverse=True, return_counts=True)
        mf.append(cnts.max().item() / x.shape[0])
    print(f"  [embedding] tỉ lệ vector duy nhất / số nút = {np.mean(ur):.3f}; nút trùng vector phổ biến nhất = {np.mean(mf)*100:.1f}%")
    print("  -> nếu tỉ lệ duy nhất << 1 (vd < 0.8) hoặc 'trùng phổ biến nhất' > 10%: embedding đang là tra từ vựng tĩnh/unk,")
    print("     chưa phải embedding ngữ cảnh -> đây là hướng cải tiến lớn tiếp theo (xem phần cuối notebook).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file"); ap.add_argument("--val_file"); ap.add_argument("--test_files", nargs="*", default=[])
    ap.add_argument("--seeds", nargs="+", type=int, default=[2024])
    ap.add_argument("--tag", default="plus")
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("--num_rel", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=256); ap.add_argument("--num_bases", type=int, default=8)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3); ap.add_argument("--in_dropout", type=float, default=0.1)
    ap.add_argument("--edge_drop", type=float, default=0.1)
    ap.add_argument("--struct", type=int, default=1); ap.add_argument("--gfeat", type=int, default=0)
    ap.add_argument("--readout", default="pool3", choices=["pool3", "meanlogit"])
    ap.add_argument("--ema", type=int, default=1); ap.add_argument("--ema_decay", type=float, default=0.998)
    ap.add_argument("--lr", type=float, default=5e-4); ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--label_smooth", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--select", default="loss", choices=["loss", "lossaug", "auc", "acc"])
    ap.add_argument("--aug", default="none", choices=["none", "swap", "mask", "both"])
    ap.add_argument("--aug_max", type=float, default=0.35)
    ap.add_argument("--cons", type=float, default=0.0)
    ap.add_argument("--amp", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.diagnose:
        return diagnose(args)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Thiết bị:", dev, "| cấu hình:", {k: v for k, v in vars(args).items() if k not in ("test_files",)})
    R = args.num_rel or None
    tr = GraphSet(f"graph_data/{args.train_file}.pkl", R=R, store_dev="cpu")
    R, mode = tr.R, tr.mode
    if args.struct and getattr(tr, "mode_frac", 1.0) < 0.9:
        print("!! Cảnh báo: không chắc chiều cạnh -> tắt đặc trưng cấu trúc"); args.struct = 0
    store = "cuda" if (dev.type == "cuda" and tr.X.numel() * 2 < 3.5e9) else "cpu"
    tr.to(store)
    va = GraphSet(f"graph_data/{args.val_file}.pkl", R=R, mode=mode, store_dev=store)
    tests = {n: GraphSet(f"graph_data/{n}.pkl", R=R, mode=mode, store_dev=store) for n in args.test_files}
    results = []
    for s in args.seeds:
        results.append(train_one(args, s, tr, va, tests, dev, R))
    os.makedirs("preds", exist_ok=True)
    out = args.out or f"preds/{args.tag}.pkl"
    with open(out, "wb") as f:
        pickle.dump(dict(args=vars(args), results=results), f)
    print("Đã lưu dự đoán:", out)


if __name__ == "__main__":
    main()
```

**Kết quả:**

```text
Writing rgcn_plus.py
```

## 1. Kiểm thử đơn vị (chạy trước khi tốn thời gian huấn luyện)

```python
import torch, importlib, rgcn_plus; importlib.reload(rgcn_plus)
from rgcn_plus import Net, RGCNLayer
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# (a) lớp R-GCN so với bản tham chiếu
torch.manual_seed(0)
N, d, R, nb, E = 40, 16, 6, 3, 200
lyr = RGCNLayer(d, R, nb); x = torch.randn(N, d)
src = torch.randint(0, N, (E,)); dst = torch.randint(0, N, (E,)); rel = torch.randint(0, R, (E,))
cnt = torch.bincount(dst * R + rel, minlength=N * R).float(); norm = 1 / cnt[dst * R + rel]
out = lyr(x, src, dst, rel, norm); ref = lyr.W0(x).clone(); W = torch.einsum('rb,bio->rio', lyr.comp, lyr.V)
for e in range(E): ref[dst[e]] += norm[e] * (x[src[e]] @ W[rel[e]])
assert (out - ref).abs().max() < 1e-4; print('R-GCN layer OK')
# (b) nhiễu mức nút + forward/backward trên rừng cây giả
R, B, per = 51, 3, 20; half = R // 2
S, D, RL = [], [], []
for g in range(B):
    for k in range(1, per):
        h = g * per + int(torch.randint(0, k, (1,))); c = g * per + k; r = int(torch.randint(0, half, (1,)))
        S += [c, h]; D += [h, c]; RL += [r, r + half]
bt = dict(x=torch.randn(B * per, 768), st=torch.zeros(B * per, 3, dtype=torch.long), src=torch.tensor(S), dst=torch.tensor(D), rel=torch.tensor(RL),
          bidx=torch.arange(B).repeat_interleave(per), B=B, gf=torch.zeros(B, 8), y=torch.tensor([0., 1., 0.]))
bt = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in bt.items()}
for mode in ('swap', 'mask', 'both'):
    net = Net(R, d=64, nb=4, aug=mode, aug_max=0.5, mode='low').to(dev); net.train()
    xa = net._augment(bt['x'], bt['src'], bt['dst'], bt['rel'], bt['bidx'], B, mode, None)
    frac = (xa != bt['x']).any(1).float().mean().item()
    o, oa = net(bt), net(bt, aug_mode=mode)
    (o.sum() + oa.sum()).backward()
    print(f'aug={mode:5s}: tỉ lệ nút bị đổi = {frac:.2f} (kỳ vọng 0–0.5), logit {tuple(o.shape)}, backward OK')
    assert o.shape == (B,) and 0.0 < frac <= 0.7
```

**Kết quả:**

```text
R-GCN layer OK
aug=swap : tỉ lệ nút bị đổi = 0.17 (kỳ vọng 0–0.5), logit (3,), backward OK
aug=mask : tỉ lệ nút bị đổi = 0.27 (kỳ vọng 0–0.5), logit (3,), backward OK
aug=both : tỉ lệ nút bị đổi = 0.30 (kỳ vọng 0–0.5), logit (3,), backward OK
```

## 2. Sweep 1 seed (epochs 10, cùng quy tắc chọn checkpoint `lossaug`)
| Biến thể | Ý nghĩa |
|---|---|
| V0_base | cấu hình C2_wide + lịch ngắn, không nhiễu (đo riêng tác dụng lịch ngắn) |
| V1_swap | nhiễu swap, không loss nhất quán |
| V2_swap_cons | swap + nhất quán (λ=1) |
| V3_both_cons | swap+mask + nhất quán (λ=1) |
| V4_both_strong | nhiễu mạnh hơn (aug_max 0.5), λ=2 |
| V5_both_nostruct | V3 nhưng bỏ đặc trưng cấu trúc (kiểm tra struct có làm giảm độ bền không) |

```python
COMMON = '--readout pool3 --gfeat 0 --ema 1 --select lossaug --hidden 384 --num_bases 12 --dropout 0.3 --in_dropout 0.1 --edge_drop 0.1 --epochs 10 --patience 10'
VAR = {
 'V0_base'        : f'{COMMON} --struct 1 --aug none',
 'V1_swap'        : f'{COMMON} --struct 1 --aug swap --cons 0',
 'V2_swap_cons'   : f'{COMMON} --struct 1 --aug swap --cons 1.0',
 'V3_both_cons'   : f'{COMMON} --struct 1 --aug both --cons 1.0',
 'V4_both_strong' : f'{COMMON} --struct 1 --aug both --cons 2.0 --aug_max 0.5',
 'V5_both_nostruct': f'{COMMON} --struct 0 --aug both --cons 1.0',
}
def run(cmd, log):
    t0 = time.time()
    with open(log, 'w') as f:
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            if line.startswith(('[Ep', '->', '   test', '===', '[dừng', 'Traceback', 'RuntimeError', 'Error')) or 'Error' in line: sys.stdout.write(line)
            f.write(line)
    print(f'--- xong {time.time()-t0:.0f}s (mã thoát {p.wait()}) ---')
os.makedirs('logs_plus', exist_ok=True); os.makedirs('preds', exist_ok=True)
SW_SEED = 2024
for k, fl in VAR.items():
    out = f'preds/p3_{k}.pkl'
    if os.path.exists(out): print('bỏ qua', k); continue
    print('\n##########', k)
    run(f'python rgcn_plus.py {DATA} --seeds {SW_SEED} {fl} --tag p3_{k} --out {out}', f'logs_plus/p3_{k}.log')
```

**Kết quả:**

```text

########## V0_base
=== seed 2024 | tham số: 4.59M | 250 bước/epoch | tag=p3_V0_base ===
[Ep 01] train_loss 0.3098 | val_loss 0.1196 acc 0.9560 auc 0.9908 | proxy-nhiễu loss 0.1338 | 20s  <- tốt nhất
[Ep 02] train_loss 0.1484 | val_loss 0.0599 acc 0.9790 auc 0.9975 | proxy-nhiễu loss 0.0930 | 37s  <- tốt nhất
[Ep 03] train_loss 0.0978 | val_loss 0.0391 acc 0.9860 auc 0.9989 | proxy-nhiễu loss 0.0751 | 53s  <- tốt nhất
[Ep 04] train_loss 0.0431 | val_loss 0.0322 acc 0.9880 auc 0.9992 | proxy-nhiễu loss 0.0777 | 69s  <- tốt nhất
[Ep 05] train_loss 0.0288 | val_loss 0.0301 acc 0.9900 auc 0.9993 | proxy-nhiễu loss 0.0794 | 85s  <- tốt nhất
[Ep 06] train_loss 0.0185 | val_loss 0.0239 acc 0.9930 auc 0.9995 | proxy-nhiễu loss 0.0695 | 102s  <- tốt nhất
[Ep 07] train_loss 0.0066 | val_loss 0.0221 acc 0.9960 auc 0.9997 | proxy-nhiễu loss 0.0680 | 118s  <- tốt nhất
[Ep 08] train_loss 0.0028 | val_loss 0.0230 acc 0.9950 auc 0.9996 | proxy-nhiễu loss 0.0660 | 134s  <- tốt nhất
[Ep 09] train_loss 0.0014 | val_loss 0.0244 acc 0.9940 auc 0.9996 | proxy-nhiễu loss 0.0677 | 150s
[Ep 10] train_loss 0.0009 | val_loss 0.0237 acc 0.9940 auc 0.9996 | proxy-nhiễu loss 0.0685 | 166s
-> epoch tốt nhất 8 | val acc@0.5 0.9950 | ngưỡng chọn trên val = 0.53 (val acc@thr 0.9950) | train 167s
   test hc3_test: acc@0.5 0.9880 | acc@thr 0.9880 | auc 0.9987 | f1 0.9881 | tpr@1%fpr 0.974
   test hc3_test_adj_5_synonym_replace: acc@0.5 0.9870 | acc@thr 0.9870 | auc 0.9989 | f1 0.9871 | tpr@1%fpr 0.974
   test hc3_test_adj_10_synonym_replace: acc@0.5 0.9840 | acc@thr 0.9850 | auc 0.9988 | f1 0.9842 | tpr@1%fpr 0.974
   test hc3_test_adj_20_synonym_replace: acc@0.5 0.9840 | acc@thr 0.9850 | auc 0.9985 | f1 0.9843 | tpr@1%fpr 0.974
   test hc3_test_adj_30_synonym_replace: acc@0.5 0.9830 | acc@thr 0.9840 | auc 0.9984 | f1 0.9833 | tpr@1%fpr 0.966
--- xong 196s (mã thoát 0) ---

########## V1_swap
=== seed 2024 | tham số: 4.59M | 250 bước/epoch | tag=p3_V1_swap ===
[Ep 01] train_loss 0.3048 | val_loss 0.0982 acc 0.9690 auc 0.9936 | proxy-nhiễu loss 0.1160 | 33s  <- tốt nhất
[Ep 02] train_loss 0.1267 | val_loss 0.0505 acc 0.9820 auc 0.9984 | proxy-nhiễu loss 0.0679 | 65s  <- tốt nhất
[Ep 03] train_loss 0.0866 | val_loss 0.0351 acc 0.9850 auc 0.9992 | proxy-nhiễu loss 0.0532 | 96s  <- tốt nhất
[Ep 04] train_loss 0.0533 | val_loss 0.0261 acc 0.9930 auc 0.9995 | proxy-nhiễu loss 0.0442 | 127s  <- tốt nhất
[Ep 05] train_loss 0.0348 | val_loss 0.0220 acc 0.9940 auc 0.9997 | proxy-nhiễu loss 0.0360 | 159s  <- tốt nhất
[Ep 06] train_loss 0.0153 | val_loss 0.0249 acc 0.9920 auc 0.9996 | proxy-nhiễu loss 0.0403 | 190s
[Ep 07] train_loss 0.0089 | val_loss 0.0253 acc 0.9920 auc 0.9996 | proxy-nhiễu loss 0.0392 | 221s
[Ep 08] train_loss 0.0062 | val_loss 0.0242 acc 0.9920 auc 0.9996 | proxy-nhiễu loss 0.0379 | 252s
[Ep 09] train_loss 0.0025 | val_loss 0.0246 acc 0.9920 auc 0.9996 | proxy-nhiễu loss 0.0390 | 283s
[Ep 10] train_loss 0.0027 | val_loss 0.0246 acc 0.9920 auc 0.9996 | proxy-nhiễu loss 0.0
```

```python
import numpy as np, pandas as pd
from rgcn_plus import metrics, pick_threshold
PAPER = [0.9878, 0.9878, 0.9872, 0.9874, 0.9864]
rows = []
for k in VAR:
    f = f'preds/p3_{k}.pkl'
    if not os.path.exists(f): continue
    r = pickle.load(open(f, 'rb'))['results'][0]; tn = list(r['tests'])
    a = [metrics(r['tests'][n]['p'], r['tests'][n]['y'])['acc'] for n in tn]
    rows.append(dict(cfg=k, best_ep=r['best_ep'], val_loss=r['val_loss'], val_aug_loss=r['val_aug_loss'], select_score=r['val_loss'] + r['val_aug_loss'],
                     val_acc_aug=r['val_acc_aug'], t0=a[0], t5=a[1], t10=a[2], t20=a[3], t30=a[4], decline=a[0] - a[4], mean5=np.mean(a)))
sw = pd.DataFrame(rows).set_index('cfg'); display(sw.round(4))
best_cfg = sw['select_score'].idxmin()
print('Chọn theo VAL (val_loss + proxy nhiễu):', best_cfg)
print('Bảng test chỉ để minh bạch; việc chọn đã cố định theo quy tắc val ở trên.')
```

**Kết quả:**

```text
                  best_ep  val_loss  val_aug_loss  select_score  val_acc_aug  \
cfg                                                                            
V0_base                 8    0.0230        0.0660        0.0890        0.980   
V1_swap                 5    0.0220        0.0360        0.0580        0.985   
V2_swap_cons            6    0.0255        0.0345        0.0600        0.988   
V3_both_cons            7    0.0317        0.0521        0.0839        0.985   
V4_both_strong          8    0.0411        0.0628        0.1039        0.977   
V5_both_nostruct       10    0.0346        0.0511        0.0856        0.984   

                     t0     t5    t10    t20    t30  decline   mean5  
cfg                                                                   
V0_base           0.988  0.987  0.984  0.984  0.983    0.005  0.9852  
V1_swap           0.993  0.993  0.992  0.991  0.989    0.004  0.9916  
V2_swap_cons      0.988  0.988  0.989  0.988  0.986    0.002  0.9878  
V3_both_cons      0.991  0.990  0.990  0.990  0.990    0.001  0.9902  
V4_both_strong    0.989  0.989  0.988  0.988  0.988    0.001  0.9884  
V5_both_nostruct  0.992  0.991  0.991  0.991  0.991    0.001  0.9912
```

**Kết quả:**

```text
Chọn theo VAL (val_loss + proxy nhiễu): V1_swap
Bảng test chỉ để minh bạch; việc chọn đã cố định theo quy tắc val ở trên.
```

## 3. Chạy chính thức 5 seed với cấu hình được chọn

```python
FINAL_SEEDS = [2021, 2022, 2023, 2024, 2025]
FINAL = best_cfg                       # có thể ghi đè tay, ví dụ 'V3_both_cons'
for s in FINAL_SEEDS:
    out = f'preds/p3fin_{FINAL}_s{s}.pkl'
    if os.path.exists(out): print('bỏ qua', s); continue
    print(f'\n########## {FINAL} seed {s}')
    run(f'python rgcn_plus.py {DATA} --seeds {s} {VAR[FINAL]} --tag p3fin_{FINAL}_s{s} --out {out}', f'logs_plus/p3fin_{FINAL}_s{s}.log')
```

**Kết quả:**

```text

########## V1_swap seed 2021
=== seed 2021 | tham số: 4.59M | 250 bước/epoch | tag=p3fin_V1_swap_s2021 ===
[Ep 01] train_loss 0.3293 | val_loss 0.0866 acc 0.9710 auc 0.9942 | proxy-nhiễu loss 0.0996 | 33s  <- tốt nhất
[Ep 02] train_loss 0.1802 | val_loss 0.0559 acc 0.9830 auc 0.9979 | proxy-nhiễu loss 0.0749 | 65s  <- tốt nhất
[Ep 03] train_loss 0.0931 | val_loss 0.0431 acc 0.9840 auc 0.9989 | proxy-nhiễu loss 0.0590 | 96s  <- tốt nhất
[Ep 04] train_loss 0.0519 | val_loss 0.0266 acc 0.9920 auc 0.9995 | proxy-nhiễu loss 0.0399 | 127s  <- tốt nhất
[Ep 05] train_loss 0.0338 | val_loss 0.0281 acc 0.9910 auc 0.9995 | proxy-nhiễu loss 0.0412 | 159s
[Ep 06] train_loss 0.0186 | val_loss 0.0242 acc 0.9890 auc 0.9996 | proxy-nhiễu loss 0.0352 | 190s  <- tốt nhất
[Ep 07] train_loss 0.0107 | val_loss 0.0221 acc 0.9930 auc 0.9996 | proxy-nhiễu loss 0.0398 | 221s
[Ep 08] train_loss 0.0063 | val_loss 0.0219 acc 0.9950 auc 0.9996 | proxy-nhiễu loss 0.0405 | 252s
[Ep 09] train_loss 0.0053 | val_loss 0.0220 acc 0.9940 auc 0.9996 | proxy-nhiễu loss 0.0413 | 283s
[Ep 10] train_loss 0.0025 | val_loss 0.0220 acc 0.9940 auc 0.9996 | proxy-nhiễu loss 0.0427 | 314s
-> epoch tốt nhất 6 | val acc@0.5 0.9890 | ngưỡng chọn trên val = 0.18 (val acc@thr 0.9950) | train 315s
   test hc3_test: acc@0.5 0.9900 | acc@thr 0.9910 | auc 0.9993 | f1 0.9901 | tpr@1%fpr 0.992
   test hc3_test_adj_5_synonym_replace: acc@0.5 0.9890 | acc@thr 0.9920 | auc 0.9993 | f1 0.9891 | tpr@1%fpr 0.988
   test hc3_test_adj_10_synonym_replace: acc@0.5 0.9900 | acc@thr 0.9910 | auc 0.9993 | f1 0.9901 | tpr@1%fpr 0.990
   test hc3_test_adj_20_synonym_replace: acc@0.5 0.9900 | acc@thr 0.9910 | auc 0.9992 | f1 0.9901 | tpr@1%fpr 0.990
   test hc3_test_adj_30_synonym_replace: acc@0.5 0.9890 | acc@thr 0.9900 | auc 0.9992 | f1 0.9891 | tpr@1%fpr 0.990
--- xong 344s (mã thoát 0) ---

########## V1_swap seed 2022
=== seed 2022 | tham số: 4.59M | 250 bước/epoch | tag=p3fin_V1_swap_s2022 ===
[Ep 01] train_loss 0.3525 | val_loss 0.1085 acc 0.9640 auc 0.9900 | proxy-nhiễu loss 0.1193 | 33s  <- tốt nhất
[Ep 02] train_loss 0.1340 | val_loss 0.0619 acc 0.9700 auc 0.9981 | proxy-nhiễu loss 0.0841 | 65s  <- tốt nhất
[Ep 03] train_loss 0.1054 | val_loss 0.0488 acc 0.9750 auc 0.9988 | proxy-nhiễu loss 0.0755 | 95s  <- tốt nhất
[Ep 04] train_loss 0.0500 | val_loss 0.0360 acc 0.9870 auc 0.9993 | proxy-nhiễu loss 0.0641 | 127s  <- tốt nhất
[Ep 05] train_loss 0.0295 | val_loss 0.0313 acc 0.9900 auc 0.9995 | proxy-nhiễu loss 0.0566 | 158s  <- tốt nhất
[Ep 06] train_loss 0.0168 | val_loss 0.0305 acc 0.9870 auc 0.9995 | proxy-nhiễu loss 0.0516 | 189s  <- tốt nhất
[Ep 07] train_loss 0.0093 | val_loss 0.0263 acc 0.9930 auc 0.9996 | proxy-nhiễu loss 0.0473 | 220s  <- tốt nhất
[Ep 08] train_loss 0.0057 | val_loss 0.0276 acc 0.9920 auc 0.9996 | proxy-nhiễu loss 0.0473 | 251s
[Ep 09] train_loss 0.0024 | val_loss 0.0290 acc 0.9890 auc 0.9996 | proxy-nhiễu loss 0.0529 | 282s
[Ep 10] train_loss 0.0026 | val_loss 0.0304 acc 0.9900 auc 0.
```

```python
res = []
for f in sorted(glob.glob(f'preds/p3fin_{FINAL}_s*.pkl')): res += pickle.load(open(f, 'rb'))['results']
tn = list(res[0]['tests']); lab = ['0%', '5%', '10%', '20%', '30%']
acc = np.array([[metrics(r['tests'][n]['p'], r['tests'][n]['y'])['acc'] for n in tn] for r in res])
ens = [np.mean([r['tests'][n]['p'] for r in res], 0) for n in tn]; y = res[0]['tests'][tn[0]]['y']
ensacc = np.array([metrics(p, y)['acc'] for p in ens])
prev = np.array([0.9896, 0.9874, 0.9862, 0.9856, 0.9832])   # giai đoạn 1: C2_wide
tab = pd.DataFrame({
  'PRDetect (bài báo)': PAPER,
  'GĐ1 C2_wide (5 seed)': prev,
  f'GĐ3 {FINAL} mean': acc.mean(0), 'GĐ3 std': acc.std(0),
  'GĐ3 − bài báo': acc.mean(0) - np.array(PAPER), 'GĐ3 − GĐ1': acc.mean(0) - prev,
  'GĐ3 ENSEMBLE 5 seed': ensacc, 'AUC (mean)': [np.mean([metrics(r['tests'][n]['p'], r['tests'][n]['y'])['auc'] for r in res]) for n in tn],
  'TPR@1%FPR (mean)': [np.mean([metrics(r['tests'][n]['p'], r['tests'][n]['y'])['tpr1'] for r in res]) for n in tn]}, index=lab).T
display(tab.round(4)); tab.to_csv('table2_hc3_phase3.csv')
print('Suy giảm 0%→30% (mean seed):', round(acc[:, 0].mean() - acc[:, 4].mean(), 4), '| bài báo: 0.0014 | GĐ1: 0.0064')
print('best_ep các seed:', [r['best_ep'] for r in res], '| thắng bài báo ở', int((acc.mean(0) > np.array(PAPER)).sum()), '/ 5 cột (so mean)')
print('Lưu ý: 1 mẫu test = 0.001; CI 95% quanh 0.99 là ±0.0067; bài báo chỉ báo cáo std trên 1 split.')
from scipy.stats import binomtest
p0, p3 = ens[0], ens[-1]; c0, c3 = (p0 >= .5) == y.astype(int), (p3 >= .5) == y.astype(int)
b, c = int((c0 & ~c3).sum()), int((~c0 & c3).sum())
print(f'McNemar ensemble 0%→30%: đúng→sai {b}, sai→đúng {c}, p = {binomtest(b, b + c, .5).pvalue if b + c else 1.0:.3f}')
```

**Kết quả:**

```text
                          0%      5%     10%     20%     30%
PRDetect (bài báo)    0.9878  0.9878  0.9872  0.9874  0.9864
GĐ1 C2_wide (5 seed)  0.9896  0.9874  0.9862  0.9856  0.9832
GĐ3 V1_swap mean      0.9902  0.9904  0.9900  0.9898  0.9892
GĐ3 std               0.0016  0.0015  0.0017  0.0013  0.0012
GĐ3 − bài báo         0.0024  0.0026  0.0028  0.0024  0.0028
GĐ3 − GĐ1             0.0006  0.0030  0.0038  0.0042  0.0060
GĐ3 ENSEMBLE 5 seed   0.9930  0.9920  0.9910  0.9910  0.9910
AUC (mean)            0.9988  0.9989  0.9988  0.9987  0.9986
TPR@1%FPR (mean)      0.9874  0.9877  0.9877  0.9885  0.9862
```

**Kết quả:**

```text
Suy giảm 0%→30% (mean seed): 0.001 | bài báo: 0.0014 | GĐ1: 0.0064
best_ep các seed: [6, 7, 6, 8, 8] | thắng bài báo ở 5 / 5 cột (so mean)
Lưu ý: 1 mẫu test = 0.001; CI 95% quanh 0.99 là ±0.0067; bài báo chỉ báo cáo std trên 1 split.
McNemar ensemble 0%→30%: đúng→sai 2, sai→đúng 0, p = 0.500
```

## 4. (Tùy chọn) Đánh giá nhiễu làm đổi cây & viết lại — Bảng 4 của bài báo
Đặt `RUN_EXT = True`. Mỗi tập được dựng → đánh giá bằng 5 checkpoint cuối → xóa pkl (tiết kiệm ổ đĩa). Mất ~15 phút.

```python
RUN_EXT = False
if RUN_EXT:
    from rgcn_plus import GraphSet, predict
    CK = []
    for f in sorted(glob.glob(f'model_plus/p3fin_{FINAL}_s*_seed*.pt')):
        c = torch.load(f, map_location='cpu', weights_only=False); a = c['args']
        m = Net(c['R'], d=a['hidden'], nb=a['num_bases'], L=a['layers'], p=a['dropout'], p_in=a['in_dropout'], edge_drop=a['edge_drop'],
                use_struct=bool(a['struct']), use_gfeat=bool(a['gfeat']), readout=a['readout'], aug=a.get('aug', 'none'),
                aug_max=a.get('aug_max', 0.35)).to(dev)
        m.load_state_dict(c['state']); m.eval(); CK.append(dict(model=m, R=c['R'], gm=c['gm'], gs=c['gs']))
    EXT = ['hc3_test_replace_25', 'hc3_test_delete_25', 'hc3_test_insert_25', 'hc3_test_repeat_25',
           'hc3_test_llama3_8b_instruct_rewrite', 'hc3_test_translate', 'hc3_test_50_100', 'hc3_test_100_150', 'hc3_test_150_200',
           'hc3_test_200_250', 'hc3_test_250_300', 'hc3_test_300_inf']
    REF = {'hc3_test_insert_25': 0.983, 'hc3_test_repeat_25': 0.982, 'hc3_test_replace_25': 0.798, 'hc3_test_delete_25': 0.747}
    out = {}
    for n in [e for e in EXT if e in names]:
        if not os.path.exists(f'graph_data/{n}.pkl'):
            subprocess.run(f'TQDM_MININTERVAL=60 python building_graph.py --file {n} > /dev/null 2>&1', shell=True)
        if not os.path.exists(f'graph_data/{n}.pkl'): print('!! không dựng được', n); continue
        ds = GraphSet(f'graph_data/{n}.pkl', R=CK[0]['R'], store_dev='cpu', verbose=False)
        ps = [predict(c['model'], ds, dev, c['gm'], c['gs']) for c in CK]; yy = ds.y.numpy()
        accs = [metrics(p, yy)['acc'] for p in ps]
        out[n.replace('hc3_test_', '')] = dict(acc=f'{np.mean(accs):.4f} ± {np.std(accs):.4f}', ens=f"{metrics(np.mean(ps, 0), yy)['acc']:.4f}",
                                              paper=REF.get(n, ''), label1=f'{yy.mean():.2f}', n=len(yy))
        os.remove(f'graph_data/{n}.pkl'); del ds
    display(pd.DataFrame(out).T)
```

Gửi lại cho mình: bảng sweep (mục 2) và bảng tổng hợp (mục 3). Nếu vẫn chưa thắng ở cột 20–30%, bước tiếp theo là **huấn luyện trên văn bản bị nhiễu thật** (sinh nhiễu cho `hc3_train.json` rồi dựng đồ thị bằng `building_graph.py`; cần 3–5 dòng đầu của `hc3_train.json` để khớp định dạng) và gộp nhãn quan hệ.
