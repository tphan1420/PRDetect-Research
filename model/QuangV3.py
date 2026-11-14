"""Inference-only port of Quang\'s PRDetect RGCN v3.

Source: docs/PRDetect_RGCN_v3_tong_hop.md (rgcn_plus.py).
Preserves the checkpoint architecture, legacy relation-half assumptions,
feature conversion and clipping. No training or architecture changes.
Use evaluate_v3_quang.py; mode is not stored in the supplied checkpoints.
"""

import math
import os
import pickle
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MAXD = 64
N_GF = 8

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
        h = h.float()
        cn = torch.zeros(B, device=h.device).index_add_(0, bidx, torch.ones(N, device=h.device)).clamp(min=1.0)
        mean = torch.zeros(B, h.size(1), device=h.device).index_add_(0, bidx, h) / cn.unsqueeze(1)
        mx = torch.full((B, h.size(1)), -1e30, device=h.device).scatter_reduce(
            0, bidx.unsqueeze(1).expand_as(h), h, "amax", include_self=True)
        w = seg_softmax(self.gate(h).squeeze(-1), bidx, B)
        att = torch.zeros(B, h.size(1), device=h.device).index_add_(0, bidx, h * w.unsqueeze(1))
        z = [mean, mx, att] + ([b["gf"]] if self.use_gfeat else [])
        return self.mlp(torch.cat(z, 1)).squeeze(-1)


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

