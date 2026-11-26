"""Fit lightweight A1 readouts on matched clean and word-replaced HC3 documents."""

import re

import numpy as np
import torch
import torch.nn.functional as F

from v3_a1_fusion import labels_from


WORDS = re.compile(r"[a-z]+(?:['’][a-z]+)*", re.IGNORECASE | re.ASCII)
ARMS = ("bce", "symmetric", "directional")


def _text_key(text):
    return " ".join(text.casefold().split())


def _replace_words(text, vocabulary, rng):
    words = list(WORDS.finditer(text))
    low, high = (len(words) + 9) // 10, len(words) // 5
    if not 1 <= low <= high:
        raise ValueError("Text too short for 10–20% replacement")
    count = int(rng.integers(low, high + 1))
    changes = sorted(rng.choice(len(words), count, replace=False))
    pieces, end = [], 0
    for index in changes:
        match = words[index]
        old = match.group()
        new = vocabulary[rng.integers(len(vocabulary))]
        while new == old.lower():
            new = vocabulary[rng.integers(len(vocabulary))]
        if old.isupper():
            new = new.upper()
        elif old[0].isupper():
            new = new.capitalize()
        pieces.extend([text[end:match.start()], new])
        end = match.end()
    pieces.append(text[end:])
    return "".join(pieces), count, len(words)


def build_pairs(train, val, train_size=1024, val_size=256, data_seed=42):
    """Sample balanced source splits, excluding duplicates and using train-only donors."""
    vocabulary = sorted({w.lower() for row in train for w in WORDS.findall(row["text"])})
    if len(vocabulary) < 2:
        raise ValueError("Need at least two different donor words in HC3 train")
    result = {}
    for split, rows, size in (("train", train, train_size), ("val", val, val_size)):
        if size < 2 or size % 2:
            raise ValueError("The pair count must be a positive even number")
        rng = np.random.default_rng([data_seed, int(split == "val")])
        seen = {_text_key(r["text"]) for r in train} if split == "val" else set()
        labels = labels_from(rows)
        candidates = [[], []]
        for i, row in enumerate(rows):
            key = _text_key(row["text"])
            if key not in seen and len(WORDS.findall(row["text"])) >= 5:
                candidates[labels[i]].append(i)
                seen.add(key)
        if min(map(len, candidates)) < size // 2:
            raise ValueError(f"Too few eligible {split} documents for {size} balanced pairs")
        chosen = sorted(int(i) for group in candidates for i in rng.choice(group, size // 2, replace=False))
        edited = []
        for i in chosen:
            text, replaced, total = _replace_words(rows[i]["text"], vocabulary, rng)
            edited.append({"index": i, "label": int(labels[i]), "text": text, "replaced": replaced, "words": total})
        result[split] = edited
    return result


@torch.inference_mode()
def pooled_graphs(model, graphs, batch_size, device):
    """Mean of the last convolution, immediately before A1's linear classification head."""
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import global_mean_pool

    model.eval()
    vectors = []
    for batch in DataLoader(graphs, batch_size=batch_size, shuffle=False):
        batch = batch.to(device)
        hidden = model.conv1(batch.x.float(), batch.edge_index, batch.edge_type).relu()
        hidden = model.conv2(hidden, batch.edge_index, batch.edge_type).relu()
        vectors.append(global_mean_pool(hidden, batch.batch, size=batch.num_graphs).cpu())
    return torch.cat(vectors)


def paired_vectors(model, clean_file, edited_file, pairs, clean_digest, edited_digest, batch_size, device):
    clean = torch.load(clean_file, weights_only=False, map_location="cpu")
    if clean["source_sha256"] != clean_digest:
        raise ValueError("Clean graphs do not match the source split")
    selected = [clean["graphs"][row["index"]] for row in pairs]
    del clean
    edited = torch.load(edited_file, weights_only=False, map_location="cpu")
    if edited["source_sha256"] != edited_digest:
        raise ValueError("Edited graphs do not match the paired source")
    labels = [row["label"] for row in pairs]
    for graphs in (selected, edited["graphs"]):
        if [g.y.item() for g in graphs] != labels:
            raise ValueError("Paired graph labels or sample counts differ")
    return (pooled_graphs(model, selected, batch_size, device),
            pooled_graphs(model, edited["graphs"], batch_size, device), torch.tensor(labels))


def document_vectors(scorer, texts):
    """Reuse the normal A1 scorer and collect its 64 features in the same forward pass."""
    vectors = []

    def capture(_layer, inputs):
        vectors.append(inputs[0].mean(dim=0).detach().cpu())

    with scorer.graph.fc.register_forward_pre_hook(capture), torch.inference_mode():
        logits = [scorer.score_doc(doc) for doc in scorer.nlp.pipe(texts, batch_size=16)]
    if len(vectors) != len(logits):
        raise ValueError("Expected one A1 document vector per logit")
    return torch.stack(vectors), np.asarray(logits)


def _margins(head, data):
    clean, edited, labels = data
    sign = labels * 2 - 1
    return sign * head(clean).flatten(), sign * head(edited).flatten()


def _bce(clean_margin, edited_margin):
    return (F.softplus(-clean_margin).mean() + F.softplus(-edited_margin).mean()) / 2


def paired_penalty(clean_margin, edited_margin, eligible, arm):
    """Symmetric and directional controls use the same pairs and unit Huber scale."""
    drop = clean_margin - edited_margin
    if arm == "directional":
        drop = drop.clamp_min(0)
    elif arm != "symmetric":
        raise ValueError(f"Unknown penalty {arm}")
    terms = F.smooth_l1_loss(drop, torch.zeros_like(drop), reduction="none")
    return (terms * eligible).sum() / eligible.sum().clamp_min(1)


def fit_heads(base_fc, train, val, steps=80, lr=0.01, strength=1.0):
    """Fit three CPU heads from identical weights. Select by clean + edited source-val BCE."""
    import copy

    train = tuple(t.detach().to(device="cpu", dtype=torch.float64) for t in train)
    val = tuple(t.detach().to(device="cpu", dtype=torch.float64) for t in val)
    base = copy.deepcopy(base_fc).cpu().double()
    with torch.no_grad():
        eligible = (base(train[0]).flatten() >= 0) == train[2].bool()
    heads, validation = {}, {}
    for arm in ARMS:
        head = copy.deepcopy(base)
        optimizer = torch.optim.Adam(head.parameters(), lr=lr)
        best = float("inf")
        for step in range(steps + 1):
            if step:
                optimizer.zero_grad(set_to_none=True)
                clean, edited = _margins(head, train)
                loss = _bce(clean, edited)
                if arm != "bce":
                    loss = loss + strength * paired_penalty(clean, edited, eligible, arm)
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                clean, edited = _margins(head, val)
                score = _bce(clean, edited).item()
                if not np.isfinite(score):
                    raise FloatingPointError(f"Non-finite source validation loss for {arm}")
                if score < best:
                    best = score
                    heads[arm] = {k: v.detach().float().clone() for k, v in head.state_dict().items()}
                    validation[arm] = {"step": step, "bce": score}
    return heads, validation
