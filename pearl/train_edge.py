"""Train q on clean-origin pairs; select by source validation edge log loss."""
import time
from collections import defaultdict
from pathlib import Path
import torch
from .config import config_cli, load_config, seed_all
from .data import assert_isolated, file_hash, read_jsonl, write_json
from .edge_features import EdgeFeatures
from .edge_predictor import EdgeRetentionPredictor, document_edge_loss
from .graph import arcs
from .metrics import distribution, edge_metrics


def load_partition(root, split):
    docs = read_jsonl(Path(root) / f"{split}_docs.jsonl.gz")
    pairs = read_jsonl(Path(root) / f"{split}_pairs.jsonl.gz")
    by_doc = defaultdict(list)
    ids = {d["doc_id"] for d in docs}
    for p in pairs:
        if p["split"] != split or p["source_doc_id"] not in ids:
            raise ValueError("Pair is outside its declared source partition")
        by_doc[p["source_doc_id"]].append(p)
    return docs, by_doc


def observation_rows(docs, pairs, encoder, include_features=True):
    xs, ys, keys, families = [], [], [], []
    for d in docs:
        x = encoder.encode(d["graph"])
        ts = d["graph"]["tokens"]
        for p in pairs[d["doc_id"]]:
            for i, a in enumerate(p["arc_labels"]):
                if a["relation_valid"]:
                    if include_features:
                        xs.append(x[i])
                    ys.append(a["relation_retained"])
                    keys.append((a["relation"], ts[a["child"]]["pos"], ts[a["head"]]["pos"]))
                    families.append(p["edit_family"])
    if not ys:
        raise ValueError("No valid retention labels")
    return torch.stack(xs) if include_features else None, ys, keys, families


def fit_model(cfg, encoder, docs, pairs, val_x, val_y, shuffled=False):
    seed_all(cfg["seed"])
    device = cfg["device"]
    model = EdgeRetentionPredictor(encoder.dim, cfg.get("edge_hidden", 64)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("edge_lr", .001))
    xs = {d["doc_id"]: encoder.encode(d["graph"]).to(device) for d in docs}
    if shuffled:
        # Global permutation of valid labels; validation remains untouched.
        import copy
        pairs = copy.deepcopy(pairs)
        labels = [a for ps in pairs.values() for p in ps for a in p["arc_labels"] if a["relation_valid"]]
        permutation = torch.randperm(len(labels)).tolist()
        targets = [a["relation_retained"] for a in labels]
        for a, j in zip(labels, permutation):
            a["relation_retained"] = targets[j]
    best, history, state = float("inf"), [], None
    for epoch in range(cfg.get("edge_epochs", 10)):
        model.train()
        losses = []
        order = torch.randperm(len(docs)).tolist()
        batch = []
        optimizer.zero_grad()
        for index in order:
            d = docs[index]
            loss = document_edge_loss(model(xs[d["doc_id"]]), pairs[d["doc_id"]])
            if loss is not None:
                batch.append(loss)
                losses.append(float(loss.detach()))
            if len(batch) == cfg.get("edge_batch_docs", 8):
                torch.stack(batch).mean().backward()
                optimizer.step()
                optimizer.zero_grad()
                batch = []
        if batch:
            torch.stack(batch).mean().backward()
            optimizer.step()
        if not losses:
            raise ValueError("No train document has valid edge labels")
        model.eval()
        with torch.no_grad():
            q = model(val_x.to(device)).sigmoid().cpu().tolist()
        metrics = edge_metrics(val_y, q)
        history.append(dict(epoch=epoch + 1, train_loss=sum(losses) / max(1, len(losses)), validation=metrics))
        if metrics["log_loss"] < best:
            best = metrics["log_loss"]
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"edge {'shuffle' if shuffled else 'MLP'} epoch {epoch + 1}: val BCE={metrics['log_loss']:.5f}", flush=True)
    if state is None:
        raise ValueError("edge_epochs must be positive")
    model.load_state_dict(state)
    model.eval()
    return model, history


def train(cfg):
    start = time.perf_counter()
    out = Path(cfg["edge_output"])
    out.mkdir(parents=True, exist_ok=True)
    if (out / "edge.pt").exists():
        raise FileExistsError(out / "edge.pt")
    docs, pairs = load_partition(cfg["prepared"], "train")
    val, vpairs = load_partition(cfg["prepared"], "validation")
    assert_isolated(docs, val)
    encoder = EdgeFeatures.fit([d["graph"] for d in docs])
    _, ty, tk, _ = observation_rows(docs, pairs, encoder, include_features=False)
    vx, vy, vk, vf = observation_rows(val, vpairs, encoder)
    prior = sum(ty) / len(ty)
    scores = {"constant_prior": [prior] * len(vy)}
    for name, keyfn in (("relation_prior", lambda k: k[0]), ("relation_pos_prior", lambda k: k)):
        counts = defaultdict(lambda: [0, 0])
        for y, key in zip(ty, tk):
            counts[keyfn(key)][0] += y
            counts[keyfn(key)][1] += 1
        scores[name] = [(counts[keyfn(k)][0] + prior * 2) / (counts[keyfn(k)][1] + 2) for k in vk]
    model, history = fit_model(cfg, encoder, docs, pairs, vx, vy)
    with torch.no_grad():
        scores["mlp"] = model(vx.to(cfg["device"])).sigmoid().cpu().tolist()
    shuffled_history = []
    if cfg.get("shuffle_control", False):
        shuffled, shuffled_history = fit_model(cfg, encoder, docs, pairs, vx, vy, shuffled=True)
        with torch.no_grad():
            scores["shuffled_label_mlp"] = shuffled(vx.to(cfg["device"])).sigmoid().cpu().tolist()
    views = defaultdict(list)
    with torch.no_grad():
        for d in val:
            clean = model(encoder.encode(d["graph"]).to(cfg["device"])).sigmoid().cpu().tolist()
            views["clean"].extend(clean)
            for p in vpairs[d["doc_id"]]:
                q = model(encoder.encode(p["edited_graph"]).to(cfg["device"])).sigmoid().cpu().tolist()
                views["edited"].extend(q)
                views[f"edited/family:{p['edit_family']}"].extend(q)
                for a, value in zip(arcs(p["edited_graph"]), q):
                    views[f"edited/relation:{a[2]}"] .append(value)
    breakdown = {}
    for field, values in (("family", vf), ("relation", [k[0] for k in vk])):
        for value in sorted(set(values)):
            ids = [i for i, v in enumerate(values) if v == value]
            breakdown[f"{field}:{value}"] = edge_metrics([vy[i] for i in ids], [scores["mlp"][i] for i in ids])
    report = dict(config=cfg, seconds=time.perf_counter() - start, metrics={k: edge_metrics(vy, v) for k, v in scores.items()},
                  breakdown=breakdown, view_distributions={k: distribution(v) for k, v in views.items()},
                  history=history, shuffled_history=shuffled_history,
                  note="Edited graphs have q diagnostics only, no G→G' labels reused as forward labels of G'.")
    manifest_hash = file_hash(Path(cfg["prepared"]) / "manifest.json")
    torch.save(dict(schema="pearl-edge-v1", state_dict=model.cpu().state_dict(), vocab=encoder.vocab,
                    hidden=cfg.get("edge_hidden", 64), config=cfg, prepared_manifest_sha256=manifest_hash), out / "edge.pt")
    write_json(out / "metrics.json", report)
    return report


def load_edge(path, device="cpu"):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    encoder = EdgeFeatures(ckpt["vocab"])
    model = EdgeRetentionPredictor(encoder.dim, ckpt["hidden"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model, encoder, ckpt


if __name__ == "__main__":
    args = config_cli("Train PEARL single-view edge predictor").parse_args()
    train(load_config(args.config))
