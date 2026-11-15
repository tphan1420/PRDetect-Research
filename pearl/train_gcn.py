"""Warm-start GCN and jointly fine-tune clean classification plus retention."""
import copy
import json
import time
from pathlib import Path
import torch
from .config import config_cli, load_config
from .data import assert_isolated, file_hash, write_json
from .edge_features import EdgeFeatures
from .edge_predictor import EdgeRetentionPredictor, document_edge_loss
from .embeddings import LegacyEncoder, embedding_cache
from .metrics import classification_metrics
from .train_edge import load_edge, load_partition
from .weighted_gcn import PEARLGCN, baseline_from_state


def save_checkpoint(path, model, metadata):
    torch.save(dict(schema="pearl-gcn-v1", base_state={k: v.detach().cpu() for k, v in model.base.state_dict().items()},
                    predictor_state={k: v.detach().cpu() for k, v in model.predictor.state_dict().items()},
                    feature_vocab=model.features.vocab, edge_hidden=model.predictor.mlp[0].out_features,
                    epsilon=model.epsilon, backend=model.backend, metadata=metadata), path)


def load_checkpoint(path, device="cpu"):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["schema"] != "pearl-gcn-v1":
        raise ValueError("Not a PEARL checkpoint")
    base = baseline_from_state(state["base_state"])
    features = EdgeFeatures(state["feature_vocab"])
    predictor = EdgeRetentionPredictor(features.dim, state["edge_hidden"])
    predictor.load_state_dict(state["predictor_state"])
    model = PEARLGCN(base, predictor, features, state["epsilon"]).to(device)
    if state["backend"] != model.backend:
        raise ValueError("Checkpoint backend differs from runtime")
    model.eval()
    return model, state["metadata"]


def train(cfg):
    import spacy
    start = time.perf_counter()
    out = Path(cfg["gcn_output"])
    out.mkdir(parents=True, exist_ok=True)
    if (out / "best.pt").exists():
        raise FileExistsError(out / "best.pt")
    docs, pairs = load_partition(cfg["prepared"], "train")
    val, _ = load_partition(cfg["prepared"], "validation")
    assert_isolated(docs, val)
    manifest = json.loads((Path(cfg["prepared"]) / "manifest.json").read_text(encoding="utf-8"))
    nlp = spacy.load(cfg["parser"])
    from .data import digest
    if digest(nlp.to_bytes().hex()) != manifest["parser_hash"]:
        raise ValueError("Parser differs from prepared pair parser")
    predictor, features, edge_state = load_edge(Path(cfg["edge_output"]) / "edge.pt", cfg["device"])
    if edge_state["prepared_manifest_sha256"] != file_hash(Path(cfg["prepared"]) / "manifest.json"):
        raise ValueError("Edge predictor was trained with a different prepared manifest")
    state = torch.load(cfg["baseline_checkpoint"], map_location="cpu", weights_only=True)
    base = baseline_from_state(state)
    model = PEARLGCN(base, predictor, features, cfg.get("epsilon", .8)).to(cfg["device"])
    encoder = LegacyEncoder(cfg)
    train_x = embedding_cache(cfg, docs, "train", encoder)
    val_x = embedding_cache(cfg, val, "validation", encoder)
    metadata = dict(config=cfg, baseline_sha256=file_hash(cfg["baseline_checkpoint"]),
                    prepared_manifest_sha256=file_hash(Path(cfg["prepared"]) / "manifest.json"),
                    edge_checkpoint_sha256=file_hash(Path(cfg["edge_output"]) / "edge.pt"),
                    encoder=encoder.metadata, parser_hash=manifest["parser_hash"], parser_meta=manifest["parser_meta"],
                    spacy_version=spacy.__version__, torch_version=str(torch.__version__),
                    selection_rule="minimum source-validation clean classification BCE", epoch=0)
    del encoder  # frozen embeddings cached once; no encoder in optimizer/checkpoint
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("gcn_lr", .0001), weight_decay=cfg.get("weight_decay", .00001))
    criterion = torch.nn.BCEWithLogitsLoss()
    best, history, max_q_grad = float("inf"), [], 0.0
    for epoch in range(cfg.get("gcn_epochs", 3)):
        model.train()
        losses, batch, valid_losses = [], [], []
        optimizer.zero_grad()
        for index in torch.randperm(len(docs)).tolist():
            d = docs[index]
            _, logits, edge_logits, _ = model(train_x[d["doc_id"]].to(cfg["device"]), d["graph"], True)
            cls = criterion(logits.view(-1), logits.new_tensor([d["label"]]))
            edge = document_edge_loss(edge_logits, pairs[d["doc_id"]])
            batch.append(cls)
            if edge is not None:
                valid_losses.append(edge)
            losses.append(float(cls.detach()))
            if len(batch) == cfg.get("gcn_batch_docs", 8):
                loss = torch.stack(batch).mean()
                if valid_losses:
                    loss = loss + cfg.get("lambda_edge", .1) * torch.stack(valid_losses).mean()
                loss.backward()
                max_q_grad = max(max_q_grad, sum(float(p.grad.abs().sum()) for p in model.predictor.parameters() if p.grad is not None))
                optimizer.step()
                optimizer.zero_grad()
                batch, valid_losses = [], []
        if batch:
            loss = torch.stack(batch).mean()
            if valid_losses:
                loss = loss + cfg.get("lambda_edge", .1) * torch.stack(valid_losses).mean()
            loss.backward()
            max_q_grad = max(max_q_grad, sum(float(p.grad.abs().sum()) for p in model.predictor.parameters() if p.grad is not None))
            optimizer.step()
        model.eval()
        with torch.no_grad():
            p = [float(model(val_x[d["doc_id"]].to(cfg["device"]), d["graph"]).item()) for d in val]
        metrics = classification_metrics([d["label"] for d in val], p)
        history.append(dict(epoch=epoch + 1, train_classification_bce=sum(losses) / len(losses), validation=metrics))
        if metrics["bce"] < best:
            best = metrics["bce"]
            metadata["epoch"] = epoch + 1
            save_checkpoint(out / "best.pt", model, copy.deepcopy(metadata))
        print(f"GCN epoch {epoch + 1}: val BCE={metrics['bce']:.5f}, acc={metrics['accuracy']:.4f}", flush=True)
    if not history:
        raise ValueError("gcn_epochs must be positive")
    # Verify saved checkpoint against its actual selected model state on one graph.
    loaded, _ = load_checkpoint(out / "best.pt", cfg["device"])
    sample = val[0]
    with torch.no_grad():
        prediction = float(loaded(val_x[sample["doc_id"]].to(cfg["device"]), sample["graph"]).item())
    report = dict(history=history, best_validation_bce=best, selected_epoch=metadata["epoch"],
                  max_predictor_gradient_l1=max_q_grad, seconds=time.perf_counter() - start,
                  reloaded_prediction=prediction, metadata=metadata)
    write_json(out / "training.json", report)
    return report


if __name__ == "__main__":
    args = config_cli("Joint PEARL-GCN fine-tuning from an unchanged baseline checkpoint").parse_args()
    train(load_config(args.config))
