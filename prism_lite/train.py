"""Train the F1 GCN-residual-fallback variant on separate PRISM caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time
import hashlib

import torch
import torch.nn as nn
import torch.optim as optim

from .data import load_cache, to_data_list
from .model import PrismF1GCN


def _loader(dataset, batch_size, shuffle):
    from torch_geometric.loader import DataLoader
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _seed(value: int):
    random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def evaluate(model, loader, device):
    criterion = nn.BCELoss()
    model.eval()
    loss_sum = correct = count = 0
    matrix = [[0, 0], [0, 0]]
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            probability = model(batch)
            label = batch.y.float().view(-1, 1)
            size = label.size(0)
            loss_sum += criterion(probability, label).item() * size
            correct += ((probability >= 0.5) == label).sum().item()
            for actual, predicted in zip(label.reshape(-1).int().tolist(), (probability >= 0.5).reshape(-1).int().tolist()):
                matrix[actual][predicted] += 1
            count += size
    return {"loss": loss_sum / count, "accuracy": correct / count,
            "recall_ai": matrix[0][0] / max(sum(matrix[0]), 1),
            "recall_human": matrix[1][1] / max(sum(matrix[1]), 1)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--val-cache", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--output-dim", type=int, default=64)
    parser.add_argument("--input-dim", type=int, default=768)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args(argv)
    if min(args.epochs, args.patience, args.batch_size, args.input_dim, args.hidden_dim, args.output_dim) < 1:
        raise ValueError("Epochs, patience, batch size, and dimensions must be positive")
    if args.output.exists():
        raise FileExistsError(f"Choose a new checkpoint path: {args.output}")
    _seed(args.seed)
    train, validation = load_cache(args.train_cache), load_cache(args.val_cache)
    if train["edge_mode"] != "dependency_plus_sequential" or validation["edge_mode"] != "dependency_plus_sequential":
        raise ValueError("F1 requires dependency_plus_sequential caches")
    if train["preprocessing_sha256"] != validation["preprocessing_sha256"]:
        raise ValueError("Train/validation preprocessing fingerprints differ")
    if any(graph["x"].size(1) != args.input_dim for cache in (train, validation) for graph in cache["graphs"]):
        raise ValueError("Feature dimension does not match --input-dim")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_loader = _loader(to_data_list(train), args.batch_size, True)
    val_loader = _loader(to_data_list(validation), args.batch_size, False)
    model = PrismF1GCN(input_dim=args.input_dim, hidden_dim=args.hidden_dim, output_dim=args.output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion, best, remaining, history = nn.BCELoss(), -1.0, args.patience, []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        train_loss = train_correct = train_count = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            probability = model(batch)
            labels = batch.y.float().view(-1, 1)
            loss = criterion(probability, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * labels.size(0)
            train_correct += ((probability >= 0.5) == labels).sum().item()
            train_count += labels.size(0)
        metrics = evaluate(model, val_loader, device)
        row = {"epoch": epoch, **metrics, "train_loss": train_loss / train_count,
               "train_accuracy": train_correct / train_count,
               "lr": optimizer.param_groups[0]["lr"], "seconds": time.perf_counter() - epoch_start,
               "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0}
        history.append(row)
        improved = metrics["accuracy"] >= best
        print(json.dumps({**row, "best": improved}), flush=True)
        if improved:
            best, remaining = metrics["accuracy"], args.patience
            torch.save({"variant": "F1", "model_state": model.state_dict(), "seed": args.seed,
                        "config": config, "best_val_accuracy": best, "history": history,
                        "epoch": epoch, "optimizer_state": optimizer.state_dict(), "scheduler_state": None,
                        "config_sha256": config_hash, "parameter_count": parameter_count,
                        "preprocessing_sha256": train["preprocessing_sha256"], "protocol": train["protocol"],
                        "source_train_sha256": train.get("source_sha256"),
                        "source_val_sha256": validation.get("source_sha256"),
                        "edge_mode": train["edge_mode"], "cache_version": train["cache_version"]}, args.output)
        else:
            remaining -= 1
            if remaining <= 0:
                break
    args.output.with_suffix(".train.json").write_text(json.dumps({
        "config": config, "history": history, "best_val_accuracy": best,
        "seconds": time.perf_counter() - started, "parameter_count": parameter_count,
        "checkpoint_rule": "val_accuracy >= best; ties save and reset patience",
    }, indent=2), encoding="utf-8")
    return args.output


if __name__ == "__main__":
    main()
