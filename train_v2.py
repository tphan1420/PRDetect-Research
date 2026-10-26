"""Train PRDetect V2 on precomputed graph_data_v2 pickles.

This script intentionally trains and validates only. Use ``test_v2.py`` later
for Kaggle evaluation on clean, perturbed or cross-dataset test sets.
"""

import argparse
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from model.PRDetectV2 import ABLATION_MODES, PRDetectV2
from v2_data import (
    infer_dimensions,
    load_payload,
    payload_to_data_list,
    resolve_graph_file,
)


def configure_console() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def binary_metrics(labels: List[int], probabilities: List[float]) -> Dict[str, float]:
    predictions = [1 if probability >= 0.5 else 0 for probability in probabilities]
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }
    if len(set(labels)) == 2:
        metrics["auc"] = float(roc_auc_score(labels, probabilities))
    else:
        metrics["auc"] = float("nan")
    return metrics


def evaluate(
    model: PRDetectV2,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> Dict[str, float]:
    model.eval()
    labels: List[int] = []
    probabilities: List[float] = []
    total_loss = 0.0
    total_samples = 0

    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            targets = batch.y.float().view(-1, 1)
            loss = criterion(logits, targets)
            batch_size = targets.size(0)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            probabilities.extend(torch.sigmoid(logits).view(-1).cpu().tolist())
            labels.extend(targets.view(-1).long().cpu().tolist())

    metrics = binary_metrics(labels, probabilities)
    metrics["loss"] = total_loss / max(total_samples, 1)
    return metrics


def checkpoint_score(metrics: Dict[str, float]) -> float:
    auc = metrics["auc"]
    return metrics["accuracy"] if math.isnan(auc) else auc


def save_checkpoint(
    path: Path,
    model: PRDetectV2,
    model_config: Dict[str, Any],
    training_config: Dict[str, Any],
    graph_metadata: Dict[str, Any],
    epoch: int,
    validation_metrics: Dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": "2.0",
            "model_state": model.state_dict(),
            "model_config": model_config,
            "training_config": training_config,
            "graph_metadata": graph_metadata,
            "epoch": epoch,
            "validation_metrics": validation_metrics,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="hc3")
    parser.add_argument("--train-file", default="hc3_train")
    parser.add_argument("--val-file", default="hc3_val")
    parser.add_argument("--graph-dir", type=Path, default=Path("graph_data_v2"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--graph-dim", type=int, default=128)
    parser.add_argument("--fusion-dim", type=int, default=256)
    parser.add_argument("--quality-hidden-dim", type=int, default=32)
    parser.add_argument("--num-bases", type=int, default=30)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--mode", choices=ABLATION_MODES, default="hybrid")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--result-dir", type=Path, default=Path("result"))
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_console()
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    train_path = resolve_graph_file(args.train_file, args.graph_dir)
    val_path = resolve_graph_file(args.val_file, args.graph_dir)
    train_payload = load_payload(train_path)
    val_payload = load_payload(val_path)
    if train_payload["encoder_name"] != val_payload["encoder_name"]:
        raise ValueError("Train và validation graph dùng encoder khác nhau.")
    if train_payload["encoder_hidden_size"] != val_payload["encoder_hidden_size"]:
        raise ValueError("Train và validation graph khác embedding dimension.")

    train_data = payload_to_data_list(train_payload)
    val_data = payload_to_data_list(val_payload)
    dimensions = infer_dimensions(train_data)
    if infer_dimensions(val_data) != dimensions:
        raise ValueError("Train và validation graph không cùng feature schema.")

    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    model_config: Dict[str, Any] = {
        "input_dim": dimensions["input_dim"],
        "hidden_dim": args.hidden_dim,
        "graph_dim": args.graph_dim,
        "fusion_dim": args.fusion_dim,
        "quality_dim": dimensions["quality_dim"],
        "quality_hidden_dim": args.quality_hidden_dim,
        "num_relations": dimensions["num_relations"],
        "num_bases": args.num_bases,
        "dropout": args.dropout,
        "mode": args.mode,
    }
    model = PRDetectV2(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()

    checkpoint_path = args.checkpoint or Path(
        "model/{}_prdetect_v2_{}_{}.pth".format(
            args.dataset, args.mode, args.seed
        )
    )
    training_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    training_config["checkpoint"] = str(checkpoint_path)
    graph_metadata = {
        "schema_version": train_payload["schema_version"],
        "encoder_name": train_payload["encoder_name"],
        "encoder_hidden_size": train_payload["encoder_hidden_size"],
        "num_relations": train_payload["num_relations"],
        "relation_labels": train_payload["relation_labels"],
    }

    best_score = -float("inf")
    epochs_without_improvement = 0
    history: List[Dict[str, Any]] = []

    print(
        "PRDetect V2 | mode={} | device={} | train={} | val={} | checkpoint={}"
        .format(args.mode, device, len(train_data), len(val_data), checkpoint_path)
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        progress = tqdm(train_loader, desc="Epoch {}/{}".format(epoch, args.epochs))
        for batch in progress:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            targets = batch.y.float().view(-1, 1)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            batch_size = targets.size(0)
            running_loss += float(loss.item()) * batch_size
            seen += batch_size
            progress.set_postfix(loss="{:.4f}".format(running_loss / max(seen, 1)))

        train_loss = running_loss / max(seen, 1)
        validation = evaluate(model, val_loader, device, criterion)
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation": validation,
        }
        history.append(epoch_record)
        print(
            "Epoch {} | train_loss={:.4f} | val_loss={:.4f} | "
            "val_acc={:.4f} | val_auc={:.4f} | val_f1={:.4f}"
            .format(
                epoch,
                train_loss,
                validation["loss"],
                validation["accuracy"],
                validation["auc"],
                validation["f1"],
            )
        )

        score = checkpoint_score(validation)
        if score > best_score:
            best_score = score
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_path,
                model,
                model_config,
                training_config,
                graph_metadata,
                epoch,
                validation,
            )
            print("  -> Saved best checkpoint (score={:.4f})".format(score))
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("Early stopping after {} stale epochs.".format(args.patience))
                break

    args.result_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.result_dir / "{}_prdetect_v2_{}_train_{}.json".format(
        args.dataset, args.mode, args.seed
    )
    summary_path.write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "checkpoint": str(checkpoint_path),
                "model_config": model_config,
                "training_config": training_config,
                "graph_metadata": graph_metadata,
                "history": history,
            },
            indent=2,
            allow_nan=True,
        ),
        encoding="utf-8",
    )
    print("Training summary: {}".format(summary_path))


if __name__ == "__main__":
    main()
