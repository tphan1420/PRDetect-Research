"""Evaluate a PRDetect V2 checkpoint on a precomputed V2 graph file."""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from model.PRDetectV2 import CHECKPOINT_FORMAT, PRDetectV2
from v2_data import load_payload, payload_to_data_list, resolve_graph_file


def configure_console() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", required=True, help="V2 graph pickle path hoặc stem.")
    parser.add_argument("--graph-dir", type=Path, default=Path("graph_data_v2"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_console()
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold phải nằm trong [0, 1].")
    if not args.checkpoint.is_file():
        raise FileNotFoundError("Không tìm thấy checkpoint: {}".format(args.checkpoint))

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    graph_path = resolve_graph_file(args.file, args.graph_dir)
    payload = load_payload(graph_path)
    data_list = payload_to_data_list(payload)
    loader = DataLoader(data_list, batch_size=args.batch_size, shuffle=False)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    if checkpoint.get("format_version") != CHECKPOINT_FORMAT:
        raise ValueError(
            "Cần checkpoint A1+A2 mới (2.1-graph). Checkpoint V2 cũ không tương thích; "
            "hãy train lại bằng train_v2.py."
        )
    graph_metadata = checkpoint["graph_metadata"]
    for key in (
        "schema_version",
        "encoder_name",
        "encoder_hidden_size",
        "num_relations",
        "relation_labels",
    ):
        if graph_metadata[key] != payload[key]:
            raise ValueError("Checkpoint và graph không khớp metadata '{}'.".format(key))

    model_config = checkpoint["model_config"]
    model = PRDetectV2(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    labels: List[int] = []
    probabilities: List[float] = []
    total_loss = 0.0
    total_samples = 0
    started = time.time()

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Testing PRDetect V2"):
            batch = batch.to(device)
            logits = model(batch)
            targets = batch.y.float().view(-1, 1)
            loss = criterion(logits, targets)
            batch_size = targets.size(0)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            probabilities.extend(torch.sigmoid(logits).view(-1).cpu().tolist())
            labels.extend(targets.view(-1).long().cpu().tolist())

    predictions = [
        1 if probability >= args.threshold else 0 for probability in probabilities
    ]
    auc = (
        float(roc_auc_score(labels, probabilities))
        if len(set(labels)) == 2
        else float("nan")
    )
    results: Dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "graph": str(graph_path),
        "checkpoint": str(args.checkpoint),
        "use_quality": model.use_quality,
        "samples": total_samples,
        "threshold": args.threshold,
        "loss": total_loss / max(total_samples, 1),
        "accuracy": float(accuracy_score(labels, predictions)),
        "auc": auc,
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "confusion_matrix_ai_human": confusion_matrix(
            labels, predictions, labels=[0, 1]
        ).tolist(),
        "predicted_human": sum(predictions),
        "mean_probability_human": sum(probabilities) / total_samples,
        "elapsed_seconds": time.time() - started,
    }

    print(json.dumps(results, indent=2, allow_nan=True))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(results, indent=2, allow_nan=True), encoding="utf-8"
        )
        print("Saved: {}".format(args.output))


if __name__ == "__main__":
    main()
