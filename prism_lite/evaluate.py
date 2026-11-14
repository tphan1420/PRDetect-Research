"""Evaluate an F1 checkpoint on a separately built PRISM cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .data import load_cache, to_data_list
from .model import PrismF1GCN
from .train import _loader


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if state.get("variant") != "F1":
        raise ValueError(f"F1 evaluator refuses checkpoint variant {state.get('variant')!r}")
    cache = load_cache(args.cache)
    if cache["edge_mode"] != "dependency_plus_sequential":
        raise ValueError("F1 evaluation requires dependency_plus_sequential cache")
    if state.get("preprocessing_sha256") != cache["preprocessing_sha256"]:
        raise ValueError("Checkpoint and evaluation cache preprocessing fingerprints differ")
    config = state["config"]
    model = PrismF1GCN(input_dim=config["input_dim"], hidden_dim=config["hidden_dim"], output_dim=config["output_dim"]).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()
    predictions, labels, sample_ids = [], [], []
    with torch.no_grad():
        for batch in _loader(to_data_list(cache), args.batch_size, False):
            batch = batch.to(device)
            predictions.extend(model(batch).reshape(-1).cpu().tolist())
            labels.extend(batch.y.reshape(-1).cpu().int().tolist())
            sample_ids.extend(batch.sample_id.reshape(-1).cpu().tolist())
    predicted = [int(value >= 0.5) for value in predictions]
    matrix = [[sum(actual == row and inferred == column for actual, inferred in zip(labels, predicted))
               for column in (0, 1)] for row in (0, 1)]
    accuracy = sum(actual == inferred for actual, inferred in zip(labels, predicted)) / len(labels)
    from sklearn.metrics import f1_score, roc_auc_score
    auc = float(roc_auc_score(labels, predictions)) if len(set(labels)) == 2 else None
    f1 = float(f1_score(labels, predicted, zero_division=0))
    report = {"variant": "F1", "checkpoint": str(Path(args.checkpoint).resolve()),
              "cache": str(Path(args.cache).resolve()), "samples": len(labels), "threshold": 0.5,
              "accuracy": accuracy, "auc": auc, "f1_human": f1,
              "f1_macro": float(f1_score(labels, predicted, labels=[0, 1], average="macro", zero_division=0)),
              "seed": state["seed"], "preprocessing_sha256": cache["preprocessing_sha256"],
              "recall_ai": matrix[0][0] / max(sum(matrix[0]), 1),
              "recall_human": matrix[1][1] / max(sum(matrix[1]), 1),
              "confusion_matrix_rows_true_columns_pred": matrix,
              "probabilities_human": predictions, "labels": labels, "sample_ids": sample_ids}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"probabilities_human", "labels"}}, indent=2))
    return report


if __name__ == "__main__":
    main()
