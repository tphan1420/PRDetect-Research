"""Evaluate one LISA-GCN checkpoint on a labeled raw JSONL split."""

import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from infer_lisa_gcn import document_logits
from lisa_gcn.data import build_document, label_of, read_jsonl
from lisa_gcn.model import LisaGCN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--window-chunk-size", type=int, default=16)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    config = state["config"]
    model = LisaGCN(hidden_dim=config["hidden_dim"], output_dim=config["output_dim"], dropout=config["dropout"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    records = read_jsonl(args.input_json)
    predictions = []
    with torch.no_grad():
        for index, record in enumerate(records):
            document = build_document(str(record["text"]), label=label_of(record), document_index=index)
            logit = document_logits(model, [document], 128, device, args.window_chunk_size)
            predictions.append({"index": index, "y_true": int(label_of(record)), "p_human": float(torch.sigmoid(logit).item())})
    truth = [row["y_true"] for row in predictions]
    probabilities = [row["p_human"] for row in predictions]
    predicted = [int(value >= 0.5) for value in probabilities]
    matrix = [[sum(y == actual and p == inferred for y, p in zip(truth, predicted)) for inferred in (0, 1)]
              for actual in (0, 1)]
    metrics = {"samples": len(predictions), "threshold": 0.5,
               "accuracy": float(accuracy_score(truth, predicted)),
               "auc": float(roc_auc_score(truth, probabilities)),
               "f1": float(f1_score(truth, predicted, zero_division=0)),
               "recall_ai": matrix[0][0] / max(sum(matrix[0]), 1),
               "recall_human": matrix[1][1] / max(sum(matrix[1]), 1),
               "confusion_matrix_rows_true_columns_pred": matrix,
               "checkpoint": str(Path(args.checkpoint).resolve()), "input": str(Path(args.input_json).resolve())}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "predictions.json").write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
