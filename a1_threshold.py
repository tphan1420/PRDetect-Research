"""Validation-only threshold selection for an existing A1-only RGCN2 checkpoint.

Commands: prepare (validation perturbations), score (one graph), fit (validation
scores only), evaluate (fixed threshold). No model training or test-time fitting.
Heavy dependencies are imported only by prepare/score.
"""

import argparse
import hashlib
import inspect
import json
import math
import pickle
import random
import re
from collections import defaultdict
from pathlib import Path


CONDITIONS = ("clean", "insert", "delete", "repeat")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding="utf-8")


def auc_from_logits(labels, scores):
    pairs = sorted(zip(scores, labels))
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError("Both AI and Human are required for AUROC.")
    rank_sum = 0.0
    left = 0
    while left < len(pairs):
        right = left + 1
        while right < len(pairs) and pairs[right][0] == pairs[left][0]:
            right += 1
        rank_sum += ((left + 1 + right) / 2) * sum(y for _, y in pairs[left:right])
        left = right
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def metrics(records, threshold):
    if not records or not math.isfinite(threshold):
        raise ValueError("Require nonempty records and a finite threshold.")
    labels = [int(row["label"]) for row in records]
    scores = [float(row["logit_human"]) for row in records]
    if set(labels) != {0, 1} or not all(math.isfinite(s) for s in scores):
        raise ValueError("Require finite logits and both binary labels.")
    predictions = [int(s >= threshold) for s in scores]
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    return {
        "samples": len(labels),
        "accuracy": (tn + tp) / len(labels),
        "balanced_accuracy": (tn / (tn + fp) + tp / (tp + fn)) / 2,
        "auc_logit": auc_from_logits(labels, scores),
        "f1_human": 2 * tp / max(2 * tp + fp + fn, 1),
        "f1_ai": 2 * tn / max(2 * tn + fp + fn, 1),
        "recall_ai": tn / (tn + fp),
        "recall_human": tp / (tp + fn),
        "predicted_human": sum(predictions),
        "confusion_matrix_ai_human": [[tn, fp], [fn, tp]],
    }


def choose_threshold(groups):
    """Exact sweep over tied logits; equal condition/class weights, not a grid."""
    events = defaultdict(float)
    for rows in groups:
        metrics(rows, 0.0)  # Validate finite logits and class coverage.
        positives = sum(int(row["label"]) for row in rows)
        negatives = len(rows) - positives
        for row in rows:
            contribution = (1 / positives if row["label"] == 1 else -1 / negatives)
            events[float(row["logit_human"])] += contribution / (2 * len(groups))

    # Above the largest score everything is AI: balanced accuracy = 0.5.
    threshold = math.nextafter(max(events), math.inf)
    best_score = sum(metrics(rows, 0.0)["balanced_accuracy"] for rows in groups) / len(groups)
    best_threshold = 0.0
    current_score = 0.5
    if current_score > best_score + 1e-12:
        best_score, best_threshold = current_score, threshold
    for threshold in sorted(events, reverse=True):
        current_score += events[threshold]
        if (current_score > best_score + 1e-12 or
                (abs(current_score - best_score) <= 1e-12 and
                 abs(threshold) < abs(best_threshold))):
            best_score, best_threshold = current_score, threshold
    return best_threshold


def prepare(args):
    import spacy
    from building_graph_v2 import read_records

    if "val" not in args.source.stem.lower() or "test" in args.source.stem.lower():
        raise ValueError("--source must be a validation split, never a test split.")
    records = read_records(args.source)
    if not records or not 0 < args.percent < 100:
        raise ValueError("Require nonempty validation data and 0 < percent < 100.")
    paths = {c: args.output_dir / ("hc3_val_cal_{}{}.json".format(
        c, "" if c == "clean" else "_{}".format(args.percent))) for c in CONDITIONS}
    if any(p.exists() for p in paths.values()) or (args.output_dir / "manifest.json").exists():
        raise FileExistsError("Validation outputs exist; reuse them or choose another directory.")
    nlp = spacy.load(args.spacy_model)
    tokens = [[token.text for token in doc]
              for doc in nlp.pipe([str(r["text"]) for r in records])]
    vocabulary = sorted({t for words in tokens for t in words if 1 < len(t) <= 20})
    if not vocabulary:
        raise ValueError("No validation vocabulary for insert perturbations.")
    rng = random.Random(args.seed)
    masked = [" ".join("[[" + t + "]]" if len(t) > 1 and
                      rng.random() < args.percent / 100 else t for t in words)
              for words in tokens]
    rng = random.Random(args.seed)
    output = {"clean": records, "insert": [], "delete": [], "repeat": []}
    # Follow other_perturb.ipynb's global replace behavior, not per-token EDA.
    # The insertion vocabulary comes exclusively from validation; sorting makes it reproducible.
    for record, text in zip(records, masked):
        views = {c: text for c in CONDITIONS[1:]}
        for word in re.findall(r"\[\[(.*?)\]\]", text):
            marker = "[[" + word + "]]"
            views["delete"] = views["delete"].replace(marker, "")
            views["repeat"] = views["repeat"].replace(marker, word + " " + word)
            views["insert"] = views["insert"].replace(marker, rng.choice(vocabulary) + " " + word)
        for condition, value in views.items():
            row = dict(record)
            row["text"] = re.sub(r"\s{2,}", " ", value)
            output[condition].append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for condition, path in paths.items():
        with path.open("w", encoding="utf-8") as handle:
            for row in output[condition]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(args.output_dir / "manifest.json", {
        "role": "validation", "source": str(args.source), "source_sha256": digest(args.source),
        "seed": args.seed, "percent": args.percent, "spacy_version": spacy.__version__,
        "spacy_model_version": nlp.meta.get("version"),
        "files": {c: {"path": str(p), "sha256": digest(p)} for c, p in paths.items()},
    })
    print("Created four validation JSONL files (no graph files):", args.output_dir)


def score(args):
    import torch
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import global_mean_pool
    from model.RGCN import RGCN2

    with args.file.open("rb") as handle:
        graph = pickle.load(handle)
    metadata = graph.get("ablation_metadata", {})
    if metadata.get("variant") != "a1_only_on_v1":
        raise ValueError("Require an A1-only pickle made by export_a1_v1.py.")
    source = Path(metadata["source"])
    if args.role == "validation" and ("val" not in source.stem.lower() or "test" in source.stem.lower()):
        raise ValueError("Cannot mark a test graph as validation.")
    if not source.is_file():
        raise FileNotFoundError("Need source JSON for provenance: {}".format(source))
    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = True
    state = torch.load(args.checkpoint, **kwargs)
    if "model_state" in state or "conv3.weight" in state:
        raise ValueError("Require a plain RGCN2 A1-only state_dict, not a V2 checkpoint.")
    if "conv1.comp" not in state:
        raise ValueError("This scorer expects the basis-decomposed PyG RGCN2 used by the ablation.")
    relations, bases = state["conv1.comp"].shape
    model = RGCN2(
        input_dim=state["conv1.root"].size(0),
        hidden_dim=state["conv1.root"].size(1),
        output_dim=state["fc.weight"].size(1),
        num_relations=relations, num_bases=bases,
    )
    model.load_state_dict(state, strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = model.to(device).eval()
    labels = torch.as_tensor(graph["y"]).view(-1).long()
    count = len(labels)
    for key in ("all_token_embeddings", "all_edge_index", "all_edge_type"):
        if len(graph[key]) != count:
            raise ValueError("Graph/label length mismatch.")
    data = []
    for i in range(count):
        edge_type = graph["all_edge_type"][i].long()
        if edge_type.numel() and (edge_type.min() < 0 or edge_type.max() >= relations):
            raise ValueError("Relation ID outside checkpoint schema.")
        data.append(Data(x=graph["all_token_embeddings"][i].float(),
                         edge_index=graph["all_edge_index"][i].long(), edge_type=edge_type,
                         y=labels[i].view(1)))
    # Capture fc output in the actual unchanged V1 forward pass, then use its mean pooling.
    captured = {}
    def capture_fc(module, inputs, output):
        captured["nodes"] = output.detach()
    hook = model.fc.register_forward_hook(capture_fc)
    records = []
    try:
        with torch.inference_mode():
            for batch in DataLoader(data, batch_size=args.batch_size, shuffle=False):
                batch = batch.to(device)
                probabilities = model(batch)
                logits = global_mean_pool(captured.pop("nodes"), batch.batch, size=batch.num_graphs)
                if not torch.allclose(torch.sigmoid(logits), probabilities, atol=1e-7, rtol=1e-6):
                    raise RuntimeError("Captured logit does not reproduce V1 forward output.")
                for label, value in zip(batch.y.cpu().tolist(), logits.view(-1).double().cpu().tolist()):
                    records.append({"index": len(records), "label": int(label), "logit_human": value})
    finally:
        hook.remove()
    baseline = metrics(records, 0.0)
    write_json(args.output, {
        "role": args.role, "condition": args.condition, "graph": str(args.file),
        "source": str(source), "source_sha256": digest(source),
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": digest(args.checkpoint),
        "relation_vocab_sha256": digest(Path("dep_vocab.json")),
        "score_space": "human_logit", "records": records, "baseline": baseline,
    })
    print(json.dumps(baseline, indent=2))


def fit(args):
    if args.output.exists():
        raise FileExistsError("Threshold is already frozen; choose another output for a new experiment.")
    manifest = read_json(args.manifest)
    if manifest.get("role") != "validation":
        raise ValueError("Require a validation manifest.")
    payloads = [read_json(path) for path in args.scores]
    if len(payloads) != 4 or {p["condition"] for p in payloads} != set(CONDITIONS):
        raise ValueError("Require exactly clean/insert/delete/repeat validation scores.")
    for p in payloads:
        if p["role"] != "validation" or p["score_space"] != "human_logit":
            raise ValueError("Threshold fitting accepts validation logits only, not test scores.")
        if p["source_sha256"] != manifest["files"][p["condition"]]["sha256"]:
            raise ValueError("Scores do not match the generated validation manifest.")
    for key in ("checkpoint_sha256", "relation_vocab_sha256"):
        if len({p[key] for p in payloads}) != 1:
            raise ValueError("Validation scores use different checkpoints or relation vocabularies.")
    label_orders = [[r["label"] for r in p["records"]] for p in payloads]
    if any(labels != label_orders[0] for labels in label_orders):
        raise ValueError("Validation conditions must have identical label order and sample counts.")
    threshold = choose_threshold([p["records"] for p in payloads])
    report = {
        "score_space": "human_logit", "threshold_logit": threshold,
        "checkpoint_sha256": payloads[0]["checkpoint_sha256"],
        "relation_vocab_sha256": payloads[0]["relation_vocab_sha256"],
        "objective": "mean balanced accuracy; equal weight to each validation condition",
        "manifest": str(args.manifest), "manifest_sha256": digest(args.manifest),
        "validation": {p["condition"]: {
            "baseline": metrics(p["records"], 0.0),
            "calibrated": metrics(p["records"], threshold),
        } for p in payloads},
    }
    write_json(args.output, report)
    print("Frozen shared logit threshold:", threshold)
    for condition, values in report["validation"].items():
        print(condition, "balanced accuracy:", values["baseline"]["balanced_accuracy"],
              "->", values["calibrated"]["balanced_accuracy"])


def evaluate(args):
    threshold = read_json(args.threshold)
    scores = read_json(args.scores)
    for key in ("checkpoint_sha256", "relation_vocab_sha256", "score_space"):
        if threshold[key] != scores[key]:
            raise ValueError("Threshold/score mismatch: {}".format(key))
    report = {
        "condition": scores["condition"], "role": scores["role"],
        "threshold_logit": threshold["threshold_logit"],
        "baseline": metrics(scores["records"], 0.0),
        "calibrated": metrics(scores["records"], threshold["threshold_logit"]),
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("data/perturbed_text/calibration_validation"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--percent", type=int, default=25)
    p.add_argument("--spacy-model", default="en_core_web_sm")
    p.set_defaults(function=prepare)
    p = commands.add_parser("score")
    p.add_argument("--file", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--role", choices=("validation", "test"), required=True)
    p.add_argument("--condition", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--cpu", action="store_true")
    p.set_defaults(function=score)
    p = commands.add_parser("fit")
    p.add_argument("--scores", type=Path, nargs="+", required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(function=fit)
    p = commands.add_parser("evaluate")
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--threshold", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(function=evaluate)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
