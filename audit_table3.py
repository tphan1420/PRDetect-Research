"""Audit saved Table 3 predictions without loading models or fitting a threshold.

Uses only the standard library. The JSON output records archive hashes, checks
reported metrics against individual predictions, and describes source lengths.
Length is whitespace-delimited word count, NOT spaCy/BPE token count.
"""

import argparse
import ast
import hashlib
import io
import json
import math
import sys
from pathlib import Path
from statistics import mean, median
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def auc(truth, scores):
    """Mann-Whitney AUC with half credit for ties; human is positive."""
    ordered = sorted(zip(scores, truth))
    negatives = sum(y == 0 for y in truth)
    positives = len(truth) - negatives
    if not negatives or not positives:
        raise ValueError("AUC requires both classes")
    credit = lower_negatives = 0
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][0] == ordered[start][0]:
            end += 1
        group = ordered[start:end]
        neg = sum(y == 0 for _, y in group)
        pos = len(group) - neg
        credit += pos * (lower_negatives + 0.5 * neg)
        lower_negatives += neg
        start = end
    return credit / (negatives * positives)


def verify(report):
    y, p = report["y_true"], report["p_human"]
    assert len(y) == len(p) == report["samples"]
    assert set(y) == {0, 1} and report["threshold"] == 0.5
    assert all(math.isfinite(v) and 0 <= v <= 1 for v in p)
    matrix = [[0, 0], [0, 0]]
    for label, prob in zip(y, p):
        matrix[int(label)][int(prob >= 0.5)] += 1
    values = dict(accuracy=(matrix[0][0] + matrix[1][1]) / len(y),
                  auc=auc(y, p), recall_ai=matrix[0][0] / sum(matrix[0]),
                  recall_human=matrix[1][1] / sum(matrix[1]))
    assert matrix == report["confusion_matrix_rows_true_columns_pred"]
    for key, value in values.items():
        assert math.isclose(value, report[key], abs_tol=1e-8), (key, value, report[key])
    return dict(seed=report["seed"], split=report["split"], samples=len(y),
                confusion=matrix, **values)


def reports(archive, prefix):
    found = []
    for name in archive.namelist():
        if name.startswith(prefix) and name.endswith(".json"):
            report = json.loads(archive.read(name))
            if "p_human" in report:
                verify(report)
                found.append(report)
    return found


def label(row):
    value = row["label"]
    return int(value) if isinstance(value, (int, float)) else int("human" in value.lower())


def source_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summary(items):
    return {key: mean(r[key] for r in items)
            for key in ("accuracy", "auc", "recall_ai", "recall_human")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/archive/results/Table3_Audit_Data.json")
    args = parser.parse_args()
    paths = {
        "old_new": ROOT / "result/table3_rgcn_old_vs_new_2026.zip",
        "three_models": ROOT / "result/table3_hc3_to_gpt35_3models.zip",
        "v3": ROOT / "result/Cmix/v3_cmixstyle_ablation_results.zip",
    }
    with ZipFile(paths["old_new"]) as z:
        old, new = reports(z, "old_"), reports(z, "new_")
    with ZipFile(paths["three_models"]) as z:
        gcn = reports(z, "gcn_")
    with ZipFile(paths["v3"]) as z:
        with ZipFile(io.BytesIO(z.read("eval/baseline/v3_baseline_full_results.zip"))) as inner:
            v3 = reports(inner, "result/table3_")
            t4 = reports(inner, "result/table4_")
            metadata = json.loads(inner.read("result/checkpoint_metadata.json"))
        training = [json.loads(z.read(f"model/v3_baseline_seed{s}.train.json")) for s in (2025, 2026)]
    groups = {"gcn": gcn, "rgcn_old": old, "rgcn_retrained": new, "v3": v3}
    assert len(old) == len(new) == 5 and len(v3) == len(gcn) == 10
    expected = {r["split"]: r["y_true"] for r in old}
    for items in groups.values():
        for r in items:
            assert r["y_true"] == expected[r["split"]], "Label order differs"
    sources = {
        "hc3_train": ROOT / "data/original_text/benchmarks/hc3/hc3_train.json",
        "hc3_val": ROOT / "data/original_text/benchmarks/hc3/hc3_val.json",
        "hc3_test": ROOT / "data/original_text/benchmarks/hc3/hc3_test.json",
        "gpt35_test": ROOT / "data/original_text/benchmarks/gpt35_mixed/gpt3.5_mixed_test_split.json",
    }
    # Git checkout uses CRLF on Windows; Kaggle provenance hashes LF bytes.
    provenance_matches = {}
    reference = training[0]["provenance"]
    for name in ("building_graph.py", "dep_vocab.py", "dep_vocab.json"):
        normalized = (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
        provenance_matches[name] = hashlib.sha256(normalized).hexdigest() == reference["code_sha256"][name]
    for name in ("hc3_train", "hc3_val"):
        previous = next(v for k, v in reference["data"].items() if k.endswith(f"/{name}.json"))
        normalized = sources[name].read_bytes().replace(b"\r\n", b"\n")
        provenance_matches[name] = hashlib.sha256(normalized).hexdigest() == previous
    assert all(provenance_matches.values()), "Source/builder drift since the V3 run"
    lengths = {}
    for name, path in sources.items():
        rows = source_rows(path)
        lengths[name] = {}
        for cls in (0, 1):
            values = [len(r["text"].split()) for r in rows if label(r) == cls]
            lengths[name][str(cls)] = dict(n=len(values), mean=mean(values), median=median(values),
                                           fraction_above_512=mean(v > 512 for v in values))
    target = source_rows(sources["gpt35_test"])
    assert [label(r) for r in target] == expected["gpt3.5_mixed_test_split"]
    by_length = []
    for method, items in groups.items():
        for r in items:
            if r["split"] != "gpt3.5_mixed_test_split":
                continue
            for cls in (0, 1):
                for low, high in ((0, 256), (256, 512), (512, math.inf)):
                    ids = [i for i, row in enumerate(target) if label(row) == cls
                           and low <= len(row["text"].split()) < high]
                    by_length.append(dict(method=method, seed=r["seed"], label=cls,
                                          words=f"[{low}, {high})", n=len(ids),
                                          recall=mean(int(r["p_human"][i] >= .5) == cls for i in ids)
                                          if ids else None))
    shifts = []
    for before in old:
        after = next(r for r in new if r["split"] == before["split"])
        changes = {"correct_to_wrong": 0, "wrong_to_correct": 0}
        for y, p, q in zip(before["y_true"], before["p_human"], after["p_human"]):
            a, b = int(p >= .5) == y, int(q >= .5) == y
            changes["correct_to_wrong"] += int(a and not b)
            changes["wrong_to_correct"] += int(b and not a)
        medians = {str(cls): {tag: median(p for y, p in zip(r["y_true"], r["p_human"]) if y == cls)
                              for tag, r in (("old", before), ("new", after))} for cls in (0, 1)}
        shifts.append(dict(split=before["split"], medians_p_human=medians, **changes))
    # Execute the actual scalar-compatible function without importing torch/numpy.
    tree = ast.parse((ROOT / "model/QuangV3.py").read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "fwd_mask")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "QuangV3.fwd_mask", "exec"), namespace)
    vocab = json.loads((ROOT / "dep_vocab.json").read_text())["dep2id"]
    relation_audit = []
    for ckpt in metadata["checkpoints"]:
        R = ckpt["R"]
        excluded = {k: v for k, v in vocab.items() if v < R
                    and not namespace["fwd_mask"](v, R, ckpt["struct_mode_used"])}
        relation_audit.append(dict(seed=ckpt["seed"], R=R, excluded_forward_labels=excluded))
    old_hashes = {r["checkpoint_sha256"] for r in old}
    local_hash = sha256(ROOT / "model/hc3_rgcn_model_2026.pth")
    assert old_hashes == {local_hash}, "Local old checkpoint does not match old/new reports"
    output = dict(
        scope="Retrospective descriptive audit; no training, fitted thresholds or causal claims.",
        caveat="Matching labels/split names alone does not prove identical embeddings across archives.",
        archive_sha256={k: sha256(v) for k, v in paths.items()},
        source_sha256={k: sha256(v) for k, v in sources.items()},
        v3_provenance_matches_after_CRLF_to_LF=provenance_matches,
        audited_code_sha256={k: sha256(ROOT / k) for k in
                            ("audit_table3.py", "model/QuangV3.py", "dep_vocab.json")},
        old_checkpoint_sha256=local_hash,
        per_seed={k: {str(seed): summary([r for r in items if r["seed"] == seed])
                      for seed in sorted({r["seed"] for r in items})} for k, items in groups.items()},
        aggregate={k: summary(v) for k, v in groups.items()},
        per_split={k: [verify(r) for r in v] for k, v in groups.items()},
        v3_table4_noise=summary([r for r in t4 if r["split"] != "hc3_test"]),
        old_new_shifts=shifts, length_unit="whitespace words, not tokenizer tokens",
        source_lengths=lengths, target_recall_by_length=by_length,
        relation_audit=relation_audit,
        v3_training=[dict(seed=d["provenance"]["seed"], best_epoch=d["best_epoch"],
                          config=d["provenance"]["config"]) for d in training],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"per_seed": output["per_seed"], "aggregate": output["aggregate"],
                      "verified_reports": sum(map(len, groups.values())) + len(t4)}, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
