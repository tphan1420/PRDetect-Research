"""Replay the source-fitted router on the three notebook 07 test sets and paired GCN predictions."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import joblib
import numpy as np

from benchmark_v3_stability import accuracy_stats, paired_interval
from evaluate_v3_quang import REPO, sha256, write_json
from v3_a1_fusion import labels_from, paired_scores, read_source, source_digest
from v3_a1_routing import fit_router, predict_human, use_v3


CASES = {
    "hc3": "data/original_text/benchmarks/hc3/hc3_test.json",
    "gpt35_0pct": "data/original_text/benchmarks/gpt35_mixed/gpt3.5_mixed_test_split.json",
    "hc3_replace25": "data/perturbed_text/hc3/structural_noise/hc3_test_replace_25.json",
}


def author_reports(root, seeds):
    reports = {}
    for archive_name, suite, split in (("author_table2_results.zip", "table2", "hc3_test"),
                                       ("author_table3_table4_results.zip", "table4", "hc3_test_replace_25")):
        with ZipFile(root / archive_name) as archive:
            for row in json.loads(archive.read("results/per_seed_reports.json")):
                if row["suite"] == suite and row["split"] == split:
                    reports[row["seed"], split] = row
    for seed in seeds:
        with ZipFile(root / f"author_gcn_hc3_to_gpt35_0pct_seed{seed}.zip") as archive:
            row = json.loads(archive.read("gpt35_0pct_report.json"))
            reports[seed, row["test_split"].removesuffix(".json")] = row
    return reports


def replay(path, manifest, cases, author):
    seed = manifest["seed"]
    rows = []
    differences = {}
    with ZipFile(path) as archive:
        for name, (source, truth, mask) in cases.items():
            v3 = json.loads(archive.read(f"results/{name}_v3.json"))
            a1 = json.loads(archive.read(f"results/scores_{source.stem}.json"))
            if (v3["seed"] != seed or v3["checkpoint_sha256"] != manifest["v3_sha256"]
                    or a1["checkpoint_sha256"] != manifest["a1_sha256"]):
                raise ValueError(f"Checkpoint mismatch in {path.name} / {name}")
            product = paired_scores(source, truth, v3, a1)
            logits = np.array([r["logit_human"] for r in a1["records"]])
            threshold = v3["source_human95_threshold"]
            p = np.array(v3["p_human"])
            prediction = predict_human(mask, p, logits, threshold)
            control = author[seed, source.stem]
            if control["seed"] != seed or control["threshold"] != 0.5 or not np.array_equal(control["y_true"], truth):
                raise ValueError(f"Unpaired GCN baseline for {seed} / {name}")
            correct = (np.array(control["p_human"]) >= 0.5) == truth
            if not np.isclose(correct.mean(), control["accuracy"]):
                raise ValueError(f"Inconsistent baseline accuracy for {seed} / {name}")
            differences[name] = (prediction == truth).astype(int) - correct.astype(int)
            rows.append({
                "case": name, "samples": len(truth), "v3_documents": int(mask.sum()),
                "gcn": float(correct.mean()), "v3_human95": float(((p >= threshold) == truth).mean()),
                "a1": float(((logits >= 0) == truth).mean()), "product": float(((product >= 0.5) == truth).mean()),
                "routed": float((prediction == truth).mean()), "predicted_human": prediction.astype(int).tolist(),
            })
    return ({"seed": seed, "archive": path.name, "dual_training": "v3_ema_sha256" in manifest,
             "v3_sha256": manifest["v3_sha256"], "a1_sha256": manifest["a1_sha256"], "results": rows}, differences)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", type=Path, nargs="+", help="One notebook 07 RAW+A1 archive per seed")
    parser.add_argument("--output-dir", type=Path, default=REPO / "result/Result_v3/v3_a1_routing")
    parser.add_argument("--author-dir", type=Path, default=REPO / "result/author")
    args = parser.parse_args()
    hc3 = REPO / "data/original_text/benchmarks/hc3"
    train, val = hc3 / "hc3_train.json", hc3 / "hc3_val.json"
    router = fit_router([r["text"] for r in read_source(train)], [r["text"] for r in read_source(val)])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    router_path = args.output_dir / "hc3_router.joblib"
    joblib.dump(router, router_path, compress=3)
    cases = {}
    for name, relative in CASES.items():
        source = REPO / relative
        rows = read_source(source)
        cases[name] = (source, labels_from(rows), use_v3(router, [r["text"] for r in rows]))
    inputs = {}
    for path in args.archives:
        with ZipFile(path) as archive:
            manifest = json.loads(archive.read("results/manifest.json"))
        if manifest["seed"] in inputs:
            raise ValueError(f"Two archives for seed {manifest['seed']}; provide one paired run per seed")
        inputs[manifest["seed"]] = (path, manifest)
    author = author_reports(args.author_dir, inputs)
    runs, differences = [], defaultdict(dict)
    for seed, (path, manifest) in sorted(inputs.items()):
        run, deltas = replay(path, manifest, cases, author)
        runs.append(run)
        for name, delta in deltas.items():
            differences[name][seed] = delta
    summary = {}
    groups = {"all": runs}
    if any(r["dual_training"] for r in runs) and not all(r["dual_training"] for r in runs):
        groups["dual_only"] = [r for r in runs if r["dual_training"]]
    for group, selected in groups.items():
        summary[group] = {}
        for name, (_, truth, _) in cases.items():
            rows = [row for run in selected for row in run["results"] if row["case"] == name]
            stats = {k: accuracy_stats([row[k] for row in rows]) for k in ("gcn", "a1", "v3_human95", "product", "routed")}
            interval = paired_interval([differences[name][r["seed"]] for r in selected], truth)
            summary[group][name] = {**stats, **interval, "seed_wins": sum(row["routed"] > row["gcn"] for row in rows)}
            print(f"{group:10} {name:16} routed {stats['routed']['mean']:.2%}   GCN {stats['gcn']['mean']:.2%}")
    write_json(args.output_dir / "paired_results.json", {
        "fit_sources": {p.name: source_digest(p) for p in (train, val)},
        "test_sources": {k: source_digest(v[0]) for k, v in cases.items()},
        "router_code_sha256": sha256(REPO / "v3_a1_routing.py"), "router_sha256": sha256(router_path),
        "cutoff": router["cutoff"], "validation": router["source_validation"], "runs": runs, "summary": summary,
    })


if __name__ == "__main__":
    main()
