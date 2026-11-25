"""Replay the project's paired HC3 -> GPT-3.5 and HC3 robustness experiments."""

import argparse
import csv
import hashlib
import io
import json
import pickle
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.special import expit

from evaluate_v3_quang import REPO, TRAINING_DOMAINS, benchmark_cases, human95_threshold, score_probabilities, sha256, write_json
from v3_a1_fusion import labels_from, paired_scores, read_source
from calibrate_v3 import calibrated_probabilities


COHORTS = {"deterministic": "Current version ( determinism )",
           "non_deterministic": "noDesterminsm", "historical": "dual_rawema"}
SEEDS = (2020, 2022, 2024, 2025, 2026)


def reports(root):
    cases = benchmark_cases("full")
    clean = next(c for c in cases if c.suite == "table3" and c.noise_percent == 0)
    for cohort, folder in COHORTS.items():
        paths = sorted((root / folder).glob("v3_dual_ema_raw_hc3_to_gpt35_0pct_seed*_batch1.zip"))
        if cohort == "non_deterministic":
            paths += sorted((root / folder).glob("v3_dual_ema_raw_hc3_to_gpt35_0pct_seed*_batch1 (1).zip"))
        seen = set()
        for path in paths:
            with ZipFile(path) as archive:
                report = json.loads(archive.read("gpt35_0pct_raw.json"))
                key, seed = report["checkpoint_sha256"], report["seed"]
                config = json.loads(archive.read("config_raw.json"))
                history = json.loads(archive.read(f"v3_baseline_dual_seed{seed}.train.json"))
                if (config["checkpoint_sha256"][str(seed)] != key or config["weight_source"] != "raw"
                        or history["checkpoint_sha256"]["raw"] != key or not history["complete"]):
                    raise ValueError(f"Invalid RAW archive {path}")
                run = history["provenance"]["config"]
                if cohort != "historical" and run["deterministic"] != (cohort == "deterministic"):
                    raise ValueError(f"Incorrect determinism cohort for {path}")
                if key in seen:
                    continue
                seen.add(key)
                group = cohort + ("_repeat" if "(1)" in path.name else "")
                yield group, clean, report, path.name
    suite = root / COHORTS["historical"] / "v3_baseline_all_checkpoints_table2_table3_table4.zip"
    with ZipFile(suite) as outer:
        for seed in SEEDS:
            name = next(n for n in outer.namelist() if n.startswith(f"v3_baseline_raw_seed{seed}/"))
            with ZipFile(io.BytesIO(outer.read(name))) as archive:
                config = json.loads(archive.read("result/manifest.json"))
                for case in cases:
                    if case == clean:
                        continue
                    member = f"result/{case.suite}_{seed}_{case.split}.json"
                    report = json.loads(archive.read(member))
                    if report["seed"] != seed or report["checkpoint_sha256"] != config["checkpoint_sha256"][str(seed)]:
                        raise ValueError(f"Checkpoint mismatch in {member}")
                    yield "historical", case, report, f"{suite.name}/{name}/{member}"


def author_accuracy(root):
    scores = {}
    for name in ("author_table2_results.zip", "author_table3_table4_results.zip"):
        with ZipFile(root / name) as archive:
            data = archive.read("results/per_seed_metrics.csv").decode()
            for row in csv.DictReader(io.StringIO(data)):
                scores[int(row["seed"]), row["split"]] = float(row["accuracy"])
    for seed in SEEDS:
        with ZipFile(root / f"author_gcn_hc3_to_gpt35_0pct_seed{seed}.zip") as archive:
            row = json.loads(archive.read("gpt35_0pct_report.json"))
            scores[seed, "gpt3.5_mixed_test_split"] = row["accuracy"]
    return scores


def accuracy_stats(values):
    return {"mean": float(np.mean(values)), "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "min": float(min(values)), "max": float(max(values))}


def paired_interval(differences, labels):
    """Bootstrap documents within class. Seeds share the same resampled documents."""
    rng = np.random.default_rng(210)
    delta = np.mean(differences, axis=0)
    totals = np.zeros(2000)
    for label in (0, 1):
        rows = delta[labels == label]
        totals += rng.choice(rows, size=(len(totals), len(rows)), replace=True).sum(axis=1)
    return {"mean_pp": float(delta.mean() * 100),
            "document_bootstrap_95pct_pp": np.percentile(totals * (100 / len(labels)), [2.5, 97.5]).tolist()}


def suite_averages(records):
    """Average conditions inside each seed first. Omit incomplete suites or methods."""
    result = []
    for suite in ("table2", "table3", "table4"):
        expected = {c.condition for c in benchmark_cases(suite) if suite != "table4" or c.noise_percent > 0}
        rows = [r for r in records if r["cohort"] == "historical" and r["suite"] == suite
                and r["condition"] in expected]
        per_seed = {seed: [r for r in rows if r["seed"] == seed] for seed in SEEDS}
        if any({r["condition"] for r in group} != expected or len(group) != len(expected)
               for group in per_seed.values()):
            raise ValueError(f"Incomplete historical RAW suite: {suite}")
        methods = set.intersection(*(set(r["metrics"]) for r in rows))
        scores = {m: [np.mean([r["metrics"][m]["accuracy"] for r in group]) for group in per_seed.values()]
                  for m in sorted(methods)}
        scores["author_gcn"] = [np.mean([r["author_gcn_accuracy"] for r in group]) for group in per_seed.values()]
        result.append({"suite": suite, "conditions": sorted(expected), "seeds": list(per_seed),
                       "accuracy": {m: accuracy_stats(v) for m, v in scores.items()}})
    return result


def evaluate_holdout(root, graph):
    """Confirmation on the separate target validation split with frozen source thresholds."""
    import torch
    from model.QuangV3 import predict
    from train_v3 import load_graph
    from calibrate_v3 import load_v3_checkpoint

    source = REPO / TRAINING_DOMAINS["gpt35"]["val_source"]
    test_source = REPO / benchmark_cases("table3")[0].source
    test_texts = {" ".join(row["text"].split()) for row in read_source(test_source)}
    mask = np.array([" ".join(row["text"].split()) not in test_texts for row in read_source(source)])
    rows, dataset = [], None
    for path in sorted((root / COHORTS["non_deterministic"]).glob("v3_dual*hc3_to_gpt35*batch1.zip")):
        model, ckpt, key = load_v3_checkpoint(path)
        fitted = json.loads((root / "source_calibration/parameters" / f"{key}.json").read_bytes())
        if fitted["checkpoint_sha256"] != key:
            raise ValueError("Wrong holdout calibration")
        if dataset is None:
            dataset = load_graph(graph, source, ckpt["R"], ckpt["struct_mode"])
        if (ckpt["R"], ckpt["struct_mode"]) != (dataset.R, dataset.mode):
            raise ValueError("Incompatible checkpoint schema")
        p = predict(model, dataset, torch.device("cpu"), ckpt["gm"], ckpt["gs"], bs=2)
        truth = dataset.y.numpy()
        rows.append({"seed": ckpt["args"]["seeds"][0], "v3_checkpoint": key, "samples": int(mask.sum()),
                     "raw": score_probabilities(truth[mask], p[mask], 0.5),
                     "human95": score_probabilities(truth[mask], p[mask], fitted["human95_threshold"]),
                     "source_human95_threshold": fitted["human95_threshold"],
                     "p_human": p[mask].tolist()})
        print(f"Holdout seed {rows[-1]['seed']}: {rows[-1]['human95']['accuracy']:.3f}", flush=True)
    if not rows:
        raise FileNotFoundError("No non-deterministic batch-1 checkpoints found")
    return {"source": str(source.relative_to(REPO)), "source_sha256": sha256(source),
            "excluded_test_duplicates": np.flatnonzero(~mask).tolist(), "y_true": truth[mask].tolist(),
            "graph_sha256": sha256(graph), "per_run": rows}


def calibrate_author(root, graph):
    """Give the retrained author GCN the same validation split and decision rule as V3."""
    import torch
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from model.GCN2 import GCN2

    source = REPO / TRAINING_DOMAINS["hc3"]["val_source"]
    truth = labels_from(read_source(source))
    with graph.open("rb") as stream:
        cache = pickle.load(stream)
    if not np.array_equal(truth, cache["y"]) or len(cache["all_edge_index"]) != len(truth) or len(cache["all_token_embeddings"]) != len(truth):
        raise ValueError("Author GCN calibration cache is not aligned with HC3 validation")
    examples = [Data(x=x.float(), edge_index=edge.long()) for x, edge in zip(cache["all_token_embeddings"], cache["all_edge_index"])]
    loader = DataLoader(examples, batch_size=8, shuffle=False)
    folder = root.parent / "author"
    tests = []
    for filename in ("author_table2_results.zip", "author_table3_table4_results.zip"):
        with ZipFile(folder / filename) as archive:
            tests.extend(json.loads(archive.read("results/per_seed_reports.json")))
    label_cache = {c.split: labels_from(read_source(REPO / c.source)) for c in benchmark_cases("full")}
    rows = []
    for seed in SEEDS:
        with ZipFile(folder / f"author_gcn_hc3_to_gpt35_0pct_seed{seed}.zip") as archive:
            members = [name for name in archive.namelist() if name.endswith(".pth")]
            if len(members) != 1:
                raise ValueError(f"Expected the single GCN checkpoint of training run {seed}")
            weights = archive.read(members[0])
            clean = json.loads(archive.read("gpt35_0pct_report.json"))
        digest = hashlib.sha256(weights).hexdigest()
        # The earliest clean report (2026) ships alongside its weights but lacks
        # the digest field. The independent evaluation archives below carry it.
        clean.setdefault("checkpoint_sha256", digest)
        model = GCN2(768, 256, 64).eval()
        model.load_state_dict(torch.load(io.BytesIO(weights), map_location="cpu", weights_only=True))
        with torch.inference_mode():
            scores = torch.cat([model(batch).flatten() for batch in loader]).numpy()
        threshold = human95_threshold(truth, scores)
        clean.update(split="gpt3.5_mixed_test_split", suite="table3", condition="Original")
        for report in [clean, *(r for r in tests if r["seed"] == seed)]:
            if report["checkpoint_sha256"] != digest or not np.array_equal(report["y_true"], label_cache[report["split"]]):
                raise ValueError(f"Unpaired author report for seed {seed}")
            p = np.asarray(report["p_human"])
            rows.append({"seed": seed, "split": report["split"], "suite": report["suite"], "condition": report["condition"],
                         "checkpoint_sha256": digest, "source_human95_threshold": threshold, "raw_accuracy": report["accuracy"],
                         "human95": score_probabilities(label_cache[report["split"]], p, threshold)})
    return {"val_source_sha256": sha256(source), "val_graph_sha256": sha256(graph), "per_run": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO / "result/Result_v3")
    parser.add_argument("--output", type=Path, required=True)
    audit = parser.add_mutually_exclusive_group()
    audit.add_argument("--holdout-graph", type=Path, help="Evaluate frozen thresholds on separate GPT-3.5 validation")
    audit.add_argument("--author-val-graph", type=Path, help="Apply the same HC3 source rule to the author's retrained GCN")
    args = parser.parse_args()
    if args.holdout_graph:
        write_json(args.output, evaluate_holdout(args.root, args.holdout_graph))
        return
    if args.author_val_graph:
        write_json(args.output, calibrate_author(args.root, args.author_val_graph))
        return
    source_cache, a1_cache, records = {}, {}, []
    clean_differences = defaultdict(list)
    author = author_accuracy(args.root.parent / "author")
    control_file = args.root / "source_calibration/author_human95_control.json"
    control = {}
    if control_file.is_file():
        control = {r["checkpoint_sha256"]: r["source_human95_threshold"]
                   for r in json.loads(control_file.read_bytes())["per_run"]}
    a1_digest = sha256(REPO / "model/hc3_a1_only_rgcn_model_2026.pth")
    for cohort, case, report, origin in reports(args.root):
        source = REPO / case.source
        if source not in source_cache:
            rows = read_source(source)
            source_cache[source] = labels_from(rows), [max(1, len(r["text"].split())) for r in rows]
            path = args.root.parent / "a1_threshold" / f"scores_test_{case.split}.txt"
            if not path.exists():
                path = args.root / "v3_a1/a1_scores" / f"scores_{case.split}.json"
            if path.exists():
                a1_cache[source] = json.loads(path.read_bytes()), path.name
        truth, lengths = source_cache[source]
        if (report["split"] != source.stem or not np.array_equal(report["y_true"], truth)
                or report["samples"] != len(truth) or report["threshold"] != 0.5):
            raise ValueError(f"Cannot pair {origin} with {source}")
        p = np.asarray(report["p_human"])
        key = report["checkpoint_sha256"]
        fitted = json.loads((args.root / "source_calibration/parameters" / f"{key}.json").read_bytes())
        if fitted["checkpoint_sha256"] != key or fitted["seed"] != report["seed"]:
            raise ValueError("Source calibration belongs to another checkpoint")
        floor = fitted["human95_threshold"]
        variants = {"raw": (p, 0.5), "human95": (p, floor)}
        if "lengths" in fitted:
            variants["length"] = calibrated_probabilities(p, lengths, fitted), 0.5
        a1_name = None
        if source in a1_cache:
            a1, a1_name = a1_cache[source]
            if a1["checkpoint_sha256"] != a1_digest:
                raise ValueError("Mismatched A1 checkpoint")
            variants["fusion"] = paired_scores(source, truth, report, a1), 0.5
            variants["a1"] = expit([r["logit_human"] for r in a1["records"]]), 0.5
        row = {"cohort": cohort, "seed": report["seed"], "suite": case.suite, "condition": case.condition,
               "split": case.split, "v3_checkpoint": key, "archive": origin, "samples": len(truth),
               "author_gcn_accuracy": author[report["seed"], case.split], "source_human95_threshold": floor,
               "a1_scores": a1_name}
        row["metrics"] = {name: score_probabilities(truth, values, threshold)
                          for name, (values, threshold) in variants.items()}
        if cohort == "non_deterministic" and case.suite == "table3" and case.noise_percent == 0:
            seed = report["seed"]
            filename = args.root.parent / "author" / f"author_gcn_hc3_to_gpt35_0pct_seed{seed}.zip"
            with ZipFile(filename) as archive:
                reference = json.loads(archive.read("gpt35_0pct_report.json"))
            if reference["seed"] != seed or not np.array_equal(reference["y_true"], truth):
                raise ValueError("Author and V3 test records are not paired")
            correct = ((p >= floor) == truth).astype(int)
            clean_differences["human95_minus_raw"].append(correct - ((p >= .5) == truth))
            ref_p = np.asarray(reference["p_human"])
            clean_differences["human95_minus_author_gcn"].append(correct - ((ref_p >= .5) == truth))
            if control:
                ref_floor = control[reference["checkpoint_sha256"]]
                clean_differences["human95_minus_author_human95"].append(correct - ((ref_p >= ref_floor) == truth))
        if not np.isclose(row["metrics"]["raw"]["accuracy"], report["accuracy"]):
            raise ValueError("Replayed accuracy differs from the archive")
        records.append(row)
    groups = defaultdict(list)
    for row in records:
        groups[(row["cohort"], row["suite"], row["condition"])].append(row)
    summary = []
    for (cohort, suite, condition), rows in groups.items():
        seeds = [r["seed"] for r in rows]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"Duplicate seeds in {cohort}")
        methods = sorted(set.intersection(*(set(r["metrics"]) for r in rows)))
        values = {name: [r["metrics"][name]["accuracy"] for r in rows] for name in methods}
        values["author_gcn"] = [r["author_gcn_accuracy"] for r in rows]
        summary.append({"cohort": cohort, "suite": suite, "condition": condition, "seeds": seeds,
                        "accuracy": {name: accuracy_stats(v) for name, v in values.items()}})
    clean = next(c for c in benchmark_cases("table3") if c.noise_percent == 0)
    labels, _ = source_cache[REPO / clean.source]
    intervals = {key: paired_interval(deltas, labels) for key, deltas in clean_differences.items()}
    write_json(args.output, {"a1_checkpoint_sha256": a1_digest, "per_run": records, "summary": summary,
                             "non_deterministic_clean_paired_intervals": intervals,
                             "historical_suite_averages": suite_averages(records)})


if __name__ == "__main__":
    main()
