"""Test Quang's V3 or controlled baseline/CMixStyle checkpoints; never train.

The orchestrator uses only the standard library. Graph building and inference
run in separate processes so the frozen encoder and multiple classifiers never
remain on the GPU together. Only trusted, locally generated pickle graphs are
supported. Checkpoint .zip files are loaded as PyTorch files, not extracted.
"""

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import subprocess
import sys
from zipfile import ZIP_DEFLATED, ZipFile


REPO = Path(__file__).resolve().parent
MODEL_SOURCE = REPO / "model/QuangV3.py"
DEFAULT_SEEDS = [2021, 2022, 2023, 2024, 2025]
METRICS = ("accuracy", "auc", "f1", "recall_ai", "recall_human")

# A training domain fixes both the source train/validation split and which
# direction Table 3 represents.  Keeping this mapping explicit prevents a
# GPT-3.5 checkpoint from accidentally being evaluated as HC3 -> GPT-3.5.
TRAINING_DOMAINS = {
    "hc3": {
        "train_file": "hc3_train",
        "val_file": "hc3_val",
        "train_source": "data/original_text/benchmarks/hc3/hc3_train.json",
        "val_source": "data/original_text/benchmarks/hc3/hc3_val.json",
    },
    "gpt35": {
        "train_file": "gpt3.5_mixed_train_split",
        "val_file": "gpt3.5_mixed_val_split",
        "train_source": "data/original_text/benchmarks/gpt35_mixed/gpt3.5_mixed_train_split.json",
        "val_source": "data/original_text/benchmarks/gpt35_mixed/gpt3.5_mixed_val_split.json",
    },
}


@dataclass(frozen=True)
class Case:
    suite: str
    condition: str
    noise_percent: int
    source: str

    @property
    def split(self):
        return Path(self.source).stem


def benchmark_cases(suite="all", train_domain="hc3"):
    if train_domain not in TRAINING_DOMAINS:
        raise ValueError(f"Unknown training domain: {train_domain}")
    if train_domain == "hc3":
        in_domain = ("data/original_text/benchmarks/hc3/hc3_test.json",
                     "data/perturbed_text/hc3/lexical_synonym/hc3_test_adj_{rate}_synonym_replace.json")
        cross_domain = ("data/original_text/benchmarks/gpt35_mixed/gpt3.5_mixed_test_split.json",
                        "data/perturbed_text/gpt35_mixed/lexical_synonym/"
                        "gpt3.5_mixed_test_adj_{rate}_synonym_replace.json")
    else:
        in_domain = ("data/original_text/benchmarks/gpt35_mixed/gpt3.5_mixed_test_split.json",
                     "data/perturbed_text/gpt35_mixed/lexical_synonym/"
                     "gpt3.5_mixed_test_adj_{rate}_synonym_replace.json")
        cross_domain = ("data/original_text/benchmarks/hc3/hc3_test.json",
                        "data/perturbed_text/hc3/lexical_synonym/hc3_test_adj_{rate}_synonym_replace.json")
    table2 = [Case("table2", "Original", 0, in_domain[0])] + [Case(
        "table2", f"Synonym {rate}%", rate, in_domain[1].format(rate=rate),
    ) for rate in (5, 10, 20, 30)]
    table3 = [Case("table3", "Original", 0, cross_domain[0])] + [Case(
        "table3", f"Synonym {rate}%", rate, cross_domain[1].format(rate=rate),
    ) for rate in (5, 10, 20, 30)]
    table4 = [Case(
        "table4", "Original", 0,
        "data/original_text/benchmarks/hc3/hc3_test.json",
    )] + [Case(
        "table4", f"{kind.capitalize()} 25%", 25,
        f"data/perturbed_text/hc3/structural_noise/hc3_test_{kind}_25.json",
    ) for kind in ("insert", "repeat", "replace", "delete")]
    if suite == "table2":
        return table2
    if suite == "table2_table3":
        return table2 + table3
    if suite == "full":
        return table2 + table3 + table4
    if suite == "table3":
        return table3
    if suite == "table4":
        return table4
    if suite == "all":
        return table3 + table4
    raise ValueError(f"Unknown suite: {suite}")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # A failed write must not leave a report that a resumed run can reuse.
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def environment():
    result = {"python": sys.version.split()[0]}
    for name in ("torch", "numpy", "spacy", "en-core-web-sm", "transformers", "scikit-learn"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def find_checkpoints(root, seeds, variant="quang"):
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {root}")
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Provide at least one seed, without duplicates")
    found = {}
    for seed in seeds:
        if variant not in ("quang", "baseline", "cmixstyle"):
            raise ValueError(f"Unknown V3 variant: {variant}")
        stem = (f"p3fin_V1_swap_s{seed}_seed{seed}" if variant == "quang"
                else f"v3_{variant}_seed{seed}")
        # Kaggle preserves some uploaded PyTorch checkpoints without a file
        # extension (for example ``p3fin_V1_swap_s2021_seed2021``).  PyTorch
        # identifies checkpoints from their contents, not their suffix.
        paths = sorted(p for p in root.rglob(stem + "*")
                       if p.is_file() and p.name in (stem, stem + ".pt", stem + ".pth", stem + ".zip"))
        if not paths:
            raise FileNotFoundError(f"No {stem} (or .pt/.pth/.zip) under {root}")
        if len({sha256(path) for path in paths}) != 1:
            raise ValueError(f"Conflicting checkpoint copies for seed {seed}: {paths}")
        found[str(seed)] = str(paths[0])
    return found


def make_config(args):
    variant = args.variant
    if variant == "quang" and args.train_domain != "hc3":
        raise ValueError("Legacy Quang checkpoints only support --train-domain hc3")
    paths = find_checkpoints(args.checkpoint_dir, args.seeds, variant)
    files = [MODEL_SOURCE, REPO / "building_graph.py", REPO / "dep_vocab.py",
             REPO / "dep_vocab.json"]
    if variant != "quang":
        files += [REPO / "model/V3CMixStyle.py", REPO / "model/v3_mixstyle.py"]
    return {
        "model": "Quang_v3_V1_swap" if variant == "quang" else f"v3_{variant}",
        "variant": variant, "train_domain": args.train_domain, "seeds": args.seeds,
        "checkpoint_paths": paths,
        "checkpoint_sha256": {s: sha256(p) for s, p in paths.items()},
        "code_sha256": {str(p.relative_to(REPO)): sha256(p) for p in files if p.is_file()},
        "batch_size": args.batch_size, "device": args.device,
        "struct_mode": args.struct_mode,
        "struct_mode_note": ("Legacy checkpoint does not store mode; fixed CLI value, not inferred from test."
                             if variant == "quang" else "Must match the explicit mode stored during training."),
        "threshold": 0.5, "ensemble": False, "environment": environment(),
        "graph_budget_root": args.graph_budget_root,
    }


def _load_model(config, seed, device):
    import torch
    from model.QuangV3 import Net

    path = Path(config["checkpoint_paths"][str(seed)])
    if sha256(path) != config["checkpoint_sha256"][str(seed)]:
        raise ValueError(f"Checkpoint changed: {path}")
    for name, expected_hash in config["code_sha256"].items():
        if sha256(REPO / name) != expected_hash:
            raise ValueError(f"Source changed: {name}; start a new run")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not {"state", "args", "R", "gm", "gs", "thr"} <= checkpoint.keys():
        raise ValueError(f"Not a Quang V3 checkpoint: {path}")
    a = checkpoint["args"]
    expected_train = TRAINING_DOMAINS[config["train_domain"]]["train_file"]
    if a["train_file"] != expected_train or a["seeds"] != [seed]:
        raise ValueError(f"Checkpoint training dataset/seed mismatch: {path}")
    variant = config.get("variant", "quang")
    extra = {}
    if variant == "quang":
        if a.get("aug") != "swap" or a.get("cons", 0) != 0:
            raise ValueError(f"Expected V1_swap without consistency: {path}")
        if checkpoint.get("method") in ("v3_baseline", "v3_cmixstyle"):
            raise ValueError("New ablation checkpoint requires its explicit --variant")
    else:
        if checkpoint.get("method") != f"v3_{variant}":
            raise ValueError(f"Checkpoint/variant mismatch: {path}")
        if checkpoint.get("training_complete") is not True:
            raise ValueError(f"Training did not complete: {path}; saved best epoch is incomplete")
        if checkpoint.get("struct_mode") != config["struct_mode"]:
            raise ValueError("--struct-mode differs from the checkpoint training mode")
        if variant == "cmixstyle":
            from model.V3CMixStyle import CMixStyleNet
            Net = CMixStyleNet
            extra = {f"mixstyle_{k}": v for k, v in checkpoint["mixstyle"].items()}
    model = Net(
        int(checkpoint["R"]), d=a["hidden"], nb=a["num_bases"], L=a["layers"],
        p=a["dropout"], p_in=a["in_dropout"], edge_drop=a["edge_drop"],
        use_struct=bool(a["struct"]), use_gfeat=bool(a["gfeat"]),
        readout=a["readout"], aug=a.get("aug", "none"),
        aug_max=a.get("aug_max", 0.35), mode=config["struct_mode"], **extra,
    )
    model.load_state_dict(checkpoint["state"], strict=True)
    return path, checkpoint, model.to(device).eval()


def inspect_checkpoints(config):
    import gc

    rows = []
    for seed in config["seeds"]:
        path, checkpoint, model = _load_model(config, seed, "cpu")
        row = {
            "seed": seed, "checkpoint": str(path),
            "checkpoint_sha256": config["checkpoint_sha256"][str(seed)],
            "R": int(checkpoint["R"]), "validation_threshold": float(checkpoint["thr"]),
            "arguments": checkpoint["args"], "struct_mode_used": config["struct_mode"],
            "parameters": sum(p.numel() for p in model.parameters()),
            "method": checkpoint.get("method", "Quang_v3_V1_swap"),
            "mixstyle": checkpoint.get("mixstyle"),
            "best_epoch": checkpoint.get("best_epoch"),
        }
        rows.append(row)
        print(f"Seed {seed}: strict load OK | R={row['R']} | "
              f"val threshold={row['validation_threshold']:.3f}", flush=True)
        del checkpoint, model
        gc.collect()
    return {"checkpoints": rows, "environment": config["environment"],
            "struct_mode_note": config["struct_mode_note"]}


def score_probabilities(truth, probabilities, threshold):
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    pred = (probabilities >= threshold).astype(int)
    matrix = [[int(((truth == i) & (pred == j)).sum()) for j in (0, 1)] for i in (0, 1)]
    return {
        "accuracy": float(accuracy_score(truth, pred)),
        "auc": float(roc_auc_score(truth, probabilities)),
        "f1": float(f1_score(truth, pred, zero_division=0)),
        "recall_ai": matrix[0][0] / sum(matrix[0]),
        "recall_human": matrix[1][1] / sum(matrix[1]),
        "predicted_human": int(np.sum(pred)),
        "confusion_matrix_rows_true_columns_pred": matrix,
    }


def infer_checkpoint(config, seed, graph_path, expected_samples, signature):
    import numpy as np
    import torch
    from model import QuangV3 as v3

    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU, or explicitly pass --device cpu")
    v3.set_seed(seed)
    path, checkpoint, model = _load_model(config, seed, device)
    relation_count = int(checkpoint["R"])
    observed = {}
    original_read = v3.read_raw

    def checked_read(path):
        emb, edges, relations, labels = original_read(path)
        if not len(emb) == len(edges) == len(relations) == len(labels) == expected_samples:
            raise ValueError("Graph/source sample count mismatch; no dropped samples allowed")
        if not np.isin(labels, [0, 1]).all():
            raise ValueError("Graph labels must be binary")
        clipped = total = 0
        for x, edge, rel in zip(emb, edges, relations):
            edge, rel = np.asarray(edge), np.asarray(rel)
            if (x.ndim != 2 or x.shape[0] == 0 or x.shape[1] != 768
                    or edge.shape != (2, rel.size) or rel.size == 0):
                raise ValueError("Invalid V1 graph tensor dimensions")
            if rel.min() < 0 or edge.min() < 0 or edge.max() >= x.shape[0]:
                raise ValueError("Invalid graph indices")
            clipped += int((rel >= relation_count).sum())
            total += int(rel.size)
        observed.update(clipped_edges=clipped, total_edges=total)
        if clipped:
            print(f"WARNING: original V3 code clamps {clipped}/{total} edges to "
                  f"R-1={relation_count - 1}; verify graph vocabulary", flush=True)
        return emb, edges, relations, labels

    v3.read_raw = checked_read
    try:
        dataset = v3.GraphSet(str(graph_path), R=relation_count,
                              mode=config["struct_mode"], store_dev="cpu")
    finally:
        v3.read_raw = original_read
    probabilities = v3.predict(
        model, dataset, device, checkpoint["gm"], checkpoint["gs"],
        bs=config["batch_size"], amp=bool(checkpoint["args"].get("amp", 0)), aug_mode=None,
    )
    truth = dataset.y.cpu().numpy().astype(int)
    if (len(truth) != expected_samples or len(probabilities) != expected_samples
            or set(truth.tolist()) != {0, 1} or not np.isfinite(probabilities).all()
            or not ((probabilities >= 0) & (probabilities <= 1)).all()):
        raise ValueError("Invalid predictions/labels")
    threshold = float(checkpoint["thr"])
    return {
        "model": config["model"], "train": "hc3", "seed": seed,
        "split": Path(graph_path).stem, "checkpoint": str(path),
        "checkpoint_sha256": config["checkpoint_sha256"][str(seed)],
        "run_signature": signature, "samples": expected_samples, "threshold": 0.5,
        "batch_size": config["batch_size"], "R": relation_count,
        "struct_mode_used": config["struct_mode"], **observed,
        "label_convention": {"0": "ai", "1": "human"},
        **score_probabilities(truth, probabilities, 0.5),
        "validation_threshold": threshold,
        "at_validation_threshold": score_probabilities(truth, probabilities, threshold),
        "y_true": truth.tolist(), "p_human": probabilities.tolist(),
    }


def case_signature(config, case, expected_samples):
    value = {"config": config, "case": asdict(case), "samples": expected_samples,
             "source_sha256": sha256(REPO / case.source)}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def cached_report_matches(path, signature, seed, checkpoint_hash):
    try:
        report = read_json(path)
        required = set(METRICS) | {"validation_threshold", "at_validation_threshold", "clipped_edges"}
        return (required <= report.keys()
                and set(METRICS) <= report["at_validation_threshold"].keys()
                and report["run_signature"] == signature and report["seed"] == seed
                and report["checkpoint_sha256"] == checkpoint_hash
                and report["threshold"] == 0.5
                and len(report["y_true"]) == len(report["p_human"]) == report["samples"])
    except (OSError, ValueError, KeyError, TypeError):
        return False


def report_path(results, case, seed):
    return results / f"{case.suite}_{seed}_{case.split}.json"


def run_child(*args, cwd=None):
    subprocess.run([sys.executable, str(Path(__file__).resolve()), *map(str, args)],
                   cwd=cwd, check=True)


def evaluate_case(case, config, config_path, work_dir, max_graphs, force=False):
    source = REPO / case.source
    with source.open(encoding="utf-8") as stream:
        expected = sum(bool(line.strip()) for line in stream)
    if expected == 0:
        raise ValueError(f"Empty source: {source}")
    signature = case_signature(config, case, expected)
    results = work_dir / "result"
    pending = []
    for seed in config["seeds"]:
        output = report_path(results, case, seed)
        if not force and output.exists():
            if not cached_report_matches(output, signature, seed,
                                         config["checkpoint_sha256"][str(seed)]):
                raise ValueError(f"Stale report: {output}. Use a new --work-dir or --force")
            print(f"Reuse report: {output.name}", flush=True)
        else:
            pending.append((seed, output))
    if not pending:
        return

    graph_dir = work_dir / "graph_data"
    graph_dir.mkdir(parents=True, exist_ok=True)
    cache = graph_dir / f"{case.split}.pkl"
    metadata = cache.with_suffix(".meta.json")
    provenance = {"source_sha256": sha256(source), "code_sha256": config["code_sha256"],
                  "environment": config["environment"]}
    if cache.exists():
        if not metadata.exists() or read_json(metadata) != provenance:
            raise ValueError(f"Unverified/stale graph: {cache}. Use a new --work-dir")
    else:
        roots = {graph_dir, REPO / "graph_data", REPO / "TacGia/PRDetect/graph_data",
                 work_dir.parent / "paper_table4_original/graph_data"}
        if config.get("graph_budget_root"):
            roots.add(Path(config["graph_budget_root"]))
        existing = {p.resolve() for root in roots if root.exists() for p in root.rglob("*.pkl")}
        if len(existing) >= max_graphs:
            raise ValueError(f"Graph limit {max_graphs} reached; clean unused caches first: "
                             + ", ".join(map(str, sorted(existing))))
        print(f"Build {case.split} ({expected} samples)", flush=True)
        subprocess.run([sys.executable, str(REPO / "building_graph.py"), "--file", str(source)],
                       cwd=work_dir, check=True)
        if not cache.is_file():
            raise FileNotFoundError(cache)
        write_json(metadata, provenance)
    print(f"{case.suite} {case.condition}: cache {cache.stat().st_size / 2**30:.2f} GiB", flush=True)
    for seed, output in pending:
        run_child("_test", "--config", config_path, "--seed", seed,
                  "--graph", cache, "--report", output,
                  "--expected-samples", expected, "--signature", signature)
    # Only this runner's verified, rebuildable cache; never user source/weights.
    cache.unlink()
    metadata.unlink()
    print(f"Removed rebuildable cache: {cache.name}", flush=True)


def summarize(cases, results, seeds):
    details, summaries = [], {False: [], True: []}
    all_reports = {}
    for case in cases:
        reports = [read_json(report_path(results, case, seed)) for seed in seeds]
        labels = reports[0]["y_true"]
        for seed, report in zip(seeds, reports):
            if report["seed"] != seed or report["y_true"] != labels:
                raise ValueError(f"Unpaired labels/seed: {case.split}")
            for use_val in (False, True):
                metrics = report["at_validation_threshold"] if use_val else report
                details.append({
                    "suite": case.suite, "condition": case.condition,
                    "noise_percent": case.noise_percent, "seed": seed,
                    "threshold_mode": "validation" if use_val else "fixed_0.5",
                    "threshold": report["validation_threshold"] if use_val else 0.5,
                    **{metric: metrics[metric] for metric in METRICS},
                    "clipped_edges": report["clipped_edges"],
                })
        all_reports[case.suite, case.split] = reports
        for use_val in (False, True):
            row = {"suite": case.suite, "condition": case.condition,
                   "noise_percent": case.noise_percent, "num_seeds": len(seeds)}
            for metric in METRICS:
                values = [(r["at_validation_threshold"] if use_val else r)[metric] for r in reports]
                row[f"{metric}_mean"] = statistics.mean(values)
                row[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else None
            summaries[use_val].append(row)
    noise_cases = [c for c in cases if c.suite == "table4" and c.noise_percent == 25]
    if len(noise_cases) == 4:
        for use_val in (False, True):
            row = {"suite": "table4", "condition": "Average (4 noise sets)",
                   "noise_percent": 25, "num_seeds": len(seeds)}
            for metric in METRICS:
                values = []
                for i in range(len(seeds)):
                    per_seed = []
                    for case in noise_cases:
                        report = all_reports[case.suite, case.split][i]
                        per_seed.append((report["at_validation_threshold"] if use_val else report)[metric])
                    values.append(statistics.mean(per_seed))
                row[f"{metric}_mean"] = statistics.mean(values)
                row[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else None
            summaries[use_val].append(row)
    for name, rows in (("per_seed.csv", details),
                       ("summary_threshold_0.5.csv", summaries[False]),
                       ("summary_validation_threshold.csv", summaries[True])):
        with (results / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for row in summaries[False]:
        spread = f" +/- {100 * row['accuracy_std']:.2f}" if row["accuracy_std"] is not None else ""
        print(f"{row['suite']:6s} | {row['condition']:25s} | "
              f"Acc@0.5 {100 * row['accuracy_mean']:.2f}{spread}%", flush=True)


def export_archive(results, archive, cases, seeds, extra_cases=()):
    """Do not export stale reports from a previous seed/suite in the same folder."""
    paths = {report_path(results, case, seed) for case in [*cases, *extra_cases] for seed in seeds}
    paths.update(results / name for name in (
        "per_seed.csv", "summary_threshold_0.5.csv", "summary_validation_threshold.csv",
        "checkpoint_metadata.json", "manifest.json",
    ))
    with ZipFile(archive, "w", ZIP_DEFLATED) as output:
        for path in sorted(paths):
            output.write(path, str(Path("result") / path.relative_to(results)))
        output.write(Path(__file__), "evaluate_v3_quang.py")
        output.write(MODEL_SOURCE, "model/QuangV3.py")
        for relative in ("model/V3CMixStyle.py", "model/v3_mixstyle.py", "train_v3.py"):
            source = REPO / relative
            if source.is_file():
                output.write(source, relative)


def run_benchmarks(args):
    work = Path(args.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    results = work / "result"
    results.mkdir(exist_ok=True)
    config = make_config(args)
    cases = benchmark_cases(args.suite, args.train_domain)
    for case in cases:
        if not (REPO / case.source).is_file():
            raise FileNotFoundError(REPO / case.source)
    config_path = work / "config.json"
    write_json(config_path, config)
    print("No training. Fixed threshold 0.5; validation thresholds reported separately.", flush=True)
    print(f"Variant {args.variant}, structural mode {args.struct_mode!r}. "
          + config["struct_mode_note"], flush=True)
    run_child("inspect", "--config", config_path, "--output", results / "checkpoint_metadata.json")

    execution = list(cases)
    extra_cases = []
    if not args.skip_clean_check:
        clean = next((c for c in cases if c.suite == "table2" and c.noise_percent == 0), None)
        if clean is None:
            original = benchmark_cases("table2", args.train_domain)[0]
            clean = Case("sanity", original.condition, 0, original.source)
            extra_cases.append(clean)
        evaluate_case(clean, config, config_path, work, args.max_graphs, args.force)
        mean = statistics.mean(read_json(report_path(results, clean, s))["accuracy"] for s in args.seeds)
        print(f"{args.train_domain} in-domain Original mean: {mean:.4f}", flush=True)
        if (args.variant == "quang" and args.train_domain == "hc3" and sorted(args.seeds) == DEFAULT_SEEDS
                and abs(mean - 0.9902) > args.clean_tolerance):
            raise ValueError("Clean HC3 differs substantially from Quang's report. Check parser, "
                             "graph vocabulary and --struct-mode; do not tune on test. "
                             "--skip-clean-check is an explicit diagnostic override only.")
        execution = [c for c in cases if c != clean]
    for case in execution:
        evaluate_case(case, config, config_path, work, args.max_graphs, args.force)
    summarize(cases, results, args.seeds)
    manifest = {**config, "suite": args.suite, "cases": [asdict(c) for c in cases],
                "sanity_cases": [asdict(c) for c in extra_cases],
                "clean_check_skipped": args.skip_clean_check,
                "std_definition": "sample standard deviation across seeds (ddof=1)"}
    write_json(results / "manifest.json", manifest)
    name = ("v3_quang_table3_table4_results.zip" if args.variant == "quang" and args.suite == "all"
            else f"v3_{args.variant}_{args.suite}_results.zip")
    archive = work / name
    export_archive(results, archive, cases, args.seeds, extra_cases)
    print(f"Results: {results}\nArchive: {archive}", flush=True)


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("Must be positive")
    return value


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("inspect", "run"):
        p = sub.add_parser(command)
        p.add_argument("--variant", choices=("quang", "baseline", "cmixstyle"), default="quang")
        p.add_argument("--train-domain", choices=tuple(TRAINING_DOMAINS), default="hc3",
                       help="Training domain; determines in-domain Table 2 and Table 3 direction")
        p.add_argument("--checkpoint-dir", default=str(REPO / "model/V3_quang"))
        p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
        p.add_argument("--batch-size", type=positive_int, default=4)
        p.add_argument("--struct-mode", choices=("low", "high"), default="low")
        p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        p.add_argument("--graph-budget-root", help="Also count caches recursively under this workflow root")
        if command == "inspect":
            p.add_argument("--config")
            p.add_argument("--output")
        else:
            p.add_argument("--suite", choices=("all", "table2", "table3", "table4", "table2_table3", "full"), default="all",
                           help="table2_table3: in-domain Table 2 plus cross-domain Table 3; full also adds Table 4")
            p.add_argument("--work-dir", default=str(REPO / "result/v3_quang_eval"))
            p.add_argument("--max-graphs", type=positive_int, default=4)
            p.add_argument("--force", action="store_true", help="Rerun reports; do not reuse old predictions")
            p.add_argument("--skip-clean-check", action="store_true")
            p.add_argument("--clean-tolerance", type=float, default=0.005)
    p = sub.add_parser("_test", help="Internal isolated inference worker")
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--graph", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--expected-samples", type=positive_int, required=True)
    p.add_argument("--signature", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "run":
        if not 0 <= args.clean_tolerance < 1:
            raise ValueError("--clean-tolerance must be a probability difference in [0,1)")
        run_benchmarks(args)
    elif args.command == "inspect":
        config = read_json(args.config) if args.config else make_config(args)
        metadata = inspect_checkpoints(config)
        if args.output:
            write_json(args.output, metadata)
    else:
        config = read_json(args.config)
        report = infer_checkpoint(config, args.seed, args.graph, args.expected_samples, args.signature)
        write_json(args.report, report)
        print(f"Seed {args.seed} | {report['split']} | Acc@0.5={report['accuracy']:.4f} | "
              f"Acc@val-thr={report['at_validation_threshold']['accuracy']:.4f}", flush=True)


if __name__ == "__main__":
    main()
