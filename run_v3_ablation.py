"""Kaggle workflow: build two source graphs, train, release caches, benchmark.

Every GPU-heavy operation is isolated in a child process. Only verified caches
created by this workflow are removed; checkpoints and source data are retained.
"""

import argparse
import csv
from pathlib import Path
import shutil
import subprocess
import sys
from zipfile import ZIP_DEFLATED, ZipFile

from evaluate_v3_quang import REPO, environment, positive_int, read_json, sha256, write_json


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=REPO / "result/v3_cmixstyle")
    parser.add_argument("--variants", nargs="+", choices=("baseline", "classswap", "cmixstyle"),
                        default=["baseline", "cmixstyle"])
    parser.add_argument("--experiment-name", default="v3_cmixstyle",
                        help="Output prefix; for example v3_classswap")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2025, 2026])
    parser.add_argument("--stage", choices=("all", "train", "evaluate"), default="all")
    parser.add_argument("--suite", choices=("table2", "table3", "table4", "all", "full"), default="full")
    parser.add_argument("--batch-size", type=positive_int, default=32)
    parser.add_argument("--micro-batch-size", type=positive_int, default=8)
    parser.add_argument("--eval-batch-size", type=positive_int, default=4)
    parser.add_argument("--val-batch-size", type=positive_int, default=64)
    parser.add_argument("--epochs", type=positive_int, default=10)
    parser.add_argument("--patience", type=positive_int, default=10)
    parser.add_argument("--mixstyle-p", type=float, default=0.5)
    parser.add_argument("--mixstyle-alpha", type=float, default=0.1)
    parser.add_argument("--mixstyle-strength", type=float, default=0.3)
    parser.add_argument("--struct-mode", choices=("low", "high"), default="low")
    parser.add_argument("--max-graphs", type=positive_int, default=4)
    parser.add_argument("--amp", type=int, choices=(0, 1), default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser


def child(script, *arguments, cwd=None):
    subprocess.run([sys.executable, str(REPO / script), *map(str, arguments)], cwd=cwd, check=True)


def graph_files(work):
    roots = {work, REPO / "graph_data", REPO / "TacGia/PRDetect/graph_data"}
    return {p.resolve() for root in roots if root.exists() for p in root.rglob("*.pkl")}


def training_cache(source, work, max_graphs):
    folder = work / "training"
    folder.mkdir(parents=True, exist_ok=True)
    cache = folder / "graph_data" / f"{source.stem}.pkl"
    meta = cache.with_suffix(".meta.json")
    provenance = {"owner": "run_v3_ablation", "source": str(source), "source_sha256": sha256(source),
                  "builder_sha256": {name: sha256(REPO / name) for name in
                                     ("building_graph.py", "dep_vocab.py", "dep_vocab.json")},
                  "environment": environment()}
    if cache.exists():
        if not meta.is_file() or read_json(meta) != provenance:
            raise ValueError(f"Unverified/stale cache: {cache}. Use a new --work-dir.")
        return cache
    existing = graph_files(work)
    if len(existing) >= max_graphs:
        raise ValueError(f"Graph limit {max_graphs} reached: {sorted(map(str, existing))}")
    free_gib = shutil.disk_usage(work).free / 2**30
    if free_gib < 2:
        raise OSError(f"Only {free_gib:.2f} GiB free; release unused data before building")
    print(f"Build {source.stem}; {len(existing)} caches, free {free_gib:.2f} GiB", flush=True)
    child("building_graph.py", "--file", source, cwd=folder)
    if not cache.is_file():
        raise FileNotFoundError(cache)
    write_json(meta, provenance)
    print(f"Cache size: {cache.stat().st_size/2**30:.2f} GiB", flush=True)
    return cache


def release_training_cache(cache, work):
    expected_parent = (work / "training/graph_data").resolve()
    meta = cache.with_suffix(".meta.json")
    if (cache.resolve().parent != expected_parent or cache.stem not in ("hc3_train", "hc3_val")
            or not meta.is_file() or read_json(meta).get("owner") != "run_v3_ablation"):
        raise ValueError(f"Refusing to remove an unowned cache: {cache}")
    cache.unlink()
    meta.unlink()
    print(f"Removed verified, rebuildable training cache: {cache.name}", flush=True)


def comparison(work, variants, seeds):
    rows = []
    for variant in variants:
        path = work / "eval" / variant / "result/summary_threshold_0.5.csv"
        with path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                if int(row["num_seeds"]) != len(seeds):
                    raise ValueError("Summary seed count differs from the requested run")
                rows.append({"variant": variant, **row})
    baselines = {(r["suite"], r["condition"]): float(r["accuracy_mean"])
                 for r in rows if r["variant"] == "baseline"}
    for row in rows:
        base = baselines.get((row["suite"], row["condition"]))
        row["delta_accuracy_vs_baseline_pp"] = (100 * (float(row["accuracy_mean"]) - base)
                                                 if base is not None else "")
    output = work / "comparison.csv"
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.max_graphs > 4:
        raise ValueError("This Kaggle workflow enforces at most four graph files")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.variants)) != len(args.variants):
        raise ValueError("Duplicate seeds/variants")
    # Fail before building expensive graphs if a training setting is invalid.
    from train_v3 import build_parser as training_parser, validate_args

    train_arguments = ["--variants", *args.variants, "--seeds", *map(str, args.seeds)]
    for name in ("batch_size", "micro_batch_size", "val_batch_size", "epochs", "patience",
                 "mixstyle_p", "mixstyle_alpha", "mixstyle_strength", "struct_mode", "amp", "device"):
        train_arguments += ["--" + name.replace("_", "-"), str(getattr(args, name))]
    validate_args(training_parser().parse_args(train_arguments))
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = work / "model"
    if args.stage in ("all", "train"):
        sources = [REPO / f"data/original_text/benchmarks/hc3/{split}.json"
                   for split in ("hc3_train", "hc3_val")]
        caches = [training_cache(source, work, args.max_graphs) for source in sources]
        child("train_v3.py", *train_arguments, "--graph-dir", work / "training/graph_data",
              "--output-dir", checkpoint_dir, "--skip-completed")
        for cache in caches:
            release_training_cache(cache, work)
    if args.stage in ("all", "evaluate"):
        for variant in args.variants:
            child("evaluate_v3_quang.py", "run", "--variant", variant,
                  "--checkpoint-dir", checkpoint_dir, "--seeds", *args.seeds,
                  "--suite", args.suite, "--batch-size", args.eval_batch_size,
                  "--struct-mode", args.struct_mode, "--device", args.device,
                  "--max-graphs", args.max_graphs, "--graph-budget-root", work,
                  "--work-dir", work / "eval" / variant)
        summary = comparison(work, args.variants, args.seeds)
        archive = work / f"{args.experiment_name}_ablation_results.zip"
        with ZipFile(archive, "w", ZIP_DEFLATED) as output:
            output.write(summary, summary.name)
            for variant in args.variants:
                result = work / "eval" / variant / f"v3_{variant}_{args.suite}_results.zip"
                output.write(result, str(result.relative_to(work)))
                for seed in args.seeds:
                    checkpoint = checkpoint_dir / f"v3_{variant}_seed{seed}.pt"
                    output.write(checkpoint, str(checkpoint.relative_to(work)))
                    metadata = checkpoint.with_suffix(".train.json")
                    output.write(metadata, str(metadata.relative_to(work)))
        print(f"Comparison: {summary}\nArchive (includes trained checkpoints): {archive}", flush=True)


if __name__ == "__main__":
    main()
