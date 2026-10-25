"""Reproduce the PRDetect paper's Table 3/4 evaluation suites on Kaggle.

Run from the repository root after attaching this repository (and one dataset
containing the new RGCN checkpoint) to a Kaggle notebook:

    !python kaggle_paper_tables_pipeline.py \
        --checkpoint /kaggle/input/my-weights/my_model.pth \
        --model-name new_model --seed 2026 --suite all

The script deliberately evaluates a fixed manifest rather than every JSON file
in ``data/``.  It makes the paper comparison auditable and prevents accidental
evaluation of train/validation splits.  Outputs are written to
``result/paper_tables/<model-name>/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class EvaluationCase:
    """One paper-table column and its source JSON relative to the repo root."""

    table: str
    benchmark: str
    condition: str
    source: str


# Table 3: clean and adjective-synonym replacement conditions reported for HC3
# and GPT-3.5-mixed.  The manuscript reports 0/5/10/30%; the files below are
# the repository's corresponding processed test splits (not train or val).
TABLE_3 = (
    EvaluationCase("Table 3", "HC3", "0% (clean)",
                   "data/original_text/benchmarks/hc3/hc3_test.json"),
    EvaluationCase("Table 3", "HC3", "5% synonym", "data/perturbed_text/hc3/lexical_synonym/hc3_test_adj_5_synonym_replace.json"),
    EvaluationCase("Table 3", "HC3", "10% synonym", "data/perturbed_text/hc3/lexical_synonym/hc3_test_adj_10_synonym_replace.json"),
    EvaluationCase("Table 3", "HC3", "30% synonym", "data/perturbed_text/hc3/lexical_synonym/hc3_test_adj_30_synonym_replace.json"),
    EvaluationCase("Table 3", "GPT-3.5-mixed", "0% (clean)",
                   "data/original_text/benchmarks/gpt35_mixed/gpt3.5_mixed_test_split.json"),
    EvaluationCase("Table 3", "GPT-3.5-mixed", "5% synonym", "data/perturbed_text/gpt35_mixed/lexical_synonym/gpt3.5_mixed_test_adj_5_synonym_replace.json"),
    EvaluationCase("Table 3", "GPT-3.5-mixed", "10% synonym", "data/perturbed_text/gpt35_mixed/lexical_synonym/gpt3.5_mixed_test_adj_10_synonym_replace.json"),
    EvaluationCase("Table 3", "GPT-3.5-mixed", "30% synonym", "data/perturbed_text/gpt35_mixed/lexical_synonym/gpt3.5_mixed_test_adj_30_synonym_replace.json"),
)

# Table 4: the paper's concrete perturbation tests.  Keep the perturbation
# family and rate in the output instead of averaging unlike conditions.
TABLE_4 = (
    EvaluationCase("Table 4", "HC3", "insert 15%", "data/perturbed_text/hc3/structural_noise/hc3_test_insert_15.json"),
    EvaluationCase("Table 4", "HC3", "insert 25%", "data/perturbed_text/hc3/structural_noise/hc3_test_insert_25.json"),
    EvaluationCase("Table 4", "HC3", "delete 15%", "data/perturbed_text/hc3/structural_noise/hc3_test_delete_15.json"),
    EvaluationCase("Table 4", "HC3", "delete 25%", "data/perturbed_text/hc3/structural_noise/hc3_test_delete_25.json"),
    EvaluationCase("Table 4", "HC3", "replace 15%", "data/perturbed_text/hc3/structural_noise/hc3_test_replace_15.json"),
    EvaluationCase("Table 4", "HC3", "replace 25%", "data/perturbed_text/hc3/structural_noise/hc3_test_replace_25.json"),
    EvaluationCase("Table 4", "HC3", "repeat 15%", "data/perturbed_text/hc3/structural_noise/hc3_test_repeat_15.json"),
    EvaluationCase("Table 4", "HC3", "back-translation", "data/perturbed_text/hc3/translation/hc3_test_translate.json"),
    EvaluationCase("Table 4", "HC3", "LLaMA-3 rewrite", "data/perturbed_text/hc3/rewrite/hc3_test_llama3_8b_instruct_rewrite.json"),
)

METRICS = re.compile(
    r"Test Loss:\s*([0-9.]+).*?Test Acc:\s*([0-9.]+).*?ROC-AUC:\s*([0-9.]+).*?F1-score:\s*([0-9.]+)",
    re.DOTALL,
)
SAMPLES = re.compile(r"Tổng mẫu hợp lệ:\s*(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="Path to the new compatible RGCN .pth checkpoint.")
    parser.add_argument("--model-name", required=True,
                        help="Identifier used in result files; only letters, numbers, _ and -.")
    parser.add_argument("--seed", default="2026")
    parser.add_argument("--suite", choices=("table3", "table4", "all"), default="all")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-bases", type=int, default=30)
    parser.add_argument("--keep-graphs", action="store_true",
                        help="Keep graph_data/*.pkl. Default removes each temporary graph.")
    return parser.parse_args()


def select_cases(suite: str) -> tuple[EvaluationCase, ...]:
    if suite == "table3":
        return TABLE_3
    if suite == "table4":
        return TABLE_4
    return TABLE_3 + TABLE_4


def run(command: list[str], cwd: Path) -> str:
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    output = f"{completed.stdout}\n{completed.stderr}"
    if completed.returncode:
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(command)}\n{output[-4000:]}")
    return output


def graph_name(source: Path) -> str:
    return source.stem


def evaluate_case(repo: Path, case: EvaluationCase, args: argparse.Namespace, checkpoint_name: str) -> dict[str, object]:
    source = repo / case.source
    if not source.is_file():
        raise FileNotFoundError(f"Missing required paper test split: {source}")
    name = graph_name(source)
    graph = repo / "graph_data" / f"{name}.pkl"
    started = time.perf_counter()
    build_log = run([sys.executable, "building_graph.py", "--file", str(source)], repo)
    test_log = run([
        sys.executable, "test_rgcn.py", "--file", name, "--dataset", args.model_name,
        "--seed", args.seed, "--batch_size", str(args.batch_size),
        "--num_layers", str(args.num_layers), "--num_bases", str(args.num_bases),
    ], repo)
    if not args.keep_graphs and graph.exists():
        graph.unlink()
    found = METRICS.search(test_log)
    if not found:
        raise RuntimeError(f"Could not parse metrics for {name}. Last output:\n{test_log[-4000:]}")
    loss, accuracy, auc, f1 = (float(value) for value in found.groups())
    sample_match = SAMPLES.search(build_log)
    return {
        "table": case.table, "benchmark": case.benchmark, "condition": case.condition,
        "source": case.source, "graph": name, "checkpoint": checkpoint_name,
        "samples": int(sample_match.group(1)) if sample_match else "unknown",
        "loss": loss, "accuracy": accuracy, "roc_auc": auc, "f1": f1,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }


def write_outputs(rows: Iterable[dict[str, object]], output_dir: Path, args: argparse.Namespace) -> None:
    rows = list(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({"created_utc": datetime.now(timezone.utc).isoformat(), "arguments": vars(args), "results": rows}, handle, indent=2, default=str)
    lines = ["# PRDetect paper-table evaluation", "", "| Table | Benchmark | Condition | n | Accuracy | ROC-AUC | F1 |", "|---|---|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['table']} | {row['benchmark']} | {row['condition']} | {row['samples']} | {row['accuracy']:.4f} | {row['roc_auc']:.4f} | {row['f1']:.4f} |")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.model_name):
        raise ValueError("--model-name may contain only letters, numbers, _ and -")
    repo = Path.cwd().resolve()
    if not (repo / "building_graph.py").is_file():
        raise FileNotFoundError("Run this script from the PRDetect repository root.")
    checkpoint = args.checkpoint.resolve()
    if checkpoint.suffix != ".pth" or not checkpoint.is_file():
        raise FileNotFoundError(f"--checkpoint must be an existing .pth file: {checkpoint}")
    model_dir = repo / "model"
    model_dir.mkdir(exist_ok=True)
    target_checkpoint = model_dir / f"{args.model_name}_rgcn_model_{args.seed}.pth"
    shutil.copy2(checkpoint, target_checkpoint)
    output_dir = repo / "result" / "paper_tables" / args.model_name
    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for index, case in enumerate(select_cases(args.suite), start=1):
        print(f"[{index}] {case.table}: {case.benchmark} / {case.condition}", flush=True)
        try:
            rows.append(evaluate_case(repo, case, args, target_checkpoint.name))
            write_outputs(rows, output_dir, args)
        except Exception as error:
            failures.append({"table": case.table, "condition": case.condition, "error": str(error)})
            print(f"FAILED: {error}", file=sys.stderr, flush=True)
    (output_dir / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    if failures:
        raise SystemExit(f"{len(failures)} case(s) failed; see {output_dir / 'failures.json'}")
    print(f"Done. Results: {output_dir}")


if __name__ == "__main__":
    main()
