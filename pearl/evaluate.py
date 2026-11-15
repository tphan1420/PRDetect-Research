"""Stream existing official Table 2/3/4 inputs; never fit on benchmark data."""
import json
import time
from pathlib import Path
import torch
from .config import config_cli, load_config
from .data import digest, file_hash, human_label, read_jsonl, write_json
from .embeddings import LegacyEncoder
from .graph import parse_graph
from .metrics import classification_metrics, distribution
from .train_gcn import load_checkpoint


def benchmark_plan(cfg):
    root = Path(cfg.get("data_root", "data"))
    source = cfg.get("dataset", "hc3")
    if source not in ("hc3", "gpt35_mixed"):
        raise ValueError("Built-in tables support hc3/gpt35_mixed; supply evaluation_entries for another layout")
    target = "gpt35_mixed" if source == "hc3" else "hc3"
    plan = []
    for table, dataset in (("table2", source), ("table3", target), ("table4", "hc3")):
        prefix = "hc3_test" if dataset == "hc3" else "gpt3.5_mixed_test"
        clean = root / "original_text" / "benchmarks" / ("hc3" if dataset == "hc3" else "gpt35_mixed") / (prefix + ("_split.json" if dataset != "hc3" else ".json"))
        plan.append(dict(table=table, dataset=dataset, condition="original", path=str(clean)))
        if table in ("table2", "table3"):
            for ratio in (5, 10, 20, 30):
                path = root / "perturbed_text" / dataset / "lexical_synonym" / f"{prefix}_adj_{ratio}_synonym_replace.json"
                plan.append(dict(table=table, dataset=dataset, condition=f"syn{ratio}", path=str(path)))
        else:
            for family in ("insert", "repeat", "replace", "delete"):
                path = root / "perturbed_text" / "hc3" / "structural_noise" / f"{prefix}_{family}_25.json"
                plan.append(dict(table=table, dataset=dataset, condition=f"{family}25", path=str(path)))
    return cfg.get("evaluation_entries", plan)


def verify_runtime(cfg, metadata, nlp, encoder):
    if digest(nlp.to_bytes().hex()) != metadata["parser_hash"]:
        raise ValueError("Inference parser does not match training parser")
    if encoder.metadata != metadata["encoder"]:
        raise ValueError("Inference RoBERTa does not match training encoder metadata")


def evaluate(cfg, tables=None, limit=None, output=None):
    import spacy
    start = time.perf_counter()
    output = Path(output or Path(cfg["gcn_output"]) / "evaluation.json")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model, metadata = load_checkpoint(Path(cfg["gcn_output"]) / "best.pt", cfg["device"])
    nlp = spacy.load(cfg["parser"])
    encoder = LegacyEncoder(cfg)
    verify_runtime(cfg, metadata, nlp, encoder)
    results = []
    reference_labels = {}
    for entry in benchmark_plan(cfg):
        if tables and entry["table"] not in tables:
            continue
        path = Path(entry["path"])
        if not path.exists():
            results.append(dict(**entry, status="missing"))
            continue
        rows = read_jsonl(path)
        labels = [human_label(r) for r in rows]
        if entry["condition"] == "original":
            reference_labels[entry["dataset"]] = labels
        elif entry["dataset"] in reference_labels and labels != reference_labels[entry["dataset"]]:
            raise ValueError(f"Benchmark row count/label ordering differs from original: {path}")
        # Same deterministic, label-balanced row indices for clean/perturbed files
        # when their label ordering agrees. No selection based on model scores.
        if limit:
            indices = []
            for cls in (0, 1):
                ids = [i for i, r in enumerate(rows) if human_label(r) == cls]
                ids.sort(key=lambda i: digest(f"{cfg['seed']}:{i}"))
                indices.extend(ids[:limit // 2 + (limit % 2 if cls == 0 else 0)])
            rows = [rows[i] for i in sorted(indices)]
        p, y, qs, records = [], [], [], []
        for row in rows:
            graph = parse_graph(nlp, row["text"])
            x = encoder.encode(graph).to(cfg["device"])
            with torch.no_grad():
                prob, _, edges, _ = model(x, graph, True)
            value = float(prob.item())
            p.append(value)
            y.append(human_label(row))
            qs.extend(edges.sigmoid().cpu().tolist())
            records.append(dict(text_hash=digest(row["text"]), label=y[-1], p_human=value))
        result = dict(**entry, status="ok", raw_sha256=file_hash(path), metrics=classification_metrics(y, p),
                      q=distribution(qs), predictions=records)
        results.append(result)
        print(f"{entry['table']} {entry['dataset']} {entry['condition']}: n={len(y)}, acc={result['metrics']['accuracy']:.4f}", flush=True)
        write_json(output, dict(config=cfg, smoke_limit=limit, results=results, seconds=time.perf_counter() - start,
                               note="Subset scores are smoke tests, not full benchmark results." if limit else "Full input files evaluated."))
    write_json(output, dict(config=cfg, smoke_limit=limit, results=results, seconds=time.perf_counter() - start,
                           checkpoint_sha256=file_hash(Path(cfg["gcn_output"]) / "best.pt"),
                           note="Subset scores are smoke tests, not full benchmark results." if limit else "Full input files evaluated."))
    return results


if __name__ == "__main__":
    parser = config_cli("Evaluate PEARL on Table 2/3/4 existing benchmark text files")
    parser.add_argument("--tables", nargs="*", choices=["table2", "table3", "table4"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()
    evaluate(load_config(args.config), args.tables, args.limit, args.output)
