"""Read-only audit for a V3-CorrectedStruct run and its graph caches.

This script never builds graphs, trains, evaluates, deletes, or overwrites a
checkpoint.  Give it the two checkpoints and the train/validation/test caches
that were retained from the Kaggle run.
"""

import argparse
import json
from pathlib import Path


def inspect_checkpoint(path):
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    args = payload.get("args", {})
    required = ("state", "args", "R", "gm", "gs", "thr", "method", "struct_mode", "training_complete")
    return {
        "path": str(path.resolve()),
        "missing_required_fields": [key for key in required if key not in payload],
        "method": payload.get("method"),
        "struct_mode": payload.get("struct_mode"),
        "training_complete": payload.get("training_complete"),
        "R": payload.get("R"),
        "readout": args.get("readout"),
        "arguments": {key: args.get(key) for key in (
            "hidden", "num_bases", "layers", "dropout", "in_dropout", "edge_drop",
            "struct", "gfeat", "aug", "aug_max", "ema", "ema_decay", "lr", "wd",
            "batch_size", "micro_batch_size", "epochs", "patience", "select", "seeds",
        )},
        "validation_threshold": payload.get("thr"),
        "best_epoch": payload.get("best_epoch"),
        "run_signature": payload.get("run_signature"),
        "provenance": payload.get("provenance"),
        "state_parameter_count": sum(value.numel() for value in payload.get("state", {}).values()),
        "checks": {
            "is_correctedstruct_method": payload.get("method") == "v3_structmap",
            "records_directed_mode": payload.get("struct_mode") == "directed",
            "completed": payload.get("training_complete") is True,
        },
    }


def inspect_cache(path, relation_count, sample_graphs):
    import numpy as np
    from model.QuangV3 import parent_array, read_raw, struct_features

    embeddings, edges, relations, labels = read_raw(path)
    if not len(embeddings) == len(edges) == len(relations) == len(labels):
        raise ValueError(f"Cache arrays have inconsistent lengths: {path}")
    summary = {
        "path": str(Path(path).resolve()), "graphs": len(labels), "relation_count_expected": relation_count,
        "edges": 0, "nonself_edges": 0, "self_loops": 0, "invalid_relation_edges": 0,
        "invalid_index_graphs": 0, "multi_parent_graphs": 0, "reverse_edge_pairs": 0,
        "relations_seen": set(), "directed_low_structural_differences": 0,
    }
    for index, (embedding, edge_index, edge_type) in enumerate(zip(embeddings, edges, relations)):
        n = int(np.asarray(embedding).shape[0])
        edge_index, edge_type = np.asarray(edge_index), np.asarray(edge_type).reshape(-1)
        if edge_index.shape != (2, edge_type.size):
            raise ValueError(f"Invalid edge/relation shapes in graph {index}: {path}")
        src, dst = edge_index[0].astype(np.int64), edge_index[1].astype(np.int64)
        if src.size and (src.min() < 0 or dst.min() < 0 or src.max() >= n or dst.max() >= n):
            summary["invalid_index_graphs"] += 1
            continue
        nonself = src != dst
        summary["edges"] += int(src.size)
        summary["nonself_edges"] += int(nonself.sum())
        summary["self_loops"] += int((~nonself).sum())
        summary["invalid_relation_edges"] += int(((edge_type < 0) | (edge_type >= relation_count)).sum())
        summary["relations_seen"].update(map(int, edge_type.tolist()))
        _, counts = np.unique(src[nonself], return_counts=True)
        summary["multi_parent_graphs"] += int((counts > 1).any())
        pairs = {(int(a), int(b)) for a, b in zip(src[nonself], dst[nonself])}
        summary["reverse_edge_pairs"] += sum((b, a) in pairs for a, b in pairs) // 2
        if index < sample_graphs and not ((edge_type < 0) | (edge_type >= relation_count)).any():
            parent_array(n, src, dst, edge_type, relation_count, "directed")
            directed, _ = struct_features(n, src, dst, edge_type, relation_count, "directed")
            low, _ = struct_features(n, src, dst, edge_type, relation_count, "low")
            summary["directed_low_structural_differences"] += int((directed != low).any(axis=1).sum())
    summary["relations_seen"] = sorted(summary["relations_seen"])
    summary["checks"] = {
        "all_relation_ids_in_range": summary["invalid_relation_edges"] == 0,
        "every_graph_has_at_most_one_nonself_parent_per_dependent": summary["multi_parent_graphs"] == 0,
        "no_observed_reverse_edge_pair": summary["reverse_edge_pairs"] == 0,
    }
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2025, 2026])
    parser.add_argument("--graph-cache", type=Path, nargs="+", required=True)
    parser.add_argument("--sample-graphs", type=int, default=100,
                        help="Graphs for the structural low-vs-directed comparison")
    parser.add_argument("--output", type=Path, required=True,
                        help="New JSON report path; refuses to overwrite an existing file")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite audit report: {args.output}")

    from evaluate_v3_quang import find_checkpoints

    checkpoint_paths = find_checkpoints(args.checkpoint_dir, args.seeds, "structmap")
    checkpoints = {seed: inspect_checkpoint(Path(path)) for seed, path in checkpoint_paths.items()}
    relation_counts = {item["R"] for item in checkpoints.values()}
    if len(relation_counts) != 1 or None in relation_counts:
        raise ValueError(f"Seeds do not agree on checkpoint R: {relation_counts}")
    relation_count = int(relation_counts.pop())
    report = {
        "checkpoints": checkpoints,
        "graph_caches": [inspect_cache(path, relation_count, args.sample_graphs) for path in args.graph_cache],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
