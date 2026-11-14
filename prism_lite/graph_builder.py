"""Build F1 graph caches with sentence-local sequential fallback edges.

This module deliberately reuses the legacy ``building_graph`` spaCy and
RoBERTa pipeline.  It writes a new cache schema and never changes legacy
``graph_data/*.pkl`` files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import importlib.metadata
from typing import Iterable

import torch


DEPENDENCY, FALLBACK = 0, 1
CACHE_VERSION = "prism-f1-sentence-fallback-v2"


def protocol_metadata():
    """Fingerprint the actual baseline assets and parser used for this build."""
    import building_graph

    def digest_file(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    versions = {}
    for name in ("torch", "torch-geometric", "spacy", "transformers"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    model = building_graph.model
    tokenizer = building_graph.tokenizer
    vocab = tokenizer.get_vocab()
    return {
        "feature_protocol": "legacy-spacy-text-convert_tokens_to_ids-512-float32",
        "sentence_protocol": "spacy-doc.sents-original-document-indices",
        "graph_unit": "one-document; 512-token chunks affect embeddings only",
        "dependency_direction": "dependent-to-head-including-root-self-loops",
        "edge_version": CACHE_VERSION,
        "baseline_builder_sha256": digest_file(building_graph.__file__),
        "builder_sha256": digest_file(__file__),
        "parser": dict(building_graph.nlp.meta),
        "tokenizer_name": str(tokenizer.name_or_path),
        "tokenizer_vocab_sha256": hashlib.sha256(json.dumps(vocab, sort_keys=True).encode()).hexdigest(),
        "encoder_config": model.config.to_dict(),
        "encoder_revision": str(getattr(model.config, "_commit_hash", None)),
        "encoder_training": model.training,
        "versions": versions,
    }


def metadata_hash(metadata):
    return hashlib.sha256(json.dumps(metadata, sort_keys=True, default=str).encode()).hexdigest()


def add_sequential_fallback(dep_edges: Iterable[tuple[int, int]], sentence_node_lists: Iterable[Iterable[int]]):
    """Return deduplicated typed edges, prioritising dependency edges.

    ``dep_edges`` use the legacy PRDetect source-to-destination convention.
    Fallback edges link only adjacent nodes inside each supplied sentence.
    """
    # Keep existing root self-loops: F1 uses the baseline GCNConv convention.
    # Fallback generation never adds any new self-loop.
    typed = {(int(source), int(destination)): DEPENDENCY
             for source, destination in dep_edges}
    for sentence in sentence_node_lists:
        nodes = [int(node) for node in sentence]
        for source, destination in zip(nodes, nodes[1:]):
            if source == destination:
                continue
            typed.setdefault((source, destination), FALLBACK)
            typed.setdefault((destination, source), FALLBACK)
    pairs = sorted(typed)
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.long)
    return (torch.tensor(pairs, dtype=torch.long).t().contiguous(),
            torch.tensor([typed[pair] for pair in pairs], dtype=torch.long))


def _legacy_embeddings(doc):
    """Mirror the frozen legacy feature construction exactly, token by token."""
    import building_graph

    tokens = [token.text for token in doc]
    outputs = []
    for start in range(0, len(tokens), 512):
        ids = building_graph.tokenizer.convert_tokens_to_ids(tokens[start:start + 512])
        input_ids = torch.tensor(ids).unsqueeze(0).to(building_graph.device)
        with torch.no_grad():
            outputs.append(building_graph.model(input_ids).last_hidden_state[0])
    return torch.cat(outputs, dim=0).cpu()


def _label(record: dict) -> float:
    value = record.get("label", "")
    return float(value) if isinstance(value, (int, float)) else float("human" in str(value).lower())


def build_records(records: Iterable[dict], edge_mode: str = "dependency_plus_sequential") -> dict:
    """Build a serializable PRISM cache from raw parsed JSON records."""
    if edge_mode not in {"dependency_only", "dependency_plus_sequential"}:
        raise ValueError(f"unsupported edge mode: {edge_mode}")
    import building_graph

    building_graph.init_models()
    if building_graph.model.training:
        raise ValueError("Legacy RoBERTa must be in eval mode for reproducible frozen embeddings")
    graphs, labels, stats, skipped = [], [], [], []
    for sample_index, record in enumerate(records):
        doc = building_graph.nlp(str(record["text"]))
        if not len(doc):
            skipped.append(sample_index)
            continue
        dependency = [(token.i, token.head.i) for token in doc]
        sentences = [[token.i for token in sentence] for sentence in doc.sents]
        if edge_mode == "dependency_only":
            edge_index, edge_type = add_sequential_fallback(dependency, [])
        else:
            edge_index, edge_type = add_sequential_fallback(dependency, sentences)
        graphs.append({
            "x": _legacy_embeddings(doc), "edge_index": edge_index,
            "edge_type": edge_type, "sentence_node_lists": sentences,
            "sample_index": sample_index,
        })
        label = _label(record)
        if label not in (0., 1.):
            raise ValueError(f"Non-binary label at sample {sample_index}: {label}")
        labels.append(label)
        sequential = {(u, v) for sentence in sentences for a, b in zip(sentence, sentence[1:])
                      for u, v in ((a, b), (b, a))}
        overlap = len(sequential.intersection(dependency))
        stats.append({"nodes": len(doc), "dependency_edges": len(set(dependency)),
                      "fallback_edges": int((edge_type == FALLBACK).sum().item()),
                      "sequential_candidates": len(sequential), "overlap_edges": overlap,
                      "overlap_fraction": overlap / max(len(sequential), 1),
                      "edge_growth_fraction": int((edge_type == FALLBACK).sum()) / max(len(set(dependency)), 1)})
    if not graphs:
        raise ValueError("no non-empty documents were built")
    protocol = protocol_metadata()
    return {"cache_version": CACHE_VERSION, "edge_mode": edge_mode, "graphs": graphs,
            "y": torch.tensor(labels, dtype=torch.float32), "stats": stats,
            "skipped_sample_indices": skipped, "protocol": protocol,
            "preprocessing_sha256": metadata_hash(protocol)}


def source_hash(records: Iterable[dict]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(json.dumps(record, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def read_records(path: str | Path) -> list[dict]:
    records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError(f"no records in {path}")
    return records


def build_file(input_path: str | Path, output_path: str | Path, edge_mode: str = "dependency_plus_sequential"):
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Choose a new output path rather than replacing {output_path}")
    records = read_records(input_path)
    cache = build_records(records, edge_mode=edge_mode)
    cache["source_sha256"] = source_hash(records)
    cache["source_path"] = str(Path(input_path).resolve())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(cache, handle)
    return cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--edge-mode", choices=("dependency_only", "dependency_plus_sequential"),
                        default="dependency_plus_sequential")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.cpu:
        import building_graph
        building_graph.init_models()
        building_graph.device = torch.device("cpu")
        building_graph.model.to("cpu")
    cache = build_file(args.input_json, args.output, args.edge_mode)
    fallback = sum(row["fallback_edges"] for row in cache["stats"])
    dependency = sum(row["dependency_edges"] for row in cache["stats"])
    sequential = sum(row["sequential_candidates"] for row in cache["stats"])
    overlap = sum(row["overlap_edges"] for row in cache["stats"])
    print({"output": args.output, "documents": len(cache["graphs"]), "dependency_edges": dependency,
           "fallback_edges": fallback, "edge_growth_fraction": fallback / max(dependency, 1),
           "sequential_overlap_fraction": overlap / max(sequential, 1),
           "skipped_sample_indices": cache["skipped_sample_indices"],
           "preprocessing_sha256": cache["preprocessing_sha256"], "cache_version": CACHE_VERSION})


if __name__ == "__main__":
    main()
