"""Cache loading and PyG conversion for PRISM-Lite F1."""

from __future__ import annotations

from pathlib import Path
import pickle

import torch

from .graph_builder import CACHE_VERSION, metadata_hash


def load_cache(path: str | Path) -> dict:
    with Path(path).open("rb") as handle:
        cache = pickle.load(handle)
    if cache.get("cache_version") != CACHE_VERSION:
        raise ValueError(f"expected {CACHE_VERSION}, got {cache.get('cache_version')!r}")
    validate_cache(cache)
    return cache


def validate_cache(cache):
    if cache.get("preprocessing_sha256") != metadata_hash(cache.get("protocol", {})):
        raise ValueError("Missing or inconsistent preprocessing fingerprint")
    if cache.get("edge_mode") not in {"dependency_only", "dependency_plus_sequential"}:
        raise ValueError("Invalid edge mode")
    if not cache["graphs"] or len(cache["graphs"]) != len(cache["y"]):
        raise ValueError("Graph/label counts must match and be non-empty")
    if not torch.isin(cache["y"], torch.tensor([0., 1.])).all():
        raise ValueError("Labels must use AI=0, Human=1")
    from .graph_builder import add_sequential_fallback
    for graph in cache["graphs"]:
        x, edges, types = graph["x"], graph["edge_index"], graph["edge_type"]
        if x.ndim != 2 or x.size(0) < 1 or x.dtype != torch.float32 or not torch.isfinite(x).all():
            raise ValueError("Invalid float32 node features")
        if edges.dtype != torch.long or edges.ndim != 2 or edges.size(0) != 2:
            raise ValueError("edge_index must be long [2,E]")
        if types.dtype != torch.long or tuple(types.shape) != (edges.size(1),):
            raise ValueError("edge_type must align with edge_index")
        if edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= x.size(0)):
            raise ValueError("Edge index outside graph")
        sentences = graph["sentence_node_lists"]
        if [node for sentence in sentences for node in sentence] != list(range(x.size(0))):
            raise ValueError("Sentence metadata must partition nodes in token order")
        dependency = edges[:, types == 0].t().tolist()
        expected_edges, expected_types = add_sequential_fallback(
            dependency, sentences if cache["edge_mode"] == "dependency_plus_sequential" else [])
        if not torch.equal(edges, expected_edges) or not torch.equal(types, expected_types):
            raise ValueError("Invalid, duplicated, or out-of-sentence fallback edges")


def to_data_list(cache: dict):
    from torch_geometric.data import Data
    labels = cache["y"]
    result = []
    for graph, label in zip(cache["graphs"], labels):
        result.append(Data(x=graph["x"], edge_index=graph["edge_index"], edge_type=graph["edge_type"],
                           sentence_id=torch.tensor([i for i, sentence in enumerate(graph["sentence_node_lists"])
                                                     for _ in sentence]),
                           sample_id=torch.tensor([graph["sample_index"]]),
                           y=torch.as_tensor(label, dtype=torch.float32).reshape(1)))
    return result
