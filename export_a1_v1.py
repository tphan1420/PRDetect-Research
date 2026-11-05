"""Replace only V1 node embeddings with A1-aligned embeddings for ablation.

V1 topology, relation IDs, sample order and labels are copied unchanged from
reference graph pickles. No hierarchical nodes or quality features are exported.
"""

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict

import torch

from relation_vocab_v2 import FORWARD_DEPENDENCIES, WORD_NODE
from v2_data import load_payload, resolve_graph_file


def export_aligned_features(
    payload: Dict[str, Any], reference: Dict[str, Any]
) -> Dict[str, Any]:
    """Fail closed if sample/token alignment with the V1 reference is uncertain."""
    graphs = payload["graphs"]
    count = len(graphs)
    if payload.get("failures"):
        raise ValueError("A1 preprocessing skipped samples; cannot form a paired ablation.")
    if payload.get("source_indices") != list(range(count)):
        raise ValueError("A1 samples are not a complete, ordered source split.")
    for key in ("all_token_embeddings", "all_edge_index", "all_edge_type"):
        if key not in reference or len(reference[key]) != count:
            raise ValueError("V1 reference has missing/mismatched {}.".format(key))
    reference_labels = torch.as_tensor(reference["y"]).float().view(-1).cpu()
    if not torch.equal(reference_labels, payload["y"].float().view(-1).cpu()):
        raise ValueError("V1/A1 labels or sample order differ.")

    aligned = []
    for index, graph in enumerate(graphs):
        word_mask = graph["node_type"] == WORD_NODE
        features = graph["x"][word_mask].float().cpu()
        original = reference["all_token_embeddings"][index]
        if tuple(features.shape) != tuple(original.shape):
            raise ValueError(
                "Sample #{}: word count/embedding dimension differs from V1: {} vs {}. "
                "Use the same SpaCy version, model and source split."
                .format(index, tuple(features.shape), tuple(original.shape))
            )
        if not torch.isfinite(features).all():
            raise ValueError("Sample #{}: non-finite A1 embeddings.".format(index))
        forward_mask = graph["edge_type"] < len(FORWARD_DEPENDENCIES)
        a1_dependencies = set(map(tuple, graph["edge_index"][:, forward_mask].t().tolist()))
        reference_dependencies = set(map(tuple, reference["all_edge_index"][index].t().tolist()))
        if not a1_dependencies.issubset(reference_dependencies):
            raise ValueError(
                "Sample #{}: SpaCy dependency/token alignment differs from V1. "
                "Use matching preprocessing versions, not regenerated V1 references."
                .format(index)
            )
        aligned.append(features)

    exported = dict(reference)
    exported["all_token_embeddings"] = aligned
    exported["ablation_metadata"] = {
        "variant": "a1_only_on_v1",
        "encoder_name": payload["encoder_name"],
        "source": payload.get("source"),
        "changed": "all_token_embeddings only",
    }
    return exported


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", nargs="+", required=True,
                        help="V2 split stems; matching V1 reference stems are required.")
    parser.add_argument("--graph-dir", type=Path, default=Path("graph_data_v2"))
    parser.add_argument("--reference-dir", type=Path, default=Path("graph_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("graph_data"))
    parser.add_argument("--prefix", default="a1_")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.prefix or any(char in args.prefix for char in ("/", "\\")):
        raise ValueError("--prefix must be a nonempty filename prefix, not a path.")

    for split in args.file:
        graph_path = resolve_graph_file(split, args.graph_dir)
        reference_path = args.reference_dir / (graph_path.stem + ".pkl")
        output_path = args.output_dir / (args.prefix + graph_path.stem + ".pkl")
        if output_path.resolve() in (reference_path.resolve(), graph_path.resolve()):
            raise ValueError("Output must not overwrite an input graph.")
        if output_path.exists() and not args.overwrite:
            raise FileExistsError("{} already exists; use --overwrite.".format(output_path))
        payload = load_payload(graph_path)
        with reference_path.open("rb") as handle:
            reference = pickle.load(handle)
        exported = export_aligned_features(payload, reference)
        exported["ablation_metadata"]["reference_graph"] = str(reference_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as handle:
            pickle.dump(exported, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("A1-only: {} samples -> {}".format(len(exported["y"]), output_path))


if __name__ == "__main__":
    main()
