"""Dataset loading and schema validation shared by PRDetect V2 scripts."""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from torch_geometric.data import Data

from building_graph_v2 import SCHEMA_VERSION
from relation_vocab_v2 import DOCUMENT_NODE, NUM_RELATIONS, RELATION_LABELS


def resolve_graph_file(graph: str, graph_dir: Path) -> Path:
    """Resolve an explicit pickle path or a stem inside graph_data_v2."""

    requested = Path(graph)
    if requested.is_file():
        return requested.resolve()
    clean_name = requested.stem if requested.suffix == ".pkl" else requested.name
    candidate = graph_dir / "{}.pkl".format(clean_name)
    if not candidate.is_file():
        raise FileNotFoundError("Không tìm thấy graph V2: {}".format(candidate))
    return candidate.resolve()


def load_payload(path: Path) -> Dict[str, Any]:
    """Load a V2 pickle and validate its immutable schema metadata."""

    with path.open("rb") as handle:
        payload = pickle.load(handle)

    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            "Graph {} dùng schema {!r}, cần {!r}."
            .format(path, payload.get("schema_version"), SCHEMA_VERSION)
        )
    if int(payload.get("num_relations", -1)) != NUM_RELATIONS:
        raise ValueError("Số relation trong graph không khớp code V2.")
    if list(payload.get("relation_labels", [])) != list(RELATION_LABELS):
        raise ValueError("Thứ tự relation label trong graph không khớp code V2.")

    graphs = payload.get("graphs")
    labels = payload.get("y")
    if not isinstance(graphs, list) or not isinstance(labels, torch.Tensor):
        raise ValueError("Payload V2 thiếu graphs hoặc tensor y.")
    if len(graphs) != labels.numel():
        raise ValueError("Số graph và số label không khớp.")
    if not graphs:
        raise ValueError("Graph payload rỗng.")
    return payload


def payload_to_data_list(payload: Dict[str, Any]) -> List[Data]:
    """Convert serialized graph dictionaries to PyG Data objects."""

    data_list: List[Data] = []
    labels = payload["y"]
    for index, graph in enumerate(payload["graphs"]):
        x = graph["x"].float()
        edge_index = graph["edge_index"].long()
        edge_type = graph["edge_type"].long()
        node_type = graph["node_type"].long()
        sequence_embedding = graph["sequence_embedding"].float().view(1, -1)
        quality_features = graph["quality_features"].float().view(1, -1)

        if x.dim() != 2:
            raise ValueError("Graph #{}: x không có rank 2.".format(index))
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError("Graph #{}: edge_index không hợp lệ.".format(index))
        if edge_type.numel() != edge_index.size(1):
            raise ValueError("Graph #{}: số edge_type không khớp.".format(index))
        if node_type.numel() != x.size(0):
            raise ValueError("Graph #{}: node_type không khớp x.".format(index))
        if int((node_type == DOCUMENT_NODE).sum().item()) != 1:
            raise ValueError("Graph #{}: cần đúng một document node.".format(index))
        if sequence_embedding.size(1) != x.size(1):
            raise ValueError(
                "Graph #{}: sequence và node embedding khác chiều.".format(index)
            )

        data_list.append(
            Data(
                x=x,
                edge_index=edge_index,
                edge_type=edge_type,
                node_type=node_type,
                sequence_embedding=sequence_embedding,
                quality_features=quality_features,
                y=labels[index].float().view(1),
            )
        )
    return data_list


def infer_dimensions(data_list: Sequence[Data]) -> Dict[str, int]:
    """Infer serialized feature dimensions instead of duplicating constants."""

    if not data_list:
        raise ValueError("Không thể suy luận dimension từ dataset rỗng.")
    first = data_list[0]
    return {
        "input_dim": int(first.x.size(1)),
        "quality_dim": int(first.quality_features.size(1)),
        "num_relations": NUM_RELATIONS,
    }
