"""Raw-text, sentence-aware graph construction for LISA-GCN.

Legacy ``graph_data/*.pkl`` records nodes and edges but not verified sentence
boundaries.  LISA deliberately rebuilds from raw JSONL instead of splitting
those caches by node position.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

from .windowing import plan_sentence_windows, sentence_spans_from_doc


@dataclass
class LisaDocument:
    label: float
    token_count: int
    graphs_by_scale: dict[int, list]


def read_jsonl(path: str | Path) -> list[dict]:
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    if not records:
        raise ValueError(f"no JSONL records in {path}")
    return records


def label_of(record: dict) -> float:
    label = record.get("label")
    if isinstance(label, (int, float)):
        return float(label)
    return 1.0 if "human" in str(label).lower() else 0.0


def _legacy_embed(tokens):
    """Use exactly the frozen legacy RoBERTa token embedding convention."""
    import torch
    import building_graph

    building_graph.init_models()
    outputs = []
    for start in range(0, len(tokens), 512):
        ids = building_graph.tokenizer.convert_tokens_to_ids(tokens[start:start + 512])
        input_ids = torch.tensor(ids).unsqueeze(0).to(building_graph.device)
        with torch.no_grad():
            outputs.append(building_graph.model(input_ids).last_hidden_state[0])
    return torch.cat(outputs, dim=0).cpu()


def _window_graph(doc, start: int, end: int, label: float, document_index: int, split_sentence: bool):
    import torch
    from torch_geometric.data import Data

    original_indices = list(range(start, end))
    local_index = {original: local for local, original in enumerate(original_indices)}
    edges = []
    for original in original_indices:
        head = doc[original].head.i
        if head in local_index:
            edges.append((local_index[original], local_index[head]))
    edge_index = (torch.tensor(edges, dtype=torch.long).t().contiguous()
                  if edges else torch.empty((2, 0), dtype=torch.long))
    return Data(
        x=_legacy_embed([doc[index].text for index in original_indices]),
        edge_index=edge_index,
        y=torch.tensor([label], dtype=torch.float32),
        document_index=torch.tensor([document_index], dtype=torch.long),
        token_count=torch.tensor([end - start], dtype=torch.long),
        split_sentence=torch.tensor([int(split_sentence)], dtype=torch.long),
    )


def build_document(text: str, label: float, document_index: int, window_sizes=(96, 128, 160)) -> LisaDocument:
    import building_graph

    building_graph.init_models()
    doc = building_graph.nlp(text)
    spans = sentence_spans_from_doc(doc)
    if not spans:
        raise ValueError("spaCy produced no tokens for a document")
    graphs_by_scale = {}
    for scale in window_sizes:
        graphs_by_scale[scale] = [_window_graph(doc, window.start, window.end, label, document_index, window.split_sentence)
                                  for window in plan_sentence_windows(len(doc), spans, scale)]
    return LisaDocument(label=label, token_count=len(doc), graphs_by_scale=graphs_by_scale)


def build_documents(records: Iterable[dict], window_sizes=(96, 128, 160)) -> list[LisaDocument]:
    return [build_document(str(record["text"]), label_of(record), index, window_sizes)
            for index, record in enumerate(records)]


class DiskDocumentDataset(Sequence):
    """Keep only scalar metadata in RAM; load graph tensors on each access.

    Cache files contain tensor dictionaries rather than pickled PyG classes,
    so they can be read with the weights-only checkpoint loader.
    """

    def __init__(self, directory: Path, metadata: list[tuple[float, int]]):
        self.directory = Path(directory)
        self.metadata = metadata

    @property
    def token_counts(self):
        return [count for _, count in self.metadata]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        from torch_geometric.data import Data
        from .checkpoint import load_lisa_checkpoint

        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        label, token_count = self.metadata[index]
        state = load_lisa_checkpoint(self.directory / f"{index}.pt", map_location="cpu")
        graphs = {scale: [Data.from_dict(graph) for graph in view]
                  for scale, view in state.items()}
        return LisaDocument(label=label, token_count=token_count, graphs_by_scale=graphs)


def build_disk_documents(records: Iterable[dict], directory: str | Path,
                         window_sizes=(96, 128, 160)) -> DiskDocumentDataset:
    """Build and release one document at a time into a per-run cache directory."""
    import torch

    directory = Path(directory)
    # A fresh directory prevents accidental reuse of graphs from another run.
    directory.mkdir(parents=True, exist_ok=False)
    metadata = []
    for index, record in enumerate(records):
        document = build_document(str(record["text"]), label_of(record), index, window_sizes)
        torch.save({scale: [graph.to_dict() for graph in view]
                    for scale, view in document.graphs_by_scale.items()}, directory / f"{index}.pt")
        metadata.append((document.label, document.token_count))
        del document
    if not metadata:
        raise ValueError("non-empty records are required")
    return DiskDocumentDataset(directory, metadata)


def length_quantile_groups(token_counts: list[int], groups: int = 4):
    """Fit deterministic source-only boundaries and return one group per count."""
    import torch

    if groups < 1 or not token_counts:
        raise ValueError("positive groups and non-empty training counts are required")
    values = torch.tensor(token_counts, dtype=torch.float32)
    boundaries = torch.quantile(values, torch.arange(1, groups, dtype=torch.float32) / groups).tolist()
    assignments = [sum(value > boundary for boundary in boundaries) for value in token_counts]
    return boundaries, assignments


def apply_length_groups(token_counts: list[int], boundaries: list[float]) -> list[int]:
    return [sum(value > boundary for boundary in boundaries) for value in token_counts]
