"""PRDetect V2 hybrid sequence + hierarchical relation graph classifier."""

from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphNorm, RGCNConv, global_max_pool, global_mean_pool

from relation_vocab_v2 import DOCUMENT_NODE, NUM_RELATIONS


ABLATION_MODES = ("hybrid", "sequence_only", "graph_only")


class PRDetectV2(nn.Module):
    """Fuse a sequence representation with a two-layer hierarchical RGCN."""

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        graph_dim: int = 128,
        fusion_dim: int = 256,
        quality_dim: int = 5,
        quality_hidden_dim: int = 32,
        num_relations: int = NUM_RELATIONS,
        num_bases: int = 30,
        dropout: float = 0.3,
        mode: str = "hybrid",
    ) -> None:
        super().__init__()
        if mode not in ABLATION_MODES:
            raise ValueError(
                "mode must be one of {}, got {!r}.".format(ABLATION_MODES, mode)
            )
        self.input_dim = input_dim
        self.fusion_dim = fusion_dim
        self.num_relations = num_relations
        self.mode = mode

        self.conv1 = RGCNConv(
            input_dim,
            hidden_dim,
            num_relations=num_relations,
            num_bases=num_bases,
            root_weight=True,
        )
        self.graph_norm1 = GraphNorm(hidden_dim)
        self.residual1 = nn.Linear(input_dim, hidden_dim, bias=False)

        self.conv2 = RGCNConv(
            hidden_dim,
            graph_dim,
            num_relations=num_relations,
            num_bases=num_bases,
            root_weight=True,
        )
        self.graph_norm2 = GraphNorm(graph_dim)
        self.residual2 = nn.Linear(hidden_dim, graph_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.graph_projection = nn.Sequential(
            nn.Linear(graph_dim * 3, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.sequence_projection = nn.Sequential(
            nn.Linear(input_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.quality_encoder = nn.Sequential(
            nn.LayerNorm(quality_dim),
            nn.Linear(quality_dim, quality_hidden_dim),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 2 + quality_hidden_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim + quality_hidden_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, 1),
        )

    @staticmethod
    def _document_pool(
        x: torch.Tensor,
        batch: torch.Tensor,
        node_type: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        document_mask = node_type == DOCUMENT_NODE
        if int(document_mask.sum().item()) != num_graphs:
            raise ValueError("Mỗi graph trong batch phải có đúng một document node.")

        document_vectors = x[document_mask]
        document_batches = batch[document_mask]
        pooled = torch.zeros(
            (num_graphs, x.size(-1)), device=x.device, dtype=x.dtype
        )
        pooled.index_copy_(0, document_batches, document_vectors)
        return pooled

    def forward(
        self, data: Any, return_aux: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        sequence_embedding = data.sequence_embedding.view(-1, self.input_dim)
        quality_features = data.quality_features.view(sequence_embedding.size(0), -1)
        num_graphs = sequence_embedding.size(0)
        quality_vector = self.quality_encoder(quality_features)

        graph_vector = sequence_embedding.new_zeros(
            (num_graphs, self.fusion_dim)
        )
        if self.mode != "sequence_only":
            x = data.x
            batch = data.batch
            first = self.conv1(x, data.edge_index, data.edge_type)
            first = self.graph_norm1(first, batch)
            first = F.gelu(first + self.residual1(x))
            first = self.dropout(first)

            second = self.conv2(first, data.edge_index, data.edge_type)
            second = self.graph_norm2(second, batch)
            second = F.gelu(second + self.residual2(first))
            second = self.dropout(second)

            mean_pool = global_mean_pool(second, batch, size=num_graphs)
            max_pool = global_max_pool(second, batch, size=num_graphs)
            document_pool = self._document_pool(
                second, batch, data.node_type, num_graphs=num_graphs
            )
            graph_vector = self.graph_projection(
                torch.cat([mean_pool, max_pool, document_pool], dim=-1)
            )

        sequence_vector = sequence_embedding.new_zeros(
            (num_graphs, self.fusion_dim)
        )
        if self.mode != "graph_only":
            sequence_vector = self.sequence_projection(sequence_embedding)

        if self.mode == "hybrid":
            gate_input = torch.cat(
                [graph_vector, sequence_vector, quality_vector], dim=-1
            )
            gate_weights = torch.softmax(self.gate(gate_input), dim=-1)
            fused = (
                gate_weights[:, 0:1] * graph_vector
                + gate_weights[:, 1:2] * sequence_vector
            )
        elif self.mode == "graph_only":
            gate_weights = sequence_embedding.new_tensor([1.0, 0.0]).expand(
                num_graphs, -1
            )
            fused = graph_vector
        else:
            gate_weights = sequence_embedding.new_tensor([0.0, 1.0]).expand(
                num_graphs, -1
            )
            fused = sequence_vector

        logits = self.classifier(torch.cat([fused, quality_vector], dim=-1))

        if not return_aux:
            return logits
        return logits, {
            "gate_weights": gate_weights,
            "graph_vector": graph_vector,
            "sequence_vector": sequence_vector,
        }
