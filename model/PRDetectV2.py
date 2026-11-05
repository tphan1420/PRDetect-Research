"""PRDetect V2: aligned embeddings and a hierarchical relation graph (A1+A2)."""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphNorm, RGCNConv, global_max_pool, global_mean_pool

from relation_vocab_v2 import DOCUMENT_NODE, NUM_RELATIONS


CHECKPOINT_FORMAT = "2.1-graph"


class PRDetectV2(nn.Module):
    """Two-layer hierarchical RGCN with optional quality features."""

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        graph_dim: int = 128,
        projection_dim: int = 256,
        quality_dim: int = 5,
        quality_hidden_dim: int = 32,
        num_relations: int = NUM_RELATIONS,
        num_bases: int = 30,
        dropout: float = 0.3,
        use_quality: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_relations = num_relations
        self.use_quality = use_quality
        self.quality_hidden_dim = quality_hidden_dim

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
            nn.Linear(graph_dim * 3, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.quality_encoder = nn.Sequential(
            nn.LayerNorm(quality_dim),
            nn.Linear(quality_dim, quality_hidden_dim),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(projection_dim + quality_hidden_dim, projection_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim // 2, 1),
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

    def forward(self, data: Any) -> torch.Tensor:
        x = data.x
        batch = data.batch
        num_graphs = data.num_graphs
        first = self.conv1(x, data.edge_index, data.edge_type)
        first = self.graph_norm1(first, batch)
        first = self.dropout(F.gelu(first + self.residual1(x)))
        second = self.conv2(first, data.edge_index, data.edge_type)
        second = self.graph_norm2(second, batch)
        second = self.dropout(F.gelu(second + self.residual2(first)))
        graph_vector = self.graph_projection(torch.cat([
            global_mean_pool(second, batch, size=num_graphs),
            global_max_pool(second, batch, size=num_graphs),
            self._document_pool(second, batch, data.node_type, num_graphs),
        ], dim=-1))

        # Keep head shape/initialization identical in the quality ablation.
        # Disabled quality features never enter the classifier or encoder.
        quality_vector = graph_vector.new_zeros(
            (num_graphs, self.quality_hidden_dim)
        )
        if self.use_quality:
            quality_vector = self.quality_encoder(
                data.quality_features.view(num_graphs, -1)
            )
        return self.classifier(torch.cat([graph_vector, quality_vector], dim=-1))
