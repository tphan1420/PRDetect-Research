"""F1: legacy two-layer GCN with residual paths and fallback-edge input."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, global_mean_pool


class PrismF1GCN(nn.Module):
    """A separately parameterized R0/F1 encoder, preserving baseline readout."""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, output_dim: int = 64, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.skip1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.skip2 = nn.Linear(hidden_dim, output_dim, bias=False)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(output_dim, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, "batch", None)
        x = self.norm1(self.skip1(x) + self.conv1(x, edge_index))
        x = self.dropout(F.relu(x))
        x = self.norm2(self.skip2(x) + self.conv2(x, edge_index))
        x = self.dropout(F.relu(x))
        x = self.fc(x)
        x = global_mean_pool(x, batch) if batch is not None else x.mean(dim=0, keepdim=True)
        return torch.sigmoid(x)
