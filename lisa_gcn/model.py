"""Two-layer GCN with per-window logits and document mean-logit readout."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.GCN2 import GCNConv, global_mean_pool


class LisaGCN(nn.Module):
    """Independent GCN baseline preserving PRDetect's two-layer backbone.

    Unlike ``GCN2``, this model returns logits so documents can correctly use
    the arithmetic mean of window logits before one final sigmoid.
    """

    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, output_dim: int = 64, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(output_dim, 1)

    def forward(self, batch):
        x = F.relu(self.conv1(batch.x, batch.edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, batch.edge_index))
        x = self.dropout(x)
        return self.fc(global_mean_pool(x, batch.batch)).squeeze(-1)


def mean_document_logits(window_logits: torch.Tensor, document_ids: torch.Tensor, document_count: int) -> torch.Tensor:
    if window_logits.ndim != 1 or document_ids.ndim != 1 or window_logits.numel() != document_ids.numel():
        raise ValueError("window logits and document ids must be aligned vectors")
    if document_count < 1 or window_logits.numel() == 0:
        raise ValueError("a non-empty set of windows is required")
    sums = window_logits.new_zeros(document_count).index_add_(0, document_ids, window_logits)
    counts = window_logits.new_zeros(document_count).index_add_(0, document_ids, torch.ones_like(window_logits))
    if bool((counts == 0).any()):
        raise ValueError("every document must have at least one window")
    return sums / counts
