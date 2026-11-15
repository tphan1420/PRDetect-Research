import torch
from torch import nn


class EdgeRetentionPredictor(nn.Module):
    def __init__(self, feature_dim, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, features):
        return self.mlp(features).squeeze(-1)


def document_edge_loss(logits, pairs):
    """One document, all its valid arcs/pairs; unknowns never enter BCE."""
    total = logits.sum() * 0
    count = 0
    for pair in pairs:
        # labels and features both follow non-root spaCy token order
        valid = [i for i, a in enumerate(pair["arc_labels"]) if a["relation_valid"]]
        if valid:
            labels = logits.new_tensor([pair["arc_labels"][i]["relation_retained"] for i in valid])
            total = total + nn.functional.binary_cross_entropy_with_logits(logits[valid], labels, reduction="sum")
            count += len(valid)
    return total / count if count else None
