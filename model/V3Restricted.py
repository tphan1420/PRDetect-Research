"""Structure-Guided Restricted Embeddings adapter for the V3 R-GCN.

The adapter keeps the V3 graph encoder and R-GCN unchanged.  It learns a
separate channel-wise gate for the RoBERTa-derived node features and for the
projected structural features (depth, children, distance).  A soft semantic
budget discourages retaining every semantic channel, while the structural
channels remain independently learnable.
"""

import math

import torch
from torch import nn

from model.QuangV3 import Net


def _logit(probability):
    return math.log(probability / (1.0 - probability))


class StructureGuidedRestrictedNet(Net):
    """V3 with learned semantic/structural channel gates.

    Semantic gates are in ``(0, 1)``.  Structural gates are in ``(0, 2)`` and
    initialized to one, so they can increase or decrease structural evidence
    without silently removing it.  The semantic budget is applied only during
    optimization through :meth:`restriction_loss`; inference is deterministic.
    """

    def __init__(self, *args, semantic_init=0.9, structural_init=1.0,
                 semantic_budget=0.8, semantic_penalty=0.1, **kwargs):
        values = (semantic_init, structural_init, semantic_budget, semantic_penalty)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Restricted-embedding settings must be finite")
        if not 0 < semantic_init < 1 or not 0 < structural_init < 2:
            raise ValueError("semantic_init must be in (0,1); structural_init in (0,2)")
        if not 0 <= semantic_budget <= 1 or semantic_penalty < 0:
            raise ValueError("semantic_budget must be in [0,1]; semantic_penalty must be nonnegative")
        super().__init__(*args, **kwargs)
        if not self.use_struct:
            raise ValueError("Structure-Guided Restricted Embeddings requires use_struct=True")
        width = self.proj.out_features
        self.semantic_logits = nn.Parameter(torch.full((width,), _logit(float(semantic_init))))
        self.structural_logits = nn.Parameter(torch.full(
            (width,), _logit(float(structural_init) / 2.0)))
        self.semantic_budget = float(semantic_budget)
        self.semantic_penalty = float(semantic_penalty)

    def semantic_gate(self):
        return torch.sigmoid(self.semantic_logits)

    def structural_gate(self):
        return 2.0 * torch.sigmoid(self.structural_logits)

    def _restrict_semantic(self, features, _batch):
        return features * self.semantic_gate().to(features.dtype).unsqueeze(0)

    def _restrict_structural(self, features, _batch):
        return features * self.structural_gate().to(features.dtype).unsqueeze(0)

    def restriction_loss(self):
        """Penalize semantic capacity above the configured retained budget."""
        overflow = (self.semantic_gate().mean() - self.semantic_budget).clamp_min(0.0)
        return self.semantic_penalty * overflow.square()

    @torch.no_grad()
    def restriction_stats(self):
        semantic = self.semantic_gate()
        structural = self.structural_gate()
        return {
            "semantic_mean": float(semantic.mean()),
            "semantic_below_0_5": int((semantic < 0.5).sum()),
            "structural_mean": float(structural.mean()),
        }

    def forward(self, b, aug_mode=None, gen=None):
        return super().forward(
            b, aug_mode=aug_mode, gen=gen,
            feature_transform=self._restrict_semantic,
            structural_transform=self._restrict_structural,
        )
