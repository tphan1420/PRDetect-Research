"""Training-only, class-conditional MixStyle for variable-size graph batches.

Adapted from Domain Generalization with MixStyle (ICLR 2021):
https://arxiv.org/abs/2104.02008 .  The graph/class conditioning is experimental,
not a claim that the image-domain results transfer to syntax graphs.
"""

import math

import torch
from torch import nn


class ClassConditionalGraphMixStyle(nn.Module):
    """Mix detached per-graph channel moments with another same-label graph.

    Each node keeps its normalized residual. Donors form random derangements
    within each class; singleton classes and graphs with <2 nodes are unchanged.
    ``strength`` bounds the donor contribution, unlike unrestricted MixStyle.
    There are no learnable parameters or running statistics. Evaluation never
    reads labels, samples randomness, or changes features.
    """

    def __init__(self, p=0.5, alpha=0.1, strength=0.3, eps=1e-6):
        super().__init__()
        values = (p, alpha, strength, eps)
        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError("MixStyle settings must be finite")
        if not 0 <= p <= 1 or not 0 <= strength <= 1 or alpha <= 0 or eps <= 0:
            raise ValueError("Require p/strength in [0,1], alpha>0, eps>0")
        self.p, self.alpha, self.strength, self.eps = map(float, values)
        self.last_stats = {"graphs": 0, "eligible": 0, "applied": 0}

    def forward(self, features, batch_index, labels=None, num_graphs=None):
        self.last_stats = {"graphs": 0, "eligible": 0, "applied": 0}
        # These exits must not change the RNG stream of a baseline experiment.
        if not self.training or self.p == 0 or self.strength == 0:
            return features
        if features.ndim != 2 or features.shape[0] == 0:
            raise ValueError("Expected nonempty [num_nodes, num_channels] features")
        if labels is None:
            raise ValueError("Training CMixStyle requires source graph labels")
        labels = labels.detach().reshape(-1).to(features.device)
        B = labels.numel() if num_graphs is None else int(num_graphs)
        if B != labels.numel() or not torch.isin(labels, labels.new_tensor([0, 1])).all():
            raise ValueError("Expected one binary AI/human label per graph")
        if batch_index.shape != (features.shape[0],) or batch_index.dtype != torch.long:
            raise ValueError("Expected int64 graph index for every node")
        if batch_index.device != features.device:
            raise ValueError("Graph indices and features must use the same device")
        if int(batch_index.min()) < 0 or int(batch_index.max()) >= B:
            raise ValueError("Graph index outside batch")
        count = torch.bincount(batch_index, minlength=B)
        donor = torch.arange(B, device=features.device)
        eligible = torch.zeros(B, dtype=torch.bool, device=features.device)
        for label in (0, 1):
            members = torch.nonzero((labels == label) & (count >= 2), as_tuple=False).flatten()
            if members.numel() >= 2:
                order = members[torch.randperm(members.numel(), device=features.device)]
                donor[order] = order.roll(1)
                eligible[members] = True
        self.last_stats = {"graphs": B, "eligible": int(eligible.sum()), "applied": 0}
        if not bool(eligible.any()):
            return features
        active = eligible & (torch.rand(B, device=features.device) < self.p)
        self.last_stats["applied"] = int(active.sum())
        if not bool(active.any()):
            return features
        # Accumulate moments in FP32, including when the encoder uses autocast.
        h = features.float()
        denominator = count.clamp_min(1).float().unsqueeze(1)
        mean = h.new_zeros(B, h.shape[1]).index_add_(0, batch_index, h) / denominator
        centered = h - mean[batch_index]
        variance = h.new_zeros(B, h.shape[1]).index_add_(0, batch_index, centered.square()) / denominator
        mean, std = mean.detach(), (variance + self.eps).sqrt().detach()
        beta = torch.distributions.Beta(self.alpha, self.alpha)
        # Beta's CPU RNG works identically for this scalar distribution across
        # CPU/GPU batches; only one coefficient per graph, never per node.
        weight = (self.strength * (1 - beta.sample((B,)))).to(h.device).unsqueeze(1)
        mixed_mean = mean + weight * (mean[donor] - mean)
        mixed_std = std + weight * (std[donor] - std)
        transformed = (h - mean[batch_index]) / std[batch_index]
        transformed = transformed * mixed_std[batch_index] + mixed_mean[batch_index]
        return torch.where(active[batch_index, None], transformed, h).to(features.dtype)

