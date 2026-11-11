"""Class-preserving Syntax-Aware Swap adapter for the V3 R-GCN.

The base V3 model is not copied or changed.  This adapter only replaces the
donor-selection rule used by the existing training-time ``swap`` augmentation.
"""

import torch

from model.QuangV3 import Net


class ClassPreservingSyntaxSwapNet(Net):
    """Swap only from a different graph with the same label and syntax role.

    A node with no eligible donor is left unchanged.  The adapter owns no new
    learnable parameters, so a V3 baseline state dict strict-loads into it.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_swap_stats = {"selected": 0, "eligible": 0, "swapped": 0}

    def _augment(self, x, src, dst, rel, bidx, B, mode, gen, labels=None):
        if labels is None or labels.numel() != B:
            raise ValueError("Class-preserving swap requires one graph label per batch item")

        N, dev = x.size(0), x.device
        half = self.R // 2
        node_rel = torch.full((N,), half, device=dev, dtype=torch.long)
        if self.mode == "low":
            edge_mask, edge_rel = (rel < half) & (src != dst), rel
        else:
            edge_mask, edge_rel = (rel >= half) & (rel < 2 * half) & (src != dst), rel - half
        node_rel[src[edge_mask]] = edge_rel[edge_mask]

        rho = torch.rand(B, device=dev, generator=gen) * self.aug_max
        selected = torch.rand(N, device=dev, generator=gen) < rho[bidx]
        coin = torch.rand(N, device=dev, generator=gen) < 0.5
        if mode == "mask":
            use_mask = selected
        elif mode == "both":
            use_mask = selected & coin
        else:
            use_mask = torch.zeros_like(selected)
        use_swap = selected & ~use_mask

        # Group by dependency relation and graph class.  Within each group a
        # receiver can only draw from a *different* graph, preserving the
        # existing syntax-role condition and adding the requested class guard.
        node_label = labels.to(device=dev, dtype=torch.long)[bidx]
        group = node_rel * 2 + node_label
        donor = torch.arange(N, device=dev)
        eligible = torch.zeros(N, dtype=torch.bool, device=dev)
        for group_id in torch.unique(group).tolist():
            members = torch.nonzero(group == group_id, as_tuple=False).flatten()
            member_graphs = bidx[members]
            for graph_id in torch.unique(member_graphs).tolist():
                receivers = members[member_graphs == graph_id]
                candidates = members[member_graphs != graph_id]
                if candidates.numel():
                    choices = torch.randint(candidates.numel(), (receivers.numel(),),
                                            device=dev, generator=gen)
                    donor[receivers] = candidates[choices]
                    eligible[receivers] = True

        use_swap = use_swap & eligible
        x_new = torch.where(use_swap.unsqueeze(1), x[donor], x)
        if mode in ("mask", "both"):
            x_new = torch.where(use_mask.unsqueeze(1),
                                self.mask_emb.to(x.dtype).unsqueeze(0).expand_as(x), x_new)
        self.last_swap_stats = {
            "selected": int(selected.sum().item()),
            "eligible": int(eligible.sum().item()),
            "swapped": int(use_swap.sum().item()),
        }
        return x_new
