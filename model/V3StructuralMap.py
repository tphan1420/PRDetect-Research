"""Corrected Structural Mapping adapter for the V3 R-GCN.

The adapter preserves V3's learnable architecture.  It only changes syntax
role assignment for augmentation when the graph schema is explicitly
``directed``: every non-self edge is dependent -> head and its actual relation
ID is the syntax role.  GraphSet validates the same schema before training or
inference.
"""

import torch

from model.QuangV3 import Net


class CorrectedStructuralMappingNet(Net):
    """Use actual dependency IDs for swap groups in directed graph caches.

    This class owns no learnable parameters, so a V3 baseline state dict can
    strict-load for controlled architecture checks.  A checkpoint must still
    be trained from scratch because its structural features differ.
    """

    def _augment(self, x, src, dst, rel, bidx, B, mode, gen, labels=None):
        if self.mode != "directed":
            raise ValueError("Corrected Structural Mapping requires mode='directed'")

        N, dev = x.size(0), x.device
        # R is a sentinel that cannot collide with real relation IDs [0, R).
        node_rel = torch.full((N,), self.R, device=dev, dtype=torch.long)
        edge_mask = src != dst
        node_rel[src[edge_mask]] = rel[edge_mask]

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

        # Preserve the legacy V3 donor policy: same dependency role, sampled
        # across the current micro-batch.  Class preservation remains a
        # separate classswap ablation.
        key = node_rel.float() * 2.0 + torch.rand(N, device=dev, generator=gen)
        order = torch.argsort(key)
        nxt = order.roll(-1)
        same = node_rel[order] == node_rel[nxt]
        donor = torch.empty_like(order)
        donor[order] = torch.where(same, nxt, order)
        x_new = torch.where(use_swap.unsqueeze(1), x[donor], x)
        if mode in ("mask", "both"):
            x_new = torch.where(use_mask.unsqueeze(1),
                                self.mask_emb.to(x.dtype).unsqueeze(0).expand_as(x), x_new)
        return x_new
