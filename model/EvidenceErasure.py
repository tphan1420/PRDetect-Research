"""Signed node evidence with budgeted, graph-local erasure during training."""

import math

import torch
from torch import nn
import torch.nn.functional as F

from model.QuangV3 import Net


def candidate_masks(node_count, src, dst, *, fraction, candidates, hops):
    """Largest complete undirected ball (radius <= hops) fitting the budget.

    Centers are evenly spaced in document node order. A ball that exceeds the
    budget keeps its previous radius. Self loops have no effect. The search is
    over these candidates, not every possible deletion set. All tensors are local
    to one document. No labels or randomness enter candidate construction.
    """
    budget = min(node_count - 1, math.floor(fraction * node_count))
    if budget < 1:
        return torch.empty((0, node_count), device=src.device)
    centers = torch.linspace(0, node_count - 1, min(candidates, node_count),
                             device=src.device).long()
    masks = torch.zeros((centers.numel(), node_count), device=src.device)
    masks[torch.arange(centers.numel(), device=src.device), centers] = 1
    for _ in range(hops):
        expanded = masks.clone()
        expanded.scatter_reduce_(1, dst.expand(centers.numel(), -1), masks[:, src],
                                 reduce="amax", include_self=True)
        expanded.scatter_reduce_(1, src.expand(centers.numel(), -1), masks[:, dst],
                                 reduce="amax", include_self=True)
        fits = expanded.sum(1) <= budget
        masks = torch.where(fits[:, None], expanded, masks)
    return masks


def erased_predictions(evidence, batch, *, mode, fraction, candidates, hops):
    """Choose a training mask, then differentiate through the retained evidence.

    Worst masks maximize BCE among the candidates, using training labels only.
    Random masks use the same candidates/budget as a controlled ablation.
    Graphs too small to erase any node contribute zero auxiliary loss.
    """
    logits, fractions, active = [], [], []
    for graph in range(batch["B"]):
        nodes = torch.where(batch["bidx"] == graph)[0]
        values = evidence[nodes]
        # GraphSet batches concatenate nodes in document order.
        edges = batch["bidx"][batch["src"]] == graph
        src = batch["src"][edges] - nodes[0]
        dst = batch["dst"][edges] - nodes[0]
        masks = candidate_masks(len(nodes), src, dst, fraction=fraction,
                                candidates=candidates, hops=hops)
        if masks.shape[0] == 0:
            logits.append(values.mean())
            fractions.append(values.new_zeros(()))
            active.append(False)
            continue
        remaining = len(nodes) - masks.sum(1)
        predictions = (values.sum() - (masks * values[None, :]).sum(1)) / remaining
        if mode == "worst":
            losses = F.binary_cross_entropy_with_logits(
                predictions.detach(), batch["y"][graph].expand_as(predictions), reduction="none")
            chosen = losses.argmax()
        elif mode == "random":
            chosen = torch.randint(masks.shape[0], (), device=values.device)
        else:
            raise ValueError("Erasure mode must be random or worst")
        logits.append(predictions[chosen])
        fractions.append(masks[chosen].mean())
        active.append(True)
    return (torch.stack(logits), torch.stack(fractions),
            torch.tensor(active, device=evidence.device, dtype=torch.bool))


class EvidenceNet(Net):
    def __init__(self, R, **kwargs):
        super().__init__(R, **kwargs)
        if self.use_gfeat:
            raise ValueError("Evidence readout requires gfeat=0")
        del self.gate, self.mlp
        self.evidence_head = nn.Sequential(
            nn.Linear(self.d, self.d), nn.GELU(), nn.Dropout(self.drop.p), nn.Linear(self.d, 1))

    def node_evidence(self, batch, aug_mode=None, gen=None):
        h = self.encode_nodes(batch, aug_mode=aug_mode, gen=gen)
        return self.evidence_head(h).squeeze(-1).float()

    @staticmethod
    def aggregate(evidence, batch):
        total = evidence.new_zeros(batch["B"]).index_add_(0, batch["bidx"], evidence)
        counts = torch.bincount(batch["bidx"], minlength=batch["B"]).clamp_min(1)
        return total / counts

    def forward(self, batch, aug_mode=None, gen=None):
        return self.aggregate(self.node_evidence(batch, aug_mode, gen), batch)

    def training_objective(self, batch, *, mode, fraction, weight, candidates, aug_mode=None):
        evidence = self.node_evidence(batch, aug_mode=aug_mode)
        logits = self.aggregate(evidence, batch)
        clean = F.binary_cross_entropy_with_logits(logits, batch["y"])
        auxiliary = clean.new_zeros(())
        erased_fraction = clean.new_zeros(())
        active_fraction = clean.new_zeros(())
        if mode != "none" and weight > 0:
            erased, fractions, active = erased_predictions(
                evidence, batch, mode=mode, fraction=fraction,
                candidates=candidates, hops=len(self.layers))
            per_graph = F.binary_cross_entropy_with_logits(erased, batch["y"], reduction="none")
            auxiliary = (per_graph * active).mean()
            erased_fraction = fractions.mean()
            active_fraction = active.float().mean()
        stats = dict(clean_bce=clean.detach(), erased_bce=auxiliary.detach(),
                     erased_fraction=erased_fraction.detach(), erasure_active_fraction=active_fraction.detach())
        return logits, clean + weight * auxiliary, stats
