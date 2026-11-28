"""Shared R-GCN readout from a graph and a relation-label permutation."""

import hashlib

import torch
from torch import nn

from model.QuangV3 import Net


class RelationResponseNet(Net):
    def __init__(self, R, **kwargs):
        super().__init__(R, **kwargs)
        if self.use_struct or self.use_gfeat or self.aug != "none":
            raise ValueError("Relation response requires struct=0, gfeat=0 and aug=none")
        width = self.mlp[0].in_features
        self.mlp[0] = nn.Linear(2 * width, self.d)

    def permuted_relations(self, batch):
        """Shuffle within each document, retaining self-loops (ROOT in HC3 graphs).

        Evaluation uses a CPU generator seeded by the local graph. Its permutation
        is independent of device, batch offsets, graph order and global RNG state.
        Empty graphs and single-relation graphs naturally have a zero response.
        """
        src, dst, rel = batch["src"], batch["dst"], batch["rel"]
        result = rel.clone()
        for graph in range(batch["B"]):
            indices = torch.where((batch["bidx"][src] == graph) & (src != dst))[0]
            if indices.numel() < 2:
                continue
            if self.training:
                order = torch.randperm(indices.numel(), device=rel.device)
            else:
                offset = torch.where(batch["bidx"] == graph)[0][0]
                local = torch.stack((src[indices] - offset, dst[indices] - offset,
                                     rel[indices])).cpu().numpy().astype("<i8")
                digest = hashlib.sha256(local.tobytes()).digest()
                generator = torch.Generator(device="cpu")
                generator.manual_seed(int.from_bytes(digest[:8], "little") % (2**63))
                order = torch.randperm(indices.numel(), generator=generator).to(rel.device)
            result[indices] = rel[indices[order]]
        return result

    def forward(self, batch, aug_mode=None, gen=None):
        # The original validation swap proxy is applied once to both views.
        if aug_mode and aug_mode != "none":
            batch = {**batch, "x": self._augment(
                batch["x"], batch["src"], batch["dst"], batch["rel"],
                batch["bidx"], batch["B"], aug_mode, gen)}
        control = {**batch, "rel": self.permuted_relations(batch)}
        if self.training:
            cpu_state = torch.get_rng_state()
            device = batch["x"].device
            devices = [device.index] if device.type == "cuda" else []
            cuda_state = torch.cuda.get_rng_state(device) if devices else None
            actual = self.encode(batch)
            # Restore the pre-encode RNG for the control only, then return to the
            # state after the actual view. Dropout/edge dropout match both passes.
            with torch.random.fork_rng(devices=devices):
                torch.set_rng_state(cpu_state)
                if devices:
                    torch.cuda.set_rng_state(cuda_state, device)
                counterfactual = self.encode(control)
        else:
            actual = self.encode(batch)
            counterfactual = self.encode(control)
        return self.mlp(torch.cat((actual, actual - counterfactual), dim=1)).squeeze(-1)
