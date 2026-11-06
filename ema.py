"""Exponential moving average of model weights; no inference architecture changes."""

from copy import deepcopy
import math

import torch


class ModelEMA:
    """Keep a detached shadow model, updated once after each optimizer step."""

    def __init__(self, model, decay=0.99):
        if not math.isfinite(decay) or not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be finite and satisfy 0 <= decay < 1.")
        self.decay = float(decay)
        self.num_updates = 0
        self.module = deepcopy(model).eval()
        self.module.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        """Average parameters; copy buffers (RGCN currently has no BatchNorm)."""
        source_parameters = dict(model.named_parameters())
        for name, averaged in self.module.named_parameters():
            current = source_parameters[name].detach()
            if averaged.is_floating_point() or averaged.is_complex():
                averaged.mul_(self.decay).add_(current, alpha=1.0 - self.decay)
            else:
                averaged.copy_(current)
        source_buffers = dict(model.named_buffers())
        for name, averaged in self.module.named_buffers():
            averaged.copy_(source_buffers[name].detach())
        self.num_updates += 1
