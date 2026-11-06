"""R-Drop for the original sigmoid RGCN; no encoder/graph/model changes."""

import math


def validate_alpha(alpha):
    alpha = float(alpha)
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError('rdrop_alpha must be finite and non-negative.')
    return alpha


def require_v1_graph(graph):
    """Reject known non-V1 cache formats instead of silently testing alignment."""
    if any(key in graph for key in (
        'encoder_metadata', 'schema_version', 'all_cls_embeddings', 'all_quality', 'quality_features',
    )):
        raise ValueError('This experiment requires legacy V1 graphs, not aligned/V2 caches.')


def symmetric_bernoulli_kl(first, second, eps=1e-6):
    """Mean 0.5 * (KL(Ber(p)||Ber(q)) + KL(Ber(q)||Ber(p))).

    RGCN returns a single sigmoid probability, NOT a vector of class logits.
    Softmax/KL over a final dimension of size one would always yield zero.
    Both distributions remain in the autograd graph; neither is detached.
    """
    import torch

    if first.shape != second.shape or first.numel() == 0:
        raise ValueError('R-Drop probabilities must have identical non-empty shapes.')
    if not 0 < eps < 0.5:
        raise ValueError('eps must be between 0 and 0.5.')
    # Promote half precision before logarithms; preserve double for grad checks.
    p = first if first.dtype == torch.float64 else first.float()
    q = second if second.dtype == torch.float64 else second.float()
    p = p.clamp(eps, 1.0 - eps)
    q = q.clamp(eps, 1.0 - eps)
    logit_p = p.log() - torch.log1p(-p)
    logit_q = q.log() - torch.log1p(-q)
    return (0.5 * (p - q) * (logit_p - logit_q)).mean()


def training_objective(model, batch, targets, criterion, alpha=0.0):
    """Return total loss, metric probabilities, detached BCE and detached KL.

    alpha=0 performs exactly ONE forward and the original BCE. alpha>0 uses
    two independent dropout forwards, one averaged BCE and one optimizer step.
    This function does not toggle model.train()/eval() or reset any RNG state.
    """
    alpha = validate_alpha(alpha)
    first = model(batch)
    first_bce = criterion(first, targets)
    if alpha == 0.0:
        return first_bce, first, first_bce.detach(), first.new_zeros(())
    second = model(batch)
    bce = 0.5 * (first_bce + criterion(second, targets))
    kl = symmetric_bernoulli_kl(first, second)
    total = bce + alpha * kl
    metric_probabilities = 0.5 * (first.detach() + second.detach())
    return total, metric_probabilities, bce.detach(), kl.detach()
