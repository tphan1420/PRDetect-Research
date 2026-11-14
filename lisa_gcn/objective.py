"""Source-only scale consistency and numerically stable GroupDRO."""

import torch
import torch.nn.functional as F


class GroupDRO:
    def __init__(self, groups: int = 4, eta: float = 0.01):
        if groups < 1 or eta <= 0:
            raise ValueError("groups and eta must be positive")
        self.eta = float(eta)
        self.log_q = torch.zeros(groups)

    def to(self, device):
        self.log_q = self.log_q.to(device)
        return self

    @property
    def weights(self):
        return torch.softmax(self.log_q, dim=0)

    def loss(self, per_document_loss: torch.Tensor, group_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return weighted source loss, group means, and present-group mask.

        Missing groups are excluded rather than silently assigned zero loss.
        ``log_q`` is updated from detached observed losses, so gradients still
        flow through every document loss included in the objective.
        """
        if per_document_loss.ndim != 1 or group_ids.shape != per_document_loss.shape:
            raise ValueError("losses and group ids must be aligned vectors")
        count = self.log_q.numel()
        if group_ids.numel() == 0 or int(group_ids.min()) < 0 or int(group_ids.max()) >= count:
            raise ValueError("invalid GroupDRO group ids")
        sums = per_document_loss.new_zeros(count).index_add_(0, group_ids, per_document_loss)
        sizes = per_document_loss.new_zeros(count).index_add_(0, group_ids, torch.ones_like(per_document_loss))
        present = sizes > 0
        means = sums / sizes.clamp_min(1)
        with torch.no_grad():
            self.log_q[present] += self.eta * means[present].detach()
            self.log_q -= self.log_q.max()
        # Normalize in log-space over observed groups only. Normalizing global
        # probabilities can underflow when the dominant group is absent.
        current = torch.softmax(self.log_q[present], dim=0).to(per_document_loss.device)
        return (current * means[present]).sum(), means, present


def scale_consistency(first_logits: torch.Tensor, second_logits: torch.Tensor) -> torch.Tensor:
    if first_logits.shape != second_logits.shape:
        raise ValueError("both scale views must contain the same documents")
    return (torch.sigmoid(first_logits) - torch.sigmoid(second_logits)).square().mean()


def classification_losses(first_logits: torch.Tensor, second_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return 0.5 * (F.binary_cross_entropy_with_logits(first_logits, labels, reduction="none") +
                  F.binary_cross_entropy_with_logits(second_logits, labels, reduction="none"))
