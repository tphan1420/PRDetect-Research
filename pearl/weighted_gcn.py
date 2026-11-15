"""Small adapter over unchanged baseline layers, pooling and classifier."""
import torch
from torch import nn
from model.GCN2 import GCN2, global_mean_pool
from .graph import arcs, tensor_edges


def expand_arc_weights(graph, q, epsilon, bidirectional=False):
    if not 0 < epsilon <= 1:
        raise ValueError("epsilon must be in (0, 1]")
    edges, mapping = tensor_edges(graph, bidirectional)
    if len(q) != len(arcs(graph)):
        raise ValueError("Arc prediction count mismatch")
    columns, indices = [], []
    for i, (child, _, _) in enumerate(arcs(graph)):
        for column in mapping[child]:
            columns.append(column)
            indices.append(i)
    weights = q.new_ones(len(edges))
    if columns:
        weights = weights.index_copy(0, torch.tensor(columns, device=q.device),
                                     epsilon + (1 - epsilon) * q[indices])
    # ROOT remains 1; GCNConv adds other unit self-loops using baseline defaults.
    return torch.tensor(edges, dtype=torch.long, device=q.device).T.contiguous(), weights


def weighted_conv(conv, x, edge_index, weight):
    if hasattr(conv, "lin"):  # native PyG; normalization/loops/bias are unchanged
        return conv(x, edge_index, edge_weight=weight)
    if not hasattr(conv, "linear"):
        raise TypeError("Unrecognized baseline GCN backend")
    # Preserve root fallback's existing double ROOT self-loop behavior exactly.
    src, dst = edge_index
    degree = x.new_zeros(len(x)).index_add(0, dst, weight) + 1
    inv = degree.pow(-.5)
    norm = inv[src] * weight * inv[dst]
    out = torch.zeros_like(x).index_add(0, dst, x[src] * norm[:, None])
    out = out + x * inv.square()[:, None]
    return conv.linear(out)


class PEARLGCN(nn.Module):
    def __init__(self, base, predictor, features, epsilon=.8):
        super().__init__()
        self.base, self.predictor = base, predictor
        self.features, self.epsilon = features, epsilon
        self.backend = "pyg" if hasattr(base.conv1, "lin") else "legacy_fallback"

    def forward(self, x, graph, return_details=False, batch=None, bidirectional=False):
        features = self.features.encode(graph).to(x.device)
        logits_edge = self.predictor(features)
        q = logits_edge.sigmoid()
        edge_index, weight = expand_arc_weights(graph, q, self.epsilon, bidirectional)
        h = self.base.dropout(torch.relu(weighted_conv(self.base.conv1, x, edge_index, weight)))
        h = self.base.dropout(torch.relu(weighted_conv(self.base.conv2, h, edge_index, weight)))
        node_logits = self.base.fc(h)
        logits = global_mean_pool(node_logits, batch) if batch is not None else node_logits.mean(dim=0, keepdim=True)
        p = logits.sigmoid()
        return (p, logits, logits_edge, weight) if return_details else p


def baseline_from_state(state):
    key = "conv1.lin.weight" if "conv1.lin.weight" in state else "conv1.linear.weight"
    second = "conv2.lin.weight" if "conv2.lin.weight" in state else "conv2.linear.weight"
    model = GCN2(state[key].shape[1], state[key].shape[0], state[second].shape[0])
    model.load_state_dict(state, strict=True)
    return model
