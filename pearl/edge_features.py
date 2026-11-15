"""Whitelisted single-view features. No text, provenance, edits, labels or IDs."""
import math
import torch
from .graph import arcs


class EdgeFeatures:
    def __init__(self, vocab):
        self.vocab = vocab
        self.dim = sum(len(vocab[k]) + 1 for k in ("pos", "pos", "dep")) + 4

    @classmethod
    def fit(cls, graphs):
        return cls({key: sorted({t[key] for g in graphs for t in g["tokens"]}) for key in ("pos", "dep")})

    def encode(self, graph):
        tokens = graph["tokens"]
        degree = [0] * len(tokens)
        edges = arcs(graph)
        for child, head, _ in edges:
            degree[child] += 1
            degree[head] += 1
        x = torch.zeros((len(edges), self.dim), dtype=torch.float32)
        for i, (child, head, dep) in enumerate(edges):
            offset = 0
            for key, value in (("pos", tokens[child]["pos"]), ("pos", tokens[head]["pos"]), ("dep", dep)):
                vocab = self.vocab[key]
                index = vocab.index(value) + 1 if value in vocab else 0
                x[i, offset + index] = 1
                offset += len(vocab) + 1
            x[i, offset:] = torch.tensor([math.log1p(abs(child - head)), math.log1p(degree[child]),
                                          math.log1p(degree[head]), float(child > head)])
        return x
