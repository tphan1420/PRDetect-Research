"""Unit tests for the Class-Preserving Syntax Swap V3 adapter."""

import importlib.util
import unittest
from unittest.mock import patch


HAS_RUNTIME = importlib.util.find_spec("torch") and importlib.util.find_spec("numpy")


@unittest.skipUnless(HAS_RUNTIME, "PyTorch/NumPy numerical runtime unavailable")
class ClassPreservingSwapTests(unittest.TestCase):
    def _model(self):
        from model.V3ClassSwap import ClassPreservingSyntaxSwapNet

        return ClassPreservingSyntaxSwapNet(4, d=4, nb=2, L=1, aug="swap", aug_max=1.0)

    def _fixture(self):
        import torch

        # Four two-node graphs: graphs 0/1 are AI, graphs 2/3 are Human.
        # All nodes have dependency role 1, making the donor constraint easy
        # to verify independently of the model's learned parameters.
        x = torch.arange(8, dtype=torch.float32).unsqueeze(1)
        bidx = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        labels = torch.tensor([0, 0, 1, 1])
        src = torch.arange(8)
        dst = torch.tensor([1, 0, 3, 2, 5, 4, 7, 6])
        rel = torch.ones(8, dtype=torch.long)
        return x, src, dst, rel, bidx, labels

    def _forced_random_draws(self, B, N):
        import torch

        # rho=1 selects every node; the other draws deterministically choose
        # swap rather than mask.  This avoids a probabilistic test.
        return [torch.ones(B), torch.zeros(N), torch.zeros(N)]

    def test_swap_donor_is_same_class_same_relation_and_other_graph(self):
        import torch

        model = self._model()
        x, src, dst, rel, bidx, labels = self._fixture()
        with patch("model.V3ClassSwap.torch.rand",
                   side_effect=self._forced_random_draws(labels.numel(), x.size(0))):
            swapped = model._augment(
                x, src, dst, rel, bidx, labels.numel(), "swap",
                torch.Generator().manual_seed(7), labels,
            )

        donor = swapped.squeeze(1).to(torch.long)
        self.assertEqual(model.last_swap_stats, {"selected": 8, "eligible": 8, "swapped": 8})
        self.assertTrue(torch.equal(labels[bidx[donor]], labels[bidx]))
        self.assertTrue(torch.all(bidx[donor] != bidx))

    def test_nodes_without_same_class_peer_are_not_swapped(self):
        import torch

        model = self._model()
        x, src, dst, rel, bidx, _ = self._fixture()
        labels = torch.tensor([0, 1, 2, 3])
        with patch("model.V3ClassSwap.torch.rand",
                   side_effect=self._forced_random_draws(labels.numel(), x.size(0))):
            swapped = model._augment(
                x, src, dst, rel, bidx, labels.numel(), "swap",
                torch.Generator().manual_seed(7), labels,
            )

        torch.testing.assert_close(swapped, x)
        self.assertEqual(model.last_swap_stats, {"selected": 8, "eligible": 0, "swapped": 0})

    def test_baseline_state_loads_strictly_into_adapter(self):
        import torch
        from model.QuangV3 import Net

        base = Net(4, d=4, nb=2, L=1, aug="swap")
        adapter = self._model()
        adapter.load_state_dict(base.state_dict(), strict=True)
        base.eval()
        adapter.eval()
        x, src, dst, rel, bidx, labels = self._fixture()
        batch = {"x": x.repeat(1, 4), "src": src, "dst": dst, "rel": rel,
                 "bidx": bidx, "B": labels.numel(), "y": labels,
                 "st": torch.zeros(x.size(0), 3, dtype=torch.long)}
        torch.testing.assert_close(adapter(batch), base(batch))


if __name__ == "__main__":
    unittest.main()
