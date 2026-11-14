import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
class PrismF1ModelTests(unittest.TestCase):
    def test_empty_edge_forward_and_backward(self):
        import torch
        from types import SimpleNamespace
        from prism_lite.model import PrismF1GCN

        model = PrismF1GCN(input_dim=4, hidden_dim=3, output_dim=2, dropout=0.0)
        graph = SimpleNamespace(x=torch.randn(2, 4), edge_index=torch.empty((2, 0), dtype=torch.long),
                                batch=torch.tensor([0, 0]))
        output = model(graph)
        self.assertEqual(tuple(output.shape), (1, 1))
        output.sum().backward()
        self.assertIsNotNone(model.skip1.weight.grad)


if __name__ == "__main__":
    unittest.main()
