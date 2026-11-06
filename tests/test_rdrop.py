"""Offline configuration tests and optional PyTorch numerical/RGCN tests."""

import importlib.util
import unittest

from rdrop import require_v1_graph, symmetric_bernoulli_kl, training_objective, validate_alpha


class ConfigurationTests(unittest.TestCase):
    def test_alpha_zero_and_positive(self):
        self.assertEqual(validate_alpha(0), 0.0)
        self.assertEqual(validate_alpha('0.1'), 0.1)

    def test_invalid_alpha(self):
        for alpha in [-1, float('nan'), float('inf'), -float('inf')]:
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                validate_alpha(alpha)

    def test_legacy_graph_accepted(self):
        require_v1_graph({'all_token_embeddings': [], 'all_edge_type': []})

    def test_known_aligned_and_v2_caches_rejected(self):
        for key in ['encoder_metadata', 'schema_version', 'all_cls_embeddings',
                    'all_quality', 'quality_features']:
            with self.subTest(key=key), self.assertRaises(ValueError):
                require_v1_graph({key: {}})


@unittest.skipUnless(importlib.util.find_spec('torch') is not None, 'PyTorch not installed locally')
class TorchTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch

    def test_kl_matches_two_class_distribution(self):
        torch = self.torch
        p = torch.tensor([[0.2], [0.7]], dtype=torch.float64)
        q = torch.tensor([[0.4], [0.9]], dtype=torch.float64)
        pp, qq = torch.cat([1-p, p], dim=1), torch.cat([1-q, q], dim=1)
        reference = 0.5 * ((pp * (pp.log()-qq.log())).sum(1)
                           + (qq * (qq.log()-pp.log())).sum(1)).mean()
        torch.testing.assert_close(symmetric_bernoulli_kl(p, q), reference)
        self.assertGreater(reference.item(), 0)

    def test_zero_for_identical_and_symmetric_nonnegative(self):
        torch = self.torch
        p, q = torch.tensor([[0.1], [0.8]]), torch.tensor([[0.9], [0.2]])
        self.assertEqual(symmetric_bernoulli_kl(p, p).item(), 0)
        left, right = symmetric_bernoulli_kl(p, q), symmetric_bernoulli_kl(q, p)
        torch.testing.assert_close(left, right)
        self.assertGreater(left.item(), 0)

    def test_extremes_finite_and_gradients_reach_both_views(self):
        torch = self.torch
        p = torch.tensor([[0.0], [1.0], [0.3]], requires_grad=True)
        q = torch.tensor([[1.0], [0.0], [0.7]], requires_grad=True)
        loss = symmetric_bernoulli_kl(p, q)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for tensor in [p, q]:
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertNotEqual(tensor.grad[2].item(), 0)

    def test_kl_gradcheck(self):
        torch = self.torch
        p = torch.tensor([[0.2], [0.4]], dtype=torch.float64, requires_grad=True)
        q = torch.tensor([[0.6], [0.8]], dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(symmetric_bernoulli_kl, (p, q)))

    def test_bad_shapes_and_eps(self):
        torch = self.torch
        with self.assertRaises(ValueError):
            symmetric_bernoulli_kl(torch.ones(2), torch.ones(3))
        with self.assertRaises(ValueError):
            symmetric_bernoulli_kl(torch.empty(0), torch.empty(0))
        for eps in [0, 0.5, float('nan')]:
            with self.assertRaises(ValueError):
                symmetric_bernoulli_kl(torch.ones(2), torch.ones(2), eps)

    def make_model(self):
        torch = self.torch
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 1)
                self.dropout = torch.nn.Dropout(0.5)
                self.calls = 0

            def forward(self, features):
                self.calls += 1
                return torch.sigmoid(self.linear(self.dropout(features)))
        return Model()

    def test_alpha_zero_matches_original_bce_gradients_and_rng(self):
        torch = self.torch
        from copy import deepcopy
        direct = self.make_model().train()
        experiment = deepcopy(direct).train()
        features, labels = torch.ones(8, 4), torch.ones(8, 1)
        criterion = torch.nn.BCELoss()
        torch.manual_seed(2026)
        reference = criterion(direct(features), labels)
        reference.backward()
        rng = torch.get_rng_state().clone()
        torch.manual_seed(2026)
        actual, _, bce, kl = training_objective(experiment, features, labels, criterion, 0)
        actual.backward()
        self.assertEqual(experiment.calls, 1)
        self.assertEqual(kl.item(), 0)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        torch.testing.assert_close(actual.detach(), bce, rtol=0, atol=0)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        for left, right in zip(direct.parameters(), experiment.parameters()):
            torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)

    def test_positive_alpha_two_forwards_and_correct_total(self):
        torch = self.torch
        model = self.make_model().train()
        features, labels = torch.ones(32, 4), torch.ones(32, 1)
        torch.manual_seed(2026)
        loss, probability, bce, kl = training_objective(model, features, labels, torch.nn.BCELoss(), 0.1)
        self.assertEqual(model.calls, 2)
        torch.testing.assert_close(loss.detach(), bce + 0.1 * kl)
        self.assertFalse(probability.requires_grad)
        self.assertTrue(torch.isfinite(probability).all())
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_no_dropout_zero_kl(self):
        torch = self.torch
        model = self.make_model().eval()
        _, _, _, kl = training_objective(model, torch.ones(8, 4), torch.ones(8, 1), torch.nn.BCELoss(), 0.1)
        self.assertEqual(kl.item(), 0)

    def test_original_rgcn_checkpoint_compatible(self):
        torch = self.torch
        from types import SimpleNamespace
        from model.RGCN import RGCN2
        model = RGCN2(input_dim=3, hidden_dim=4, output_dim=2, num_relations=2, num_bases=1)
        model.train()
        batch = SimpleNamespace(x=torch.randn(6, 3),
                                edge_index=torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 2, 4, 5, 5]]),
                                edge_type=torch.tensor([0, 1, 0, 1, 0, 1]),
                                batch=torch.tensor([0, 0, 0, 1, 1, 1]))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        loss, _, _, _ = training_objective(model, batch, torch.tensor([[0.], [1.]]),
                                          torch.nn.BCELoss(), 0.1)
        loss.backward()
        optimizer.step()
        restored = RGCN2(input_dim=3, hidden_dim=4, output_dim=2, num_relations=2, num_bases=1)
        restored.load_state_dict(model.state_dict())
        model.eval()
        restored.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(batch), restored(batch))


if __name__ == '__main__':
    unittest.main()
