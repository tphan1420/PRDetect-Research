"""Numerical invariants of CMixStyle, plus a tiny CPU training/inference smoke test."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import train_v3
import evaluate_v3_quang as evaluator
import run_v3_ablation as workflow


HAS_RUNTIME = all(importlib.util.find_spec(name) for name in ("torch", "numpy", "sklearn"))


class WorkflowTests(unittest.TestCase):
    def test_cache_cleanup_only_removes_verified_training_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "training/graph_data"
            folder.mkdir(parents=True)
            cache = folder / "hc3_train.pkl"
            meta = cache.with_suffix(".meta.json")
            cache.write_bytes(b"fixture")
            checkpoint = root / "model.pt"
            checkpoint.write_bytes(b"must survive")
            with self.assertRaises(ValueError):
                workflow.release_training_cache(cache, root)
            self.assertTrue(cache.exists())
            meta.write_text(json.dumps({"owner": "run_v3_ablation"}))
            workflow.release_training_cache(cache, root)
            self.assertFalse(cache.exists())
            self.assertTrue(checkpoint.exists())

    def test_graph_limit_stops_builder_without_deleting_other_caches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "hc3_train.json"
            source.write_text('{}\n')
            for i in range(4):
                (root / f"other{i}.pkl").write_bytes(b"fixture")
            with patch.object(workflow, "child") as builder:
                with self.assertRaisesRegex(ValueError, "Graph limit"):
                    workflow.training_cache(source, root, 4)
                builder.assert_not_called()
            self.assertEqual(len(list(root.glob("other*.pkl"))), 4)


@unittest.skipUnless(HAS_RUNTIME, "Requires PyTorch, NumPy and scikit-learn")
class MixStyleTests(unittest.TestCase):
    def test_moments_only_mix_same_class_graphs_and_stats_are_detached(self):
        import torch
        from model.v3_mixstyle import ClassConditionalGraphMixStyle

        h = torch.tensor([[0., 2.], [2., 4.], [10., 20.], [12., 26.], [14., 32.],
                          [100., 110.], [104., 120.], [200., 220.], [206., 240.]], requires_grad=True)
        bidx = torch.tensor([0, 0, 1, 1, 1, 2, 2, 3, 3])
        y = torch.tensor([0., 0., 1., 1.])
        module = ClassConditionalGraphMixStyle(p=1, alpha=0.1, strength=1)
        with patch("torch.distributions.Beta.sample", return_value=torch.full((4,), 0.5)):
            out = module(h, bidx, y, 4)
        for graph, donor in enumerate([1, 0, 3, 2]):
            own = h.detach()[bidx == graph]
            other = h.detach()[bidx == donor]
            mean, sigma = own.mean(0), (own.var(0, unbiased=False) + module.eps).sqrt()
            mixed_mean = (mean + other.mean(0)) / 2
            mixed_std = (sigma + (other.var(0, unbiased=False) + module.eps).sqrt()) / 2
            expected = (own - mean) / sigma * mixed_std + mixed_mean
            torch.testing.assert_close(out[bidx == graph], expected)
        self.assertEqual(module.last_stats, {"graphs": 4, "eligible": 4, "applied": 4})
        out[bidx == 0].sum().backward()
        self.assertTrue(torch.isfinite(h.grad).all())
        self.assertTrue(bool((h.grad[bidx != 0] == 0).all()))

    def test_eval_and_disabled_are_exact_identity_without_labels_or_rng_use(self):
        import torch
        from model.v3_mixstyle import ClassConditionalGraphMixStyle

        h, bidx = torch.randn(6, 3), torch.tensor([0, 0, 1, 1, 2, 2])
        for module in (ClassConditionalGraphMixStyle().eval(),
                       ClassConditionalGraphMixStyle(p=0),
                       ClassConditionalGraphMixStyle(strength=0)):
            state = torch.get_rng_state().clone()
            self.assertIs(module(h, bidx, None), h)
            self.assertTrue(torch.equal(torch.get_rng_state(), state))

    def test_singleton_class_or_single_node_graph_has_no_cross_class_fallback(self):
        import torch
        from model.v3_mixstyle import ClassConditionalGraphMixStyle

        module = ClassConditionalGraphMixStyle(p=1)
        h = torch.randn(5, 3)
        bidx = torch.tensor([0, 0, 1, 1, 2])
        self.assertIs(module(h, bidx, torch.tensor([0, 1, 1]), 3), h)
        self.assertEqual(module.last_stats["eligible"], 0)
        with self.assertRaises(ValueError):
            module(h, bidx, None, 3)

    def test_constant_features_and_half_precision_are_finite(self):
        import torch
        from model.v3_mixstyle import ClassConditionalGraphMixStyle

        h = torch.tensor([[1., 2.], [1., 2.], [5., 8.], [5., 8.]], dtype=torch.float16, requires_grad=True)
        module = ClassConditionalGraphMixStyle(p=1)
        out = module(h, torch.tensor([0, 0, 1, 1]), torch.tensor([0, 0]), 2)
        out.float().square().sum().backward()
        self.assertEqual(out.dtype, h.dtype)
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.isfinite(h.grad).all())

    def test_adapter_matches_base_checkpoint_and_clean_branch_exactly(self):
        import torch
        from model.QuangV3 import Net
        from model.V3CMixStyle import CMixStyleNet

        base = Net(4, d=8, nb=2, L=1)
        adapter = CMixStyleNet(4, d=8, nb=2, L=1)
        adapter.load_state_dict(base.state_dict(), strict=True)
        batch = dict(x=torch.randn(4, 768), src=torch.arange(4), dst=torch.arange(4),
                     rel=torch.ones(4, dtype=torch.long), bidx=torch.tensor([0, 0, 1, 1]),
                     B=2, st=torch.zeros(4, 3, dtype=torch.long), y=torch.tensor([0., 0.]))
        torch.manual_seed(42)
        expected = base(batch)
        torch.manual_seed(42)
        actual = adapter(batch, mixstyle=False)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        base.eval(); adapter.eval()
        del batch["y"]
        torch.testing.assert_close(adapter(batch), base(batch), rtol=0, atol=0)

    def test_cpu_train_checkpoint_strict_load_and_inference(self):
        import numpy as np
        import pickle
        import torch
        from model.QuangV3 import GraphSet

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Six source graphs, both labels have multiple donor candidates.
            raw = dict(all_token_embeddings=[torch.randn(3, 768) for _ in range(6)],
                       all_edge_index=[torch.tensor([[0, 1, 2], [1, 1, 1]]) for _ in range(6)],
                       all_edge_type=[torch.tensor([2, 1, 2]) for _ in range(6)],
                       y=torch.tensor([0., 0., 0., 1., 1., 1.]))
            graph = root / "hc3_train.pkl"
            with graph.open("wb") as stream:
                pickle.dump(raw, stream)
            dataset = GraphSet(graph, R=3, mode="low", verbose=False)
            args = train_v3.build_parser().parse_args([
                "--variants", "cmixstyle", "--seeds", "2025", "--device", "cpu",
                "--epochs", "2", "--hidden", "8", "--num-bases", "2", "--layers", "1",
                "--batch-size", "6", "--micro-batch-size", "6", "--eval-batch-size", "6",
                "--mixstyle-p", "1", "--output-dir", str(root),
            ])
            train_v3.validate_args(args)
            train_v3.train_one(args, "cmixstyle", 2025, dataset, dataset, "fixture", {})
            checkpoint = root / "v3_cmixstyle_seed2025.pt"
            record = checkpoint.with_suffix(".train.json")
            self.assertTrue(train_v3.completed_run(checkpoint, record, "fixture"))
            metadata = json.loads(record.read_text())
            self.assertGreater(sum(r["mixstyle"]["applied"] for r in metadata["history"]), 0)
            eval_args = evaluator.build_parser().parse_args([
                "inspect", "--variant", "cmixstyle", "--checkpoint-dir", str(root),
                "--seeds", "2025", "--device", "cpu",
            ])
            config = evaluator.make_config(eval_args)
            report = evaluator.infer_checkpoint(config, 2025, graph, 6, "fixture")
            self.assertEqual(report["samples"], 6)
            self.assertTrue(np.isfinite(report["p_human"]).all())
            self.assertEqual(report["threshold"], 0.5)
            self.assertFalse(train_v3.completed_run(checkpoint, record, "different"))


if __name__ == "__main__":
    unittest.main()
