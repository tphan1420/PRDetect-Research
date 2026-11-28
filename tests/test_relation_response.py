"""Checks for graph pairing, shared stochastic masks and checkpoint reload."""

import copy
import pickle
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import numpy as np

import evaluate_v3_quang as evaluator
from model.RelationResponse import RelationResponseNet
from train_v3 import build_model, build_parser, validate_args


def graph():
    return dict(x=torch.randn(6, 768), src=torch.arange(6),
                dst=torch.tensor([0, 0, 1, 1, 2, 3]),
                rel=torch.tensor([0, 1, 2, 3, 1, 2]),
                bidx=torch.zeros(6, dtype=torch.long), B=1)


def combine(left, right):
    offset = len(left["x"])
    return dict(x=torch.cat((left["x"], right["x"])),
                src=torch.cat((left["src"], right["src"] + offset)),
                dst=torch.cat((left["dst"], right["dst"] + offset)),
                rel=torch.cat((left["rel"], right["rel"])),
                bidx=torch.cat((left["bidx"], right["bidx"] + 1)), B=2)


class RelationResponseTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = RelationResponseNet(4, d=8, nb=2, L=2, use_struct=False,
                                         p=0.3, p_in=0.2, edge_drop=0.3)

    def test_shuffle_preserves_document_histograms_and_roots(self):
        first, second = graph(), graph()
        second["rel"] = torch.tensor([0, 3, 3, 1, 3, 1])
        batch = combine(first, second)
        before = copy.deepcopy(batch)
        for training in (True, False):
            self.model.train(training)
            shuffled = self.model.permuted_relations(batch)
            for start in (0, 6):
                self.assertEqual(shuffled[start].item(), batch["rel"][start].item())
                torch.testing.assert_close(shuffled[start:start+6].sort().values,
                                           batch["rel"][start:start+6].sort().values)
        for key in ("x", "src", "dst", "rel"):
            torch.testing.assert_close(batch[key], before[key])

    def test_evaluation_is_independent_of_batch_order_and_global_rng(self):
        self.model.eval()
        first, second = graph(), graph()
        individual = torch.cat((self.model(first), self.model(second)))
        torch.rand(100)
        torch.testing.assert_close(self.model(combine(first, second)), individual)
        torch.testing.assert_close(self.model(combine(second, first)), individual.flip(0))
        perms = self.model.permuted_relations(combine(first, second))
        torch.testing.assert_close(perms[:6], self.model.permuted_relations(first))
        torch.testing.assert_close(perms[6:], self.model.permuted_relations(second))

    def test_identity_control_has_zero_difference_despite_dropout(self):
        self.model.train()
        batch = graph()
        captured = []
        handle = self.model.mlp.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
        with patch.object(self.model, "permuted_relations", return_value=batch["rel"]):
            self.model(batch)
        handle.remove()
        difference = captured[0][:, 3 * self.model.d:]
        torch.testing.assert_close(difference, torch.zeros_like(difference), rtol=0, atol=0)

    def test_gradients_reach_shared_encoder_and_both_readout_inputs(self):
        self.model.eval()
        batch = graph()
        control = batch["rel"].clone()
        control[1:] = control[1:].roll(1)
        with patch.object(self.model, "permuted_relations", return_value=control):
            self.model(batch).square().sum().backward()
        self.assertGreater(self.model.layers[0].comp.grad.abs().sum().item(), 0)
        self.assertGreater(self.model.mlp[0].weight.grad[:, 3*self.model.d:].abs().sum().item(), 0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None))

    def test_single_node_and_single_relation_graphs(self):
        self.model.eval()
        batch = dict(x=torch.randn(1, 768), src=torch.tensor([0]), dst=torch.tensor([0]),
                     rel=torch.tensor([0]), bidx=torch.tensor([0]), B=1)
        self.assertTrue(torch.isfinite(self.model(batch)).all())
        batch["src"] = batch["dst"] = batch["rel"] = torch.empty(0, dtype=torch.long)
        self.assertTrue(torch.isfinite(self.model(batch)).all())
        batch = graph()
        batch["rel"].fill_(0)
        torch.testing.assert_close(self.model.permuted_relations(batch), batch["rel"])

    def test_checkpoint_loader_selects_architecture_and_supports_old_args(self):
        args = build_parser().parse_args([
            "--struct", "0", "--aug", "none", "--relation-response",
            "--hidden", "8", "--num-bases", "2", "--seeds", "2026",
        ])
        validate_args(args)
        with TemporaryDirectory() as tmp:
            for enabled in (False, True):
                args.relation_response = enabled
                model = build_model(args, "baseline", 4).eval()
                saved = vars(args).copy()
                for key, value in saved.items():
                    if isinstance(value, Path):
                        saved[key] = str(value)
                if not enabled:
                    del saved["relation_response"]  # legacy checkpoints lack this field
                path = Path(tmp) / "v3_baseline_ema_seed2026.pt"
                torch.save(dict(state=model.state_dict(), args=saved, R=4, gm=torch.zeros(10),
                                gs=torch.ones(10), thr=0.5, weight_source="ema",
                                method="v3_baseline", training_complete=True, struct_mode="low"), path)
                config = evaluator.make_config(SimpleNamespace(
                    variant="baseline", weights="ema", train_domain="hc3", checkpoint_dir=Path(tmp),
                    seeds=[2026], device="cpu", batch_size=1, struct_mode="low", graph_budget_root=None))
                _, _, loaded = evaluator._load_model(config, 2026, torch.device("cpu"))
                batch = graph()
                torch.testing.assert_close(loaded(batch), model(batch), rtol=0, atol=0)
                self.assertEqual(isinstance(loaded, RelationResponseNet), enabled)

    def test_dual_training_and_normal_inference_entry_point(self):
        from model.QuangV3 import GraphSet
        from train_v3_dual_best import train_one

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "hc3_test.pkl"
            rng = np.random.default_rng(42)
            payload = dict(
                all_token_embeddings=[rng.normal(size=(5, 768)).astype("f") for _ in range(4)],
                all_edge_index=[np.array([[0, 1, 2, 3, 4], [0, 0, 1, 1, 3]])] * 4,
                all_edge_type=[np.array([0, 1, 2, 3, 1])] * 4,
                y=[0, 1, 0, 1],
            )
            path.write_bytes(pickle.dumps(payload))
            data = GraphSet(path, R=4, mode="low", verbose=False)
            args = build_parser().parse_args([
                "--device", "cpu", "--epochs", "1", "--seeds", "2026",
                "--hidden", "8", "--num-bases", "2", "--layers", "1",
                "--struct", "0", "--aug", "none", "--relation-response",
                "--output-dir", tmp, "--batch-size", "2", "--micro-batch-size", "1",
                "--val-batch-size", "2",
            ])
            validate_args(args)
            train_one(args, "baseline", 2026, data, data, "test", {})
            for weights in ("raw", "ema"):
                config = evaluator.make_config(SimpleNamespace(
                    variant="baseline", weights=weights, train_domain="hc3",
                    checkpoint_dir=Path(tmp), seeds=[2026], device="cpu",
                    batch_size=2, struct_mode="low", graph_budget_root=None))
                report = evaluator.infer_checkpoint(config, 2026, path, 4, "test")
                self.assertTrue(report["relation_response"])
                self.assertEqual(report["y_true"], payload["y"])
                config["batch_size"] = 1
                individual = evaluator.infer_checkpoint(config, 2026, path, 4, "test")
                np.testing.assert_allclose(individual["p_human"], report["p_human"], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
