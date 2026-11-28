"""Check mask budgets, single-encoder training, and checkpoint entry points."""

import json
import pickle
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import torch
import torch.nn.functional as F

import evaluate_v3_quang as ev
from model.EvidenceErasure import candidate_masks, erased_predictions
from model.QuangV3 import GraphSet
from train_v3 import build_model, build_parser, training_forward, validate_args
from train_v3_dual_best import train_one


def make_dataset(root):
    rng = np.random.default_rng(19)
    edges = np.array([[0, 1, 2, 3, 4, 5], [0, 0, 1, 2, 3, 4]])
    payload = dict(all_token_embeddings=[rng.normal(size=(6, 768)).astype("f") for _ in range(4)],
                   all_edge_index=[edges] * 4, all_edge_type=[np.arange(6) % 3] * 4, y=[0, 1, 0, 1])
    path = root / "tiny.pkl"
    path.write_bytes(pickle.dumps(payload))
    return path, GraphSet(path, R=3, mode="low", verbose=False)


def arguments(root, mode="worst"):
    return build_parser().parse_args([
        "--readout", "evidence", "--erasure-mode", mode, "--erasure-fraction", "0.5",
        "--struct", "0", "--aug", "none", "--device", "cpu", "--epochs", "1",
        "--seeds", "2026", "--hidden", "8", "--num-bases", "2", "--layers", "2",
        "--batch-size", "2", "--micro-batch-size", "1", "--val-batch-size", "2",
        "--output-dir", str(root)])


class EvidenceErasureTests(unittest.TestCase):
    def test_regions_are_complete_balls_within_budget(self):
        src, dst = torch.arange(9), torch.arange(1, 10)
        masks = candidate_masks(10, src, dst, fraction=.4, candidates=10, hops=2)
        self.assertTrue(torch.all(masks.sum(1) <= 4))
        self.assertEqual(torch.where(masks[0])[0].tolist(), [0, 1, 2])
        self.assertEqual(torch.where(masks[5])[0].tolist(), [4, 5, 6])
        for mask in masks:
            nodes = torch.where(mask)[0]
            self.assertEqual(int(nodes[-1] - nodes[0] + 1), len(nodes))
        self.assertEqual(candidate_masks(1, torch.tensor([0]), torch.tensor([0]),
                                        fraction=.2, candidates=32, hops=2).shape, (0, 1))

    def test_worst_region_uses_label_and_gradients_only_for_retained_nodes(self):
        values = torch.tensor([9., 3., -2., -6.], requires_grad=True)
        batch = dict(B=1, bidx=torch.zeros(4, dtype=torch.long), y=torch.tensor([1.]),
                     src=torch.empty(0, dtype=torch.long), dst=torch.empty(0, dtype=torch.long))
        pred, fraction, active = erased_predictions(values, batch, mode="worst",
                                                    fraction=.25, candidates=4, hops=2)
        torch.testing.assert_close(pred, torch.tensor([-5 / 3]))
        self.assertEqual(fraction.item(), .25)
        self.assertTrue(active.item())
        pred.sum().backward()
        torch.testing.assert_close(values.grad, torch.tensor([0., 1/3, 1/3, 1/3]))
        batch["y"] = torch.tensor([0.])
        pred, _, _ = erased_predictions(values, batch, mode="worst", fraction=.25, candidates=4, hops=2)
        torch.testing.assert_close(pred, torch.tensor([10 / 3]))

    def test_random_and_worst_use_same_candidates_and_short_graph_is_safe(self):
        values = torch.tensor([9., 3., -2., -6.])
        batch = dict(B=1, bidx=torch.zeros(4, dtype=torch.long), y=torch.tensor([1.]),
                     src=torch.empty(0, dtype=torch.long), dst=torch.empty(0, dtype=torch.long))
        worst, _, _ = erased_predictions(values, batch, mode="worst", fraction=.25, candidates=4, hops=2)
        torch.manual_seed(12)
        random, fraction, active = erased_predictions(
            values, batch, mode="random", fraction=.25, candidates=4, hops=2)
        self.assertGreaterEqual(F.binary_cross_entropy_with_logits(worst, batch["y"]).item(),
                                F.binary_cross_entropy_with_logits(random, batch["y"]).item())
        self.assertEqual(fraction.item(), .25)
        plain, fraction, active = erased_predictions(
            values, batch, mode="worst", fraction=0, candidates=4, hops=2)
        torch.testing.assert_close(plain, values.mean().reshape(1))
        self.assertEqual(fraction.item(), 0)
        self.assertFalse(active.item())

    def test_single_encoder_backward_and_inference_independent_of_labels_and_batch(self):
        with TemporaryDirectory() as tmp:
            _, ds = make_dataset(Path(tmp))
            args = arguments(Path(tmp))
            model = build_model(args, "baseline", ds.R)
            batch = ds.batch(torch.tensor([0, 1]), "cpu")
            with patch.object(model, "encode_nodes", wraps=model.encode_nodes) as encode:
                logits, loss, stats = training_forward(model, batch, args)
                self.assertEqual(encode.call_count, 1)
            torch.testing.assert_close(loss.detach(), stats["clean_bce"] + .5 * stats["erased_bce"])
            self.assertGreater(stats["erased_fraction"].item(), 0)
            loss.backward()
            for param in (model.layers[0].V, model.evidence_head[0].weight):
                self.assertTrue(torch.isfinite(param.grad).all())
                self.assertGreater(param.grad.abs().sum().item(), 0)
            model.eval()
            with torch.no_grad():
                joint = model(batch)
                independent = torch.cat([model(ds.batch(torch.tensor([i]), "cpu")) for i in (0, 1)])
                torch.testing.assert_close(joint, independent, atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(joint, model({k: v for k, v in batch.items() if k != "y"}))
                torch.testing.assert_close(joint.flip(0), model(ds.batch(torch.tensor([1, 0]), "cpu")))

    def test_all_training_modes_export_raw_ema_and_load_through_evaluator(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph, ds = make_dataset(root)
            for mode in ("none", "random", "worst"):
                out = root / mode
                out.mkdir()
                args = arguments(out, mode)
                validate_args(args)
                train_one(args, "baseline", 2026, ds, ds, mode, {})
                record = json.loads((out / "v3_baseline_dual_seed2026.train.json").read_text())
                self.assertTrue(record["complete"])
                stats = record["history"][0]["evidence"]
                self.assertEqual(stats["erased_fraction"] > 0, mode != "none")
                for weights in ("ema", "raw"):
                    config = ev.make_config(SimpleNamespace(
                        variant="baseline", weights=weights, train_domain="hc3", checkpoint_dir=out,
                        seeds=[2026], device="cpu", batch_size=2, struct_mode="low", graph_budget_root=None))
                    report = ev.infer_checkpoint(config, 2026, graph, 4, "unit")
                    self.assertEqual(report["readout"], "evidence")
                    self.assertEqual(report["erasure_mode"], mode)
                    self.assertTrue(np.isfinite(report["p_human"]).all())
                    single = ev.infer_checkpoint({**config, "batch_size": 1}, 2026, graph, 4, "unit")
                    np.testing.assert_allclose(report["p_human"], single["p_human"], atol=1e-6)

    def test_reject_invalid_settings_and_preserve_baseline_defaults(self):
        for options in (["--erasure-mode", "worst"], ["--erasure-fraction", "nan"],
                        ["--erasure-fraction", "1"], ["--erasure-weight", "-1"],
                        ["--readout", "evidence", "--gfeat", "1"]):
            with self.assertRaises(ValueError):
                validate_args(build_parser().parse_args(options))
        args = build_parser().parse_args([])
        self.assertEqual(args.readout, "pool3")
        self.assertEqual(args.erasure_mode, "none")

    def test_v4_reuses_a1_and_packages_training_config_and_both_weights(self):
        import train_v4
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            a1 = root / "a1_seed2026.pt"
            a1.write_bytes(b"existing A1")
            def train_command(command, **kwargs):
                args = build_parser().parse_args(command[2:])
                validate_args(args)
                self.assertEqual((args.readout, args.erasure_mode), ("evidence", "worst"))
                self.assertEqual((args.struct, args.aug), (0, "none"))
                ema = args.output_dir / "v3_baseline_ema_seed2026.pt"
                raw = args.output_dir / "v3_baseline_raw_seed2026.pt"
                ema.write_bytes(b"EMA")
                raw.write_bytes(b"RAW")
                ev.write_json(args.output_dir / "v3_baseline_dual_seed2026.train.json",
                              dict(complete=True, checkpoint_sha256={"ema": ev.sha256(ema)},
                                   provenance={"config": {k: v for k, v in vars(args).items()
                                                          if not isinstance(v, Path)}}))
            command = ["train_v4.py", "--seed", "2026", "--output-dir", str(root / "out"),
                       "--feature-cache", str(root / "cache"), "--v3-readout", "evidence",
                       "--erasure-mode", "worst", "--a1-checkpoint", str(a1)]
            with patch("sys.argv", command), patch.object(train_v4.torch.cuda, "is_available", return_value=True), \
                 patch.object(train_v4.AutoConfig, "from_pretrained", return_value=SimpleNamespace(_commit_hash="test")), \
                 patch.object(train_v4.AlignedScorer, "load_a1"), \
                 patch.object(train_v4, "prepare_graphs", return_value=root) as graphs, \
                 patch.object(train_v4, "fit_a1") as fit, \
                 patch.object(train_v4.subprocess, "run", side_effect=train_command):
                train_v4.main()
            self.assertFalse(graphs.call_args.kwargs["prepare_a1"])
            fit.assert_not_called()
            dest = root / "out/seed2026"
            self.assertEqual((dest / a1.name).read_bytes(), a1.read_bytes())
            with ZipFile(dest / "v4_evidence_worst_hc3_seed2026.zip") as z:
                self.assertTrue({"v4_train.json", "v3_baseline_dual_seed2026.train.json",
                                 "v3_baseline_raw_seed2026.pt", "v3_baseline_ema_seed2026.pt",
                                 a1.name, "manifest.json"} <= set(z.namelist()))


if __name__ == "__main__":
    unittest.main()
