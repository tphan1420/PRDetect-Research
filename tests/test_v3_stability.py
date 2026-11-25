import csv
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from calibrate_v3 import union_pools
from evaluate_v3_quang import METRICS, benchmark_cases, human95_threshold, report_path, summarize, write_json
from v3_a1_fusion import human_score, paired_scores, source_digest


class StabilityTests(unittest.TestCase):
    def test_training_saves_threshold_for_the_actual_selected_weights(self):
        from calibrate_v3 import load_v3_checkpoint
        from model.QuangV3 import GraphSet, predict
        from train_v3 import build_parser, train_one

        with TemporaryDirectory() as temp:
            root = Path(temp)
            graph = root / "tiny.pkl"
            rng = np.random.default_rng(42)
            payload = {"all_token_embeddings": [rng.normal(size=(3, 768)).astype("f") for _ in range(4)],
                       "all_edge_index": [np.array([[0, 1, 2], [0, 0, 1]])] * 4,
                       "all_edge_type": [np.array([0, 1, 2])] * 4, "y": [0, 1, 0, 1]}
            graph.write_bytes(pickle.dumps(payload))
            data = GraphSet(graph, R=3, mode="low", verbose=False)
            args = build_parser().parse_args(["--device", "cpu", "--epochs", "1", "--seeds", "5",
                                              "--hidden", "8", "--num-bases", "2", "--layers", "1",
                                              "--output-dir", temp, "--batch-size", "2", "--val-batch-size", "2"])
            train_one(args, "baseline", 5, data, data, "unit", {})
            model, checkpoint, _ = load_v3_checkpoint(root / "v3_baseline_seed5.pt")
            probs = predict(model, data, torch.device("cpu"), checkpoint["gm"], checkpoint["gs"], bs=2)
            self.assertEqual(checkpoint["human95_threshold"], human95_threshold(data.y.numpy(), probs))

    def test_source_threshold_uses_only_humans_and_includes_ties(self):
        human = np.arange(101, dtype=float)
        labels = [0, 0] + [1] * len(human)
        self.assertEqual(human95_threshold(labels, [-100, 1000, *human]), 5)
        self.assertGreaterEqual(np.mean(human >= 5), .95)
        self.assertEqual(human95_threshold([0, 1, 1], [100, 2, 2]), 2)
        with self.assertRaises(ValueError):
            human95_threshold([0], [.1])

    def test_summary_averages_noise_conditions_within_each_seed(self):
        cases = [c for c in benchmark_cases("table4") if c.noise_percent]
        with TemporaryDirectory() as temp:
            root = Path(temp)
            for i, case in enumerate(cases):
                for seed in (1, 2):
                    baseline = dict.fromkeys(METRICS, .5)
                    calibrated = dict.fromkeys(METRICS, .1 * (seed + i))
                    write_json(report_path(root, case, seed), {
                        **baseline, "seed": seed, "y_true": [0, 1], "clipped_edges": 0, "threshold": .5,
                        "at_validation_threshold": baseline, "validation_threshold": .4,
                        "at_source_human95": calibrated, "source_human95_threshold": .9})
            tables = summarize(cases, root, [1, 2])
            self.assertIn("summary_source_human95.csv", tables)
            with (root / "summary_source_human95.csv").open(encoding="utf-8", newline="") as stream:
                row = list(csv.DictReader(stream))[-1]
            self.assertEqual(row["condition"], "Average (4 noise sets)")
            self.assertAlmostEqual(float(row["accuracy_mean"]), .30)
            self.assertAlmostEqual(float(row["accuracy_std"]), np.std([.25, .35], ddof=1))

    def test_source_graph_union_matches_joint_readout(self):
        nodes = [torch.tensor([[1., 3.], [2., -1.]]), torch.tensor([[-2., 4.]])]

        def pooled(h):
            gates = h[:, 0]
            return torch.cat((h.mean(0), h.amax(0), (gates.softmax(0)[:, None] * h).sum(0)))

        summaries = torch.stack([pooled(h) for h in nodes])
        masses = torch.stack([h[:, 0].logsumexp(0) for h in nodes])
        joined = union_pools(summaries, masses, torch.tensor([2, 1]), torch.tensor([[0, 1]]))
        torch.testing.assert_close(joined[0], pooled(torch.cat(nodes)))

    def test_fusion_is_per_document_and_rejects_mispaired_archives(self):
        with TemporaryDirectory() as root:
            path = Path(root) / "split.json"
            path.write_bytes(b'{"text":"one","label":0}\r\n{"text":"two","label":1}\r\n')
            labels = np.array([0, 1])
            a1 = {"condition": "split", "source_sha256": source_digest(path), "score_space": "human_logit",
                  "records": [{"index": 0, "label": 0, "logit_human": 0.0},
                              {"index": 1, "label": 1, "logit_human": 2.0}]}
            v3 = {"split": "split", "train": "hc3", "y_true": [0, 1], "p_human": [.8, .9],
                  "samples": 2, "label_convention": {"0": "ai", "1": "human"}}
            p = paired_scores(path, labels, v3, a1)
            self.assertAlmostEqual(p[0], .4)
            self.assertAlmostEqual(p[1], float(human_score(.9, 2)))
            a1["records"].reverse()
            with self.assertRaises(ValueError):
                paired_scores(path, labels, v3, a1)
