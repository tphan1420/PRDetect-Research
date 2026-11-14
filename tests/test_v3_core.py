"""Offline contract checks for the standalone V3 Core runner.

Numerical tests are skipped when the project environment does not have the
optional PyTorch/Numpy stack installed.  They are intentionally small: their
purpose is to catch wiring regressions, not to alter research evaluation.
"""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import evaluate_v3_quang as evaluator
import train_v3


ROOT = Path(__file__).resolve().parents[1]


class V3CoreCliTests(unittest.TestCase):
    def test_training_defaults_are_the_core_baseline(self):
        args = train_v3.build_parser().parse_args([])
        self.assertEqual(args.variants, ["baseline"])
        self.assertEqual(args.readout, "pool3")
        self.assertEqual(args.struct_mode, "low")

    def test_training_rejects_removed_variants(self):
        with self.assertRaises(SystemExit):
            train_v3.build_parser().parse_args(["--variants", "meanlogit"])

    def test_evaluation_exposes_only_core_variants_and_modes(self):
        args = evaluator.build_parser().parse_args(["inspect"])
        self.assertEqual(args.variant, "quang")
        self.assertEqual(args.struct_mode, "low")
        with self.assertRaises(SystemExit):
            evaluator.build_parser().parse_args(["inspect", "--variant", "cmixstyle"])
        with self.assertRaises(SystemExit):
            evaluator.build_parser().parse_args(["inspect", "--struct-mode", "directed"])

    def test_table_suites_remain_available(self):
        self.assertEqual(len(evaluator.benchmark_cases("table2")), 5)
        self.assertEqual(len(evaluator.benchmark_cases("table3")), 5)
        self.assertEqual(len(evaluator.benchmark_cases("table4")), 5)

    def test_checkpoint_discovery_accepts_legacy_quang_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in (2021, 2022):
                (root / f"p3fin_V1_swap_s{seed}_seed{seed}.pt").write_bytes(b"core")
            paths = evaluator.find_checkpoints(root, [2021, 2022], "quang")
            self.assertEqual(set(paths), {"2021", "2022"})

    def test_core_source_has_no_removed_experiment_hooks(self):
        sources = [ROOT / "model/QuangV3.py", ROOT / "train_v3.py", ROOT / "evaluate_v3_quang.py"]
        removed = ("meanlogit", "cmixstyle", "classpreserving", "restricted", "structuralmap")
        combined = "\n".join(path.read_text(encoding="utf-8").lower() for path in sources)
        self.assertFalse(any(token in combined for token in removed))


@unittest.skipUnless(
    importlib.util.find_spec("torch") is not None and importlib.util.find_spec("numpy") is not None,
    "requires the optional torch and numpy packages",
)
class V3CoreNumericalTests(unittest.TestCase):
    def test_forward_backward_optimizer_and_syntax_swap(self):
        import torch

        from model.QuangV3 import Net

        torch.manual_seed(7)
        model = Net(4, d=8, nb=2, L=1, p=0.1, p_in=0.0, aug="swap", aug_max=1.0)
        batch = {
            "x": torch.randn(4, 768),
            "src": torch.tensor([0, 1, 2, 3]),
            "dst": torch.tensor([1, 0, 3, 2]),
            "rel": torch.tensor([0, 0, 1, 1]),
            "bidx": torch.tensor([0, 0, 1, 1]),
            "B": 2,
            "st": torch.tensor([[0, 0, 0], [1, 1, 1], [0, 0, 0], [1, 1, 1]]),
            "gf": torch.zeros(2, 10),
        }
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        output = model(batch, aug_mode="swap", gen=torch.Generator().manual_seed(8))
        self.assertEqual(tuple(output.shape), (2,))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(output, torch.tensor([0.0, 1.0]))
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    unittest.main()
