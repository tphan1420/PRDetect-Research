"""Offline runner tests; optional numerical checks when PyTorch is installed."""

import ast
import csv
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import evaluate_v3_quang as runner


def fixture_report(seed, accuracy, signature="sig", checkpoint_hash="hash"):
    metrics = {metric: accuracy for metric in runner.METRICS}
    return {
        **metrics, "seed": seed, "threshold": 0.5, "samples": 2,
        "run_signature": signature, "checkpoint_sha256": checkpoint_hash,
        "y_true": [0, 1], "p_human": [0.2, 0.8], "clipped_edges": 0,
        "validation_threshold": 0.3,
        "at_validation_threshold": {metric: min(accuracy + 0.05, 1) for metric in runner.METRICS},
    }


class ConfigurationTests(unittest.TestCase):
    def test_table3_is_only_hc3_to_gpt35_with_correct_rates(self):
        cases = runner.benchmark_cases("table3")
        self.assertEqual([c.noise_percent for c in cases], [0, 5, 10, 20, 30])
        self.assertEqual(cases[0].split, "gpt3.5_mixed_test_split")
        self.assertTrue(all("gpt35_mixed" in c.source for c in cases))
        self.assertTrue(all(c.suite == "table3" for c in cases))

    def test_table4_has_only_original_and_four_25_percent_noises(self):
        cases = runner.benchmark_cases("table4")
        self.assertEqual([c.split for c in cases], [
            "hc3_test", "hc3_test_insert_25", "hc3_test_repeat_25",
            "hc3_test_replace_25", "hc3_test_delete_25",
        ])
        self.assertEqual([c.noise_percent for c in cases], [0, 25, 25, 25, 25])
        for case in runner.benchmark_cases():
            self.assertTrue((runner.REPO / case.source).is_file(), case.source)
        with self.assertRaises(ValueError):
            runner.benchmark_cases("reverse")

    def test_cli_defaults_and_positive_sizes(self):
        args = runner.build_parser().parse_args(["run"])
        self.assertEqual(args.seeds, [2021, 2022, 2023, 2024, 2025])
        self.assertEqual(args.batch_size, 4)
        self.assertEqual(args.max_graphs, 4)
        self.assertEqual(args.struct_mode, "low")
        with self.assertRaises(Exception):
            runner.positive_int("0")

    def test_help_does_not_require_torch(self):
        result = subprocess.run(
            [sys.executable, str(runner.REPO / "evaluate_v3_quang.py"), "run", "--help"],
            capture_output=True, text=True, check=True,
        )
        self.assertIn("--checkpoint-dir", result.stdout)

    def test_zip_checkpoints_are_discovered_without_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "p3fin_V1_swap_s2025_seed2025.zip"
            path.write_bytes(b"checkpoint fixture; discovery must not unzip it")
            copy = root / "copy"
            copy.mkdir()
            (copy / path.name).write_bytes(path.read_bytes())
            found = runner.find_checkpoints(root, [2025])
            self.assertEqual(runner.sha256(found["2025"]), runner.sha256(path))
            self.assertEqual(len(list(root.rglob("*"))), 3)
            with self.assertRaises(FileNotFoundError):
                runner.find_checkpoints(root, [2026])
            with self.assertRaises(ValueError):
                runner.find_checkpoints(root, [2025, 2025])
            (copy / path.name).write_bytes(b"different checkpoint")
            with self.assertRaises(ValueError):
                runner.find_checkpoints(root, [2025])

    def test_report_identity_and_incomplete_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            runner.write_json(path, fixture_report(2025, 0.8))
            self.assertTrue(runner.cached_report_matches(path, "sig", 2025, "hash"))
            self.assertFalse(runner.cached_report_matches(path, "other", 2025, "hash"))
            self.assertFalse(runner.cached_report_matches(path, "sig", 2026, "hash"))
            incomplete = fixture_report(2025, 0.8)
            del incomplete["at_validation_threshold"]
            runner.write_json(path, incomplete)
            self.assertFalse(runner.cached_report_matches(path, "sig", 2025, "hash"))
            path.write_text("{", encoding="utf-8")
            self.assertFalse(runner.cached_report_matches(path, "sig", 2025, "hash"))

    def test_archive_contains_only_selected_seed_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory) / "result"
            results.mkdir()
            case = runner.benchmark_cases("table3")[0]
            for seed in [2025, 2026]:
                runner.write_json(runner.report_path(results, case, seed), fixture_report(seed, 0.8))
            for name in ["per_seed.csv", "summary_threshold_0.5.csv", "summary_validation_threshold.csv",
                         "checkpoint_metadata.json", "manifest.json"]:
                (results / name).write_text("fixture", encoding="utf-8")
            archive = Path(directory) / "results.zip"
            runner.export_archive(results, archive, [case], [2025])
            with ZipFile(archive) as stream:
                names = stream.namelist()
            self.assertTrue(any("table3_2025_" in name for name in names))
            self.assertFalse(any("2026" in name for name in names))

    def test_table4_average_excludes_original_and_uses_seed_averages(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            cases = runner.benchmark_cases("table4")
            seeds = [2025, 2026]
            for case in cases:
                for seed, accuracy in zip(seeds, [0.6, 0.8]):
                    if case.noise_percent == 0:
                        accuracy = 0.99
                    runner.write_json(runner.report_path(results, case, seed),
                                      fixture_report(seed, accuracy))
            runner.summarize(cases, results, seeds)
            with (results / "summary_threshold_0.5.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            average = rows[-1]
            self.assertEqual(average["condition"], "Average (4 noise sets)")
            self.assertAlmostEqual(float(average["accuracy_mean"]), 0.7)
            self.assertAlmostEqual(float(average["accuracy_std"]), statistics.stdev([0.6, 0.8]))
            with (results / "summary_validation_threshold.csv").open() as stream:
                val_rows = list(csv.DictReader(stream))
            self.assertAlmostEqual(float(val_rows[-1]["accuracy_mean"]), 0.75)
            # Test probabilities, not a seed ensemble, are aggregated as metrics.
            self.assertNotIn("ensemble", average)

    def test_single_seed_std_is_blank_and_table3_has_no_table4_average(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            cases = runner.benchmark_cases("table3")
            for case in cases:
                runner.write_json(runner.report_path(results, case, 2025), fixture_report(2025, 0.7))
            runner.summarize(cases, results, [2025])
            with (results / "summary_threshold_0.5.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 5)
            self.assertTrue(all(row["accuracy_std"] == "" for row in rows))

    def test_unpaired_labels_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            case = runner.benchmark_cases("table3")[0]
            for seed in [2025, 2026]:
                report = fixture_report(seed, 0.8)
                if seed == 2026:
                    report["y_true"] = [1, 0]
                runner.write_json(runner.report_path(results, case, seed), report)
            with self.assertRaises(ValueError):
                runner.summarize([case], results, [2025, 2026])

    def test_graph_limit_does_not_delete_existing_caches_or_start_builder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.json"
            source.write_text('{}\n{}\n', encoding="utf-8")
            work = root / "work"
            graph = work / "graph_data"
            graph.mkdir(parents=True)
            for i in range(4):
                (graph / f"old{i}.pkl").write_bytes(b"fixture")
            config = {"seeds": [2025], "checkpoint_sha256": {"2025": "hash"},
                      "code_sha256": {}, "environment": {}}
            case = runner.Case("table3", "Original", 0, "input.json")
            with patch.object(runner, "REPO", root), patch.object(runner.subprocess, "run") as child:
                with self.assertRaisesRegex(ValueError, "Graph limit"):
                    runner.evaluate_case(case, config, root / "config.json", work, 4)
                child.assert_not_called()
            self.assertEqual(len(list(graph.glob("*.pkl"))), 4)

    def test_same_report_is_reused_without_building_graph(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.json").write_text('{}\n{}\n', encoding="utf-8")
            work = root / "work"
            config = {"seeds": [2025], "checkpoint_sha256": {"2025": "hash"}}
            case = runner.Case("table3", "Original", 0, "input.json")
            with patch.object(runner, "REPO", root):
                signature = runner.case_signature(config, case, 2)
                report = fixture_report(2025, 0.8, signature)
                runner.write_json(runner.report_path(work / "result", case, 2025), report)
                with patch.object(runner.subprocess, "run") as child:
                    runner.evaluate_case(case, config, root / "config.json", work, 4)
                    child.assert_not_called()

    def test_ported_computation_matches_quang_source_when_reference_available(self):
        reference = runner.REPO / "docs/PRDetect_RGCN_v3_tong_hop.md"
        if not reference.is_file():
            self.skipTest("Optional original Markdown is not needed to run the evaluator")
        text = reference.read_text(encoding="utf-8-sig")
        original = text.split("%%writefile rgcn_plus.py", 1)[1].lstrip("\r\n").split("\n```", 1)[0]
        ref = {n.name: n for n in ast.parse(original).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        port = ast.parse(runner.MODEL_SOURCE.read_text(encoding="utf-8"))
        for node in port.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                self.assertEqual(ast.dump(node), ast.dump(ref[node.name]), node.name)


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("numpy"),
                     "PyTorch/NumPy numerical runtime unavailable")
class NumericalTests(unittest.TestCase):
    def test_layer_matches_relation_normalized_reference(self):
        import torch
        from model.QuangV3 import RGCNLayer

        torch.manual_seed(0)
        n, d, relations, bases, edges = 8, 4, 6, 2, 20
        layer = RGCNLayer(d, relations, bases)
        x = torch.randn(n, d)
        src, dst = torch.randint(n, (edges,)), torch.randint(n, (edges,))
        rel = torch.randint(relations, (edges,))
        counts = torch.bincount(dst * relations + rel, minlength=n * relations).float()
        norm = 1 / counts[dst * relations + rel]
        weights = torch.einsum("rb,bio->rio", layer.comp, layer.V)
        expected = layer.W0(x).clone()
        for i in range(edges):
            expected[dst[i]] += norm[i] * (x[src[i]] @ weights[rel[i]])
        torch.testing.assert_close(layer(x, src, dst, rel, norm), expected)


if __name__ == "__main__":
    unittest.main()
