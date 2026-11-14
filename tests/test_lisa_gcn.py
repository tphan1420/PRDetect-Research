import importlib.util
import json
import math
from argparse import Namespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import weakref

from lisa_gcn.windowing import SentenceSpan, plan_sentence_windows


class WindowPlannerTests(unittest.TestCase):
    def test_preserves_complete_sentences_when_possible(self):
        windows = plan_sentence_windows(10, [SentenceSpan(0, 3), SentenceSpan(3, 7), SentenceSpan(7, 10)], 7)
        self.assertEqual(windows[0].start, 0)
        self.assertEqual(windows[0].end, 7)
        self.assertEqual(windows[1].start, 7)
        self.assertEqual(windows[1].end, 10)
        self.assertFalse(any(window.split_sentence for window in windows))

    def test_long_sentence_is_marked_and_exactly_covered(self):
        windows = plan_sentence_windows(9, [SentenceSpan(0, 9)], 4)
        self.assertEqual([(item.start, item.end) for item in windows], [(0, 4), (4, 8), (8, 9)])
        self.assertTrue(all(item.split_sentence for item in windows))

    def test_rejects_unverified_gaps(self):
        with self.assertRaises(ValueError):
            plan_sentence_windows(5, [SentenceSpan(0, 2), SentenceSpan(3, 5)], 4)


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
class CheckpointTests(unittest.TestCase):
    def test_checkpoint_loader_accepts_legacy_path_metadata(self):
        import torch
        from lisa_gcn.checkpoint import checkpoint_config, load_lisa_checkpoint

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save({"config": {"output": Path("model/lisa_gcn")}, "tensor": torch.tensor([1.0])}, path)
            state = load_lisa_checkpoint(path, map_location="cpu")
        self.assertEqual(Path(state["config"]["output"]), Path("model/lisa_gcn"))
        self.assertEqual(checkpoint_config(Namespace(output=Path("model/lisa_gcn")))["output"], str(Path("model/lisa_gcn")))


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
class GroupDRORegressionTests(unittest.TestCase):
    def test_single_present_group_keeps_full_loss_and_gradient(self):
        import torch
        from lisa_gcn.objective import GroupDRO

        for gap in (20.0, 1000.0):
            with self.subTest(gap=gap):
                dro = GroupDRO(groups=4)
                dro.log_q = torch.tensor([0.0, -gap, -gap, -gap])
                losses = torch.tensor([2.0, 4.0], requires_grad=True)
                objective, _, _ = dro.loss(losses, torch.tensor([1, 1]))
                torch.testing.assert_close(objective, torch.tensor(3.0))
                objective.backward()
                torch.testing.assert_close(losses.grad, torch.tensor([0.5, 0.5]))

    def test_multiple_present_groups_normalize_without_absent_dominant_group(self):
        import torch
        from lisa_gcn.objective import GroupDRO

        dro = GroupDRO(groups=3)
        dro.log_q = torch.tensor([0.0, -1000.0, -1002.0])
        losses = torch.tensor([2.0, 4.0], requires_grad=True)
        objective, _, present = dro.loss(losses, torch.tensor([1, 2]))
        expected_weights = torch.softmax(dro.log_q[present], dim=0)
        torch.testing.assert_close(objective, (expected_weights * losses).sum())
        objective.backward()
        torch.testing.assert_close(losses.grad, expected_weights)


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("torch_geometric"), "requires torch and torch_geometric")
class LisaNumericalTests(unittest.TestCase):
    @staticmethod
    def make_document(text, label, document_index, window_sizes=(96, 128, 160)):
        import torch
        from torch_geometric.data import Data
        from lisa_gcn.data import LisaDocument

        graphs = {}
        for scale in window_sizes:
            graphs[scale] = [Data(
                x=torch.arange(12, dtype=torch.float32).reshape(4, 3) / scale + document_index + offset,
                edge_index=torch.tensor([[0, 1, 2, 3], [0, 0, 1, 2]]),
                y=torch.tensor([label]),
                document_index=torch.tensor([document_index]),
                token_count=torch.tensor([4]),
                split_sentence=torch.tensor([0]),
            ) for offset in (0, 1)]
        return LisaDocument(label, 8, graphs)

    def test_disk_cache_preserves_graphs_logits_and_gradients(self):
        import torch
        from lisa_gcn.data import build_disk_documents
        from lisa_gcn.model import LisaGCN
        from train_lisa_gcn import paired_document_logits

        records = [{"text": "first", "label": 1}, {"text": "second", "label": 0}]
        original = [self.make_document(row["text"], float(row["label"]), i)
                    for i, row in enumerate(records)]
        with tempfile.TemporaryDirectory() as directory:
            with patch("lisa_gcn.data.build_document", side_effect=self.make_document):
                cached = build_disk_documents(records, Path(directory) / "train")
            self.assertEqual(cached.token_counts, [8, 8])
            for expected, actual in zip(original, cached):
                self.assertEqual((expected.label, expected.token_count), (actual.label, actual.token_count))
                for scale, view in expected.graphs_by_scale.items():
                    for first, second in zip(view, actual.graphs_by_scale[scale]):
                        self.assertEqual(set(first.keys()), set(second.keys()))
                        for key in first.keys():
                            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)

            torch.manual_seed(123)
            model = LisaGCN(input_dim=3, hidden_dim=4, output_dim=2, dropout=0.0)
            first = paired_document_logits(model, original, 96, "cpu", 2, 8)
            sum(value.sum() for value in first).backward()
            gradients = [p.grad.clone() for p in model.parameters()]
            model.zero_grad(set_to_none=True)
            second = paired_document_logits(model, cached[:], 96, "cpu", 1, 1)
            sum(value.sum() for value in second).backward()
            for expected, actual in zip(first, second):
                torch.testing.assert_close(expected, actual)
            for expected, parameter in zip(gradients, model.parameters()):
                torch.testing.assert_close(expected, parameter.grad)
            self.assertEqual(cached[-1].label, 0.0)
            with self.assertRaises(IndexError):
                cached[len(cached)]

    def test_cache_builder_releases_graphs_between_documents(self):
        from lisa_gcn.data import build_disk_documents

        references = []

        def build(*args):
            self.assertTrue(all(reference() is None for reference in references))
            document = self.make_document(*args)
            references.append(weakref.ref(document))
            return document

        with tempfile.TemporaryDirectory() as directory:
            with patch("lisa_gcn.data.build_document", side_effect=build):
                cached = build_disk_documents([{"text": "sample", "label": 1}] * 20,
                                              Path(directory) / "train", window_sizes=(128,))
            self.assertTrue(all(reference() is None for reference in references))
            self.assertEqual(cached.token_counts, [8] * 20)
            self.assertEqual(set(cached[0].graphs_by_scale), {128})

    def test_trainer_uses_and_cleans_disk_cache(self):
        from lisa_gcn.checkpoint import load_lisa_checkpoint
        from train_lisa_gcn import main

        def build(*args):
            document = self.make_document(*args)
            for view in document.graphs_by_scale.values():
                for graph in view:
                    graph.x = graph.x.repeat(1, 256)  # Trainer's legacy input dimension is 768.
            return document

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [{"text": "human sample", "label": 1}, {"text": "ai sample", "label": 0}]
            source = root / "source.jsonl"
            source.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
            cache_parent = root / "cache"
            args = ["--train-json", str(source), "--val-json", str(source), "--output", str(root / "model"),
                    "--cache-dir", str(cache_parent), "--epochs", "2", "--patience", "2", "--cpu",
                    "--hidden-dim", "4", "--output-dim", "2", "--dropout", "0", "--window-chunk-size", "1"]
            with patch("lisa_gcn.data.build_document", side_effect=build) as builder:
                main(args)
            self.assertEqual([call.args[-1] for call in builder.call_args_list],
                             [(96, 128, 160)] * 2 + [(128,)] * 2)
            state = load_lisa_checkpoint(root / "model" / "lisa_gcn_seed2025.pt", map_location="cpu")
            self.assertTrue(all(math.isfinite(row["val_loss"]) for row in state["history"]))
            self.assertEqual(list(cache_parent.iterdir()), [])

            with patch("lisa_gcn.data.build_document", side_effect=RuntimeError("preprocessing failed")):
                with self.assertRaisesRegex(RuntimeError, "preprocessing failed"):
                    main(args)
            self.assertEqual(list(cache_parent.iterdir()), [])

    def test_mean_logits_and_groupdro_keep_gradients(self):
        import torch
        from lisa_gcn.model import mean_document_logits
        from lisa_gcn.objective import GroupDRO, classification_losses, scale_consistency

        first = torch.tensor([0.5, -0.5, 0.25], requires_grad=True)
        second = torch.tensor([0.4, -0.2, 0.1], requires_grad=True)
        labels = torch.tensor([1.0, 0.0, 1.0])
        objective, means, present = GroupDRO(groups=4).loss(classification_losses(first, second, labels), torch.tensor([0, 3, 3]))
        total = objective + 0.1 * scale_consistency(first, second)
        total.backward()
        self.assertIsNotNone(first.grad)
        self.assertEqual(present.tolist(), [True, False, False, True])
        document = mean_document_logits(torch.tensor([1.0, 3.0, -2.0]), torch.tensor([0, 0, 1]), 2)
        self.assertTrue(torch.allclose(document, torch.tensor([2.0, -2.0])))


if __name__ == "__main__":
    unittest.main()
