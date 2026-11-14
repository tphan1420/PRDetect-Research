import importlib.util
from argparse import Namespace
from pathlib import Path
import tempfile
import unittest

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

        with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
            torch.save({"config": {"output": Path("model/lisa_gcn")}, "tensor": torch.tensor([1.0])}, handle.name)
            state = load_lisa_checkpoint(handle.name, map_location="cpu")
        self.assertEqual(str(state["config"]["output"]), "model/lisa_gcn")
        self.assertEqual(checkpoint_config(Namespace(output=Path("model/lisa_gcn")))["output"], "model/lisa_gcn")


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("torch_geometric"), "requires torch and torch_geometric")
class LisaNumericalTests(unittest.TestCase):
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
