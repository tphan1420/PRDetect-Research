"""Offline integration tests: real graph/training code, controlled encoder fixtures."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch_geometric.data import Batch

import building_graph
from prism_lite.data import load_cache, to_data_list
from prism_lite.graph_builder import build_file, build_records, metadata_hash
from prism_lite.model import PrismF1GCN
from prism_lite.train import main as train
from prism_lite.evaluate import main as evaluate
from prism_lite.infer import main as infer
from prism_lite.data import validate_cache


class EncoderFixture(torch.nn.Module):
    """Deterministic non-RoBERTa encoder; proves call-path parity, not pretrained accuracy."""

    def forward(self, ids):
        return SimpleNamespace(last_hidden_state=ids.float().unsqueeze(-1).expand(-1, -1, 4).contiguous())


def parsed_doc(text):
    words = text.split()
    tokens = [SimpleNamespace(i=i, text=word, dep_="ROOT" if i % 3 == 0 else "dep")
              for i, word in enumerate(words)]
    for token in tokens:
        token.head = tokens[(token.i // 3) * 3]

    class Doc(list):
        @property
        def sents(self):
            return [self[start:start + 3] for start in range(0, len(self), 3)]

    return Doc(tokens)


class PrismPipelineTests(unittest.TestCase):
    def setUp(self):
        tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda words: [len(word) for word in words])
        self.patches = [
            patch.object(building_graph, "init_models"),
            patch.object(building_graph, "nlp", side_effect=parsed_doc),
            patch.object(building_graph, "tokenizer", tokenizer),
            patch.object(building_graph, "model", EncoderFixture().eval()),
            patch.object(building_graph, "device", torch.device("cpu")),
            patch("prism_lite.graph_builder.protocol_metadata", return_value={"test_fixture": "controlled-4d-encoder"}),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def test_baseline_feature_and_dependency_parity_across_embedding_chunks(self):
        record = {"text": " ".join(["a", "four", "tokens"] * 174), "label": "human"}
        baseline_x, baseline_edges, baseline_y = building_graph.build_graph([json.dumps(record)], return_edge_type=False)
        cache = build_records([record])
        graph = cache["graphs"][0]
        self.assertTrue(torch.equal(graph["x"], baseline_x[0]))
        self.assertTrue(torch.equal(cache["y"], baseline_y))
        baseline_pairs = set(map(tuple, baseline_edges[0].t().tolist()))
        dependency_pairs = set(map(tuple, graph["edge_index"][:, graph["edge_type"] == 0].t().tolist()))
        self.assertEqual(dependency_pairs, baseline_pairs)

    def test_build_train_checkpoint_and_evaluate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "records.jsonl"
            records = [{"text": "a human sentence two more words", "label": "human"},
                       {"text": "machine generated text", "label": "machine"}]
            source.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
            cache_path = root / "cache.pkl"
            build_file(source, cache_path)
            cache = load_cache(cache_path)
            batch = Batch.from_data_list(to_data_list(cache))
            self.assertEqual(batch.sample_id.tolist(), [0, 1])
            self.assertTrue(torch.equal(batch.batch[batch.edge_index[0]], batch.batch[batch.edge_index[1]]))
            checkpoint = root / "f1.pt"
            train(["--train-cache", str(cache_path), "--val-cache", str(cache_path), "--output", str(checkpoint),
                   "--epochs", "2", "--batch-size", "2", "--input-dim", "4", "--hidden-dim", "5", "--output-dim", "3", "--cpu"])
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            # Baseline ties count as improvement: the second tied epoch must save.
            self.assertEqual(state["epoch"], 2)
            self.assertIn("optimizer_state", state)
            self.assertIsNone(state["scheduler_state"])
            model = PrismF1GCN(input_dim=4, hidden_dim=5, output_dim=3).eval()
            model.load_state_dict(state["model_state"])
            with torch.no_grad():
                expected = model(batch).reshape(-1)
            report = evaluate(["--checkpoint", str(checkpoint), "--cache", str(cache_path),
                               "--output", str(root / "report.json"), "--cpu"])
            self.assertTrue(torch.allclose(expected, torch.tensor(report["probabilities_human"]), atol=1e-7))
            raw_prediction = infer(["--checkpoint", str(checkpoint), "--text", records[0]["text"], "--cpu"])
            self.assertAlmostEqual(raw_prediction["p_human"], expected[0].item(), places=6)
            self.assertEqual(report["sample_ids"], [0, 1])
            self.assertEqual(report["threshold"], .5)
            self.assertTrue(checkpoint.with_suffix(".train.json").is_file())
            with self.assertRaises(FileExistsError):
                build_file(source, cache_path)
            cache["protocol"] = {"different": "parser"}
            cache["preprocessing_sha256"] = metadata_hash(cache["protocol"])
            import pickle
            other = root / "other.pkl"
            with other.open("wb") as handle:
                pickle.dump(cache, handle)
            with self.assertRaisesRegex(ValueError, "fingerprints differ"):
                evaluate(["--checkpoint", str(checkpoint), "--cache", str(other),
                          "--output", str(root / "bad.json"), "--cpu"])

    def test_rejects_sentence_crossing_fallback_and_retains_sample_ids(self):
        cache = build_records([{"text": "", "label": 0}, {"text": "a b c d e f", "label": 1}])
        self.assertEqual(cache["skipped_sample_indices"], [0])
        self.assertEqual(to_data_list(cache)[0].sample_id.tolist(), [1])
        graph = cache["graphs"][0]
        graph["edge_index"] = torch.cat([graph["edge_index"], torch.tensor([[2], [3]])], dim=1)
        graph["edge_type"] = torch.cat([graph["edge_type"], torch.tensor([1])])
        with self.assertRaisesRegex(ValueError, "out-of-sentence"):
            validate_cache(cache)

    def test_original_gcn_still_runs_and_can_take_optimizer_step(self):
        from model.GCN2 import GCN2
        cache = build_records([{"text": "a small graph", "label": 1}])
        graph = to_data_list(cache)[0]
        graph.edge_index = graph.edge_index[:, graph.edge_type == 0]
        model = GCN2(4, 5, 3)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        probability = model(graph)
        loss = torch.nn.functional.binary_cross_entropy(probability, torch.ones_like(probability))
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
