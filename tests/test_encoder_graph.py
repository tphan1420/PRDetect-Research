"""Small offline checks; run with python -m unittest discover -s tests -v."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from building_graph_encoder import FrozenWordEncoder, main, parse_label, resolve_source
from encoder_graph import (PIPELINE, check_checkpoint_graph, check_graph_pair,
                           checkpoint_manifest_path, fingerprint, pack_subwords)


def make_graph(encoder='microsoft/deberta-v3-base'):
    configuration = {'pipeline': PIPELINE, 'encoder': encoder,
                     'hidden_size': 768, 'num_relations': 60}
    return {'encoder_metadata': {'configuration': configuration,
                                'fingerprint': fingerprint(configuration)}}


class AlignmentTests(unittest.TestCase):
    def test_many_subwords_per_node(self):
        self.assertEqual(pack_subwords([10, 11, 12], [0, 0, 1], ['longword', '!'], 3),
                         ([10, 11, 12], [0, 0, 1], 0))

    def test_whitespace_node_is_not_dropped(self):
        self.assertEqual(pack_subwords([10, 12], [0, 2], ['one', '\n', 'two'], 3),
                         ([10, 3, 12], [0, 1, 2], 1))

    def test_missing_nonempty_node_fails(self):
        with self.assertRaises(ValueError):
            pack_subwords([1], [0], ['one', 'two'], 3)

    def test_bad_alignment_fails(self):
        for ids, mapping in [([1], []), ([1], [None]), ([1], [2]), ([1, 2], [1, 0])]:
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                pack_subwords(ids, mapping, ['one', 'two'], 3)

    def test_label_convention(self):
        for label in ['hc3human', 'human', 1, 1.0, '1']:
            self.assertEqual(parse_label(label), 1)
        for label in ['hc3chatgpt', 'machine', 'ai', 0, 0.0, '0']:
            self.assertEqual(parse_label(label), 0)
        with self.assertRaises(ValueError):
            parse_label('unrecognized')
        with self.assertRaises(ValueError):
            parse_label(2)

    def test_source_full_path(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'one.json'
            source.write_text('{}', encoding='utf-8')
            self.assertEqual(resolve_source(str(source)), source.resolve())

    def test_graph_cap_counts_other_encoder_and_legacy_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'graphs'
            (root / 'roberta_aligned').mkdir(parents=True)
            for index in range(4):
                (root / 'roberta_aligned' / f'old{index}.pkl').touch()
            source = Path(directory) / 'test.json'
            source.write_text('{}', encoding='utf-8')
            with patch('sys.argv', ['builder', '--file', str(source), '--graph-root', str(root)]):
                with self.assertRaisesRegex(RuntimeError, 'Graph limit 4'):
                    main()  # Fails before importing torch or downloading encoders.

    def test_cache_is_not_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'graphs'
            (root / 'deberta_aligned').mkdir(parents=True)
            (root / 'deberta_aligned' / 'test.pkl').touch()
            source = Path(directory) / 'test.json'
            source.write_text('{}', encoding='utf-8')
            with patch('sys.argv', ['builder', '--file', str(source), '--graph-root', str(root)]):
                with self.assertRaises(FileExistsError):
                    main()


class ProvenanceTests(unittest.TestCase):
    def test_fingerprint_key_order(self):
        self.assertEqual(fingerprint({'a': 1, 'b': 2}), fingerprint({'b': 2, 'a': 1}))

    def test_same_encoder(self):
        self.assertIsNotNone(check_graph_pair(make_graph(), make_graph(), 768, 60))

    def test_legacy_pair_unchanged(self):
        self.assertIsNone(check_graph_pair({}, {}, 768, 60))

    def test_mixed_encoders_or_legacy_rejected(self):
        for other in [make_graph('roberta-base'), {}]:
            with self.assertRaises(ValueError):
                check_graph_pair(make_graph(), other, 768, 60)

    def test_dimensions_and_vocab_rejected(self):
        for width, relations in [(1024, 60), (768, 61)]:
            with self.assertRaises(ValueError):
                check_graph_pair(make_graph(), make_graph(), width, relations)

    def test_tampered_metadata_rejected(self):
        graph = make_graph()
        graph['encoder_metadata']['configuration']['encoder'] = 'roberta-base'
        with self.assertRaises(ValueError):
            check_graph_pair(graph, graph, 768, 60)

    def test_checkpoint_sidecar_required(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'weights.pth'
            with self.assertRaises(ValueError):
                check_checkpoint_graph(checkpoint, make_graph(), 768, 60)
            check_checkpoint_graph(checkpoint, {}, 768, 60)

    def test_checkpoint_encoder_match(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'weights.pth'
            checkpoint_manifest_path(checkpoint).write_text(json.dumps({
                'encoder_metadata': make_graph()['encoder_metadata'],
            }), encoding='utf-8')
            check_checkpoint_graph(checkpoint, make_graph(), 768, 60)
            for graph in [make_graph('roberta-base'), {}]:
                with self.assertRaises(ValueError):
                    check_checkpoint_graph(checkpoint, graph, 768, 60)


@unittest.skipUnless(importlib.util.find_spec('torch') is not None, 'PyTorch not installed locally')
class EncoderNumericsTests(unittest.TestCase):
    def test_chunk_boundaries_mean_pooling_and_freezing(self):
        import torch
        from types import SimpleNamespace

        class Encoding(dict):
            def word_ids(self):
                return [0, 0, 0, 1, 2]

        class Tokenizer:
            unk_token_id = 99

            def __call__(self, words, **kwargs):
                return Encoding(input_ids=[1, 3, 5, 7, 9])

            def build_inputs_with_special_tokens(self, ids):
                return [100] + ids + [101]

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(hidden_size=2)
                self.scale = torch.nn.Parameter(torch.ones(()))
                self.dropout = torch.nn.Dropout(0.9)
                self.lengths = []

            def forward(self, input_ids, attention_mask):
                self.lengths.append(input_ids.shape[1])
                self.assertions(input_ids, attention_mask)
                states = self.dropout(input_ids.float()).unsqueeze(-1).repeat(1, 1, 2) * self.scale
                return SimpleNamespace(last_hidden_state=states)

            def assertions(self, ids, mask):
                assert not self.training and not torch.is_grad_enabled()
                assert torch.equal(mask, torch.ones_like(ids))

        encoder = FrozenWordEncoder.__new__(FrozenWordEncoder)
        encoder.torch, encoder.device = torch, 'cpu'
        encoder.tokenizer, encoder.model = Tokenizer(), Model()
        encoder.model.requires_grad_(False)
        encoder.max_length = 4  # 2 content pieces; first word crosses a boundary.
        features, stats = encoder.encode(['longword', 'two', 'three'])
        torch.testing.assert_close(features, torch.tensor([[3., 3.], [7., 7.], [9., 9.]]))
        self.assertEqual(encoder.model.lengths, [4, 4, 3])
        self.assertEqual(stats['subwords'], 5)
        self.assertFalse(features.requires_grad)
        self.assertTrue(all(not p.requires_grad for p in encoder.model.parameters()))
        repeated, _ = encoder.encode(['longword', 'two', 'three'])
        torch.testing.assert_close(features, repeated)


if __name__ == '__main__':
    unittest.main()
