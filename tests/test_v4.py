import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from v4 import CheckpointPair, score_a1_split, source_digest


class V4Tests(unittest.TestCase):
    def test_each_text_is_encoded_once_and_scores_keep_row_and_seed_order(self):
        scorer = Mock()
        scorer.nlp.pipe.side_effect = lambda texts, batch_size: (SimpleNamespace(text=t) for t in texts)
        scorer.aligned_graph.side_effect = lambda doc: len(doc.text)
        scorer.a1_logit.side_effect = lambda head, length: head * length
        rows = [{"text": t} for t in ("aaa", "b", "aaa")]
        self.assertEqual(score_a1_split(scorer, {2020: 1, 2026: -2}, rows),
                         {2020: [3, 1, 3], 2026: [-6, -2, -6]})
        self.assertEqual(scorer.aligned_graph.call_count, 2)

    def test_fusion_checks_checkpoint_identity_and_includes_threshold_ties(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.json"
            path.write_text('{"text":"a","label":1}\n{"text":"b","label":0}\n', encoding="utf-8")
            pair = CheckpointPair(2020, {"checkpoint_sha256": {"2020": "ema"}}, Path("a1.pt"), "a1")
            v3 = dict(seed=2020, checkpoint_sha256="ema", split="sample", train="hc3", samples=2,
                      label_convention={"0": "ai", "1": "human"}, y_true=[1, 0], p_human=[1., .8])
            a1 = dict(checkpoint_sha256="a1", condition="sample", source_sha256=source_digest(path),
                      score_space="human_logit", records=[dict(index=i, label=y, logit_human=0.)
                                                          for i, y in enumerate([1, 0])])
            truth = np.array([1, 0])
            result = pair.fuse(path, truth, v3, a1)
            self.assertEqual(result["p_human_fusion"], [.5, .4])
            self.assertEqual(result["metrics"]["accuracy"], 1.)
            for other, head in (({**v3, "seed": 2026}, a1), ({**v3, "checkpoint_sha256": "raw"}, a1),
                                (v3, {**a1, "checkpoint_sha256": "other"})):
                with self.assertRaises(ValueError):
                    pair.fuse(path, truth, other, head)
