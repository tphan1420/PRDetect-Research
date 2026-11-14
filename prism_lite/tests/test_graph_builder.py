import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
class FallbackEdgeTests(unittest.TestCase):
    def test_adds_only_in_sentence_bidirectional_edges(self):
        from prism_lite.graph_builder import DEPENDENCY, FALLBACK, add_sequential_fallback
        index, types = add_sequential_fallback([(1, 0)], [[0, 1, 2], [3, 4]])
        edges = {tuple(pair): int(kind) for pair, kind in zip(index.t().tolist(), types.tolist())}
        self.assertEqual(edges[(1, 0)], DEPENDENCY)
        self.assertEqual(edges[(0, 1)], FALLBACK)
        self.assertEqual(edges[(1, 2)], FALLBACK)
        self.assertEqual(edges[(2, 1)], FALLBACK)
        self.assertNotIn((2, 3), edges)

    def test_keeps_legacy_root_edges_but_never_creates_new_self_edges(self):
        from prism_lite.graph_builder import add_sequential_fallback
        index, types = add_sequential_fallback([(0, 0)], [[0]])
        self.assertEqual(index.t().tolist(), [[0, 0]])
        self.assertEqual(types.tolist(), [0])
        index, types = add_sequential_fallback([], [[0]])
        self.assertEqual(tuple(index.shape), (2, 0))
        self.assertEqual(tuple(types.shape), (0,))


if __name__ == "__main__":
    unittest.main()
