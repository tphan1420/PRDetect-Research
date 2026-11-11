"""Focused invariants for the Corrected Structural Mapping V3 variant."""

import importlib.util
import unittest
from unittest.mock import patch


HAS_RUNTIME = importlib.util.find_spec("torch") and importlib.util.find_spec("numpy")


@unittest.skipUnless(HAS_RUNTIME, "PyTorch/NumPy numerical runtime unavailable")
class CorrectedStructuralMappingTests(unittest.TestCase):
    def test_directed_mode_keeps_every_non_self_dependency_label(self):
        import numpy as np
        from model.QuangV3 import fwd_mask, parent_array, struct_features

        # 0 is the root self-loop; both 25 and 50 were formerly excluded by
        # the legacy R//2 convention for R=51.
        src = np.array([0, 1, 2])
        dst = np.array([0, 0, 1])
        rel = np.array([1, 25, 50])
        self.assertTrue(fwd_mask(rel, 51, "directed").all())
        parent, edges, nodes = parent_array(3, src, dst, rel, 51, "directed")
        np.testing.assert_array_equal(parent, [-1, 0, 1])
        self.assertEqual((edges, nodes), (2, 2))
        structural, _ = struct_features(3, src, dst, rel, 51, "directed")
        np.testing.assert_array_equal(structural[:, 0], [0, 1, 2])

    def test_directed_features_cover_high_relation_ids_and_differ_from_legacy_low(self):
        import numpy as np
        from model.QuangV3 import fwd_mask, parent_array, struct_features

        # Two roots (0 and 6) represent two sentences.  The five dependency
        # labels deliberately span both sides of the old R//2 split.
        src = np.array([0, 1, 2, 3, 4, 5, 6])
        dst = np.array([0, 0, 0, 2, 2, 4, 6])
        rel = np.array([1, 12, 25, 34, 42, 50, 1])
        directed_parent, edges, nodes = parent_array(7, src, dst, rel, 51, "directed")
        np.testing.assert_array_equal(directed_parent, [-1, 0, 0, 2, 2, 4, -1])
        self.assertEqual((edges, nodes), (5, 5))

        directed_st, _ = struct_features(7, src, dst, rel, 51, "directed")
        np.testing.assert_array_equal(directed_st[:, 0], [0, 1, 1, 2, 2, 3, 0])
        np.testing.assert_array_equal(directed_st[:, 1], [2, 0, 2, 0, 1, 0, 0])
        np.testing.assert_array_equal(directed_st[:, 2], [0, 1, 1, 1, 1, 1, 0])

        # Legacy low admits only relation 12 here; directed must retain all
        # valid relation IDs rather than confusing IDs >= 25 with roots.
        self.assertTrue(fwd_mask(rel, 51, "directed").all())
        low_parent, _, _ = parent_array(7, src, dst, rel, 51, "low")
        np.testing.assert_array_equal(low_parent, [-1, 0, -1, -1, -1, -1, -1])
        low_st, _ = struct_features(7, src, dst, rel, 51, "low")
        self.assertFalse(np.array_equal(directed_st, low_st))

    def test_directed_swap_never_crosses_high_relation_ids(self):
        import torch
        from model.V3StructuralMap import CorrectedStructuralMappingNet

        # Nodes 1 and 3 share role 25; node 2 has role 34.  All nodes are
        # selected deterministically, so a faulty legacy else branch could
        # group 25 and 34 as its fallback role.  The corrected variant may
        # swap node 1 with node 3, but node 2 must remain its own donor.
        model = CorrectedStructuralMappingNet(51, d=8, nb=2, L=1, mode="directed", aug_max=1.0)
        x = torch.tensor([[10.0], [20.0], [30.0], [40.0]])
        src = torch.tensor([0, 1, 2, 3])
        dst = torch.tensor([0, 0, 0, 0])
        rel = torch.tensor([1, 25, 34, 25])
        bidx = torch.zeros(4, dtype=torch.long)
        random_draws = [
            torch.ones(1),                 # rho: select every node
            torch.ones(4),                 # selected
            torch.zeros(4),                # no masking in swap mode
            torch.tensor([0.4, 0.1, 0.2, 0.3]),  # deterministic donor order
        ]
        with patch("model.V3StructuralMap.torch.rand", side_effect=random_draws):
            actual = model._augment(x, src, dst, rel, bidx, 1, "swap", gen=None)
        self.assertEqual(float(actual[2, 0]), 30.0)
        self.assertIn(float(actual[1, 0]), (20.0, 40.0))
        self.assertNotEqual(float(actual[1, 0]), 30.0)

    def test_legacy_low_and_high_masks_remain_exact(self):
        import numpy as np
        from model.QuangV3 import fwd_mask, parent_array

        R = 51
        src = np.array([0, 1, 2, 3, 4])
        dst = np.array([0, 0, 1, 1, 2])
        rel = np.array([1, 12, 25, 34, 50])
        for mode, expected_mask in (
            ("low", np.array([True, True, False, False, False])),
            ("high", np.array([False, False, True, True, False])),
        ):
            actual_mask = fwd_mask(rel, R, mode)
            np.testing.assert_array_equal(actual_mask, expected_mask)
            expected_parent = -np.ones(5, dtype=np.int64)
            usable = expected_mask & (src != dst)
            expected_parent[src[usable]] = dst[usable]
            actual_parent, _, _ = parent_array(5, src, dst, rel, R, mode)
            np.testing.assert_array_equal(actual_parent, expected_parent)

    def test_directed_mode_rejects_multiple_non_self_parents(self):
        import numpy as np
        from model.QuangV3 import parent_array

        with self.assertRaisesRegex(ValueError, "at most one"):
            parent_array(3, np.array([1, 1]), np.array([0, 2]),
                         np.array([25, 50]), 51, "directed")

    def test_directed_mode_rejects_parent_cycles_before_computing_depth(self):
        import numpy as np
        from model.QuangV3 import parent_array

        # A dependency parse is a forest.  Without this check, struct_features
        # would walk this loop until MAXD and fabricate a large depth.
        with self.assertRaisesRegex(ValueError, "acyclic"):
            parent_array(2, np.array([0, 1]), np.array([1, 0]),
                         np.array([25, 50]), 51, "directed")

    def test_directed_variant_uses_same_parameter_shapes_as_baseline(self):
        from model.QuangV3 import Net
        from model.V3StructuralMap import CorrectedStructuralMappingNet

        baseline = Net(51, d=8, nb=2, L=1, mode="low")
        corrected = CorrectedStructuralMappingNet(51, d=8, nb=2, L=1, mode="directed")
        corrected.load_state_dict(baseline.state_dict(), strict=True)


if __name__ == "__main__":
    unittest.main()
