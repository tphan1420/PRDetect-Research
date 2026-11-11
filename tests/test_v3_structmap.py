"""Focused invariants for the Corrected Structural Mapping V3 variant."""

import importlib.util
import unittest


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

    def test_directed_mode_rejects_multiple_non_self_parents(self):
        import numpy as np
        from model.QuangV3 import parent_array

        with self.assertRaisesRegex(ValueError, "at most one"):
            parent_array(3, np.array([1, 1]), np.array([0, 2]),
                         np.array([25, 50]), 51, "directed")

    def test_directed_variant_uses_same_parameter_shapes_as_baseline(self):
        from model.QuangV3 import Net
        from model.V3StructuralMap import CorrectedStructuralMappingNet

        baseline = Net(51, d=8, nb=2, L=1, mode="low")
        corrected = CorrectedStructuralMappingNet(51, d=8, nb=2, L=1, mode="directed")
        corrected.load_state_dict(baseline.state_dict(), strict=True)


if __name__ == "__main__":
    unittest.main()
