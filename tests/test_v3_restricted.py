"""Numerical checks for Structure-Guided Restricted Embeddings."""

import importlib.util
import unittest


HAS_RUNTIME = importlib.util.find_spec("torch") and importlib.util.find_spec("numpy")


@unittest.skipUnless(HAS_RUNTIME, "PyTorch/NumPy numerical runtime unavailable")
class RestrictedEmbeddingTests(unittest.TestCase):
    def test_gates_have_expected_initial_values_and_gradients(self):
        import torch
        from model.V3Restricted import StructureGuidedRestrictedNet

        model = StructureGuidedRestrictedNet(
            4, d=4, nb=2, L=1, semantic_init=0.9, structural_init=1.0,
            semantic_budget=0.8, semantic_penalty=0.1,
        )
        features = torch.ones(3, 4)
        torch.testing.assert_close(model._restrict_semantic(features, None), features * 0.9)
        torch.testing.assert_close(model._restrict_structural(features, None), features)
        penalty = model.restriction_loss()
        torch.testing.assert_close(penalty, torch.tensor(0.001))
        penalty.backward()
        self.assertTrue(torch.isfinite(model.semantic_logits.grad).all())
        self.assertIsNone(model.structural_logits.grad)

    def test_budget_loss_is_zero_below_budget(self):
        import torch
        from model.V3Restricted import StructureGuidedRestrictedNet

        model = StructureGuidedRestrictedNet(4, d=4, nb=2, L=1,
                                             semantic_init=0.4, semantic_budget=0.8)
        torch.testing.assert_close(model.restriction_loss(), torch.tensor(0.0))

    def test_rejects_missing_structural_features(self):
        from model.V3Restricted import StructureGuidedRestrictedNet

        with self.assertRaises(ValueError):
            StructureGuidedRestrictedNet(4, d=4, nb=2, L=1, use_struct=False)


if __name__ == "__main__":
    unittest.main()
