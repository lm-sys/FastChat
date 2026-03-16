"""Unit tests for position bias and Cohen's kappa metrics."""
import unittest

from fastchat.llm_judge.compute_agreement import (
    compute_position_bias,
    compute_cohens_kappa,
    interpret_kappa,
)


class TestPositionBias(unittest.TestCase):
    def test_no_bias(self):
        """Perfect agreement means zero bias."""
        g1 = ["model_1", "model_2", "tie"]
        g2 = ["model_1", "model_2", "tie"]
        rate, direction = compute_position_bias(g1, g2)
        self.assertEqual(rate, 0.0)
        self.assertEqual(direction, "none")

    def test_full_first_position_bias(self):
        """Judge always picks position A -> favor first."""
        g1 = ["model_1", "model_1", "model_1"]
        g2 = ["model_2", "model_2", "model_2"]
        rate, direction = compute_position_bias(g1, g2)
        self.assertEqual(rate, 1.0)
        self.assertEqual(direction, "first")

    def test_full_second_position_bias(self):
        """Judge always picks position B -> favor second."""
        g1 = ["model_2", "model_2"]
        g2 = ["model_1", "model_1"]
        rate, direction = compute_position_bias(g1, g2)
        self.assertEqual(rate, 1.0)
        self.assertEqual(direction, "second")

    def test_mixed_bias(self):
        """Partial disagreement with balanced direction."""
        g1 = ["model_1", "model_2", "model_1", "model_2"]
        g2 = ["model_1", "model_2", "model_2", "model_1"]
        rate, direction = compute_position_bias(g1, g2)
        self.assertAlmostEqual(rate, 0.5)
        # one favor_first (idx 2) and one favor_second (idx 3)
        self.assertEqual(direction, "none")

    def test_empty_input(self):
        rate, direction = compute_position_bias([], [])
        self.assertEqual(rate, 0.0)
        self.assertEqual(direction, "none")


class TestCohensKappa(unittest.TestCase):
    def test_perfect_agreement(self):
        g1 = ["model_1", "model_2", "tie", "model_1"]
        g2 = ["model_1", "model_2", "tie", "model_1"]
        kappa = compute_cohens_kappa(g1, g2)
        self.assertAlmostEqual(kappa, 1.0)

    def test_no_agreement_disjoint(self):
        """Disjoint categories: p_o=0 and p_e=0 -> kappa=0."""
        g1 = ["model_1", "model_1", "model_1"]
        g2 = ["model_2", "model_2", "model_2"]
        kappa = compute_cohens_kappa(g1, g2)
        self.assertAlmostEqual(kappa, 0.0)

    def test_below_chance_agreement(self):
        """Agreement below chance should give kappa < 0."""
        # Both raters use all categories, but disagree more than chance
        g1 = ["model_1", "model_2", "tie", "model_1", "model_2", "tie"]
        g2 = ["model_2", "tie", "model_1", "model_2", "tie", "model_1"]
        kappa = compute_cohens_kappa(g1, g2)
        self.assertLess(kappa, 0.0)

    def test_chance_agreement(self):
        """When agreement equals chance, kappa should be ~0."""
        # Two raters each pick model_1 50% and model_2 50%
        # but they disagree in a pattern that matches chance
        g1 = ["model_1", "model_2", "model_1", "model_2"]
        g2 = ["model_2", "model_1", "model_1", "model_2"]
        kappa = compute_cohens_kappa(g1, g2)
        # p_o = 2/4 = 0.5, p_e = 0.5*0.5 + 0.5*0.5 = 0.5
        self.assertAlmostEqual(kappa, 0.0)

    def test_empty_input(self):
        kappa = compute_cohens_kappa([], [])
        self.assertEqual(kappa, 0.0)

    def test_all_same_category(self):
        """Both raters always say the same thing."""
        g1 = ["tie", "tie", "tie"]
        g2 = ["tie", "tie", "tie"]
        kappa = compute_cohens_kappa(g1, g2)
        self.assertAlmostEqual(kappa, 1.0)


class TestInterpretKappa(unittest.TestCase):
    def test_scale(self):
        self.assertEqual(interpret_kappa(-0.1), "poor")
        self.assertEqual(interpret_kappa(0.1), "slight")
        self.assertEqual(interpret_kappa(0.3), "fair")
        self.assertEqual(interpret_kappa(0.5), "moderate")
        self.assertEqual(interpret_kappa(0.7), "substantial")
        self.assertEqual(interpret_kappa(0.9), "almost perfect")

    def test_boundaries(self):
        self.assertEqual(interpret_kappa(0.0), "slight")
        self.assertEqual(interpret_kappa(0.21), "fair")
        self.assertEqual(interpret_kappa(0.41), "moderate")
        self.assertEqual(interpret_kappa(0.61), "substantial")
        self.assertEqual(interpret_kappa(0.81), "almost perfect")


if __name__ == "__main__":
    unittest.main()
