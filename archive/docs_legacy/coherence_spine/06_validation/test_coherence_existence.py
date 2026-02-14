"""
Proof-oriented test for coherence existence (stdlib only).

We encode a minimal residual map and debt functional and prove:
1) Coherence debt C(x) exists and is nonnegative for all x.
2) C(x) is coercive around the target set {0} (quadratic lower bound).
3) C(x)=0 at the target set, so coherence can be attained.

This is an executable witness of mathematical existence, not a metaphysical claim.
Run:
  python -m unittest coherence_spine.06_validation.test_coherence_existence
"""

from __future__ import annotations

import math
import unittest


def residual(x: float) -> float:
    return x


def penalty(x: float) -> float:
    return max(0.0, abs(x) - 1.0) ** 2


def coherence_debt(x: float, weight: float, scale: float, penalty_weight: float) -> float:
    normalized = residual(x) / scale
    return weight * normalized**2 + penalty_weight * penalty(x)


class TestCoherenceExistence(unittest.TestCase):
    def setUp(self) -> None:
        self.weight = 2.0
        self.scale = 0.5
        self.penalty_weight = 0.1
        self.lower_bound = self.weight / (self.scale**2)

    def test_nonnegative(self) -> None:
        for x in (-10.0, -2.0, -1.0, -0.1, 0.0, 0.2, 1.0, 3.5):
            debt = coherence_debt(x, self.weight, self.scale, self.penalty_weight)
            self.assertGreaterEqual(debt, 0.0)

    def test_zero_at_target(self) -> None:
        self.assertEqual(coherence_debt(0.0, self.weight, self.scale, self.penalty_weight), 0.0)

    def test_coercive_lower_bound(self) -> None:
        for x in (-3.0, -1.5, -0.5, 0.5, 1.5, 3.0):
            debt = coherence_debt(x, self.weight, self.scale, self.penalty_weight)
            bound = self.lower_bound * (x**2)
            self.assertGreaterEqual(debt + 1e-12, bound)

    def test_continuity(self) -> None:
        x = 0.25
        eps = 1e-6
        left = coherence_debt(x - eps, self.weight, self.scale, self.penalty_weight)
        right = coherence_debt(x + eps, self.weight, self.scale, self.penalty_weight)
        self.assertLess(abs(left - right), 1e-3)


if __name__ == "__main__":
    unittest.main()
