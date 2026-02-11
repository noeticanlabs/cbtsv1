"""
Math-form tests for coherence as a principle (stdlib only).

This suite provides three complementary mathematical witnesses:
- Geometry: coherence debt lower-bounds squared distance to a target set in R^2.
- Algebra: small-gain-style bound for coupled residuals in scalar form.
- Calculus: smoothness of debt (gradient/Lipschitz checks for a quadratic form).

Run:
  python -m unittest coherence_spine.06_validation.test_coherence_principle_math
"""

from __future__ import annotations

import math
import unittest


def residual_vec(x: float, y: float) -> tuple[float, float]:
    return (x, y)


def coherence_debt_r2(x: float, y: float, weight: float = 1.0) -> float:
    r1, r2 = residual_vec(x, y)
    return weight * (r1 * r1 + r2 * r2)


def small_gain_bound(a: float, b: float, e_a: float, e_b: float) -> tuple[float, float]:
    denom = 1.0 - a * b
    return ((a * e_b + e_a) / denom, (b * e_a + e_b) / denom)


def grad_coherence_debt_r2(x: float, y: float, weight: float = 1.0) -> tuple[float, float]:
    return (2.0 * weight * x, 2.0 * weight * y)


def numeric_gradient(f, x: float, y: float, eps: float = 1e-6) -> tuple[float, float]:
    fx1 = f(x + eps, y)
    fx0 = f(x - eps, y)
    fy1 = f(x, y + eps)
    fy0 = f(x, y - eps)
    return ((fx1 - fx0) / (2.0 * eps), (fy1 - fy0) / (2.0 * eps))


class TestCoherencePrincipleMath(unittest.TestCase):
    def test_geometry_distance_bound(self) -> None:
        for x, y in [(-2.0, 1.5), (0.0, 0.0), (1.0, -3.0), (2.5, 2.5)]:
            debt = coherence_debt_r2(x, y, weight=1.0)
            distance_sq = x * x + y * y
            self.assertGreaterEqual(debt, distance_sq)

    def test_algebra_small_gain_bound(self) -> None:
        a, b = 0.4, 0.5
        e_a, e_b = 0.1, 0.2
        bound_a, bound_b = small_gain_bound(a, b, e_a, e_b)
        r_a = (a * e_b + e_a) / (1.0 - a * b)
        r_b = (b * e_a + e_b) / (1.0 - a * b)
        self.assertAlmostEqual(r_a, bound_a, delta=1e-12)
        self.assertAlmostEqual(r_b, bound_b, delta=1e-12)
        self.assertLessEqual(r_a, a * r_b + e_a + 1e-12)
        self.assertLessEqual(r_b, b * r_a + e_b + 1e-12)

    def test_calculus_gradient_matches_numeric(self) -> None:
        weight = 1.5
        x, y = 0.3, -0.7
        analytical = grad_coherence_debt_r2(x, y, weight=weight)
        numeric = numeric_gradient(lambda a, b: coherence_debt_r2(a, b, weight=weight), x, y)
        self.assertAlmostEqual(analytical[0], numeric[0], delta=1e-5)
        self.assertAlmostEqual(analytical[1], numeric[1], delta=1e-5)

    def test_calculus_lipschitz_gradient(self) -> None:
        weight = 2.0
        x1, y1 = 0.2, -0.1
        x2, y2 = -0.4, 0.3
        g1 = grad_coherence_debt_r2(x1, y1, weight=weight)
        g2 = grad_coherence_debt_r2(x2, y2, weight=weight)
        grad_dist = math.hypot(g1[0] - g2[0], g1[1] - g2[1])
        input_dist = math.hypot(x1 - x2, y1 - y2)
        lipschitz = 2.0 * weight
        self.assertLessEqual(grad_dist, lipschitz * input_dist + 1e-12)


if __name__ == "__main__":
    unittest.main()
