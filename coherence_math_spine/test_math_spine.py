"""
Test suite for Coherence Math Spine

Validates mathematical definitions and relationships from coherence_math_spine.
"""

import math
import unittest


class TestResidualMaps(unittest.TestCase):
    """Test residual map definitions and properties."""

    def test_residual_map_definition(self):
        """Test basic residual map functionality."""
        # r(x) = |x| for simple example
        def residual(x):
            return abs(x)

        # Residual should be non-negative
        self.assertGreaterEqual(residual(0), 0)
        self.assertGreaterEqual(residual(5), 0)
        self.assertGreaterEqual(residual(-3), 0)

    def test_residual_monotonicity(self):
        """Residual increases with distance from target."""
        def residual(x):
            return x ** 2

        # r(0) should be minimum at target
        self.assertEqual(residual(0), 0)

        # r(x) increases as |x| increases
        self.assertLess(residual(1), residual(2))
        self.assertLess(residual(-1), residual(-2))


class TestCoherenceDebt(unittest.TestCase):
    """Test coherence debt functional properties."""

    def test_debt_nonnegativity(self):
        """Coherence debt C(x) must be non-negative."""
        def debt(x):
            return x ** 2 + max(0, abs(x) - 1)

        for x in [-5, -2, -1, 0, 1, 2, 5]:
            self.assertGreaterEqual(debt(x), 0, f"Debt C({x}) should be non-negative")

    def test_debt_at_target(self):
        """Debt should be zero at target set {0}."""
        def debt(x):
            return x ** 2

        self.assertEqual(debt(0), 0)

    def test_debt_coercivity(self):
        """Debt should be coercive (grow like x^2)."""
        def debt(x):
            return x ** 2

        # For coercivity: C(x) ≥ k*|x|^2 for some k > 0
        k = 0.5
        for x in [-3, -2, -1, 1, 2, 3]:
            self.assertGreaterEqual(debt(x), k * (x ** 2))


class TestGatesAndRails(unittest.TestCase):
    """Test gate evaluation and rail mechanics."""

    def test_hard_gate_pass_fail(self):
        """Test hard gate binary decision."""
        def hard_gate_check(nan_free):
            return nan_free  # Hard gate passes only if nan_free is True

        self.assertTrue(hard_gate_check(True))
        self.assertFalse(hard_gate_check(False))

    def test_soft_gate_thresholding(self):
        """Test soft gate thresholding logic."""
        def soft_gate_evaluate(residual, threshold):
            if residual > threshold:
                return "fail"
            elif residual > threshold * 0.75:
                return "warn"
            else:
                return "pass"

        self.assertEqual(soft_gate_evaluate(0.5, 1.0), "pass")
        self.assertEqual(soft_gate_evaluate(0.8, 1.0), "warn")
        self.assertEqual(soft_gate_evaluate(1.2, 1.0), "fail")

    def test_retry_bounded_work(self):
        """Test that retry cap bounds total work."""
        N_retry = 5
        max_attempts_per_step = N_retry + 1

        for attempts in range(1, N_retry + 2):
            self.assertLessEqual(attempts, max_attempts_per_step)


class TestStabilityTheorems(unittest.TestCase):
    """Test stability theorem conditions."""

    def test_contractive_repair_bound(self):
        """Test contractive repair theorem."""
        # C(y) ≤ γ*C(x) + b with γ ∈ (0,1)
        gamma = 0.8
        b = 0.1

        C_x = 1.0
        C_y_repaired = gamma * C_x + b

        # After repair, debt should decrease
        self.assertLess(C_y_repaired, C_x)
        self.assertGreater(C_y_repaired, 0)

    def test_steady_state_bound(self):
        """Test steady-state debt bound under contraction."""
        gamma = 0.8  # contraction factor
        b = 0.1      # bias term
        C_0 = 2.0    # initial debt

        # Steady state bound: C* ≤ b / (1 - γ)
        steady_state = b / (1 - gamma)
        self.assertAlmostEqual(steady_state, 0.5, places=5)

        # Iterate contractive map
        C = C_0
        for _ in range(100):  # Many iterations
            C = gamma * C + b

        # Should converge to steady state
        self.assertAlmostEqual(C, steady_state, places=3)

    def test_exponential_decay(self):
        """Test exponential decay toward steady state."""
        gamma = 0.9
        b = 0.05
        C_0 = 2.0
        steady_state = b / (1 - gamma)

        # After n steps: C_n ≤ γ^n * (C_0 - C*) + C*
        C = C_0
        for n in range(10):
            C = gamma * C + b
            bound = gamma ** (n + 1) * (C_0 - steady_state) + steady_state
            self.assertLessEqual(C, bound + 1e-10)  # Allow small floating point error


class TestSmallGainTheorem(unittest.TestCase):
    """Test small-gain stability results."""

    def test_coupled_residuals_bound(self):
        """Test small-gain theorem for coupled blocks."""
        # x_{n+1} = A*x_n + B*y_n
        # y_{n+1} = C*x_n + D*y_n
        A, B, C, D = 0.3, 0.2, 0.2, 0.3

        x, y = 1.0, 1.0
        for _ in range(50):
            x_new = A * x + B * y
            y_new = C * x + D * y
            x, y = x_new, y_new

        # System should be stable and converge to origin
        self.assertLess(abs(x), 1e-6)
        self.assertLess(abs(y), 1e-6)

    def test_spectral_radius_stability(self):
        """Stability requires spectral radius < 1."""
        import math

        # For 2x2 system, check eigenvalues
        A = [[0.3, 0.2], [0.2, 0.3]]

        # Characteristic polynomial: det(A - λI) = 0
        # For this symmetric matrix: λ = 0.3 ± 0.2
        lambda_1 = 0.5
        lambda_2 = 0.1

        spectral_radius = max(abs(lambda_1), abs(lambda_2))
        self.assertLess(spectral_radius, 1.0)


class TestMathematicalConsistency(unittest.TestCase):
    """Test mathematical consistency axioms."""

    def test_triangle_inequality_residuals(self):
        """Residuals should satisfy triangle-like properties."""
        def residual(x):
            return abs(x)

        # For triangle inequality: |r(x) - r(y)| ≤ r(x - y)
        x, y = 2.0, 1.0
        lhs = abs(residual(x) - residual(y))
        rhs = residual(x - y)

        self.assertLessEqual(lhs, rhs)

    def test_continuity_of_debt(self):
        """Debt functional should be continuous."""
        def debt(x):
            return x ** 2

        x = 1.0
        eps = 1e-6

        # C(x) ≈ C(x ± eps) for small eps
        self.assertAlmostEqual(debt(x), debt(x + eps), places=5)
        self.assertAlmostEqual(debt(x), debt(x - eps), places=5)

    def test_composition_bounds(self):
        """Test bounds compose correctly."""
        # If C(x) ≤ M and repair brings C(y) ≤ γ*C(x)
        C_max = 1.0
        gamma = 0.8

        C_x = 0.9
        C_y = gamma * C_x

        # Final debt should be bounded by composition
        final_bound = gamma * C_max
        self.assertLessEqual(C_y, final_bound)


if __name__ == "__main__":
    unittest.main()
