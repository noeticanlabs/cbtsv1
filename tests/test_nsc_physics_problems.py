"""
NSC-M3L Real Physics Problems Tests
"""

import pytest

from tests.nsc_test_utils import compile_nsc_source
from src.nsc.ast import Model


class TestWaveEquation:
    """Test wave equation."""
    
    def test_wave_equation_1d(self):
        """Test 1D wave equation."""
        source = "@model(GEO, CALC, DISC);\nu :: Field[Scalar] on M;\nWaveEq := d²/dt²(u) - Δ(u) = 0;"
        result = compile_nsc_source(source, target_models={Model.DISC})
        assert result.success
        assert result.disc_output is not None


class TestMaxwellEquations:
    """Test Maxwell equations."""
    
    def test_maxwell_faraday_law(self):
        """Test Faraday's law."""
        source = "@model(GEO, CALC);\nE :: Field[Vector];\nB :: Field[Vector];\nFaraday := d/dt(B) + curl(E) = 0;"
        result = compile_nsc_source(source)
        assert result.success
    
    def test_maxwell_full_system(self):
        """Test full Maxwell."""
        source = "@model(GEO, CALC);\nE :: Field[Vector];\nB :: Field[Vector];\nFaraday := d/dt(B) + curl(E) = 0;\nAmpere := d/dt(E) - curl(B) = J;"
        result = compile_nsc_source(source)
        assert result.success


class TestEinsteinEquations:
    """Test Einstein equations."""
    
    def test_einstein_equation_simple(self):
        """Test simple Einstein."""
        source = "@model(GEO, CALC, LEDGER, EXEC);\nM :: Manifold(3+1, lorentzian);\ng :: Metric on M;\nEinsteinEq := G_ij(g) = 0;"
        result = compile_nsc_source(source)
        assert result.success
    
    def test_hamiltonian_constraint(self):
        """Test Hamiltonian constraint."""
        source = """
        @model(GEO, CALC, LEDGER);
        M :: Manifold(3+1, lorentzian);
        g :: Metric on M;
        K :: Field[Tensor(0,2)];
        H := R(g) + tr(K^2) - K_ij*K^ij = 0;
        @inv(N:INV.gr.hamiltonian_constraint);
        ⇒ (LEDGER, CALC, GEO);
        """
        result = compile_nsc_source(source)
        assert result.success


class TestNavierStokesEquations:
    """Test Navier-Stokes."""
    
    def test_ns_momentum_equation(self):
        """Test NS momentum."""
        source = """
        @model(CALC, LEDGER, EXEC);
        u :: Field[Vector] on (M, t);
        p :: Field[Scalar] on (M, t);
        ν :: Scalar;
        MomentumEq := d/dt(u) + (u·∇)(u) + ∇p - ν*Δ(u) = 0;
        ⇒ (LEDGER, CALC, EXEC);
        """
        result = compile_nsc_source(source)
        assert result.success
    
    def test_ns_continuity_equation(self):
        """Test NS continuity."""
        source = """
        @model(CALC, LEDGER, EXEC);
        u :: Field[Vector] on (M, t);
        ContinuityEq := div(u) = 0;
        ⇒ (LEDGER, CALC, EXEC);
        """
        result = compile_nsc_source(source)
        assert result.success
    
    def test_ns_full_system(self):
        """Test full NS."""
        source = """
        @model(CALC, LEDGER, EXEC);
        u :: Field[Vector] on (M, t);
        p :: Field[Scalar] on (M, t);
        ν :: Scalar;
        MomentumEq := d/dt(u) + (u·∇)(u) + ∇p - ν*Δ(u) = 0;
        ContinuityEq := div(u) = 0;
        ⇒ (LEDGER, CALC, EXEC);
        """
        result = compile_nsc_source(source)
        assert result.success


class TestPoissonEquation:
    """Test Poisson equation."""
    
    def test_poisson_simple(self):
        """Test simple Poisson."""
        source = "@model(CALC, DISC);\nu :: Field[Scalar] on M;\nf :: Field[Scalar] on M;\nPoissonEq := Δ(u) = f;"
        result = compile_nsc_source(source, target_models={Model.DISC})
        assert result.success
        assert result.disc_output is not None


class TestHeatEquation:
    """Test heat equation."""
    
    def test_heat_equation(self):
        """Test heat equation."""
        source = "@model(CALC, DISC);\nT :: Field[Scalar] on M;\nκ :: Scalar;\nHeatEq := d/dt(T) - κ*Δ(T) = 0;"
        result = compile_nsc_source(source, target_models={Model.DISC})
        assert result.success
        assert result.disc_output is not None


class TestYangMillsEquations:
    """Test Yang-Mills."""
    
    def test_yang_mills_equation(self):
        """Test Yang-Mills."""
        source = "@model(ALG, GEO, CALC);\nA :: Field[Vector];\nF :: Field[Tensor(0,2)];\nYMEquation := ∇^j F_ij = 0;"
        result = compile_nsc_source(source)
        assert result.success


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
