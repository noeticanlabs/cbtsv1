# test_coherence_alignment.py
# =============================================================================
# Defined Coherence Alignment Tests
# =============================================================================
#
# These tests verify that cbtsv1 correctly implements the canonical
# coherence functional per the Defined Coherence canon:
#     𝔠(x) = ⟨r̃(x), W r̃(x)⟩,  where r̃ = S⁻¹r
#
# Tests:
# 1. Core coherence computation (smoke tests)
# 2. Minkowski GR test (coherence ≈ 0)
# 3. Artificial residual test (verify hand calculation)
#
# Lexicon declarations per canon v1.2
LEXICON_IMPORTS = [
    "LoC_axiom",
    "UFE_core",
    "GR_dyn",
    "CTL_time"
]

import numpy as np
import pytest
import json
import os

# Add src to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cbtsv1.vendor.coherence_framework.coherence.core import (
    ResidualBlock, compute_coherence
)
from cbtsv1.solvers.gr.coherence_integration import (
    compute_gr_coherence,
    create_default_coherence_config
)


class TestCanonicalCoherenceCore:
    """Test the canonical coherence computation in isolation."""
    
    def test_zero_residual_gives_zero_coherence(self):
        """Zero residual should give zero coherence."""
        block = ResidualBlock(
            name="test",
            vector=np.zeros(100),
            scale=1.0,
            weight=1.0
        )
        result = compute_coherence({"test": block})
        assert result.coherence_value == 0.0
    
    def test_unit_residual_gives_unit_coherence(self):
        """Unit residual with scale=1, weight=1 should give coherence = N (size of vector)."""
        n = 10
        block = ResidualBlock(
            name="test",
            vector=np.ones(n),
            scale=1.0,
            weight=1.0
        )
        result = compute_coherence({"test": block})
        # ||1||² = n, weight * ||r̃||² = 1 * n = n
        assert np.isclose(result.coherence_value, n)
    
    def test_scaled_residual_scaling(self):
        """Scaled residual should scale coherence quadratically."""
        vec = np.array([1.0, 2.0, 3.0])
        block = ResidualBlock(
            name="test",
            vector=vec,
            scale=2.0,
            weight=1.0
        )
        result = compute_coherence({"test": block})
        # r̃ = 2 * vec = [2, 4, 6]
        # ||r̃||² = 4 + 16 + 36 = 56
        assert np.isclose(result.coherence_value, 56.0)
    
    def test_weighted_residual(self):
        """Weight should multiply the L2 squared."""
        vec = np.array([1.0, 2.0, 3.0])
        block = ResidualBlock(
            name="test",
            vector=vec,
            scale=1.0,
            weight=2.0
        )
        result = compute_coherence({"test": block})
        # ||vec||² = 1 + 4 + 9 = 14
        # weight * ||r̃||² = 2 * 14 = 28
        assert np.isclose(result.coherence_value, 28.0)
    
    def test_combined_blocks(self):
        """Multiple blocks should sum their contributions."""
        block1 = ResidualBlock(
            name="hamiltonian",
            vector=np.array([1.0, 2.0]),
            scale=1.0,
            weight=1.0
        )
        block2 = ResidualBlock(
            name="momentum", 
            vector=np.array([3.0, 4.0]),
            scale=1.0,
            weight=1.0
        )
        result = compute_coherence({"hamiltonian": block1, "momentum": block2})
        # ||[1,2]||² = 1 + 4 = 5
        # ||[3,4]||² = 9 + 16 = 25
        # Total = 5 + 25 = 30
        assert np.isclose(result.coherence_value, 30.0)
    
    def test_block_properties_computed(self):
        """Verify block properties (dim, l2, linf, hash) are computed."""
        vec = np.array([1.0, 2.0, 3.0])
        block = ResidualBlock(
            name="test",
            vector=vec,
            scale=2.0,
            weight=1.0
        )
        assert block.dim == 3
        # r̃ = [2, 4, 6], ||r̃||² = 4 + 16 + 36 = 56, ||r̃|| = sqrt(56)
        assert np.isclose(block.l2, np.sqrt(56))
        assert block.linf == 6.0
        assert len(block.hash) == 64  # SHA256 hex
    
    def test_empty_blocks(self):
        """Empty blocks dict should return zero coherence."""
        result = compute_coherence({})
        assert result.coherence_value == 0.0


class TestGRCoherenceIntegration:
    """Test coherence integration with GR solver components."""
    
    def test_default_config_creation(self):
        """Test default configuration creation."""
        config = create_default_coherence_config()
        
        assert config["version"] == "1.0.0"
        assert config["covariance_model"] == "diag"
        assert "hamiltonian" in config["blocks"]
        assert "momentum" in config["blocks"]
        assert config["blocks"]["hamiltonian"]["scale"] == 1.0
        assert config["blocks"]["hamiltonian"]["weight"] == 1.0
    
    def test_config_file_loading(self):
        """Test loading coherence config from JSON file."""
        config_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'config', 
            'defined_coherence_gr.json'
        )
        
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            
            assert "blocks" in config
            assert "hamiltonian" in config["blocks"]
            assert "momentum" in config["blocks"]
        else:
            pytest.skip("Config file not found")


class TestMinkowskiCoherence:
    """Test coherence on Minkowski (flat) spacetime."""
    
    @pytest.fixture
    def gr_solver(self):
        """Create a minimal GR solver for testing."""
        # Import here to avoid circular imports
        from cbtsv1.solvers.gr.gr_solver import GRSolver
        
        # Small grid for fast testing
        solver = GRSolver(
            Nx=8, Ny=8, Nz=8,
            dx=1.0, dy=1.0, dz=1.0,
            log_level=30  # WARNING level
        )
        
        # Initialize to Minkowski (flat spacetime)
        solver.init_minkowski()
        
        return solver
    
    def test_minkowski_coherence_approximately_zero(self, gr_solver):
        """Minkowski state should have near-zero coherence."""
        config = create_default_coherence_config()
        
        # Compute coherence
        result = compute_gr_coherence(
            gr_solver.fields,
            gr_solver.constraints,
            config
        )
        
        # Minkowski has H ≈ 0, M ≈ 0, so coherence should be ~0
        tolerance = config["tests"]["minkowski_tolerance"]
        
        assert result["coherence_value"] < tolerance, \
            f"Minkowski coherence {result['coherence_value']} not near zero (tolerance={tolerance})"
    
    def test_minkowski_block_norms_small(self, gr_solver):
        """Minkowski should have small block norms."""
        config = create_default_coherence_config()
        
        result = compute_gr_coherence(
            gr_solver.fields,
            gr_solver.constraints,
            config
        )
        
        h_l2 = result["blocks"]["hamiltonian"]["l2"]
        m_l2 = result["blocks"]["momentum"]["l2"]
        
        # Both should be very small for Minkowski
        assert h_l2 < 1e-8, f"Hamiltonian L2 = {h_l2} too large"
        assert m_l2 < 1e-8, f"Momentum L2 = {m_l2} too large"


class TestArtificialResidualCoherence:
    """Test coherence with known artificial residuals."""
    
    def test_hamiltonian_only(self):
        """Test with only Hamiltonian residual."""
        # Create mock objects
        class MockFields:
            pass
        
        class MockConstraints:
            def __init__(self, H, M):
                self.H = H
                self.M = M
        
        # Create uniform Hamiltonian residual: H = 0.1 everywhere
        shape = (8, 8, 8)
        H = np.ones(shape) * 0.1
        M = np.zeros((*shape, 3))  # Zero momentum
        
        constraints = MockConstraints(H, M)
        
        config = create_default_coherence_config()
        
        # Need fields for the adapter
        fields = MockFields()
        
        result = compute_gr_coherence(fields, constraints, config)
        
        # Expected: ||0.1||² * 1.0 * 1.0 = 0.01 * N
        n = 8 * 8 * 8
        expected = 0.01 * n
        
        assert np.isclose(result["coherence_value"], expected, rtol=1e-10), \
            f"Expected {expected}, got {result['coherence_value']}"
    
    def test_momentum_only(self):
        """Test with only momentum residual."""
        class MockFields:
            pass
        
        class MockConstraints:
            def __init__(self, H, M):
                self.H = H
                self.M = M
        
        # Zero Hamiltonian
        shape = (4, 4, 4)
        H = np.zeros(shape)
        # Uniform momentum: M = (0.1, 0.1, 0.1)
        M = np.ones((*shape, 3)) * 0.1
        
        constraints = MockConstraints(H, M)
        fields = MockFields()
        
        config = create_default_coherence_config()
        result = compute_gr_coherence(fields, constraints, config)
        
        # For momentum: vector has shape (4,4,4,3), flattened size = 4*4*4*3 = 192
        # Each component is 0.1, so ||M||² = 0.01 * 192 = 1.92
        n = 4 * 4 * 4 * 3
        expected = 0.01 * n
        
        assert np.isclose(result["coherence_value"], expected, rtol=1e-10)
    
    def test_scaled_residual_hand_calculation(self):
        """Test with explicit scale factor - verify hand calculation."""
        class MockFields:
            pass
        
        class MockConstraints:
            def __init__(self, H, M):
                self.H = H
                self.M = M
        
        # Simple 2-element vector
        H = np.array([3.0, 4.0])  # ||H|| = 5
        M = np.zeros((2, 3))  # No momentum
        
        constraints = MockConstraints(H, M)
        fields = MockFields()
        
        # Config with scale=2, weight=3
        config = {
            "blocks": {
                "hamiltonian": {"scale": 2.0, "weight": 3.0},
                "momentum": {"scale": 1.0, "weight": 1.0}
            }
        }
        
        result = compute_gr_coherence(fields, constraints, config)
        
        # r̃ = scale * r = 2 * [3, 4] = [6, 8]
        # ||r̃||² = 36 + 64 = 100
        # weight * ||r̃||² = 3 * 100 = 300
        assert np.isclose(result["coherence_value"], 300.0)


class TestBlockSummaries:
    """Test block summary generation."""
    
    def test_summary_contains_required_fields(self):
        """Verify summary has all required fields."""
        from cbtsv1.solvers.gr.defined_coherence_blocks import (
            build_residual_blocks, summarize_blocks
        )
        
        class MockFields:
            pass
        
        class MockConstraints:
            def __init__(self):
                self.H = np.ones((4, 4, 4)) * 0.1
                self.M = np.zeros((4, 4, 4, 3))
        
        config = create_default_coherence_config()
        constraints = MockConstraints()
        fields = MockFields()
        
        blocks = build_residual_blocks(fields, constraints, config)
        summary = summarize_blocks(blocks)
        
        # Check required fields
        for block_name in ["hamiltonian", "momentum"]:
            assert block_name in summary
            block_summary = summary[block_name]
            assert "dim" in block_summary
            assert "l2" in block_summary
            assert "linf" in block_summary
            assert "hash" in block_summary
            assert "scale" in block_summary
            assert "weight" in block_summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
