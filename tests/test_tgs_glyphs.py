"""
Test file for TGS Glyph implementations
Tests CCZ, CS, CONNECTION_COEFF, and LAMBDA_LAPLACIAN glyphs
"""
import numpy as np
import pytest
import sys
sys.path.insert(0, 'src')
sys.path.insert(0, '.')

from triaxis.lexicon import GLLL
from gr_solver.gr_geometry import connection_coeff, lambda_laplacian


class TestGLLLTGSGlyphs:
    """Test TGS-specific GLLL opcode definitions."""
    
    def test_ccz_opcode_exists(self):
        """Test CCZ (Controlled-Controlled-Z) opcode is defined."""
        assert hasattr(GLLL, 'CCZ')
        assert GLLL.CCZ == "H128:r70"
        assert GLLL.R70 == "H128:r70"
    
    def test_cs_opcode_exists(self):
        """Test CS (Controlled-S) opcode is defined."""
        assert hasattr(GLLL, 'CS')
        assert GLLL.CS == "H128:r71"
        assert GLLL.R71 == "H128:r71"
    
    def test_lambda_laplacian_opcode_exists(self):
        """Test Lambda Laplacian opcode is defined."""
        assert hasattr(GLLL, 'LAMBDA_LAPLACIAN')
        assert GLLL.LAMBDA_LAPLACIAN == "H128:r72"
        assert GLLL.R72 == "H128:r72"
    
    def test_connection_coeff_opcode_exists(self):
        """Test Connection Coefficient opcode is defined."""
        assert hasattr(GLLL, 'CONNECTION_COEFF')
        assert GLLL.CONNECTION_COEFF == "H128:r73"
        assert GLLL.R73 == "H128:r73"
    
    def test_h128_extensions_defined(self):
        """Test all H128 extensions are properly defined."""
        expected = {
            'R64': 'H128:r64', 'R65': 'H128:r65', 'R66': 'H128:r66',
            'R67': 'H128:r67', 'R68': 'H128:r68', 'R69': 'H128:r69',
            'R70': 'H128:r70', 'R71': 'H128:r71', 'R72': 'H128:r72',
            'R73': 'H128:r73'
        }
        for attr, expected_id in expected.items():
            assert hasattr(GLLL, attr)
            assert getattr(GLLL, attr) == expected_id


class TestConnectionCoeff:
    """Test connection coefficient (Christoffel symbol) computation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Simple 3x3x3 grid
        N = 5
        self.Nx = N
        self.Ny = N
        self.Nz = N
        self.dx = 0.1
        self.dy = 0.1
        self.dz = 0.1
        
        # Flat metric (identity) - should give zero Christoffel symbols
        self.metric_flat = np.ones((N, N, N, 6))
        self.metric_flat[..., 0] = 1.0  # g_xx
        self.metric_flat[..., 1] = 0.0  # g_xy
        self.metric_flat[..., 2] = 0.0  # g_xz
        self.metric_flat[..., 3] = 1.0  # g_yy
        self.metric_flat[..., 4] = 0.0  # g_yz
        self.metric_flat[..., 5] = 1.0  # g_zz
        
        # Coordinates - properly ordered as [Nx, Ny, Nz, 3]
        x = np.linspace(0, (N-1)*self.dx, N)
        y = np.linspace(0, (N-1)*self.dy, N)
        z = np.linspace(0, (N-1)*self.dz, N)
        # Create meshgrid and reshape to [Nx, Ny, Nz, 3]
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        self.coords = np.stack([xx, yy, zz], axis=-1)
        
    def test_flat_space_christoffel_zero(self):
        """Test that Christoffel symbols vanish in flat space."""
        # For flat metric (identity), all Christoffel symbols should be zero
        result = connection_coeff(0, 0, 0, self.metric_flat, self.coords)
        assert np.allclose(result, 0.0, atol=1e-10)
        
        result = connection_coeff(1, 0, 1, self.metric_flat, self.coords)
        assert np.allclose(result, 0.0, atol=1e-10)
        
        result = connection_coeff(2, 1, 2, self.metric_flat, self.coords)
        assert np.allclose(result, 0.0, atol=1e-10)
    
    def test_connection_coeff_shape(self):
        """Test that connection coefficient output has correct shape."""
        result = connection_coeff(0, 0, 0, self.metric_flat, self.coords)
        assert result.shape == (self.Nx, self.Ny, self.Nz)
    
    def test_connection_coeff_indices(self):
        """Test that connection coefficient works for different indices."""
        for lam in range(3):
            for mu in range(3):
                for nu in range(3):
                    result = connection_coeff(lam, mu, nu, self.metric_flat, self.coords)
                    assert result.shape == (self.Nx, self.Ny, self.Nz)


class TestLambdaLaplacian:
    """Test lambda Laplacian computation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        N = 5
        self.Nx = N
        self.Ny = N
        self.Nz = N
        self.dx = 0.1
        self.dy = 0.1
        self.dz = 0.1
        
        # Flat metric (identity)
        self.metric = np.ones((N, N, N, 6))
        self.metric[..., 0] = 1.0  # g_xx
        self.metric[..., 1] = 0.0  # g_xy
        self.metric[..., 2] = 0.0  # g_xz
        self.metric[..., 3] = 1.0  # g_yy
        self.metric[..., 4] = 0.0  # g_yz
        self.metric[..., 5] = 1.0  # g_zz
        
        # Coordinates - properly ordered as [Nx, Ny, Nz, 3]
        x = np.linspace(0, (N-1)*self.dx, N)
        y = np.linspace(0, (N-1)*self.dy, N)
        z = np.linspace(0, (N-1)*self.dz, N)
        # Create meshgrid and reshape to [Nx, Ny, Nz, 3]
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        self.coords = np.stack([xx, yy, zz], axis=-1)
    
    def test_lambda_laplacian_shape(self):
        """Test that lambda laplacian output has correct shape."""
        field = np.random.rand(self.Nx, self.Ny, self.Nz)
        lambda_param = 1.0
        result = lambda_laplacian(field, lambda_param, self.coords, self.metric)
        assert result.shape == (self.Nx, self.Ny, self.Nz)
    
    def test_lambda_zero_gives_standard_laplacian(self):
        """Test that λ=0 gives standard flat-space Laplacian."""
        # Create a simple field with known second derivative
        x = np.linspace(0, (self.Nx-1)*self.dx, self.Nx)
        y = np.linspace(0, (self.Ny-1)*self.dy, self.Ny)
        z = np.linspace(0, (self.Nz-1)*self.dz, self.Nz)
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        
        # Test function: φ = x² (second derivative = 2)
        field = xx**2
        
        lambda_param = 0.0
        result = lambda_laplacian(field, lambda_param, self.coords, self.metric)
        
        # In flat space, □φ = ∇²φ = ∂²φ/∂x² + ∂²φ/∂y² + ∂²φ/∂z²
        # For φ = x², this is 2 + 0 + 0 = 2
        # Due to periodic BCs, boundaries may have artifacts, so check interior
        interior = result[2:-2, 2:-2, 2:-2]
        expected = 2.0 * np.ones_like(interior)
        # Allow some tolerance for finite difference approximation
        assert np.allclose(interior, expected, atol=0.5)
    
    def test_lambda_coupling_term(self):
        """Test that coupling term λ²φ is added correctly."""
        # Constant field
        field = np.ones((self.Nx, self.Ny, self.Nz))
        
        # For constant field, Laplacian term is zero, so result should be λ²
        lambda_param = 2.0
        result = lambda_laplacian(field, lambda_param, self.coords, self.metric)
        
        expected = lambda_param**2 * np.ones_like(result)
        assert np.allclose(result, expected, atol=1e-10)
    
    def test_lambda_laplacian_zero_field(self):
        """Test lambda laplacian of zero field is zero."""
        field = np.zeros((self.Nx, self.Ny, self.Nz))
        lambda_param = 1.0
        result = lambda_laplacian(field, lambda_param, self.coords, self.metric)
        assert np.allclose(result, 0.0)


class TestGlyphIntegration:
    """Integration tests for TGS glyphs."""
    
    def test_triaxis_lexicon_complete(self):
        """Verify all TGS glyphs are accessible through lexicon."""
        glyphs = [
            ('CCZ', 'H128:r70'),
            ('CS', 'H128:r71'),
            ('LAMBDA_LAPLACIAN', 'H128:r72'),
            ('CONNECTION_COEFF', 'H128:r73'),
        ]
        for name, expected_id in glyphs:
            assert hasattr(GLLL, name), f"Missing GLLL.{name}"
            assert getattr(GLLL, name) == expected_id, f"GLLL.{name} has wrong ID"
    
    def test_h128_mnemonics(self):
        """Verify H128 mnemonics are properly aliased."""
        assert GLLL.KCALL == GLLL.R64
        assert GLLL.STENCIL == GLLL.R65
        assert GLLL.DERIV == GLLL.R66
        assert GLLL.LAPLACE == GLLL.R67
        assert GLLL.ADVECT == GLLL.R68
        assert GLLL.DIFFUSE == GLLL.R69


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
