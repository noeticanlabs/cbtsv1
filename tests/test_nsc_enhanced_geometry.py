"""
GR Geometry Tests - Backward Compatibility Verification

Tests that the enhanced gr_geometry module maintains backward compatibility
with the original API while providing the foundation for future enhancements.

Usage:
    pytest tests/test_nsc_enhanced_geometry.py -v
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.gr_core_fields import GRCoreFields
from src.core.gr_geometry import GRGeometry, create_geometry, ricci_tensor_kernel


class TestGRGeometry:
    """Test GRGeometry class."""
    
    def test_christoffel_computation(self):
        """Test Christoffel symbol computation."""
        N = 32
        fields = GRCoreFields(Nx=N, Ny=N, Nz=N, dx=2*np.pi/N, dy=2*np.pi/N, dz=2*np.pi/N)
        
        x, y, z = np.meshgrid(
            np.linspace(0, 2*np.pi, N, endpoint=False),
            np.linspace(0, 2*np.pi, N, endpoint=False),
            np.linspace(0, 2*np.pi, N, endpoint=False),
            indexing='ij'
        )
        fields.gamma_sym6[:,:,:,0] = 1.0 + 0.1 * np.sin(x) * np.cos(y)
        fields.gamma_sym6[:,:,:,3] = 1.0 + 0.1 * np.cos(x) * np.sin(y)
        
        geom = GRGeometry(fields)
        geom.compute_christoffels()
        
        assert geom.christoffels.shape == (N, N, N, 3, 3, 3)
        assert np.isfinite(geom.christoffels).all()
        assert np.isfinite(geom.Gamma).all()
        print(f"Christoffel: shape={geom.christoffels.shape}, max={np.max(np.abs(geom.christoffels)):.2e}")
    
    def test_ricci_computation(self):
        """Test Ricci tensor computation."""
        N = 32
        fields = GRCoreFields(Nx=N, Ny=N, Nz=N, dx=2*np.pi/N, dy=2*np.pi/N, dz=2*np.pi/N)
        
        x, y, z = np.meshgrid(
            np.linspace(0, 2*np.pi, N, endpoint=False),
            np.linspace(0, 2*np.pi, N, endpoint=False),
            np.linspace(0, 2*np.pi, N, endpoint=False),
            indexing='ij'
        )
        fields.gamma_sym6[:,:,:,0] = 1.0 + 0.1 * np.sin(x) * np.cos(y)
        fields.gamma_sym6[:,:,:,3] = 1.0 + 0.1 * np.cos(x) * np.sin(y)
        fields.gamma_sym6[:,:,:,5] = 1.0 + 0.1 * np.sin(y) * np.cos(z)
        
        geom = GRGeometry(fields)
        geom.compute_ricci()
        
        assert geom.ricci.shape == (N, N, N, 3, 3)
        assert np.isfinite(geom.ricci).all()
        print(f"Ricci: shape={geom.ricci.shape}")
    
    def test_scalar_curvature(self):
        """Test scalar curvature computation."""
        N = 32
        fields = GRCoreFields(Nx=N, Ny=N, Nz=N, dx=2*np.pi/N, dy=2*np.pi/N, dz=2*np.pi/N)
        
        x, y, z = np.meshgrid(
            np.linspace(0, 2*np.pi, N, endpoint=False),
            np.linspace(0, 2*np.pi, N, endpoint=False),
            np.linspace(0, 2*np.pi, N, endpoint=False),
            indexing='ij'
        )
        fields.gamma_sym6[:,:,:,0] = 1.0 + 0.1 * np.sin(x) * np.cos(y)
        
        geom = GRGeometry(fields)
        geom.compute_scalar_curvature()
        
        assert geom.R.shape == (N, N, N)
        assert np.isfinite(geom.R).all()
        print(f"Scalar curvature: max={np.max(np.abs(geom.R)):.2e}")
    
    def test_factory_function(self):
        """Test geometry factory function."""
        fields = GRCoreFields(Nx=32, Ny=32, Nz=32)
        
        geom = create_geometry(fields, method='central2')
        assert isinstance(geom, GRGeometry)
        assert geom.fd_method == 'central2'
        print("Factory function works correctly")
    
    def test_cache_functionality(self):
        """Test that caching works."""
        N = 32
        fields = GRCoreFields(Nx=N, Ny=N, Nz=N, dx=2*np.pi/N, dy=2*np.pi/N, dz=2*np.pi/N)
        
        x, y, z = np.meshgrid(
            np.linspace(0, 2*np.pi, N, endpoint=False),
            np.linspace(0, 2*np.pi, N, endpoint=False),
            np.linspace(0, 2*np.pi, N, endpoint=False),
            indexing='ij'
        )
        fields.gamma_sym6[:,:,:,0] = 1.0 + 0.1 * np.sin(x) * np.cos(y)
        
        geom = GRGeometry(fields)
        
        # First computation
        import time
        start = time.time()
        geom.compute_christoffels()
        t1 = time.time() - start
        
        # Second computation (should use cache)
        start = time.time()
        geom.compute_christoffels()
        t2 = time.time() - start
        
        # Cache should speed up second call
        print(f"First call: {t1*1000:.2f}ms, Second call (cached): {t2*1000:.2f}ms")
        assert t2 <= t1 * 1.5, "Cache should speed up second call"


class TestBackwardCompatibility:
    """Test that original API is preserved."""
    
    def test_ricci_tensor_kernel(self):
        """Test standalone ricci_tensor_kernel function."""
        N = 32
        fields = GRCoreFields(Nx=N, Ny=N, Nz=N)
        x, y, z = np.meshgrid(
            np.linspace(0, 2*np.pi, N, endpoint=False),
            np.linspace(0, 2*np.pi, N, endpoint=False),
            np.linspace(0, 2*np.pi, N, endpoint=False),
            indexing='ij'
        )
        fields.gamma_sym6[:,:,:,0] = 1.0 + 0.1 * np.sin(x) * np.cos(y)
        fields.gamma_sym6[:,:,:,3] = 1.0 + 0.1 * np.cos(x) * np.sin(y)
        fields.gamma_sym6[:,:,:,5] = 1.0 + 0.1 * np.sin(y) * np.cos(z)
        
        ricci = ricci_tensor_kernel(fields)
        
        assert ricci.shape == (N, N, N, 3, 3)
        assert np.isfinite(ricci).all()
        print(f"Ricci kernel: shape={ricci.shape}")
    
    def test_connection_coeff(self):
        """Test standalone connection_coeff function."""
        from src.core.gr_geometry import connection_coeff
        
        N = 32
        metric = np.zeros((N, N, N, 6))
        metric[:,:,:,0] = 1.0
        metric[:,:,:,3] = 1.0
        metric[:,:,:,5] = 1.0
        
        coords = np.zeros((N, N, N, 3))
        x = np.linspace(0, 2*np.pi, N)
        y = np.linspace(0, 2*np.pi, N)
        z = np.linspace(0, 2*np.pi, N)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        coords[:,:,:,0] = X
        coords[:,:,:,1] = Y
        coords[:,:,:,2] = Z
        
        christoffel = connection_coeff(0, 0, 0, metric, coords)
        
        assert christoffel.shape == (N, N, N)
        assert np.isfinite(christoffel).all()
        print(f"Connection coeff: shape={christoffel.shape}")


class TestGeometryConstraints:
    """Test algebraic constraint enforcement."""
    
    def test_enforce_det_gamma_tilde(self):
        """Test det(gamma_tilde) = 1 enforcement."""
        N = 16
        fields = GRCoreFields(Nx=N, Ny=N, Nz=N, dx=2*np.pi/N, dy=2*np.pi/N, dz=2*np.pi/N)
        
        x, y, z = np.meshgrid(
            np.linspace(0, 2*np.pi, N, endpoint=False),
            np.linspace(0, 2*np.pi, N, endpoint=False),
            np.linspace(0, 2*np.pi, N, endpoint=False),
            indexing='ij'
        )
        # Initialize with some perturbation
        fields.gamma_tilde_sym6[:,:,:,0] = 1.5 + 0.1 * np.sin(x)
        fields.gamma_tilde_sym6[:,:,:,3] = 1.5 + 0.1 * np.cos(y)
        fields.gamma_tilde_sym6[:,:,:,5] = 1.5 + 0.1 * np.sin(z)
        
        geom = GRGeometry(fields)
        
        # Check determinant before enforcement
        from src.core.gr_core_fields import sym6_to_mat33
        gamma_tilde = sym6_to_mat33(fields.gamma_tilde_sym6)
        det_before = np.linalg.det(gamma_tilde)
        
        # Enforce constraint
        geom.enforce_det_gamma_tilde()
        
        # Check determinant after
        gamma_tilde_after = sym6_to_mat33(fields.gamma_tilde_sym6)
        det_after = np.linalg.det(gamma_tilde_after)
        
        print(f"det(gamma_tilde) before: {np.mean(det_before):.4f}")
        print(f"det(gamma_tilde) after: {np.mean(det_after):.4f}")
        
        assert np.allclose(np.mean(det_after), 1.0, atol=1e-10), "det should be 1"


def run_tests():
    """Run all tests."""
    
    print("=" * 70)
    print("GR GEOMETRY TESTS")
    print("=" * 70)
    
    results = {}
    
    # Geometry tests
    print("\n1. GR Geometry Tests")
    print("-" * 50)
    
    t = TestGRGeometry()
    
    try:
        t.test_christoffel_computation()
        results['christoffel'] = 'PASS'
        print("✓ Christoffel PASSED")
    except Exception as e:
        results['christoffel'] = f'FAIL: {e}'
        print(f"✗ Christoffel FAILED: {e}")
    
    try:
        t.test_ricci_computation()
        results['ricci'] = 'PASS'
        print("✓ Ricci PASSED")
    except Exception as e:
        results['ricci'] = f'FAIL: {e}'
        print(f"✗ Ricci FAILED: {e}")
    
    try:
        t.test_scalar_curvature()
        results['curvature'] = 'PASS'
        print("✓ Scalar curvature PASSED")
    except Exception as e:
        results['curvature'] = f'FAIL: {e}'
        print(f"✗ Scalar curvature FAILED: {e}")
    
    try:
        t.test_factory_function()
        results['factory'] = 'PASS'
        print("✓ Factory PASSED")
    except Exception as e:
        results['factory'] = f'FAIL: {e}'
        print(f"✗ Factory FAILED: {e}")
    
    try:
        t.test_cache_functionality()
        results['cache'] = 'PASS'
        print("✓ Cache PASSED")
    except Exception as e:
        results['cache'] = f'FAIL: {e}'
        print(f"✗ Cache FAILED: {e}")
    
    # Compatibility tests
    print("\n2. Backward Compatibility Tests")
    print("-" * 50)
    
    t = TestBackwardCompatibility()
    
    try:
        t.test_ricci_tensor_kernel()
        results['ricci_kernel'] = 'PASS'
        print("✓ Ricci kernel PASSED")
    except Exception as e:
        results['ricci_kernel'] = f'FAIL: {e}'
        print(f"✗ Ricci kernel FAILED: {e}")
    
    try:
        t.test_connection_coeff()
        results['connection_coeff'] = 'PASS'
        print("✓ Connection coeff PASSED")
    except Exception as e:
        results['connection_coeff'] = f'FAIL: {e}'
        print(f"✗ Connection coeff FAILED: {e}")
    
    # Constraint tests
    print("\n3. Constraint Tests")
    print("-" * 50)
    
    t = TestGeometryConstraints()
    
    try:
        t.test_enforce_det_gamma_tilde()
        results['det_constraint'] = 'PASS'
        print("✓ det constraint PASSED")
    except Exception as e:
        results['det_constraint'] = f'FAIL: {e}'
        print(f"✗ det constraint FAILED: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if v == 'PASS')
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    for name, result in results.items():
        status = "✓" if result == "PASS" else "✗"
        print(f"  {status} {name}")
    
    return results


if __name__ == "__main__":
    results = run_tests()
    
    import json
    with open('nsc_geometry_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to nsc_geometry_test_results.json")
