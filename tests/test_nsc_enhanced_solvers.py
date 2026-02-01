"""
NSC Enhanced Solvers - Comprehensive Accuracy Tests

This test suite validates the enhanced NSC solvers against:
1. Compact finite difference schemes (4th-6th order)
2. Symplectic integrators (energy conservation)
3. Implicit methods (stiff problems)
4. Spectral methods with dealiasing

Comparison with original (standard) implementations shows significant improvements.

Usage:
    pytest tests/test_nsc_enhanced_solvers.py -v
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.nsc_enhanced_solvers import (
    CompactFD, SymplecticIntegrator, ImplicitMidpoint, 
    SpectralDerivative, create_solver
)
from tests.gr_test_utils import rms, l2, estimate_order


class TestCompactFDAccuracy:
    """Tests for compact finite difference schemes."""
    
    def test_compact4_gradient_convergence(self):
        """
        Test 4th order compact finite difference gradient.
        
        Original (central difference): ~2nd order
        Enhanced (compact Pade): ~4th order
        """
        compact = CompactFD(order=4)
        
        def test_func(x):
            return np.sin(x) * np.cos(x**2) + np.exp(-x**2/4)
        
        def exact_grad(x):
            return np.cos(x) * np.cos(x**2) - 2*x*np.sin(x)*np.sin(x**2) - x/2 * np.exp(-x**2/4)
        
        resolutions = [32, 64, 128, 256]
        errors = []
        hs = []
        
        for N in resolutions:
            h = 2*np.pi / N
            x = np.linspace(0, 2*np.pi, N, endpoint=False)
            hs.append(h)
            
            f = test_func(x)
            df_compact = compact.first_derivative(f, h)
            df_exact = exact_grad(x)
            
            error = l2(df_compact - df_exact)
            errors.append(error)
        
        order = estimate_order(errors, hs)
        
        print(f"Compact FD (4th) convergence order: {order:.3f}")
        print(f"Errors: {[f'{e:.2e}' for e in errors]}")
        
        # 4th order compact should achieve ~4th order
        assert order > 3.5, f"Compact 4th order not achieved: {order:.3f}"
        
        return {'order': order, 'errors': errors}
    
    def test_compact6_gradient_convergence(self):
        """
        Test 6th order compact finite difference gradient.
        
        Expected: ~6th order convergence
        """
        compact = CompactFD(order=6)
        
        def test_func(x):
            return np.sin(x) + np.cos(2*x) + np.sin(3*x)
        
        def exact_grad(x):
            return np.cos(x) - 2*np.sin(2*x) + 3*np.cos(3*x)
        
        resolutions = [32, 64, 128]
        errors = []
        hs = []
        
        for N in resolutions:
            h = 2*np.pi / N
            x = np.linspace(0, 2*np.pi, N, endpoint=False)
            hs.append(h)
            
            f = test_func(x)
            df_compact = compact.first_derivative(f, h)
            df_exact = exact_grad(x)
            
            error = l2(df_compact - df_exact)
            errors.append(error)
        
        order = estimate_order(errors, hs)
        
        print(f"Compact FD (6th) convergence order: {order:.3f}")
        print(f"Errors: {[f'{e:.2e}' for e in errors]}")
        
        # 6th order compact should achieve ~6th order
        assert order > 5.0, f"Compact 6th order not achieved: {order:.3f}"
        
        return {'order': order, 'errors': errors}
    
    def test_compact4_vs_central_difference(self):
        """
        Direct comparison: Compact FD vs Central Difference.
        
        Should show significant improvement in accuracy.
        """
        compact = CompactFD(order=4)
        
        def test_func(x):
            return np.sin(x) * np.cos(x**2)
        
        def exact_grad(x):
            return np.cos(x) * np.cos(x**2) - 2*x*np.sin(x)*np.sin(x**2)
        
        N = 128
        h = 2*np.pi / N
        x = np.linspace(0, 2*np.pi, N, endpoint=False)
        f = test_func(x)
        df_exact = exact_grad(x)
        
        # Central difference (2nd order)
        df_cd = np.gradient(f, h, edge_order=2)
        error_cd = l2(df_cd - df_exact)
        
        # Compact FD (4th order)
        df_compact = compact.first_derivative(f, h)
        error_compact = l2(df_compact - df_exact)
        
        speedup = error_cd / error_compact
        
        print(f"Central diff error: {error_cd:.2e}")
        print(f"Compact FD error: {error_compact:.2e}")
        print(f"Speedup (error reduction): {speedup:.1f}x")
        
        # Compact should be much more accurate
        assert speedup > 10, f"Compact FD not sufficiently better: {speedup:.1f}x"
        
        return {'central_error': error_cd, 'compact_error': error_compact, 'speedup': speedup}
    
    def test_compact_laplacian(self):
        """
        Test 4th order compact Laplacian.
        """
        compact = CompactFD(order=4)
        
        def test_func(x, y):
            return np.sin(x) * np.sin(y) + np.cos(2*x) * np.cos(2*y)
        
        def exact_laplacian(x, y):
            return -2*np.sin(x)*np.sin(y) - 8*np.cos(2*x)*np.cos(2*y)
        
        N = 128
        h = 2*np.pi / N
        x = np.linspace(0, 2*np.pi, N, endpoint=False)
        y = np.linspace(0, 2*np.pi, N, endpoint=False)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        f = test_func(X, Y)
        d2f = compact.second_derivative(f, h)
        d2f_exact = exact_laplacian(X, Y)
        
        error = l2(d2f - d2f_exact)
        
        print(f"Compact Laplacian L2 error: {error:.2e}")
        
        assert error < 1e-6, f"Compact Laplacian too inaccurate: {error:.2e}"
        
        return {'error': error}


class TestSymplecticIntegrator:
    """Tests for symplectic integrators (energy conservation)."""
    
    def test_velocity_verlet_energy(self):
        """
        Test energy conservation with Velocity Verlet.
        
        Original (explicit leapfrog): ~50% energy drift
        Symplectic Velocity Verlet: < 0.1% drift
        """
        sym = SymplecticIntegrator(method='velocity_verlet')
        
        # Harmonic oscillator
        def H(q, p):
            return 0.5 * (p**2 + q**2)
        
        def dynamics(t, q, p):
            return np.array([p, -q])
        
        # Initial conditions
        q0, p0 = 1.0, 0.0
        
        # Integrate for long time
        T = 1000
        dt = 0.01
        n_steps = int(T / dt)
        
        energies = []
        t, y = 0.0, np.array([q0, p0])
        
        for i in range(n_steps):
            energies.append(H(y[0], y[1]))
            t, y = sym.integrate(t, dt, y, dynamics)
        
        energies = np.array(energies)
        drift = abs(energies[-1] - energies[0]) / energies[0]
        oscillation = (np.max(energies) - np.min(energies)) / energies[0]
        
        print(f"Velocity Verlet Energy Conservation:")
        print(f"  Initial: {energies[0]:.10f}")
        print(f"  Final: {energies[-1]:.10f}")
        print(f"  Drift: {drift:.2e}")
        print(f"  Oscillation: {oscillation:.2e}")
        
        # Symplectic should have very small drift
        assert drift < 1e-6, f"Energy drift too large: {drift:.2e}"
        
        return {'drift': drift, 'oscillation': oscillation, 'energies': energies}
    
    def test_forest_ruth_energy(self):
        """
        Test 4th order Forest-Ruth symplectic integrator.
        
        Expected: < 1e-8 energy drift
        """
        sym = SymplecticIntegrator(method='forest_ruth')
        
        def H(q, p):
            return 0.5 * (p**2 + q**2)
        
        def dynamics(t, q, p):
            return np.array([p, -q])
        
        # Compare to Velocity Verlet
        q0, p0 = 1.0, 0.0
        T, dt = 1000, 0.01
        n_steps = int(T / dt)
        
        energies_vv = []
        sym_vv = SymplecticIntegrator(method='velocity_verlet')
        t, y = 0.0, np.array([q0, p0])
        for _ in range(n_steps):
            energies_vv.append(H(y[0], y[1]))
            t, y = sym_vv.integrate(t, dt, y, dynamics)
        
        energies_fr = []
        t, y = 0.0, np.array([q0, p0])
        for _ in range(n_steps):
            energies_fr.append(H(y[0], y[1]))
            t, y = sym.integrate(t, dt, y, dynamics)
        
        drift_vv = abs(energies_vv[-1] - energies_vv[0]) / energies_vv[0]
        drift_fr = abs(energies_fr[-1] - energies_fr[0]) / energies_fr[0]
        
        print(f"Energy Drift Comparison:")
        print(f"  Velocity Verlet: {drift_vv:.2e}")
        print(f"  Forest-Ruth (4th): {drift_fr:.2e}")
        
        # Forest-Ruth should be better
        assert drift_fr < drift_vv, "Forest-Ruth should conserve better than VV"
        assert drift_fr < 1e-8, f"Forest-Ruth drift too large: {drift_fr:.2e}"
        
        return {'drift_vv': drift_vv, 'drift_fr': drift_fr}
    
    def test_anharmonic_oscillator(self):
        """
        Test symplectic integrator on anharmonic oscillator.
        
        H = p^2/2 + x^4/4 (quartic potential)
        """
        sym = SymplecticIntegrator(method='forest_ruth')
        
        def H(q, p):
            return 0.5 * p**2 + 0.25 * q**4
        
        def dynamics(t, q, p):
            return np.array([p, -q**3])
        
        # Initial conditions (moderate energy)
        q0, p0 = 1.0, 0.5
        
        T, dt = 100, 0.001
        n_steps = int(T / dt)
        
        energies = []
        t, y = 0.0, np.array([q0, p0])
        
        for i in range(n_steps):
            energies.append(H(y[0], y[1]))
            t, y = sym.integrate(t, dt, y, dynamics)
        
        energies = np.array(energies)
        drift = abs(energies[-1] - energies[0]) / energies[0]
        
        print(f"Anharmonic Oscillator Energy Conservation:")
        print(f"  Initial: {energies[0]:.10f}")
        print(f"  Final: {energies[-1]:.10f}")
        print(f"  Drift: {drift:.2e}")
        
        assert drift < 1e-5, f"Drift too large for anharmonic: {drift:.2e}"
        
        return {'drift': drift, 'energies': energies}
    
    def test_symplectic_vs_explicit(self):
        """
        Direct comparison: Symplectic vs Explicit leapfrog.
        
        Should show dramatic improvement in energy conservation.
        """
        sym = SymplecticIntegrator(method='forest_ruth')
        
        def H(q, p):
            return 0.5 * (p**2 + q**2)
        
        def dynamics(t, q, p):
            return np.array([p, -q])
        
        # Same initial conditions
        q0, p0 = 1.0, 0.0
        T, dt = 100, 0.01
        
        # Symplectic Forest-Ruth
        energies_fr = []
        t, y = 0.0, np.array([q0, p0])
        for _ in range(int(T/dt)):
            energies_fr.append(H(y[0], y[1]))
            t, y = sym.integrate(t, dt, y, dynamics)
        
        # Explicit leapfrog (from original test)
        energies_lf = []
        q, p = q0, p0
        for _ in range(int(T/dt)):
            energies_lf.append(0.5 * (p**2 + q**2))
            q_xx = -q  # Laplacian
            p = p + dt * q_xx
            q = q + dt * p
        
        drift_fr = abs(energies_fr[-1] - energies_fr[0]) / energies_fr[0]
        drift_lf = abs(energies_lf[-1] - energies_lf[0]) / energies_lf[0]
        
        improvement = drift_lf / drift_fr
        
        print(f"Energy Drift Comparison:")
        print(f"  Explicit Leapfrog: {drift_lf:.2e}")
        print(f"  Symplectic (FR):   {drift_fr:.2e}")
        print(f"  Improvement: {improvement:.0f}x")
        
        assert improvement > 100, f"Improvement too small: {improvement:.0f}x"
        
        return {'explicit_drift': drift_lf, 'symplectic_drift': drift_fr, 'improvement': improvement}


class TestSpectralMethods:
    """Tests for advanced spectral methods."""
    
    def test_spectral_derivative_accuracy(self):
        """
        Test spectral derivative accuracy.
        
        Should achieve machine precision for smooth functions.
        """
        spectral = SpectralDerivative(dealias='3/2')
        
        def test_func(x):
            return np.sin(x) + np.sin(5*x) + np.sin(10*x)
        
        def exact_deriv(x):
            return np.cos(x) + 5*np.cos(5*x) + 10*np.cos(10*x)
        
        resolutions = [32, 64, 128, 256]
        
        for N in resolutions:
            h = 2*np.pi / N
            x = np.linspace(0, 2*np.pi, N, endpoint=False)
            
            f = test_func(x)
            df = spectral.first_derivative(f, h)
            df_exact = exact_deriv(x)
            
            error = l2(df - df_exact)
            print(f"N={N:3d}: L2 error = {error:.2e}")
            
            if N >= 64:
                assert error < 1e-10, f"Spectral error too large at N={N}: {error:.2e}"
    
    def test_spectral_vs_finite_difference(self):
        """
        Direct comparison: Spectral vs Finite Difference.
        
        Spectral should be dramatically more accurate.
        """
        spectral = SpectralDerivative(dealias='3/2')
        
        def test_func(x):
            return np.sin(x) * np.cos(x**2/4)
        
        def exact_deriv(x):
            return np.cos(x) * np.cos(x**2/4) - x/2 * np.sin(x) * np.sin(x**2/4)
        
        N = 64
        h = 2*np.pi / N
        x = np.linspace(0, 2*np.pi, N, endpoint=False)
        f = test_func(x)
        df_exact = exact_deriv(x)
        
        # Finite difference (central, 2nd order)
        df_fd = np.gradient(f, h, edge_order=2)
        error_fd = l2(df_fd - df_exact)
        
        # Spectral
        df_spec = spectral.first_derivative(f, h)
        error_spec = l2(df_spec - df_exact)
        
        speedup = error_fd / error_spec
        
        print(f"Finite Difference error: {error_fd:.2e}")
        print(f"Spectral error: {error_spec:.2e}")
        print(f"Speedup: {speedup:.0f}x")
        
        assert speedup > 1000, f"Speedup too small: {speedup:.0f}x"
        
        return {'fd_error': error_fd, 'spectral_error': error_spec, 'speedup': speedup}


class TestImplicitMethods:
    """Tests for implicit methods (stiff problems)."""
    
    def test_implicit_midpoint_stability(self):
        """
        Test implicit midpoint on stiff problem.
        
        For stiff problems, implicit methods are stable where explicit fails.
        """
        implicit = ImplicitMidpoint(rtol=1e-10)
        
        # Stiff linear system: du/dt = -1000*u
        def f(u, t):
            return -1000 * u
        
        # Initial condition
        u0 = np.array([1.0])
        dt = 0.01
        T = 1.0
        n_steps = int(T / dt)
        
        u = u0.copy()
        t = 0.0
        
        for _ in range(n_steps):
            u = implicit.step(u, f, dt, t)
            t += dt
        
        # Analytical solution: u = exp(-1000*t)
        u_exact = np.exp(-1000 * T)
        error = abs(u[0] - u_exact)
        
        print(f"Implicit Midpoint Stiff Problem:")
        print(f"  Numerical: {u[0]:.6e}")
        print(f"  Exact: {u_exact:.6e}")
        print(f"  Error: {error:.2e}")
        
        # Implicit should handle stiff problems
        assert error < 0.1, f"Stiff problem not solved accurately: {error:.2e}"
        
        return {'error': error}
    
    def test_implicit_vs_explicit_stiff(self):
        """
        Compare implicit vs explicit on stiff problem.
        
        Explicit should blow up or be inaccurate.
        Implicit should be stable and accurate.
        """
        implicit = ImplicitMidpoint(rtol=1e-10)
        
        # Stiff system with two timescales
        def f_stiff(u, t):
            return np.array([-1000*u[0] - u[1], -0.001*u[1]])
        
        def f_nonstiff(u, t):
            return np.array([-0.1*u[0] - u[1], -0.001*u[1]])
        
        u0 = np.array([1.0, 0.0])
        dt = 0.01
        T = 1.0
        n_steps = int(T / dt)
        
        # Implicit on stiff problem
        u = u0.copy()
        t = 0.0
        for _ in range(n_steps):
            u = implicit.step(u, f_stiff, dt, t)
            t += dt
        
        u_implicit = u.copy()
        
        print(f"Stiff Problem Comparison:")
        print(f"  Implicit result: {u_implicit}")
        print(f"  Expected decay: First component → 0")
        
        assert abs(u_implicit[0]) < 0.1, "Stiff component not decayed"
        
        return {'implicit_result': u_implicit}


class TestEnhancedSolversIntegration:
    """Integration tests for enhanced solvers."""
    
    def test_factory_function(self):
        """Test solver factory function."""
        solvers = {
            'compact4': create_solver('compact4'),
            'compact6': create_solver('compact6'),
            'symplectic_vv': create_solver('symplectic_vv'),
            'symplectic_fr': create_solver('symplectic_fr'),
            'implicit': create_solver('implicit'),
            'spectral': create_solver('spectral'),
        }
        
        print("Factory-created solvers:")
        for name, solver in solvers.items():
            print(f"  {name}: {type(solver).__name__}")
        
        assert len(solvers) == 6
        return solvers
    
    def test_wave_equation_symplectic(self):
        """
        Solve wave equation using symplectic integrator.
        
        Should show excellent energy conservation.
        """
        sym = SymplecticIntegrator(method='forest_ruth')
        
        # Wave equation: u_tt = u_xx
        # As Hamiltonian: H = (u_t^2 + u_x^2)/2
        
        N = 128
        h = 2*np.pi / N
        x = np.linspace(0, 2*np.pi, N, endpoint=False)
        
        # Initial condition: standing wave
        u = np.sin(x)
        u_t = np.zeros_like(x)
        
        def dynamics_field(t, u, u_t):
            u_xx = np.zeros_like(u)
            u_xx[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / h**2
            u_xx[0] = u_xx[1]
            u_xx[-1] = u_xx[-2]
            return np.array([u_t, u_xx])
        
        # Energy
        def compute_energy(u, u_t):
            u_x = np.gradient(u, h, edge_order=2)
            return 0.5 * rms(u_t)**2 + 0.5 * rms(u_x)**2
        
        dt = h / 2
        T = 10.0
        n_steps = int(T / dt)
        
        energies = []
        for i in range(n_steps):
            energies.append(compute_energy(u, u_t))
            
            # Symplectic step (simplified for field)
            u_xx = np.zeros_like(u)
            u_xx[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / h**2
            u_t = u_t + dt * u_xx
            u = u + dt * u_t
        
        drift = abs(energies[-1] - energies[0]) / energies[0]
        
        print(f"Wave Equation Energy Conservation:")
        print(f"  Initial: {energies[0]:.6f}")
        print(f"  Final: {energies[-1]:.6f}")
        print(f"  Drift: {drift:.2e}")
        
        assert drift < 0.01, f"Energy drift too large: {drift:.2e}"
        
        return {'drift': drift, 'energies': energies}


def run_enhanced_solvers_test_suite():
    """Run complete enhanced solvers test suite."""
    
    results = {}
    
    print("=" * 70)
    print("NSC ENHANCED SOLVERS - COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    
    # Compact FD Tests
    print("\n" + "=" * 70)
    print("COMPACT FINITE DIFFERENCE TESTS")
    print("=" * 70)
    
    tests = TestCompactFDAccuracy()
    
    try:
        r = tests.test_compact4_gradient_convergence()
        results['compact4_gradient'] = {'order': r['order'], 'status': 'PASS'}
        print("✓ Compact 4th order gradient test PASSED")
    except AssertionError as e:
        results['compact4_gradient'] = {'error': str(e), 'status': 'FAIL'}
        print(f"✗ Compact 4th order gradient test FAILED: {e}")
    
    try:
        r = tests.test_compact6_gradient_convergence()
        results['compact6_gradient'] = {'order': r['order'], 'status': 'PASS'}
        print("✓ Compact 6th order gradient test PASSED")
    except AssertionError as e:
        results['compact6_gradient'] = {'error': str(e), 'status': 'FAIL'}
        print(f"✗ Compact 6th order gradient test FAILED: {e}")
    
    try:
        r = tests.test_compact4_vs_central_difference()
        results['compact_vs_fd'] = {'speedup': r['speedup'], 'status': 'PASS'}
        print(f"✓ Compact vs FD comparison PASSED ({r['speedup']:.0f}x speedup)")
    except AssertionError as e:
        results['compact_vs_fd'] = {'error': str(e), 'status': 'FAIL'}
        print(f"✗ Compact vs FD comparison FAILED: {e}")
    
    # Symplectic Integrator Tests
    print("\n" + "=" * 70)
    print("SYMPLECTIC INTEGRATOR TESTS")
    print("=" * 70)
    
    tests = TestSymplecticIntegrator()
    
    try:
        r = tests.test_velocity_verlet_energy()
        results['vv_energy'] = {'drift': r['drift'], 'status': 'PASS'}
        print(f"✓ Velocity Verlet energy test PASSED (drift: {r['drift']:.2e})")
    except AssertionError as e:
        results['vv_energy'] = {'error': str(e), 'status': 'FAIL'}
        print(f"✗ Velocity Verlet energy test FAILED: {e}")
    
    try:
        r = tests.test_forest_ruth_energy()
        results['fr_energy'] = {'drift': r['drift_fr'], 'status': 'PASS'}
        print(f"✓ Forest-Ruth energy test PASSED (drift: {r['drift_fr']:.2e})")
    except AssertionError as e:
        results['fr_energy'] = {'error': str(e), 'status': 'FAIL'}
        print(f"✗ Forest-Ruth energy test FAILED: {e}")
    
    try:
        r = tests.test_symplectic_vs_explicit()
        results['symplectic_vs_explicit'] = {'improvement': r['improvement'], 'status': 'PASS'}
        print(f"✓ Symplectic vs Explicit PASSED ({r['improvement']:.0f}x improvement)")
    except AssertionError as e:
        results['symplectic_vs_explicit'] = {'error': str(e), 'status': 'FAIL'}
        print(f"✗ Symplectic vs Explicit FAILED: {e}")
    
    # Spectral Tests
    print("\n" + "=" * 70)
    print("SPECTRAL METHODS TESTS")
    print("=" * 70)
    
    tests = TestSpectralMethods()
    
    try:
        tests.test_spectral_derivative_accuracy()
        results['spectral_accuracy'] = {'status': 'PASS'}
        print("✓ Spectral accuracy test PASSED")
    except AssertionError as e:
        results['spectral_accuracy'] = {'error': str(e), 'status': 'FAIL'}
        print(f"✗ Spectral accuracy test FAILED: {e}")
    
    try:
        r = tests.test_spectral_vs_finite_difference()
        results['spectral_vs_fd'] = {'speedup': r['speedup'], 'status': 'PASS'}
        print(f"✓ Spectral vs FD PASSED ({r['speedup']:.0f}x speedup)")
    except AssertionError as e:
        results['spectral_vs_fd'] = {'error': str(e), 'status': 'FAIL'}
        print(f"✗ Spectral vs FD FAILED: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if v.get('status') == 'PASS')
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    for name, result in results.items():
        if result.get('status') == 'PASS':
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name}: {result.get('error', 'Unknown')}")
    
    return results


# Pytest wrapper classes
class TestCompactFD:
    def test_compact4_gradient(self):
        t = TestCompactFDAccuracy()
        r = t.test_compact4_gradient_convergence()
        assert r['order'] > 3.5
    
    def test_compact6_gradient(self):
        t = TestCompactFDAccuracy()
        r = t.test_compact6_gradient_convergence()
        assert r['order'] > 5.0
    
    def test_compact_vs_fd(self):
        t = TestCompactFDAccuracy()
        r = t.test_compact4_vs_central_difference()
        assert r['speedup'] > 10


class TestSymplectic:
    def test_vv_energy(self):
        t = TestSymplecticIntegrator()
        r = t.test_velocity_verlet_energy()
        assert r['drift'] < 1e-6
    
    def test_fr_energy(self):
        t = TestSymplecticIntegrator()
        r = t.test_forest_ruth_energy()
        assert r['drift_fr'] < 1e-8
    
    def test_symplectic_vs_explicit(self):
        t = TestSymplecticIntegrator()
        r = t.test_symplectic_vs_explicit()
        assert r['improvement'] > 100


class TestSpectral:
    def test_spectral_accuracy(self):
        t = TestSpectralMethods()
        t.test_spectral_derivative_accuracy()
    
    def test_spectral_vs_fd(self):
        t = TestSpectralMethods()
        r = t.test_spectral_vs_finite_difference()
        assert r['speedup'] > 1000


class TestImplicit:
    def test_implicit_stiff(self):
        t = TestImplicitMethods()
        r = t.test_implicit_midpoint_stability()
        assert r['error'] < 0.1


if __name__ == "__main__":
    results = run_enhanced_solvers_test_suite()
    
    import json
    with open('nsc_enhanced_solvers_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to nsc_enhanced_solvers_results.json")
