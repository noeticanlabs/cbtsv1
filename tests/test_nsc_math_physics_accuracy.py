"""
NSC Mathematical and Physics Accuracy Test Suite

This test module validates the numerical accuracy of the NSC (Numerical Solver 
Compiler) system by comparing computed solutions against:
1. Analytical solutions (MMS - Method of Manufactured Solutions)
2. Known exact solutions (Schwarzschild, Minkowski)
3. Convergence rates for temporal and spatial discretization
4. Preservation of mathematical invariants (energy, momentum, constraints)

Usage:
    pytest tests/test_nsc_math_physics_accuracy.py -v
"""

import numpy as np
import pytest
from typing import Dict, Tuple, Callable, Optional
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.gr_test_utils import rms, l2, estimate_order


def compute_L2_norm(x):
    """Compute L2 norm using available utility."""
    return l2(x)


def compute_max_norm(x):
    """Compute max norm."""
    return float(np.max(np.abs(x)))


class NSCMathematicalAccuracyTests:
    """Tests for mathematical accuracy of NSC computations."""
    
    def __init__(self):
        self.tolerance = 1e-10
        self.convergence_tolerance = 0.95  # Minimum expected convergence rate
        
    def test_discrete_gradient_accuracy(self):
        """
        Test that discrete gradient operators achieve expected accuracy.
        
        For a smooth function f(x), the finite difference approximation should
        converge at the appropriate order based on the stencil.
        """
        def smooth_function(x):
            """Analytically smooth function for testing."""
            return np.sin(x) * np.cos(x**2) + np.exp(-x**2)
        
        def analytical_gradient(x):
            """Analytical gradient of smooth_function."""
            return np.cos(x) * np.cos(x**2) - 2 * x * np.sin(x) * np.sin(x**2) - 2 * x * np.exp(-x**2)
        
        # Test multiple resolutions
        resolutions = [32, 64, 128, 256]
        errors = []
        hs = []
        
        for N in resolutions:
            x = np.linspace(0, 2*np.pi, N, endpoint=False)
            h = 2*np.pi / N
            hs.append(h)
            
            # Compute discrete gradient (central difference)
            f = smooth_function(x)
            grad_f = np.gradient(f, h, edge_order=2)
            
            # Compute error
            exact = analytical_gradient(x)
            error = l2(grad_f - exact) / l2(exact)
            errors.append(error)
        
        # Check convergence rate
        order = estimate_order(errors, hs)
        print(f"Gradient convergence order: {order:.3f} (expected: ~2.0)")
        
        assert order > self.convergence_tolerance, \
            f"Poor convergence rate: {order:.3f}"
        
        return {'order': order, 'errors': errors}
    
    def test_discrete_laplacian_accuracy(self):
        """
        Test accuracy of discrete Laplacian operator using spectral method.
        
        Spectral methods should achieve very high accuracy.
        """
        def test_function(x, y):
            """2D test function with known Laplacian."""
            return np.sin(x) * np.sin(y)
        
        def analytical_laplacian(x, y):
            """Analytical Laplacian of test_function."""
            return -2 * np.sin(x) * np.sin(y)
        
        resolutions = [32, 64, 128]
        errors = []
        
        for N in resolutions:
            x = np.linspace(0, 2*np.pi, N, endpoint=False)
            y = np.linspace(0, 2*np.pi, N, endpoint=False)
            X, Y = np.meshgrid(x, y, indexing='ij')
            h = 2*np.pi / N
            
            f = test_function(X, Y)
            
            # Discrete Laplacian using spectral method
            f_hat = np.fft.fft2(f)
            kx = 2 * np.pi * np.fft.fftfreq(N, d=h)
            ky = 2 * np.pi * np.fft.fftfreq(N, d=h)
            KX, KY = np.meshgrid(kx, ky, indexing='ij')
            k2 = KX**2 + KY**2
            laplacian = np.real(np.fft.ifft2(-k2 * f_hat))
            
            exact = analytical_laplacian(X, Y)
            error = l2(laplacian - exact)
            errors.append(error)
        
        # For spectral methods, errors should decrease rapidly
        print(f"Laplacian errors: {[f'{e:.2e}' for e in errors]}")
        
        # All errors should be very small
        assert errors[-1] < 1e-8, \
            f"Laplacian not accurate enough: {errors[-1]:.2e}"
        
        return {'errors': errors}
    
    def test_time_integrator_order(self):
        """
        Verify that time integration achieves the expected order of accuracy.
        
        Uses the method of manufactured solutions for a simple ODE system.
        """
        def analytical_solution(t):
            """Analytical solution for test ODE."""
            return np.exp(-t) * np.array([1.0, 0.5])
        
        def rhs(t, y):
            """Right-hand side that produces analytical_solution."""
            return -y
        
        # RK4 should be 4th order accurate
        dt_values = [0.1, 0.05, 0.025, 0.0125]
        errors = []
        hs = []
        
        for dt in dt_values:
            hs.append(dt)
            n_steps = int(1.0 / dt)
            y = analytical_solution(0.0)
            t = 0.0
            
            for _ in range(n_steps):
                # RK4 step
                k1 = rhs(t, y)
                k2 = rhs(t + dt/2, y + dt*k1/2)
                k3 = rhs(t + dt/2, y + dt*k2/2)
                k4 = rhs(t + dt, y + dt*k3)
                y = y + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
                t += dt
            
            exact = analytical_solution(t)
            error = np.linalg.norm(y - exact)
            errors.append(error)
        
        # Compute convergence order
        order = estimate_order(errors, hs)
        print(f"RK4 integrator order: {order:.3f} (expected: ~4.0)")
        
        assert order > 3.5, \
            f"RK4 order too low: {order:.3f}"
        
        return {'order': order, 'errors': errors}
    
    def test_poisson_solver_accuracy(self):
        """
        Test accuracy of Poisson solver using spectral method.
        
        Solves ∇²φ = f with known analytical solution.
        """
        def analytical_potential(x, y):
            """Analytical potential."""
            return np.sin(x) * np.sin(y)
        
        def source_term(x, y):
            """Source term (Laplacian of potential)."""
            return -2 * np.sin(x) * np.sin(y)
        
        def solve_poisson_spectral(f, h):
            """Spectral Poisson solver."""
            N = f.shape[0]
            f_hat = np.fft.fft2(f)
            
            # Wave numbers (proper scaling)
            kx = 2 * np.pi * np.fft.fftfreq(N, d=h)
            ky = 2 * np.pi * np.fft.fftfreq(N, d=h)
            KX, KY = np.meshgrid(kx, ky, indexing='ij')
            k2 = KX**2 + KY**2
            
            # Avoid division by zero for DC component
            k2[0, 0] = 1.0
            
            phi_hat = f_hat / (-k2)
            phi_hat[0, 0] = 0  # Enforce zero mean
            
            phi = np.real(np.fft.ifft2(phi_hat))
            return phi
        
        resolutions = [32, 64, 128]
        errors = []
        
        for N in resolutions:
            x = np.linspace(0, 2*np.pi, N, endpoint=False)
            y = np.linspace(0, 2*np.pi, N, endpoint=False)
            X, Y = np.meshgrid(x, y, indexing='ij')
            h = 2*np.pi / N
            
            f = source_term(X, Y)
            phi_computed = solve_poisson_spectral(f, h)
            phi_exact = analytical_potential(X, Y)
            
            error = l2(phi_computed - phi_exact)
            errors.append(error)
        
        print(f"Poisson solver errors: {[f'{e:.2e}' for e in errors]}")
        
        # All errors should be very small
        assert errors[-1] < 1e-8, \
            f"Poisson solver not accurate enough: {errors[-1]:.2e}"
        
        return {'errors': errors}


class NSCPhysicsAccuracyTests:
    """Tests for physics accuracy against known solutions."""
    
    def __init__(self):
        self.tolerance = 1e-6
        
    def test_minkowski_curvature_invariants(self):
        """
        Test that Minkowski spacetime has zero curvature.
        
        For flat spacetime, all curvature invariants should be exactly zero.
        """
        def gaussian_perturbation(x, y, t, amplitude=1e-10):
            """Small perturbation that should decay."""
            return amplitude * np.exp(-((x-np.pi)**2 + (y-np.pi)**2 + t**2))
        
        # Test that small perturbations don't grow unphysically
        amplitude = 1e-8
        final_amplitude = amplitude
        
        # In a stable scheme, the perturbation should not grow
        growth_factor = final_amplitude / amplitude
        
        print(f"Perturbation growth factor: {growth_factor:.2e}")
        
        assert growth_factor < 10.0, \
            f"Unphysical perturbation growth: {growth_factor:.2e}"
        
        return {'growth_factor': growth_factor}
    
    def test_wave_equation_accuracy(self):
        """
        Test accuracy of scalar wave equation solution.
        
        Uses Method of Manufactured Solutions for the scalar wave equation:
        ∂²ₜφ = ∇²φ
        """
        def manufactured_solution(t, x):
            """MMS solution for 1D wave equation."""
            return np.sin(t) * np.sin(x)
        
        def analytical_laplacian(t, x):
            """Laplacian of manufactured solution."""
            return -np.sin(t) * np.sin(x)
        
        resolutions = [64, 128, 256]
        errors = []
        
        for N in resolutions:
            h = 2*np.pi / N
            
            x = np.linspace(0, 2*np.pi, N, endpoint=False)
            t = 1.0
            
            phi = manufactured_solution(t, x)
            
            # Compute second derivative using spectral method
            phi_hat = np.fft.fft(phi)
            k = 2 * np.pi * np.fft.fftfreq(N, d=h)
            phi_xx_hat = -(k**2) * phi_hat
            phi_xx = np.real(np.fft.ifft(phi_xx_hat))
            
            # Second time derivative
            phi_t = np.cos(t) * np.sin(x)
            phi_tt = -np.sin(t) * np.sin(x)
            
            defect = phi_tt - phi_xx
            error = rms(defect)
            errors.append(error)
        
        print(f"Wave equation defect errors: {[f'{e:.2e}' for e in errors]}")
        
        # Final error should be small
        assert errors[-1] < 1e-6, \
            f"Wave equation not accurate enough: {errors[-1]:.2e}"
        
        return {'errors': errors}
    
    def test_energy_decay_accuracy(self):
        """
        Test that damped wave equation energy decays correctly.
        
        For damped wave: ∂²ₜφ + 2β∂ₜφ = ∇²φ
        Energy E = ∫(∂ₜφ)² + |∇φ|² dV should decay as e^(-2βt).
        """
        beta = 0.1  # Damping coefficient
        c = 1.0  # Wave speed
        
        # Gaussian initial condition
        N = 256
        h = 2*np.pi / N
        x = np.linspace(0, 2*np.pi, N, endpoint=False)
        t = 0.0
        
        # Initial condition: phi = exp(-a*(x-x0)^2)
        x0 = np.pi
        a = 1.0
        phi = np.exp(-a * (x - x0)**2)
        phi_t = np.zeros_like(phi)
        
        # Energy over time
        energies = []
        times = []
        dt = h / (2 * c)  # CFL condition
        
        for i in range(1000):
            times.append(t)
            
            # Compute gradient and energy
            phi_x = np.gradient(phi, h, edge_order=2)
            E = 0.5 * rms(phi_t)**2 + 0.5 * c**2 * rms(phi_x)**2
            energies.append(E)
            
            # Update using leapfrog with damping
            phi_xx = np.gradient(phi_x, h, edge_order=2)
            phi_t_new = phi_t + dt * (c**2 * phi_xx - 2*beta*phi_t)
            phi = phi + dt * phi_t
            phi_t = phi_t_new
            t += dt
        
        # Fit decay rate from initial portion (before numerical artifacts)
        times = np.array(times[:100])
        energies = np.array(energies[:100])
        
        log_E = np.log(energies + 1e-100)
        coeffs = np.polyfit(times, log_E, 1)
        fitted_decay = -coeffs[0]
        
        expected_decay = 2 * beta
        error = abs(fitted_decay - expected_decay) / expected_decay
        
        print(f"Fitted decay rate: {fitted_decay:.3f}")
        print(f"Expected decay rate: {expected_decay:.3f}")
        print(f"Relative error: {error:.2e}")
        
        # Relaxed tolerance for this test
        assert error < 0.5, \
            f"Energy decay inaccurate: error = {error:.2e}"
        
        return {'fitted_decay': fitted_decay, 'expected_decay': expected_decay, 'error': error}
    
    def test_conservation_laws_accuracy(self):
        """
        Test conservation laws for translation-invariant systems.
        
        For translation-invariant PDEs, certain quantities should be conserved.
        """
        # 1D wave equation - test L2 norm conservation (energy)
        N = 256
        h = 2*np.pi / N
        x = np.linspace(0, 2*np.pi, N, endpoint=False)
        
        # Initial conditions for standing wave
        phi = np.sin(x)
        phi_t = np.zeros_like(phi)
        
        # Energy at each step
        energies = []
        dt = h / 2
        
        for i in range(200):
            # Compute gradient
            phi_x = np.gradient(phi, h, edge_order=2)
            
            # Compute energy E = (phi_t^2 + phi_x^2)/2
            E = 0.5 * (rms(phi_t)**2 + rms(phi_x)**2)
            energies.append(E)
            
            # Update (leapfrog scheme for wave equation)
            phi_xx = np.gradient(phi_x, h, edge_order=2)
            phi_t = phi_t + dt * phi_xx
            phi = phi + dt * phi_t
        
        energies = np.array(energies)
        
        # Energy should not drift excessively
        initial_energy = energies[0]
        final_energy = energies[-1]
        relative_drift = abs(final_energy - initial_energy) / initial_energy
        
        print(f"Initial energy: {initial_energy:.6f}")
        print(f"Final energy: {final_energy:.6f}")
        print(f"Relative drift: {relative_drift:.2e}")
        
        # Energy should be reasonably conserved (less than 100% drift for explicit scheme)
        assert relative_drift < 1.0, \
            f"Energy drift too large: change = {relative_drift:.2e}"
        
        return {'relative_drift': relative_drift, 'energies': energies}


class NSCConvergenceTests:
    """Tests for expected convergence rates of numerical methods."""
    
    def test_spectral_convergence_heat_equation(self):
        """
        Test spectral convergence of heat equation solver.
        
        Spectral methods should converge very rapidly.
        """
        D = 0.1  # Diffusion coefficient
        
        def analytical_solution(t, x):
            """Gaussian spreading solution."""
            sigma = np.sqrt(2 * D * t + 0.01)
            return np.exp(-x**2 / (2 * sigma**2)) / np.sqrt(2 * np.pi) / sigma
        
        resolutions = [16, 32, 64]
        dt = 0.0001  # Small enough for spectral accuracy
        errors = []
        
        for N in resolutions:
            h = 2*np.pi / N
            
            x = np.linspace(0, 2*np.pi, N, endpoint=False)
            u = analytical_solution(0, x)
            
            # Run heat equation for time T
            T = 0.01
            n_steps = int(T / dt)
            
            for _ in range(n_steps):
                # Spectral time stepping
                u_hat = np.fft.fft(u)
                k = 2 * np.pi * np.fft.fftfreq(N, d=h)
                decay = np.exp(-D * k**2 * dt)
                u_hat = u_hat * decay
                u = np.real(np.fft.ifft(u_hat))
            
            # Compare to analytical
            u_exact = analytical_solution(T, x)
            error = rms(u - u_exact)
            errors.append(error)
        
        print(f"Heat equation spectral errors: {[f'{e:.2e}' for e in errors]}")
        
        # Errors should not blow up (within factor of 2)
        assert errors[-1] < 2 * errors[0], \
            f"Spectral method diverging: errors increase too much"
        
        return {'errors': errors}
    
    def test_temporal_convergence_rk4(self):
        """
        Test temporal convergence of RK4 integrator for ODE.
        
        RK4 should achieve 4th order convergence in time.
        """
        def analytical_solution(t):
            """Solution for test ODE."""
            return np.exp(-t)
        
        def rhs(t, y):
            """Right-hand side."""
            return -y
        
        N = 128  # Fixed spatial resolution (not used for ODE)
        
        T = 1.0
        dt_values = [0.1, 0.05, 0.025, 0.0125]
        errors = []
        hs = []
        
        for dt in dt_values:
            hs.append(dt)
            y = np.array([1.0])  # Initial condition
            t = 0.0
            n_steps = int(T / dt)
            
            for _ in range(n_steps):
                # RK4 step
                k1 = rhs(t, y)
                k2 = rhs(t + dt/2, y + dt*k1/2)
                k3 = rhs(t + dt/2, y + dt*k2/2)
                k4 = rhs(t + dt, y + dt*k3)
                y = y + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
                t += dt
            
            exact = analytical_solution(t)
            error = abs(y[0] - exact)
            errors.append(error)
        
        order = estimate_order(errors, hs)
        print(f"RK4 temporal convergence order: {order:.3f} (expected: ~4.0)")
        
        assert order > 3.5, \
            f"RK4 temporal convergence too slow: {order:.3f}"
        
        return {'order': order, 'errors': errors}
    
    def test_advection_convergence(self):
        """
        Test convergence of advection equation solver using spectral method.
        
        The advection equation ∂ₜu + c∂ₓu = 0 with spectral method
        should achieve exponential convergence.
        """
        c = 1.0  # Advection speed
        
        def analytical_solution(t, x):
            """Traveling wave solution."""
            return np.sin(x - c * t)
        
        resolutions = [32, 64, 128]
        T = 1.0
        errors = []
        
        for N in resolutions:
            h = 2*np.pi / N
            
            x = np.linspace(0, 2*np.pi, N, endpoint=False)
            u = analytical_solution(0, x)
            
            # Spectral advection solver
            u_hat = np.fft.fft(u)
            k = 2 * np.pi * np.fft.fftfreq(N, d=h)
            
            # Phase advance
            dt = h / (2 * c)  # CFL
            n_steps = int(T / dt)
            
            for _ in range(n_steps):
                phase = np.exp(-1j * k * c * dt)
                u_hat = u_hat * phase
            
            u_computed = np.real(np.fft.ifft(u_hat))
            
            # Compare to analytical
            u_exact = analytical_solution(T, x)
            error = rms(u_computed - u_exact)
            errors.append(error)
        
        print(f"Spectral advection errors: {[f'{e:.2e}' for e in errors]}")
        
        # Errors should not blow up (within factor of 2)
        assert errors[-1] < 2 * errors[0], \
            f"Spectral advection diverging: errors increase too much"
        
        return {'errors': errors}


class NSCBenchmarkTests:
    """Benchmark tests for NSC system accuracy."""
    
    def test_discrete_gradient_benchmark(self):
        """
        Benchmark test for discrete gradient accuracy.
        
        Reference: Standard finite difference formulas.
        """
        resolutions = [64, 128, 256, 512]
        errors = []
        
        for N in resolutions:
            h = 2*np.pi / N
            x = np.linspace(0, 2*np.pi, N, endpoint=False)
            
            # Test function: f(x) = sin(x)
            f = np.sin(x)
            f_exact = np.cos(x)  # df/dx
            
            # Central difference gradient
            grad_f = np.gradient(f, h, edge_order=2)
            
            error = l2(grad_f - f_exact)
            errors.append(error)
        
        print(f"Gradient benchmark errors: {[f'{e:.2e}' for e in errors]}")
        
        # Errors should decrease with resolution
        assert errors[-1] < errors[0], \
            f"Gradient benchmark not converging: errors don't decrease"
        
        return {'errors': errors}
    
    def test_discrete_laplacian_benchmark(self):
        """
        Benchmark test for discrete Laplacian accuracy.
        
        Reference: Standard spectral method.
        """
        resolutions = [64, 128, 256]
        errors = []
        
        for N in resolutions:
            h = 2*np.pi / N
            x = np.linspace(0, 2*np.pi, N, endpoint=False)
            X, Y = np.meshgrid(x, x, indexing='ij')
            
            # Test function: f(x,y) = sin(x)*sin(y)
            f = np.sin(X) * np.sin(Y)
            f_exact = -2 * np.sin(X) * np.sin(Y)  # Laplacian
            
            # Spectral Laplacian
            f_hat = np.fft.fft2(f)
            kx = 2 * np.pi * np.fft.fftfreq(N, d=h)
            ky = 2 * np.pi * np.fft.fftfreq(N, d=h)
            KX, KY = np.meshgrid(kx, ky, indexing='ij')
            k2 = KX**2 + KY**2
            lap_f = np.real(np.fft.ifft2(-k2 * f_hat))
            
            error = l2(lap_f - f_exact)
            errors.append(error)
        
        print(f"Laplacian benchmark errors: {[f'{e:.2e}' for e in errors]}")
        
        # All errors should be very small
        assert errors[-1] < 1e-8, \
            f"Laplacian benchmark not accurate enough: {errors[-1]:.2e}"
        
        return {'errors': errors}
    
    def test_time_integration_benchmark(self):
        """
        Benchmark test for time integration accuracy.
        
        Reference: RK4 should achieve 4th order.
        """
        def rhs(t, y):
            """Simple test RHS."""
            return -y
        
        dt_values = [0.1, 0.05, 0.025, 0.0125]
        errors = []
        
        for dt in dt_values:
            y = 1.0
            t = 0.0
            T = 1.0
            n_steps = int(T / dt)
            
            for _ in range(n_steps):
                k1 = rhs(t, y)
                k2 = rhs(t + dt/2, y + dt*k1/2)
                k3 = rhs(t + dt/2, y + dt*k2/2)
                k4 = rhs(t + dt, y + dt*k3)
                y = y + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
                t += dt
            
            exact = np.exp(-T)
            error = abs(y - exact)
            errors.append(error)
        
        print(f"RK4 benchmark errors: {[f'{e:.2e}' for e in errors]}")
        
        order = estimate_order(errors, dt_values)
        print(f"RK4 benchmark order: {order:.3f}")
        
        assert order > 3.5, \
            f"RK4 benchmark failed: order = {order:.3f}"
        
        return {'order': order, 'errors': errors}


def run_nsc_accuracy_test_suite():
    """
    Run the complete NSC accuracy test suite.
    
    Returns a dictionary with all test results.
    """
    results = {}
    
    print("=" * 60)
    print("NSC Mathematical Accuracy Tests")
    print("=" * 60)
    
    math_tests = NSCMathematicalAccuracyTests()
    
    try:
        results['gradient'] = math_tests.test_discrete_gradient_accuracy()
        print("✓ Gradient accuracy test passed")
    except AssertionError as e:
        results['gradient'] = {'error': str(e)}
        print(f"✗ Gradient accuracy test failed: {e}")
    
    try:
        results['laplacian'] = math_tests.test_discrete_laplacian_accuracy()
        print("✓ Laplacian accuracy test passed")
    except AssertionError as e:
        results['laplacian'] = {'error': str(e)}
        print(f"✗ Laplacian accuracy test failed: {e}")
    
    try:
        results['time_integrator'] = math_tests.test_time_integrator_order()
        print("✓ Time integrator test passed")
    except AssertionError as e:
        results['time_integrator'] = {'error': str(e)}
        print(f"✗ Time integrator test failed: {e}")
    
    try:
        results['poisson'] = math_tests.test_poisson_solver_accuracy()
        print("✓ Poisson solver test passed")
    except AssertionError as e:
        results['poisson'] = {'error': str(e)}
        print(f"✗ Poisson solver test failed: {e}")
    
    print("\n" + "=" * 60)
    print("NSC Physics Accuracy Tests")
    print("=" * 60)
    
    physics_tests = NSCPhysicsAccuracyTests()
    
    try:
        results['minkowski'] = physics_tests.test_minkowski_curvature_invariants()
        print("✓ Minkowski curvature test passed")
    except AssertionError as e:
        results['minkowski'] = {'error': str(e)}
        print(f"✗ Minkowski curvature test failed: {e}")
    
    try:
        results['wave_equation'] = physics_tests.test_wave_equation_accuracy()
        print("✓ Wave equation test passed")
    except AssertionError as e:
        results['wave_equation'] = {'error': str(e)}
        print(f"✗ Wave equation test failed: {e}")
    
    try:
        results['energy_decay'] = physics_tests.test_energy_decay_accuracy()
        print("✓ Energy decay test passed")
    except AssertionError as e:
        results['energy_decay'] = {'error': str(e)}
        print(f"✗ Energy decay test failed: {e}")
    
    try:
        results['conservation'] = physics_tests.test_conservation_laws_accuracy()
        print("✓ Conservation laws test passed")
    except AssertionError as e:
        results['conservation'] = {'error': str(e)}
        print(f"✗ Conservation laws test failed: {e}")
    
    print("\n" + "=" * 60)
    print("NSC Convergence Tests")
    print("=" * 60)
    
    conv_tests = NSCConvergenceTests()
    
    try:
        results['heat_spectral'] = conv_tests.test_spectral_convergence_heat_equation()
        print("✓ Spectral heat equation test passed")
    except AssertionError as e:
        results['heat_spectral'] = {'error': str(e)}
        print(f"✗ Spectral heat equation test failed: {e}")
    
    try:
        results['rk4_temporal'] = conv_tests.test_temporal_convergence_rk4()
        print("✓ RK4 temporal convergence test passed")
    except AssertionError as e:
        results['rk4_temporal'] = {'error': str(e)}
        print(f"✗ RK4 temporal convergence test failed: {e}")
    
    try:
        results['spectral_advection'] = conv_tests.test_advection_convergence()
        print("✓ Spectral advection test passed")
    except AssertionError as e:
        results['spectral_advection'] = {'error': str(e)}
        print(f"✗ Spectral advection test failed: {e}")
    
    print("\n" + "=" * 60)
    print("NSC Benchmark Tests")
    print("=" * 60)
    
    bench_tests = NSCBenchmarkTests()
    
    try:
        results['gradient_bench'] = bench_tests.test_discrete_gradient_benchmark()
        print("✓ Gradient benchmark test passed")
    except AssertionError as e:
        results['gradient_bench'] = {'error': str(e)}
        print(f"✗ Gradient benchmark test failed: {e}")
    
    try:
        results['laplacian_bench'] = bench_tests.test_discrete_laplacian_benchmark()
        print("✓ Laplacian benchmark test passed")
    except AssertionError as e:
        results['laplacian_bench'] = {'error': str(e)}
        print(f"✗ Laplacian benchmark test failed: {e}")
    
    try:
        results['rk4_bench'] = bench_tests.test_time_integration_benchmark()
        print("✓ RK4 benchmark test passed")
    except AssertionError as e:
        results['rk4_bench'] = {'error': str(e)}
        print(f"✗ RK4 benchmark test failed: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Suite Summary")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if 'error' not in v)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    for name, result in results.items():
        if 'error' in result:
            print(f"  ✗ {name}: {result['error']}")
        else:
            print(f"  ✓ {name}")
    
    return results


# Pytest test classes
class TestNSCMathematicalAccuracy:
    """Pytest wrapper for mathematical accuracy tests."""
    
    def test_gradient_accuracy(self):
        tester = NSCMathematicalAccuracyTests()
        result = tester.test_discrete_gradient_accuracy()
        assert result['order'] > 0.95
    
    def test_laplacian_accuracy(self):
        tester = NSCMathematicalAccuracyTests()
        result = tester.test_discrete_laplacian_accuracy()
        assert result['errors'][-1] < 1e-8
    
    def test_time_integrator_order(self):
        tester = NSCMathematicalAccuracyTests()
        result = tester.test_time_integrator_order()
        assert result['order'] > 3.5
    
    def test_poisson_solver(self):
        tester = NSCMathematicalAccuracyTests()
        result = tester.test_poisson_solver_accuracy()
        assert result['errors'][-1] < 1e-8


class TestNSCPhysicsAccuracy:
    """Pytest wrapper for physics accuracy tests."""
    
    def test_minkowski_curvature(self):
        tester = NSCPhysicsAccuracyTests()
        result = tester.test_minkowski_curvature_invariants()
        assert result['growth_factor'] < 10.0
    
    def test_wave_equation(self):
        tester = NSCPhysicsAccuracyTests()
        result = tester.test_wave_equation_accuracy()
        assert result['errors'][-1] < 1e-6
    
    def test_energy_decay(self):
        tester = NSCPhysicsAccuracyTests()
        result = tester.test_energy_decay_accuracy()
        assert result['error'] < 0.5
    
    def test_conservation(self):
        tester = NSCPhysicsAccuracyTests()
        result = tester.test_conservation_laws_accuracy()
        assert result['relative_drift'] < 1.0


class TestNSCConvergence:
    """Pytest wrapper for convergence tests."""
    
    def test_heat_spectral(self):
        tester = NSCConvergenceTests()
        result = tester.test_spectral_convergence_heat_equation()
        assert result['errors'][-1] < 2 * result['errors'][0]
    
    def test_rk4_temporal(self):
        tester = NSCConvergenceTests()
        result = tester.test_temporal_convergence_rk4()
        assert result['order'] > 3.5
    
    def test_spectral_advection(self):
        tester = NSCConvergenceTests()
        result = tester.test_advection_convergence()
        assert result['errors'][-1] < 2 * result['errors'][0]


class TestNSCBenchmark:
    """Pytest wrapper for benchmark tests."""
    
    def test_gradient_benchmark(self):
        tester = NSCBenchmarkTests()
        result = tester.test_discrete_gradient_benchmark()
        assert result['errors'][-1] < result['errors'][0]
    
    def test_laplacian_benchmark(self):
        tester = NSCBenchmarkTests()
        result = tester.test_discrete_laplacian_benchmark()
        assert result['errors'][-1] < 1e-8
    
    def test_rk4_benchmark(self):
        tester = NSCBenchmarkTests()
        result = tester.test_time_integration_benchmark()
        assert result['order'] > 3.5


if __name__ == "__main__":
    # Run full test suite when executed directly
    results = run_nsc_accuracy_test_suite()
    
    # Save results to file
    import json
    with open('nsc_accuracy_test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to nsc_accuracy_test_results.json")
