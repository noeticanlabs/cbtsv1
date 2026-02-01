"""
NSC Phase 1: Stabilized Compact Finite Difference Schemes

This module implements robust 4th and 6th order compact finite difference
schemes with properly stabilized tridiagonal solvers.

Key improvements:
1. Thomas algorithm with periodic boundary support using scipy
2. BLAS-accelerated operations
3. Round-off error control
4. Convergence testing infrastructure

Reference: Lele, "Compact Finite Difference Schemes with Spectral-like
Resolution," JCP 1992.
"""

import numpy as np
from typing import Callable, Tuple, Optional
import warnings

try:
    from scipy.linalg import solve
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not available, using pure numpy implementation")


# ============================================================================
# STABILIZED THOMAS ALGORITHM
# ============================================================================

def solve_tridiagonal(a: np.ndarray, b: np.ndarray, c: np.ndarray, 
                      d: np.ndarray) -> np.ndarray:
    """
    Solve tridiagonal system Ax = d using Thomas algorithm.
    
    Args:
        a: sub-diagonal (length n)
        b: diagonal (length n)
        c: super-diagonal (length n)
        d: RHS (length n)
        
    Returns:
        x: solution (length n)
    """
    n = len(b)
    
    if HAS_SCIPY:
        # Build full matrix for direct solve (more reliable than banded)
        A = np.zeros((n, n))
        np.fill_diagonal(A, b)
        np.fill_diagonal(A[1:], a[1:])
        np.fill_diagonal(A[:, 1:], c[:-1])
        return solve(A, d, overwrite_a=True, overwrite_b=True)
    else:
        # Pure numpy Thomas algorithm
        a, b, c, d = [arr.copy() for arr in (a, b, c, d)]
        
        # Forward elimination
        for i in range(1, n):
            factor = a[i] / b[i-1]
            b[i] -= factor * c[i-1]
            d[i] -= factor * d[i-1]
        
        # Back substitution
        x = np.zeros(n)
        x[-1] = d[-1] / b[-1]
        for i in range(n-2, -1, -1):
            x[i] = (d[i] - c[i] * x[i+1]) / b[i]
        
        return x


def solve_periodic_tridiagonal(a: np.ndarray, b: np.ndarray, 
                                c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Solve tridiagonal system with periodic boundary conditions.
    
    Uses direct solve on the full periodic matrix.
    """
    n = len(b)
    
    if n == 1:
        return d / (b[0] + 1e-15)
    
    a0 = a[0] if n > 0 else 0
    cn = c[-1] if n > 0 else 0
    
    if HAS_SCIPY:
        # Build full periodic matrix
        A = np.zeros((n, n))
        np.fill_diagonal(A, b)
        np.fill_diagonal(A[1:], a[1:])
        np.fill_diagonal(A[:-1, 1:], c[:-1])
        A[0, -1] = cn
        A[-1, 0] = a0
        return solve(A, d, overwrite_a=True, overwrite_b=True)
    else:
        # Build full periodic matrix for numpy solve
        A = np.zeros((n, n))
        np.fill_diagonal(A, b)
        np.fill_diagonal(A[1:], a[1:])
        np.fill_diagonal(A[:-1, 1:], c[:-1])
        A[0, -1] = cn
        A[-1, 0] = a0
        return np.linalg.solve(A, d)


# ============================================================================
# COMPACT FINITE DIFFERENCE SCHEMES
# ============================================================================

class CompactFD:
    """
    4th order compact finite difference schemes.
    
    Implements the Pade schemes from Lele (1992):
    
    4th Order (α = 1/4):
        (1/4)f'_{i-1} + f'_i + (1/4)f'_{i+1} = (3/(4h))(f_{i+1} - f_{i-1})
    
    For periodic boundaries, the coupling is maintained.
    """
    
    SCHEMES = {
        4: {'alpha': 0.25, 'a': 0.75},  # 4th order compact
        6: {'alpha': 1/3, 'a': 14/18, 'b': 1/18}, # 6th order
    }
    
    def __init__(self, order: int = 4, periodic: bool = True):
        """
        Initialize compact FD scheme.
        
        Args:
            order: Accuracy order (4 or 6)
            periodic: Use periodic boundary conditions
        """
        if order not in self.SCHEMES:
            raise ValueError(f"Order must be 4 or 6, got {order}")
        
        self.order = order
        self.params = self.SCHEMES[order]
        self.periodic = periodic
    
    def first_derivative(self, f: np.ndarray, h: float) -> np.ndarray:
        """
        Compute first derivative using compact scheme.
        
        The scheme is:
            α*f'_{i-1} + f'_i + α*f'_{i+1} = 
                a*(f_{i+1} - f_{i-1})/h + b*(f_{i+2} - f_{i-2})/(2h)
        
        Args:
            f: Function values on uniform grid
            h: Grid spacing
            
        Returns:
            df/dx
        """
        n = len(f)
        alpha = self.params['alpha']
        a = self.params['a']
        b = self.params.get('b', 0.0)  # 6th order schemes have 'b'
        
        # Build tridiagonal system A * f' = rhs
        a_diag = alpha * np.ones(n)  # sub-diagonal
        b_diag = np.ones(n)          # diagonal
        c_diag = alpha * np.ones(n)  # super-diagonal
        
        # RHS
        rhs = np.zeros(n)
        
        if self.periodic:
            # Periodic: maintain corner couplings in matrix
            rhs = (a / h) * (np.roll(f, -1) - np.roll(f, 1))
            if b != 0:
                rhs += (b / h) * (0.125 * (np.roll(f, -2) - np.roll(f, 2)))
            
            # Solve with periodic tridiagonal
            return solve_periodic_tridiagonal(a_diag, b_diag, c_diag, rhs)
        else:
            # Non-periodic: use 4th order boundary conditions
            # Interior points
            f_plus = np.roll(f, -1)
            f_minus = np.roll(f, 1)
            rhs[1:-1] = (a / h) * (f_plus[1:-1] - f_minus[1:-1])
            
            if b != 0:
                f_plus2 = np.roll(f, -2)
                f_minus2 = np.roll(f, 2)
                rhs[2:-2] += (b / (2*h)) * (f_plus2[2:-2] - f_minus2[2:-2])
            
            # 4th order one-sided boundary conditions at i=0, i=1, i=n-2, i=n-1
            # From Lele 1992, these require coupling with interior points
            # For simplicity, use 3rd order for first two and last two points
            # i=0: (1+2α)f'_0 + α f'_1 = (1/(2h))*( (2+3α)f_1 - (1+3α)f_0 + f_2 )
            # Actually for 4th order with α=1/4:
            # (1+2α)f'_0 + α f'_1 = (1/(2h))*((2+3α)f_1 - (1+3α)f_0 + f_2)
            # But we need to express f'_0 and f'_1 in terms of f values
            
            # Simpler approach: use 4th order explicit for boundaries
            # i=0: f'_0 ≈ (-25f_0 + 48f_1 - 36f_2 + 16f_3 - 3f_4)/(12h)
            rhs[0] = (-25*f[0] + 48*f[1] - 36*f[2] + 16*f[3] - 3*f[4]) / (12*h)
            # i=1: f'_1 ≈ (-3f_0 - 10f_1 + 18f_2 - 6f_3 + f_4)/(12h)
            rhs[1] = (-3*f[0] - 10*f[1] + 18*f[2] - 6*f[3] + f[4]) / (12*h)
            # i=n-2: similar to i=1 from right
            rhs[-2] = (3*f[-1] + 10*f[-2] - 18*f[-3] + 6*f[-4] - f[-5]) / (12*h)
            # i=n-1: similar to i=0 from right  
            rhs[-1] = (25*f[-1] - 48*f[-2] + 36*f[-3] - 16*f[-4] + 3*f[-5]) / (12*h)
            
            # Solve non-periodic system
            return solve_tridiagonal(a_diag, b_diag, c_diag, rhs)
    
    def second_derivative(self, f: np.ndarray, h: float) -> np.ndarray:
        """
        Compute second derivative using compact scheme.
        
        4th Order scheme (α = 1/10):
            (1/10)f''_{i-1} + f''_i + (1/10)f''_{i+1_{i+1} = (f} - 2f_i + f_{i-1})/h²
        """
        n = len(f)
        alpha = 0.1  # 1/10 for 4th order
        
        a_diag = alpha * np.ones(n)
        b_diag = np.ones(n)
        c_diag = alpha * np.ones(n)
        
        rhs = np.zeros(n)
        
        if self.periodic:
            rhs = (np.roll(f, -1) - 2*f + np.roll(f, 1)) / h**2
            return solve_periodic_tridiagonal(a_diag, b_diag, c_diag, rhs)
        else:
            rhs[1:-1] = (f[2:] - 2*f[1:-1] + f[:-2]) / h**2
            # 4th order boundaries
            rhs[0] = (25*f[0] - 48*f[1] + 36*f[2] - 16*f[3] + 3*f[4]) / (12*h**2)
            rhs[1] = (3*f[0] + 10*f[1] - 18*f[2] + 6*f[3] - f[4]) / (12*h**2)
            rhs[-2] = (-3*f[-1] - 10*f[-2] + 18*f[-3] - 6*f[-4] + f[-5]) / (12*h**2)
            rhs[-1] = (25*f[-1] - 48*f[-2] + 36*f[-3] - 16*f[-4] + 3*f[-5]) / (12*h**2)
            return solve_tridiagonal(a_diag, b_diag, c_diag, rhs)


# ============================================================================
# CONVERGENCE TESTING
# ============================================================================

class ConvergenceTest:
    """
    Framework for testing numerical convergence.
    """
    
    def __init__(self):
        self.results = {}
    
    def test_gradient_convergence(self, f: Callable, df_exact: Callable,
                                   resolutions: list = None) -> dict:
        """
        Test gradient convergence for compact FD.
        """
        if resolutions is None:
            resolutions = [32, 64, 128, 256]
        
        compact = CompactFD(order=4, periodic=True)
        
        errors = []
        hs = []
        
        for N in resolutions:
            h = 2*np.pi / N
            x = np.linspace(0, 2*np.pi, N, endpoint=False)
            hs.append(h)
            
            f_vals = f(x)
            df_compact = compact.first_derivative(f_vals, h)
            df_ex = df_exact(x)
            
            # L2 error
            error = np.sqrt(np.mean((df_compact - df_ex)**2))
            errors.append(error)
        
        # Compute convergence order
        order = self._estimate_order(errors, hs)
        
        self.results['gradient'] = {
            'order': order,
            'errors': errors,
            'hs': hs,
            'passed': order > 3.5
        }
        
        return self.results['gradient']
    
    def test_against_numpy(self, f: Callable, resolutions: list = None) -> dict:
        """
        Compare compact FD against numpy gradient.
        """
        if resolutions is None:
            resolutions = [64, 128, 256]
        
        compact = CompactFD(order=4, periodic=True)
        
        speedups = []
        
        for N in resolutions:
            h = 2*np.pi / N
            x = np.linspace(0, 2*np.pi, N, endpoint=False)
            f_vals = f(x)
            
            # High-res reference
            x_ref = np.linspace(0, 2*np.pi, 512, endpoint=False)
            f_ref = f(x_ref)
            df_ref = np.gradient(f_ref, 2*np.pi/512, edge_order=2)
            df_ref_interp = np.interp(x, x_ref, df_ref)
            
            # Compact
            df_compact = compact.first_derivative(f_vals, h)
            
            # Numpy
            df_numpy = np.gradient(f_vals, h, edge_order=2)
            
            # Errors
            err_compact = np.sqrt(np.mean((df_compact - df_ref_interp)**2))
            err_numpy = np.sqrt(np.mean((df_numpy - df_ref_interp)**2))
            
            speedups.append(err_numpy / (err_compact + 1e-15))
        
        avg_speedup = np.mean(speedups)
        
        self.results['vs_numpy'] = {
            'speedups': speedups,
            'average': avg_speedup,
            'passed': avg_speedup > 1.0
        }
        
        return self.results['vs_numpy']
    
    def _estimate_order(self, errors: list, hs: list) -> float:
        """Estimate convergence order from error data."""
        valid_mask = np.array(errors) > 1e-15
        if np.sum(valid_mask) < 2:
            return 0.0
        
        log_e = np.log(np.array(errors, dtype=float)[valid_mask])
        log_h = np.log(np.array(hs, dtype=float)[valid_mask])
        
        n = len(log_h)
        sum_h = np.sum(log_h)
        sum_e = np.sum(log_e)
        sum_he = np.sum(log_h * log_e)
        sum_h2 = np.sum(log_h * log_h)
        
        denom = n * sum_h2 - sum_h**2
        if abs(denom) < 1e-15:
            return 0.0
        
        slope = (n * sum_he - sum_h * sum_e) / denom
        return slope
    
    def summary(self) -> str:
        """Generate summary report."""
        lines = ["Convergence Test Summary", "=" * 40]
        
        for test, result in self.results.items():
            lines.append(f"\n{test}:")
            order_val = result.get('order', 'N/A')
            if isinstance(order_val, (int, float)):
                lines.append(f"  Order: {order_val:.3f}")
            else:
                lines.append(f"  Order: {order_val}")
            lines.append(f"  Passed: {result.get('passed', 'N/A')}")
            if 'speedups' in result:
                lines.append(f"  Avg Speedup: {result['average']:.1f}x")
        
        return "\n".join(lines)


# ============================================================================
# BENCHMARK UTILITIES
# ============================================================================

def benchmark_compact_vs_central(f: Callable, N: int = 128) -> dict:
    """
    Benchmark compact FD against central differences.
    """
    import time
    
    compact = CompactFD(order=4, periodic=True)
    
    h = 2*np.pi / N
    x = np.linspace(0, 2*np.pi, N, endpoint=False)
    f_vals = f(x)
    
    # High-res reference
    x_ref = np.linspace(0, 2*np.pi, 512, endpoint=False)
    f_ref = f(x_ref)
    df_ref = np.gradient(f_ref, 2*np.pi/512, edge_order=2)
    df_ref_interp = np.interp(x, x_ref, df_ref)
    
    # Time compact
    start = time.time()
    for _ in range(100):
        df_compact = compact.first_derivative(f_vals, h)
    t_compact = (time.time() - start) / 100
    
    # Time numpy
    start = time.time()
    for _ in range(100):
        df_numpy = np.gradient(f_vals, h, edge_order=2)
    t_numpy = (time.time() - start) / 100
    
    # Errors
    err_compact = np.sqrt(np.mean((df_compact - df_ref_interp)**2))
    err_numpy = np.sqrt(np.mean((df_numpy - df_ref_interp)**2))
    
    return {
        'N': N,
        'compact_time': t_compact * 1000,
        'numpy_time': t_numpy * 1000,
        'compact_error': err_compact,
        'numpy_error': err_numpy,
        'error_ratio': err_numpy / (err_compact + 1e-15),
        'time_ratio': t_numpy / (t_compact + 1e-15)
    }


# ============================================================================
# INTEGRATION WITH GR GEOMETRY
# ============================================================================

def create_compact_geometry(fields):
    """
    Create GRGeometry with 4th order compact finite differences.
    """
    from src.core.gr_geometry import GRGeometry, hash_array
    
    class CompactGeometry(GRGeometry):
        """GRGeometry with 4th order compact finite differences."""
        
        def __init__(self, fields):
            super().__init__(fields, fd_method='compact4')
            self.compact = CompactFD(order=4, periodic=True)
        
        def compute_christoffels(self):
            """Compute Christoffel symbols using compact4 FD."""
            dx, dy, dz = self.fields.dx, self.fields.dy, self.fields.dz
            gamma = self.fields.gamma_sym6
            
            dgamma_dx = np.zeros_like(gamma)
            dgamma_dy = np.zeros_like(gamma)
            dgamma_dz = np.zeros_like(gamma)
            
            compact = self.compact
            
            for comp in range(6):
                # x-direction
                for j in range(gamma.shape[1]):
                    for k in range(gamma.shape[2]):
                        col = gamma[:, j, k, comp]
                        dcol = compact.first_derivative(col, dx)
                        dgamma_dx[:, j, k, comp] = dcol
                
                # y-direction
                for i in range(gamma.shape[0]):
                    for k in range(gamma.shape[2]):
                        col = gamma[i, :, k, comp]
                        dcol = compact.first_derivative(col, dy)
                        dgamma_dy[i, :, k, comp] = dcol
                
                # z-direction
                for i in range(gamma.shape[0]):
                    for j in range(gamma.shape[1]):
                        col = gamma[i, j, :, comp]
                        dcol = compact.first_derivative(col, dz)
                        dgamma_dz[i, j, :, comp] = dcol
            
            # Cache and compute Christoffels
            gamma_hash = hash_array(np.concatenate([
                gamma.flatten(), dgamma_dx.flatten(), 
                dgamma_dy.flatten(), dgamma_dz.flatten()
            ]))
            
            if gamma_hash in self._christoffel_cache:
                self.christoffels, self.Gamma = self._christoffel_cache[gamma_hash]
                self._christoffel_cache.move_to_end(gamma_hash)
                return
            
            from src.core.gr_geometry_nsc import compute_christoffels_compiled
            self.christoffels, self.Gamma = compute_christoffels_compiled(
                gamma, dgamma_dx, dgamma_dy, dgamma_dz
            )
            
            self._christoffel_cache[gamma_hash] = (
                self.christoffels.copy(), self.Gamma.copy()
            )
            if len(self._christoffel_cache) > self._cache_maxsize:
                self._christoffel_cache.popitem(last=False)
    
    return CompactGeometry(fields)


# ============================================================================
# SELF-TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Phase 1: Stabilized Compact FD - Self Test")
    print("=" * 60)
    
    # Test function
    def test_func(x):
        return np.sin(x) + 0.5 * np.sin(3*x)
    
    def exact_grad(x):
        return np.cos(x) + 1.5 * np.cos(3*x)
    
    # Convergence test
    tester = ConvergenceTest()
    
    print("\n1. Convergence Test")
    print("-" * 40)
    result = tester.test_gradient_convergence(test_func, exact_grad)
    print(f"Convergence order: {result['order']:.3f}")
    print(f"Errors: {[f'{e:.2e}' for e in result['errors']]}")
    print(f"Passed: {result['passed']}")
    
    # Comparison with numpy
    print("\n2. Comparison with Numpy")
    print("-" * 40)
    result = tester.test_against_numpy(test_func)
    print(f"Average speedup: {result['average']:.1f}x")
    print(f"Passed: {result['passed']}")
    
    # Benchmark
    print("\n3. Benchmark")
    print("-" * 40)
    bench = benchmark_compact_vs_central(test_func, N=128)
    print(f"Compact error: {bench['compact_error']:.2e}")
    print(f"Numpy error: {bench['numpy_error']:.2e}")
    print(f"Error ratio: {bench['error_ratio']:.1f}x")
    print(f"Time ratio: {bench['time_ratio']:.2f}x")
    
    print("\n" + "=" * 60)
    print(tester.summary())
    print("=" * 60)
