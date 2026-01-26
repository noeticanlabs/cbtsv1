# Lexicon declarations per canon v1.2
LEXICON_IMPORTS = [
    "LoC_axiom",
    "UFE_core",
    "GR_dyn",
    "CTL_time"
]
LEXICON_SYMBOLS = {
    "A": "Elliptic operator",
    "MG": "Multigrid solver",
    "GMRES": "Krylov solver"
}

import numpy as np
from typing import Callable, Tuple, Optional

def arnoldi(A, v1, m):
    """Arnoldi orthogonalization for Krylov subspace."""
    n = len(v1)
    V = np.zeros((n, m+1), dtype=v1.dtype)
    H = np.zeros((m+1, m), dtype=v1.dtype)
    V[:, 0] = v1
    for j in range(m):
        w = A(V[:, j])
        for i in range(j+1):
            H[i, j] = np.dot(V[:, i].conj(), w)
            w -= H[i, j] * V[:, i]
        H[j+1, j] = np.linalg.norm(w)
        if H[j+1, j] < 1e-14:
            m_actual = j
            return V[:, :m_actual+1], H[:m_actual+1, :m_actual], m_actual
        V[:, j+1] = w / H[j+1, j]
    m_actual = m
    return V[:, :m+1], H[:m+1, :m], m_actual

def restrict_full_weight(coarse, fine):
    n, m, p = coarse.shape
    for i in range(n):
        for j in range(m):
            for k in range(p):
                sum_val = 0.0
                count = 0
                for di in range(2):
                    ii = 2 * i + di
                    if ii >= fine.shape[0]: continue
                    for dj in range(2):
                        jj = 2 * j + dj
                        if jj >= fine.shape[1]: continue
                        for dk in range(2):
                            kk = 2 * k + dk
                            if kk >= fine.shape[2]: continue
                            sum_val += fine[ii, jj, kk]
                            count += 1
                coarse[i, j, k] = sum_val / count if count > 0 else 0.0

def prolongate_adjoint(fine, coarse):
    n, m, p = coarse.shape
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = coarse[i, j, k] / 8.0
                for di in range(2):
                    ii = 2 * i + di
                    if ii >= fine.shape[0]: continue
                    for dj in range(2):
                        jj = 2 * j + dj
                        if jj >= fine.shape[1]: continue
                        for dk in range(2):
                            kk = 2 * k + dk
                            if kk >= fine.shape[2]: continue
                            fine[ii, jj, kk] += val

class MultigridSolver:
    def __init__(self, apply_A: Callable, levels=3, max_cycles=10, tolerance=1e-8, aeonic_memory=None):
        self.apply_A_base = apply_A  # Base operator A(phi, dx) -> y
        self.levels = levels
        self.max_cycles = max_cycles
        self.tolerance = tolerance
        self.aeonic_memory = aeonic_memory
        self.base_shape = None
        self.last_solution = None  # Global warm-start for previous solution

    def apply_A(self, phi, level=0):
        dx = 1.0 * (2 ** level)
        return self.apply_A_base(phi, dx)

    def _get_shape(self, level):
        return tuple(s // (2 ** level) for s in self.base_shape)

    def v_cycle(self, b, x0=None, level=0):
        """V-cycle MG iteration."""
        if level == self.levels - 1:
            # Coarsest level: simple relaxation
            return self.relax(b, x0, nu=10, level=level)
        else:
            # Pre-smoothing
            x = self.relax(b, x0, nu=2, level=level)
            # Compute residual
            r = b - self.apply_A(x, level)
            # Restrict residual
            r_coarse = self.restrict(r, level)
            # Recurse
            e_coarse = self.v_cycle(r_coarse, level=level+1)
            # Prolongate and correct
            e = self.prolongate(e_coarse, level)
            x += e
            # Post-smoothing
            x = self.relax(b, x, nu=2, level=level)
            return x

    def relax(self, b, x0, nu=1, level=0):
        """Jacobi relaxation with proper diagonal."""
        if x0 is None:
            x = np.zeros_like(b)
        else:
            x = x0.copy()
        dx = 1.0 * (2 ** level)
        diag = -6.0 / (dx ** 2)  # For Poisson equation
        for _ in range(nu):
            r = b - self.apply_A(x, level)
            x += r / diag
        return x

    def restrict(self, fine, level):
        """Restrict to coarser grid using full weighting."""
        coarse_shape = self._get_shape(level + 1)
        coarse = np.zeros(coarse_shape)
        restrict_full_weight(coarse, fine)
        return coarse

    def prolongate(self, coarse, level):
        """Prolongate to finer grid using adjoint interpolation."""
        fine_shape = self._get_shape(level)
        fine = np.zeros(fine_shape)
        prolongate_adjoint(fine, coarse)
        return fine

    def solve(self, b, x0=None, regime_hash=None):
        """Solve A x = b using MG with warm-start."""
        self.base_shape = b.shape
        if x0 is None:
            if self.aeonic_memory and regime_hash:
                x0 = self.aeonic_memory.get(regime_hash)
            elif self.last_solution is not None:
                x0 = self.last_solution
        x = x0.copy() if x0 is not None else np.zeros_like(b)
        for cycle in range(self.max_cycles):
            x_new = self.v_cycle(b, x)
            residual = np.linalg.norm(b - self.apply_A(x_new))
            if residual < self.tolerance:
                break
            x = x_new
        # Store for warm-start
        self.last_solution = x.copy()
        if self.aeonic_memory and regime_hash:
            bytes_size = x.nbytes
            self.aeonic_memory.put(
                key=regime_hash,
                tier=1,
                payload=x,
                bytes=bytes_size,
                ttl_s=3600,
                ttl_l=100,
                recompute_cost_est=1.0,
                risk_score=0.0,
                tainted=False,
                regime_hashes=[regime_hash]
            )
        return x

class KrylovSolver:
    def __init__(self, apply_A: Callable, m=20, max_iter=100, tolerance=1e-8, aeonic_memory=None):
        self.apply_A = apply_A
        self.m = m  # GMRES restart
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.aeonic_memory = aeonic_memory
        self.last_solution = None  # Warm-start fallback

    def gmres(self, b, x0=None, regime_hash=None):
        """GMRES(m) solver with restarting, convergence checks, and warm-starting."""
        b_flat = b.flatten()
        n = len(b_flat)
        # Warm-start from Aeonic memory or previous solution
        if x0 is None:
            if self.aeonic_memory and regime_hash:
                x0 = self.aeonic_memory.get(regime_hash)
            elif self.last_solution is not None:
                x0 = self.last_solution
        if x0 is not None:
            x_flat = x0.flatten()
        else:
            x_flat = np.zeros(n)
        A_op = lambda vec: self.apply_A(vec.reshape(b.shape)).flatten()
        max_restarts = self.max_iter // self.m
        if max_restarts == 0:
            max_restarts = 1
        for restart in range(max_restarts):
            r_flat = b_flat - A_op(x_flat)
            beta = np.linalg.norm(r_flat)
            if beta < self.tolerance:
                break
            v1 = r_flat / beta
            V, H, m_actual = arnoldi(A_op, v1, self.m)
            if m_actual > 0:
                e1 = np.zeros(m_actual + 1)
                e1[0] = beta
                y, residuals, rank, s = np.linalg.lstsq(H, e1, rcond=None)
                x_flat += V[:, :m_actual] @ y
            # Check convergence after update
            r_flat = b_flat - A_op(x_flat)
            beta = np.linalg.norm(r_flat)
            if beta < self.tolerance:
                break
        x = x_flat.reshape(b.shape)
        # Store solution for warm-start
        self.last_solution = x.copy()
        if self.aeonic_memory and regime_hash:
            bytes_size = x.nbytes
            self.aeonic_memory.put(
                key=regime_hash,
                tier=1,
                payload=x,
                bytes=bytes_size,
                ttl_s=3600,
                ttl_l=100,
                recompute_cost_est=1.0,
                risk_score=0.0,
                tainted=False,
                regime_hashes=[regime_hash]
            )
        return x

class EllipticSolver:
    def __init__(self, apply_A: Callable, aeonic_memory=None):
        self.apply_A = apply_A
        self.mg = MultigridSolver(apply_A, aeonic_memory=aeonic_memory)
        self.krylov = KrylovSolver(apply_A, aeonic_memory=aeonic_memory)

    def solve(self, b, method='mg', x0=None, regime_hash=None):
        """Solve A x = b using specified method."""
        if method == 'mg':
            return self.mg.solve(b, x0, regime_hash)
        elif method == 'gmres':
            return self.krylov.gmres(b, x0, regime_hash)
        else:
            raise ValueError(f"Unknown method {method}")

# Example operator for Poisson: -∇² φ = f
def apply_poisson(phi, dx=1.0):
    """Apply -∇² to phi (finite difference)."""
    lap = np.zeros_like(phi)
    lap[1:-1, 1:-1, 1:-1] = (
        (phi[2:, 1:-1, 1:-1] + phi[:-2, 1:-1, 1:-1] - 2*phi[1:-1, 1:-1, 1:-1]) / dx**2 +
        (phi[1:-1, 2:, 1:-1] + phi[1:-1, :-2, 1:-1] - 2*phi[1:-1, 1:-1, 1:-1]) / dx**2 +
        (phi[1:-1, 1:-1, 2:] + phi[1:-1, 1:-1, :-2] - 2*phi[1:-1, 1:-1, 1:-1]) / dx**2
    )
    return -lap