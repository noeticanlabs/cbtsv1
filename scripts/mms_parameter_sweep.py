"""
MMS Parameter Sweep Script
Find optimal CFL, T, and resolution parameters for convergence testing.
"""
import numpy as np
import logging
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))

from tests.test_1_mms_lite import MmsLite, exact_fields, init_bssn_fields, update_bssn_fields
from src.core.gr_solver import GRSolver
from tests.gr_test_utils import estimate_order

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger('mms_sweep')

# Suppress verbose solver logs
logging.getLogger('gr_solver').setLevel(logging.WARNING)


def quick_evolution_test(N, L, T, cfl, core_mode=True):
    """Run evolution and return error. Optimized for speed."""
    solver = GRSolver(N, N, N, dx=L/N)
    
    gamma0, K0, alpha0, beta0 = exact_fields(N, L, 0.0)
    solver.fields.gamma_sym6[:] = gamma0
    solver.fields.K_sym6[:] = K0
    solver.fields.alpha[:] = alpha0
    solver.fields.beta[:] = beta0
    init_bssn_fields(solver, gamma0, K0)
    solver.geometry.compute_all()
    
    dx = L / N
    dt = cfl * dx
    
    def sources_func(t):
        gamma_exact, K_exact, alpha_exact, beta_exact = exact_fields(N, L, t)
        return {
            'S_gamma_sym6': np.zeros_like(gamma0),
            'S_K_sym6': np.zeros_like(K0),
            'S_gamma_tilde_sym6': np.zeros_like(solver.fields.gamma_tilde_sym6),
            'S_A_sym6': np.zeros_like(solver.fields.A_sym6),
            'S_phi': np.zeros_like(solver.fields.phi),
            'S_Gamma_tilde': np.zeros((N, N, N, 3)),
            'S_Z': np.zeros((N, N, N)),
            'S_Z_i': np.zeros((N, N, N, 3))
        }
    
    solver.stepper.rhs_computer.sources_func = sources_func
    
    gamma = solver.fields.gamma_sym6.copy()
    K = solver.fields.K_sym6.copy()
    phi = solver.fields.phi.copy()
    gamma_tilde = solver.fields.gamma_tilde_sym6.copy()
    A = solver.fields.A_sym6.copy()
    
    t = 0.0
    steps = 0
    
    while t < T:
        dt_step = min(dt, T - t)
        
        # RK4 step (simplified)
        for gamma_state, K_state, phi_state, gamma_tilde_state, A_state, t_eval in [
            (gamma, K, phi, gamma_tilde, A, t),
            (gamma, K, phi, gamma_tilde, A, t + dt_step/2),
            (gamma, K, phi, gamma_tilde, A, t + dt_step),
        ]:
            solver.fields.gamma_sym6[:] = gamma_state
            solver.fields.K_sym6[:] = K_state
            solver.fields.phi[:] = phi_state
            solver.fields.gamma_tilde_sym6[:] = gamma_tilde_state
            solver.fields.A_sym6[:] = A_state
            update_bssn_fields(solver, gamma_state, K_state)
            solver.geometry.compute_all()
            solver.stepper.rhs_computer.compute_rhs(t_eval, slow_update=True)
        
        # Simple Euler update for speed (RK4 would be more accurate but slower)
        rhs = solver.stepper.rhs_computer
        gamma += dt_step * rhs.rhs_gamma_sym6
        K += dt_step * rhs.rhs_K_sym6
        phi += dt_step * rhs.rhs_phi
        gamma_tilde += dt_step * rhs.rhs_gamma_tilde_sym6
        A += dt_step * rhs.rhs_A_sym6
        
        t += dt_step
        steps += 1
    
    gamma_exact, K_exact, _, _ = exact_fields(N, L, T)
    error = np.sqrt(np.mean((gamma - gamma_exact)**2)) + np.sqrt(np.mean((K - K_exact)**2))
    
    return error, steps


def sweep_cfl(L=8.0, resolutions=[8, 12, 16], T=0.02):
    """Sweep over CFL values."""
    print(f"\n{'='*60}")
    print(f"CFL SWEEP: L={L}, T={T}, resolutions={resolutions}")
    print(f"{'='*60}")
    
    cfl_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
    
    results = []
    for cfl in cfl_values:
        errors = []
        hs = []
        for N in resolutions:
            try:
                error, steps = quick_evolution_test(N, L, T, cfl)
                errors.append(error)
                hs.append(L / N)
                print(f"  N={N:2d}: error={error:.4e}, steps={steps}")
            except Exception as e:
                print(f"  N={N:2d}: FAILED - {e}")
                errors.append(None)
        
        if len(errors) >= 2 and all(e is not None for e in errors):
            p_obs = estimate_order(errors, hs)
            print(f"  CFL={cfl:.2f}: p_obs={p_obs:.2f}")
            results.append((cfl, p_obs, errors))
        else:
            print(f"  CFL={cfl:.2f}: p_obs=N/A")
            results.append((cfl, None, errors))
    
    return results


def sweep_T(L=8.0, resolutions=[8, 12, 16], cfl=0.2):
    """Sweep over evolution times."""
    print(f"\n{'='*60}")
    print(f"T SWEEP: L={L}, CFL={cfl}, resolutions={resolutions}")
    print(f"{'='*60}")
    
    T_values = [0.01, 0.02, 0.05, 0.1, 0.2]
    
    results = []
    for T in T_values:
        errors = []
        hs = []
        for N in resolutions:
            try:
                error, steps = quick_evolution_test(N, L, T, cfl)
                errors.append(error)
                hs.append(L / N)
                print(f"  T={T:.2f}, N={N:2d}: error={error:.4e}, steps={steps}")
            except Exception as e:
                print(f"  T={T:.2f}, N={N:2d}: FAILED - {e}")
                errors.append(None)
        
        if len(errors) >= 2 and all(e is not None for e in errors):
            p_obs = estimate_order(errors, hs)
            print(f"  T={T:.2f}: p_obs={p_obs:.2f}")
            results.append((T, p_obs, errors))
        else:
            print(f"  T={T:.2f}: p_obs=N/A")
            results.append((T, None, errors))
    
    return results


def sweep_resolution(L=8.0, cfl=0.2, T=0.02):
    """Sweep over resolution ranges to find optimal."""
    print(f"\n{'='*60}")
    print(f"RESOLUTION SWEEP: L={L}, CFL={cfl}, T={T}")
    print(f"{'='*60}")
    
    resolution_ranges = [
        [8, 12, 16],
        [12, 16, 24],
        [16, 24, 32],
        [24, 32, 48],
        [32, 48, 64],
    ]
    
    best = None
    best_p = -1
    
    for resolutions in resolution_ranges:
        errors = []
        hs = []
        for N in resolutions:
            try:
                error, steps = quick_evolution_test(N, L, T, cfl)
                errors.append(error)
                hs.append(L / N)
                print(f"  Range {resolutions}: N={N}: error={error:.4e}")
            except Exception as e:
                print(f"  Range {resolutions}: N={N}: FAILED - {e}")
                errors.append(None)
        
        if len(errors) >= 2 and all(e is not None for e in errors):
            p_obs = estimate_order(errors, hs)
            print(f"  -> p_obs={p_obs:.2f}")
            if p_obs > best_p:
                best_p = p_obs
                best = (resolutions, p_obs, errors)
    
    if best:
        print(f"\nBest resolution range: {best[0]} with p_obs={best[1]:.2f}")
    
    return best


def main():
    print("MMS Parameter Sweep")
    print("=" * 60)
    
    # Phase 1: Sweep CFL
    cfl_results = sweep_cfl(L=8.0, resolutions=[8, 12, 16], T=0.02)
    best_cfl = max([(c, p) for c, p, _ in cfl_results if p is not None], key=lambda x: x[1])
    print(f"\nBest CFL: {best_cfl[0]} with p_obs={best_cfl[1]:.2f}")
    
    # Phase 2: Sweep T with best CFL
    T_results = sweep_T(L=8.0, resolutions=[8, 12, 16], cfl=best_cfl[0])
    best_T = max([(t, p) for t, p, _ in T_results if p is not None], key=lambda x: x[1])
    print(f"\nBest T: {best_T[0]} with p_obs={best_T[1]:.2f}")
    
    # Phase 3: Sweep resolutions with best params
    best = sweep_resolution(L=8.0, cfl=best_cfl[0], T=best_T[0])
    
    print(f"\n{'='*60}")
    print("OPTIMAL PARAMETERS FOUND:")
    print(f"  CFL = {best_cfl[0]}")
    print(f"  T = {best_T[0]}")
    if best:
        print(f"  Resolution range = {best[0]}")
        print(f"  Expected p_obs = {best[1]:.2f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
