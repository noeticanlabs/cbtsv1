"""
MMS Convergence Test - Verify p_obs with properly controlled time step.
"""
import numpy as np
import logging
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gr_solver.gr_solver import GRSolver
from gr_solver.gr_core_fields import det_sym6, inv_sym6, sym6_to_mat33, trace_sym6
from tests.gr_test_utils import estimate_order

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def exact_fields(N, L, t):
    dx = L / N
    x = np.arange(N) * dx
    y = np.arange(N) * dx
    z = np.arange(N) * dx
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    kx = 2.0 * np.pi / L
    ky = 2.0 * np.pi / L
    kz = 2.0 * np.pi / L
    kdotx = kx * X + ky * Y + kz * Z
    omega = 1.0
    eps = 1e-5
    
    gamma = np.zeros((N, N, N, 6))
    gamma[..., 0] = 1 + eps * np.sin(kdotx) * np.sin(omega * t)
    gamma[..., 1] = eps * np.cos(kdotx) * np.cos(omega * t)
    gamma[..., 3] = 1 + eps * np.sin(kdotx) * np.sin(omega * t)
    gamma[..., 4] = eps * np.sin(kdotx) * np.sin(omega * t)
    gamma[..., 5] = 1 + eps * np.cos(kdotx) * np.cos(omega * t)
    
    K = np.zeros((N, N, N, 6))
    K[..., 0] = eps * np.cos(kdotx) * np.sin(omega * t)
    K[..., 1] = eps * np.sin(kdotx) * np.cos(omega * t)
    K[..., 3] = eps * np.cos(kdotx) * np.sin(omega * t)
    K[..., 4] = eps * np.sin(kdotx) * np.sin(omega * t)
    K[..., 5] = eps * np.cos(kdotx) * np.sin(omega * t)
    
    return gamma, K


def exact_dt_fields(N, L, t):
    dx = L / N
    x = np.arange(N) * dx
    y = np.arange(N) * dx
    z = np.arange(N) * dx
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    kx = 2.0 * np.pi / L
    ky = 2.0 * np.pi / L
    kz = 2.0 * np.pi / L
    kdotx = kx * X + ky * Y + kz * Z
    omega = 1.0
    eps = 1e-5
    
    dt_gamma = np.zeros((N, N, N, 6))
    dt_gamma[..., 0] = eps * omega * np.cos(omega * t) * np.sin(kdotx)
    dt_gamma[..., 1] = -eps * omega * np.cos(kdotx) * np.sin(omega * t)
    dt_gamma[..., 3] = eps * omega * np.cos(omega * t) * np.sin(kdotx)
    dt_gamma[..., 4] = eps * omega * np.sin(kdotx) * np.cos(omega * t)
    dt_gamma[..., 5] = -eps * omega * np.cos(kdotx) * np.sin(omega * t)
    
    dt_K = np.zeros((N, N, N, 6))
    dt_K[..., 0] = eps * omega * np.cos(omega * t) * np.cos(kdotx)
    dt_K[..., 1] = eps * omega * np.sin(kdotx) * (-np.sin(omega * t))
    dt_K[..., 3] = eps * omega * np.cos(omega * t) * np.cos(kdotx)
    dt_K[..., 4] = eps * omega * np.sin(kdotx) * (-np.cos(omega * t))
    dt_K[..., 5] = eps * omega * np.cos(omega * t) * np.cos(kdotx)
    
    return dt_gamma, dt_K


def compute_bssn_dt(N, gamma, K, dt_gamma, dt_K):
    gamma_inv = inv_sym6(gamma)
    
    tr_dt_gamma = (
        gamma_inv[..., 0] * dt_gamma[..., 0] +
        2 * gamma_inv[..., 1] * dt_gamma[..., 1] +
        2 * gamma_inv[..., 2] * dt_gamma[..., 2] +
        gamma_inv[..., 3] * dt_gamma[..., 3] +
        2 * gamma_inv[..., 4] * dt_gamma[..., 4] +
        gamma_inv[..., 5] * dt_gamma[..., 5]
    )
    dt_phi = (1.0/12.0) * tr_dt_gamma
    
    det_g = det_sym6(gamma)
    phi = (1.0/12.0) * np.log(det_g)
    exp_minus_4phi = np.exp(-4.0 * phi)[..., np.newaxis]
    gamma_tilde = gamma * exp_minus_4phi
    dt_gamma_tilde = -4.0 * dt_phi[..., np.newaxis] * gamma_tilde + exp_minus_4phi * dt_gamma
    
    K_mat = sym6_to_mat33(K)
    dt_K_mat = sym6_to_mat33(dt_K)
    gamma_mat = sym6_to_mat33(gamma)
    gamma_inv_mat = sym6_to_mat33(gamma_inv)
    dt_gamma_mat = sym6_to_mat33(dt_gamma)
    dt_gamma_inv_mat = - np.einsum('...ij,...jk,...kl->...il', gamma_inv_mat, dt_gamma_mat, gamma_inv_mat)
    dt_K_trace = np.einsum('...ij,...ji->...', dt_gamma_inv_mat, K_mat) + np.einsum('...ij,...ji->...', gamma_inv_mat, dt_K_mat)
    
    K_trace = trace_sym6(K, gamma_inv)
    A_physical = K - (1.0/3.0) * gamma * K_trace[..., np.newaxis]
    A_tilde = A_physical * exp_minus_4phi
    term_brackets = dt_K - (1.0/3.0) * (dt_gamma * K_trace[..., np.newaxis] + gamma * dt_K_trace[..., np.newaxis])
    dt_A_tilde = -4.0 * dt_phi[..., np.newaxis] * A_tilde + exp_minus_4phi * term_brackets
    
    return dt_phi, dt_gamma_tilde, dt_K_trace, dt_A_tilde


def init_bssn_fields(solver, gamma, K):
    det_g = det_sym6(gamma)
    phi = (1.0/12.0) * np.log(det_g)
    solver.fields.phi[:] = phi
    
    exp_minus_4phi = np.exp(-4.0 * phi)[..., np.newaxis]
    solver.fields.gamma_tilde_sym6[:] = gamma * exp_minus_4phi
    
    gamma_inv = inv_sym6(gamma)
    K_trace = trace_sym6(K, gamma_inv)
    solver.fields.K_trace[:] = K_trace
    
    A_physical = K - (1.0/3.0) * gamma * K_trace[..., np.newaxis]
    solver.fields.A_sym6[:] = A_physical * exp_minus_4phi
    
    solver.fields.Gamma_tilde[:] = 0.0
    solver.fields.Z[:] = 0.0
    solver.fields.Z_i[:] = 0.0


def run_convergence_test(N, L=16.0, T=1e-5, cfl=0.01):
    """Run single-resolution test with given CFL factor."""
    solver = GRSolver(N, N, N, dx=L/N)
    
    gamma0, K0 = exact_fields(N, L, 0.0)
    
    solver.fields.gamma_sym6[:] = gamma0
    solver.fields.K_sym6[:] = K0
    solver.fields.alpha[:] = 1.0
    solver.fields.beta[:] = 0.0
    
    init_bssn_fields(solver, gamma0, K0)
    solver.geometry.compute_all()
    
    def sources_func(t):
        gamma_exact, K_exact = exact_fields(N, L, t)
        dt_gamma, dt_K = exact_dt_fields(N, L, t)
        dt_phi, dt_gamma_tilde, dt_K_trace, dt_A_tilde = compute_bssn_dt(N, gamma_exact, K_exact, dt_gamma, dt_K)
        
        return {
            'S_gamma_sym6': dt_gamma,
            'S_K_sym6': dt_K,
            'S_gamma_tilde_sym6': dt_gamma_tilde,
            'S_A_sym6': dt_A_tilde,
            'S_phi': dt_phi,
            'S_Gamma_tilde': np.zeros((N, N, N, 3)),
            'S_Z': np.zeros((N, N, N)),
            'S_Z_i': np.zeros((N, N, N, 3))
        }
    
    solver.stepper.rhs_computer.sources_func = sources_func
    
    dx = L / N
    dt = (dx**2) * cfl  # Diffusion-scale, very small
    t = 0.0
    
    gamma = solver.fields.gamma_sym6.copy()
    K = solver.fields.K_sym6.copy()
    phi = solver.fields.phi.copy()
    gamma_tilde = solver.fields.gamma_tilde_sym6.copy()
    A = solver.fields.A_sym6.copy()
    
    def eval_rhs(gamma_state, K_state, phi_state, gamma_tilde_state, A_state, t_eval):
        solver.fields.gamma_sym6[:] = gamma_state
        solver.fields.K_sym6[:] = K_state
        solver.fields.phi[:] = phi_state
        solver.fields.gamma_tilde_sym6[:] = gamma_tilde_state
        solver.fields.A_sym6[:] = A_state
        solver.fields.alpha[:] = 1.0
        solver.fields.beta[:] = 0.0
        solver.geometry.compute_all()
        solver.stepper.rhs_computer.compute_rhs(t_eval, slow_update=True)
        rhs = solver.stepper.rhs_computer
        
        return {
            'gamma': rhs.rhs_gamma_sym6,
            'K': rhs.rhs_K_sym6,
            'phi': rhs.rhs_phi,
            'gamma_tilde': rhs.rhs_gamma_tilde_sym6,
            'A': rhs.rhs_A_sym6,
        }
    
    steps = 0
    while t < T:
        dt_step = min(dt, T - t)
        
        k1 = eval_rhs(gamma, K, phi, gamma_tilde, A, t)
        
        gamma2 = gamma + 0.5 * dt_step * k1['gamma']
        K2 = K + 0.5 * dt_step * k1['K']
        phi2 = phi + 0.5 * dt_step * k1['phi']
        gamma_tilde2 = gamma_tilde + 0.5 * dt_step * k1['gamma_tilde']
        A2 = A + 0.5 * dt_step * k1['A']
        k2 = eval_rhs(gamma2, K2, phi2, gamma_tilde2, A2, t + 0.5 * dt_step)
        
        gamma3 = gamma + 0.5 * dt_step * k2['gamma']
        K3 = K + 0.5 * dt_step * k2['K']
        phi3 = phi + 0.5 * dt_step * k2['phi']
        gamma_tilde3 = gamma_tilde + 0.5 * dt_step * k2['gamma_tilde']
        A3 = A + 0.5 * dt_step * k2['A']
        k3 = eval_rhs(gamma3, K3, phi3, gamma_tilde3, A3, t + 0.5 * dt_step)
        
        gamma4 = gamma + dt_step * k3['gamma']
        K4 = K + dt_step * k3['K']
        phi4 = phi + dt_step * k3['phi']
        gamma_tilde4 = gamma_tilde + dt_step * k3['gamma_tilde']
        A4 = A + dt_step * k3['A']
        k4 = eval_rhs(gamma4, K4, phi4, gamma_tilde4, A4, t + dt_step)
        
        gamma = gamma + (dt_step / 6.0) * (k1['gamma'] + 2*k2['gamma'] + 2*k3['gamma'] + k4['gamma'])
        K = K + (dt_step / 6.0) * (k1['K'] + 2*k2['K'] + 2*k3['K'] + k4['K'])
        phi = phi + (dt_step / 6.0) * (k1['phi'] + 2*k2['phi'] + 2*k3['phi'] + k4['phi'])
        gamma_tilde = gamma_tilde + (dt_step / 6.0) * (k1['gamma_tilde'] + 2*k2['gamma_tilde'] + 2*k3['gamma_tilde'] + k4['gamma_tilde'])
        A = A + (dt_step / 6.0) * (k1['A'] + 2*k2['A'] + 2*k3['A'] + k4['A'])
        
        t += dt_step
        steps += 1
    
    gamma_exact, K_exact = exact_fields(N, L, T)
    error_gamma = np.sqrt(np.mean((gamma - gamma_exact)**2))
    error_K = np.sqrt(np.mean((K - K_exact)**2))
    
    return {
        'error_gamma': float(error_gamma),
        'error_K': float(error_K),
        'total_error': float(error_gamma + error_K),
        'steps': steps
    }


def main():
    L = 16.0
    T = 1e-5  # Very small to minimize time error
    cfl = 0.01  # Small CFL for time accuracy
    resolutions = [8, 12, 16, 24, 32]
    
    logger.info("=== MMS Convergence Test (T=1e-5, CFL=0.01) ===")
    logger.info("Using small T to minimize time integration error")
    logger.info("")
    
    errors = []
    hs = []
    
    for N in resolutions:
        result = run_convergence_test(N, L=L, T=T, cfl=cfl)
        errors.append(result['total_error'])
        hs.append(L / N)
        logger.info(f"N={N:2d}, dx={L/N:.4f}, Error={result['total_error']:.4e}, steps={result['steps']}")
    
    # Compute convergence order
    p_obs = estimate_order(errors, hs)
    logger.info(f"\nObserved convergence order: p_obs = {p_obs:.2f}")
    logger.info(f"Error ratio (N=8 to N=32): {errors[0]/errors[-1]:.2f}")
    
    # Compare with expected p=2
    if p_obs > 1.5:
        logger.info("✅ Convergence order APPROXIMATELY 2 (second-order accurate)")
    elif p_obs > 0.5:
        logger.info("⚠️  Convergence order PARTIAL but not full 2nd order")
    else:
        logger.info("❌ Convergence order POOR - at numerical floor")


if __name__ == "__main__":
    main()
