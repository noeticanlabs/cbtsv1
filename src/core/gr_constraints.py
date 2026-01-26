# Implements: LoC-1 tangency/invariance via Bianchi identities for GR constraint propagation (H,M remain zero if initially zero under ADM/BSSN dynamics).

# Lexicon declarations per canon v1.2
LEXICON_IMPORTS = [
    "LoC_axiom",
    "UFE_core",
    "GR_dyn",
    "CTL_time"
]
LEXICON_SYMBOLS = {
    "\\mathcal{H}": "GR_constraint.hamiltonian",
    "\\mathcal{M}^i": "GR_constraint.momentum"
}

import numpy as np
import logging
import time
import hashlib
import json
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Optional, Tuple, Dict, Any
from .logging_config import array_stats, Timer
from .gr_core_fields import inv_sym6, trace_sym6, norm2_sym6, sym6_to_mat33, mat33_to_sym6, det_sym6
from src.elliptic.solver import MultigridSolver, KrylovSolver, EllipticSolver
from gr_constraints_nsc import compute_hamiltonian_compiled, compute_momentum_compiled, discrete_L2_norm_compiled

logger = logging.getLogger('gr_solver.constraints')

def discrete_L2_norm(field, dx, dy, dz):
    """Compute discrete L2 norm: sqrt( sum(field^2) * dV )"""
    dV = dx * dy * dz
    return np.sqrt(np.sum(field**2) * dV)

def discrete_Linf_norm(field):
    """Compute discrete Linf norm: max(|field|)"""
    return np.max(np.abs(field))


def apply_hamiltonian_constraint_operator(phi_correction, dx=1.0):
    """
    Apply the linearized Hamiltonian constraint operator A_H for elliptic solve.
    
    For the transverse-traceless correction, the linearized Hamiltonian constraint is:
    A_H[delta_gamma, delta_K] -> delta_H
    
    This implements the operator that maps correction fields to constraint violation change.
    Uses a flat-space approximation for the operator (suitable for small corrections).
    
    Args:
        phi_correction: Scalar correction field (not used directly, shape determines grid)
        dx: Grid spacing in x (tuple for non-uniform, scalar for uniform)
    
    Returns:
        Callable that applies A_H to correction fields
    """
    if isinstance(dx, (tuple, list)):
        dx_, dy_, dz_ = dx
    else:
        dx_ = dy_ = dz_ = dx
    
    def apply_A(correction_fields, gamma_inv=None, K_trace=None):
        """
        Apply linearized Hamiltonian operator.
        
        For TT correction: -∇² δφ ≈ δH (Poisson-like operator)
        The actual GR operator is more complex, involving metric variations.
        
        Args:
            correction_fields: Can be either:
                - dict with 'delta_gamma' and/or 'delta_K' corrections
                - numpy array (raw field) for elliptic solve
            gamma_inv: Inverse metric (optional, for full GR operator)
            K_trace: Trace of extrinsic curvature (optional)
        
        Returns:
            Scalar field representing delta_H
        """
        # Handle raw numpy array (from elliptic solver)
        if isinstance(correction_fields, np.ndarray):
            field = correction_fields
            shape = field.shape[:3]
            result = np.zeros(shape, dtype=np.float64)
            
            # Laplacian of the field: -∇² δφ
            # This is the scalar Poisson operator
            result[1:-1, :, :] += (
                field[2:, :, :] + field[:-2, :, :] - 2*field[1:-1, :, :]
            ) / dx_**2
            result[:, 1:-1, :] += (
                field[:, 2:, :] + field[:, :-2, :] - 2*field[:, 1:-1, :]
            ) / dy_**2
            result[:, :, 1:-1] += (
                field[:, :, 2:] + field[:, :, :-2] - 2*field[:, :, 1:-1]
            ) / dz_**2
            
            return result
        
        # Handle dict (for residual computation)
        delta_gamma = correction_fields.get('delta_gamma', None)
        delta_K = correction_fields.get('delta_K', None)
        
        # Shape info
        if delta_gamma is not None:
            shape = delta_gamma.shape[:3]
        elif delta_K is not None:
            shape = delta_K.shape[:3]
        else:
            raise ValueError("Must provide delta_gamma or delta_K")
        
        result = np.zeros(shape, dtype=np.float64)
        
        # Linearized Hamiltonian: δH = R_lin + 2K δK - 2 K_{ij} δK^{ij}
        # Simplified: use Poisson-like operator for the scalar correction φ
        # -∇² δφ ≈ δH
        
        if delta_gamma is not None:
            # Metric contribution to linearized Hamiltonian
            # Approximate as Laplacian of conformal metric perturbation
            gamma_xx = delta_gamma[..., 0] if delta_gamma.shape[-1] >= 1 else None
            if gamma_xx is not None:
                # ∂²/∂x² contribution
                result[1:-1, :, :] += (
                    gamma_xx[2:, :, :] + gamma_xx[:-2, :, :] - 2*gamma_xx[1:-1, :, :]
                ) / dx_**2
                
            gamma_yy = delta_gamma[..., 1] if delta_gamma.shape[-1] >= 2 else None
            if gamma_yy is not None:
                result[:, 1:-1, :] += (
                    gamma_yy[:, 2:, :] + gamma_yy[:, :-2, :] - 2*gamma_yy[:, 1:-1, :]
                ) / dy_**2
                
            gamma_zz = delta_gamma[..., 2] if delta_gamma.shape[-1] >= 3 else None
            if gamma_zz is not None:
                result[:, :, 1:-1] += (
                    gamma_zz[:, :, 2:] + gamma_zz[:, :, :-2] - 2*gamma_zz[:, :, 1:-1]
                ) / dz_**2
        
        if delta_K is not None:
            # Extrinsic curvature contribution: 2K δK (trace part)
            K_comps = delta_K
            K_trace_delta = np.zeros(shape, dtype=np.float64)
            for i in range(min(3, K_comps.shape[-1])):
                K_trace_delta += K_comps[..., i]
            result += 2.0 * K_trace_delta
        
        return result
    
    return apply_A


def apply_momentum_constraint_operator(vector_correction, dx=1.0):
    """
    Apply the linearized momentum constraint operator A_M for elliptic solve.
    
    For the transverse vector correction, the linearized momentum constraint is:
    A_M[delta_beta] -> delta_M^i
    
    This implements the operator that maps vector shift correction to momentum violation.
    
    Args:
        vector_correction: Vector correction field (not used directly, shape determines grid)
        dx: Grid spacing
    
    Returns:
        Callable that applies A_M to correction fields
    """
    if isinstance(dx, (tuple, list)):
        dx_, dy_, dz_ = dx
    else:
        dx_ = dy_ = dz_ = dx
    
    def apply_A(correction_fields, christoffels=None, gamma_det=None):
        """
        Apply linearized momentum constraint operator.
        
        For vector correction δβ^i: -∇² δβ^i ≈ δM^i (vector Poisson)
        
        Args:
            correction_fields: Can be either:
                - dict with 'delta_beta' corrections
                - numpy array (raw vector field) for elliptic solve
            christoffels: Connection coefficients (optional, for full GR operator)
            gamma_det: Metric determinant (optional)
        
        Returns:
            Vector field representing delta_M
        """
        # Handle raw numpy array (from elliptic solver)
        if isinstance(correction_fields, np.ndarray):
            field = correction_fields
            shape = field.shape[:3]
            result = np.zeros(shape + (3,), dtype=np.float64)
            
            # Vector Laplacian: -∇² δβ^i ≈ δM^i
            for i in range(3):
                beta_i = field[..., i]
                # X-direction Laplacian contribution to component i
                result[1:-1, :, :, i] += (
                    beta_i[2:, :, :] + beta_i[:-2, :, :] - 2*beta_i[1:-1, :, :]
                ) / dx_**2
                # Y-direction Laplacian contribution to component i
                result[:, 1:-1, :, i] += (
                    beta_i[:, 2:, :] + beta_i[:, :-2, :] - 2*beta_i[:, 1:-1, :]
                ) / dy_**2
                # Z-direction Laplacian contribution to component i
                result[:, :, 1:-1, i] += (
                    beta_i[:, :, 2:] + beta_i[:, :, :-2] - 2*beta_i[:, :, 1:-1]
                ) / dz_**2
            
            return result
        
        # Handle dict (for residual computation)
        delta_beta = correction_fields.get('delta_beta', None)
        
        if delta_beta is None:
            raise ValueError("Must provide delta_beta for momentum constraint operator")
        
        shape = delta_beta.shape[:3]
        result = np.zeros(shape + (3,), dtype=np.float64)
        
        # Vector Laplacian: -∇² δβ^i ≈ δM^i
        for i in range(3):
            beta_i = delta_beta[..., i]
            # X-direction Laplacian contribution to component i
            result[1:-1, :, :, i] += (
                beta_i[2:, :, :] + beta_i[:-2, :, :] - 2*beta_i[1:-1, :, :]
            ) / dx_**2
            # Y-direction Laplacian contribution to component i
            result[:, 1:-1, :, i] += (
                beta_i[:, 2:, :] + beta_i[:, :-2, :] - 2*beta_i[:, 1:-1, :]
            ) / dy_**2
            # Z-direction Laplacian contribution to component i
            result[:, :, 1:-1, i] += (
                beta_i[:, :, 2:] + beta_i[:, :, :-2] - 2*beta_i[:, :, 1:-1]
            ) / dz_**2
        
        return result
    
    return apply_A


class ConstraintCleanupSolver:
    """
    Elliptic solver for constraint cleanup via transverse-traceless projection.
    
    Solves: A δφ = H (for Hamiltonian) and A δβ^i = M^i (for momentum)
    where A is the appropriate elliptic operator.
    """
    
    def __init__(self, fields, geometry, aeonic_memory=None, memory_contract=None):
        self.fields = fields
        self.geometry = geometry
        self.aeonic_memory = aeonic_memory
        self.memory_contract = memory_contract
        
        # Solver configuration
        self.mg_levels = 3
        self.mg_max_cycles = 10
        self.mg_tolerance = 1e-8
        self.gmres_max_iter = 100
        self.gmres_m = 20
        self.gmres_tolerance = 1e-8
        
        # Threshold for triggering elliptic solve
        self.hamiltonian_threshold = 1e-6
        self.momentum_threshold = 1e-6
        
        # Initialize elliptic solvers
        self._init_solvers()
        
        # Receipt emission
        self.receipts_file = "aeonic_receipts.jsonl"
        self.prev_receipt_hash = "0" * 64
        
        # Solver statistics
        self.solve_stats = {
            'hamiltonian_solves': 0,
            'momentum_solves': 0,
            'mg_used': 0,
            'gmres_used': 0,
            'total_solve_time_ms': 0.0
        }
    
    def _init_solvers(self):
        """Initialize MG and GMRES solvers."""
        dx = (self.fields.dx, self.fields.dy, self.fields.dz)
        
        # Hamiltonian solver
        self.hamiltonian_operator = apply_hamiltonian_constraint_operator(
            np.zeros((self.fields.Nx, self.fields.Ny, self.fields.Nz)), dx
        )
        self.hamiltonian_solver = EllipticSolver(
            self.hamiltonian_operator, aeonic_memory=self.aeonic_memory
        )
        
        # Momentum solver
        self.momentum_operator = apply_momentum_constraint_operator(
            np.zeros((self.fields.Nx, self.fields.Ny, self.fields.Nz, 3)), dx
        )
        self.momentum_solver = EllipticSolver(
            self.momentum_operator, aeonic_memory=self.aeonic_memory
        )
    
    def _generate_regime_hash(self, constraint_type: str, step: int, t: float) -> str:
        """Generate regime hash for AEONIC memory."""
        regime_str = f"{constraint_type}_step{step}_t{t:.6e}"
        return hashlib.sha256(regime_str.encode()).hexdigest()
    
    def _emit_elliptic_solve_receipt(self, constraint_type: str, step: int, t: float, 
                                      method: str, residual_before: float, residual_after: float,
                                      solve_time_ms: float, correction_norm: float):
        """Emit receipt for elliptic solve operation."""
        receipt = {
            'run_id': 'gr_solver_run_001',
            'step': step,
            'event': f'ELLIPTIC_SOLVE_{constraint_type.upper()}',
            't': t,
            'constraint_type': constraint_type,
            'method': method,
            'residual_before': residual_before,
            'residual_after': residual_after,
            'correction_norm': correction_norm,
            'solve_time_ms': solve_time_ms,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
        
        # Chain to previous receipt
        receipt_str = json.dumps(receipt, sort_keys=True)
        receipt_hash = hashlib.sha256(receipt_str.encode()).hexdigest()
        receipt['receipt_hash'] = receipt_hash
        receipt['prev_receipt_hash'] = self.prev_receipt_hash
        self.prev_receipt_hash = receipt_hash
        
        with open(self.receipts_file, 'a') as f:
            f.write(json.dumps(receipt) + '\n')
        
        logger.debug(f"Emitted elliptic solve receipt for {constraint_type}")
    
    def solve_hamiltonian(self, H: np.ndarray, step: int = 0, t: float = 0.0, 
                          use_mg: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Solve elliptic equation for Hamiltonian constraint correction.
        
        Solves: A_H[δφ] = H
        
        Args:
            H: Hamiltonian constraint violation (right-hand side)
            step: Current step number (for regime hash)
            t: Current time (for regime hash)
            use_mg: Use multigrid if True, GMRES if False
        
        Returns:
            Tuple of (correction_field, solve_info)
        """
        start_time = time.time()
        method = 'mg' if use_mg else 'gmres'
        
        regime_hash = self._generate_regime_hash('hamiltonian', step, t)
        
        # Warm start from AEONIC memory
        warm_start = None
        if self.aeonic_memory:
            warm_start = self.aeonic_memory.get(regime_hash)
        
        # Apply elliptic solve
        if use_mg:
            # MG requires scalar field
            correction = self.hamiltonian_solver.solve(H, method='mg', x0=warm_start, regime_hash=regime_hash)
        else:
            correction = self.hamiltonian_solver.solve(H, method='gmres', x0=warm_start, regime_hash=regime_hash)
        
        solve_time_ms = (time.time() - start_time) * 1000
        
        # Compute residual after solve
        residual_before = np.linalg.norm(H)
        residual_after = np.linalg.norm(H - self.hamiltonian_operator(correction))
        
        # Store solution in AEONIC memory
        if self.aeonic_memory and correction is not None:
            try:
                self.aeonic_memory.put(
                    key=regime_hash,
                    tier=1,
                    payload=correction,
                    bytes=correction.nbytes,
                    ttl_s=3600,
                    ttl_l=100,
                    recompute_cost_est=1.0,
                    risk_score=0.0,
                    tainted=False,
                    regime_hashes=[regime_hash]
                )
            except Exception as e:
                logger.warning(f"Failed to store solution in AEONIC memory: {e}")
        
        # Emit receipt
        self._emit_elliptic_solve_receipt(
            'hamiltonian', step, t, method,
            residual_before, residual_after, solve_time_ms,
            float(np.linalg.norm(correction))
        )
        
        # Update stats
        self.solve_stats['hamiltonian_solves'] += 1
        if use_mg:
            self.solve_stats['mg_used'] += 1
        else:
            self.solve_stats['gmres_used'] += 1
        self.solve_stats['total_solve_time_ms'] += solve_time_ms
        
        solve_info = {
            'method': method,
            'residual_before': residual_before,
            'residual_after': residual_after,
            'solve_time_ms': solve_time_ms,
            'correction_norm': float(np.linalg.norm(correction))
        }
        
        return correction, solve_info
    
    def solve_momentum(self, M: np.ndarray, step: int = 0, t: float = 0.0,
                       use_mg: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Solve elliptic equation for momentum constraint correction.
        
        Solves: A_M[δβ^i] = M^i
        
        Args:
            M: Momentum constraint violation (right-hand side)
            step: Current step number (for regime hash)
            t: Current time (for regime hash)
            use_mg: Use multigrid if True, GMRES if False
        
        Returns:
            Tuple of (correction_vector, solve_info)
        """
        start_time = time.time()
        method = 'mg' if use_mg else 'gmres'
        
        regime_hash = self._generate_regime_hash('momentum', step, t)
        
        # Warm start from AEONIC memory
        warm_start = None
        if self.aeonic_memory:
            warm_start = self.aeonic_memory.get(regime_hash)
        
        # Apply elliptic solve
        if use_mg:
            correction = self.momentum_solver.solve(M, method='mg', x0=warm_start, regime_hash=regime_hash)
        else:
            correction = self.momentum_solver.solve(M, method='gmres', x0=warm_start, regime_hash=regime_hash)
        
        solve_time_ms = (time.time() - start_time) * 1000
        
        # Compute residual after solve
        residual_before = np.linalg.norm(M)
        residual_after = np.linalg.norm(M - self.momentum_operator(correction))
        
        # Store solution in AEONIC memory
        if self.aeonic_memory and correction is not None:
            try:
                self.aeonic_memory.put(
                    key=regime_hash,
                    tier=1,
                    payload=correction,
                    bytes=correction.nbytes,
                    ttl_s=3600,
                    ttl_l=100,
                    recompute_cost_est=1.0,
                    risk_score=0.0,
                    tainted=False,
                    regime_hashes=[regime_hash]
                )
            except Exception as e:
                logger.warning(f"Failed to store solution in AEONIC memory: {e}")
        
        # Emit receipt
        self._emit_elliptic_solve_receipt(
            'momentum', step, t, method,
            residual_before, residual_after, solve_time_ms,
            float(np.linalg.norm(correction))
        )
        
        # Update stats
        self.solve_stats['momentum_solves'] += 1
        if use_mg:
            self.solve_stats['mg_used'] += 1
        else:
            self.solve_stats['gmres_used'] += 1
        self.solve_stats['total_solve_time_ms'] += solve_time_ms
        
        solve_info = {
            'method': method,
            'residual_before': residual_before,
            'residual_after': residual_after,
            'solve_time_ms': solve_time_ms,
            'correction_norm': float(np.linalg.norm(correction))
        }
        
        return correction, solve_info
    
    def threshold_triggered_cleanup(self, H: np.ndarray = None, M: np.ndarray = None, step: int = 0, t: float = 0.0, 
                                     use_mg: bool = True) -> Dict[str, Any]:
        """
        Perform threshold-triggered constraint cleanup.
        
        Only solves elliptic equations if constraints exceed thresholds.
        
        Args:
            H: Hamiltonian constraint field (right-hand side), optional
            M: Momentum constraint field (right-hand side), optional  
            step: Current step number
            t: Current time
            use_mg: Use multigrid if True, GMRES if False
        
        Returns:
            Dict with cleanup results and statistics
        """
        results = {
            'hamiltonian_solved': False,
            'momentum_solved': False,
            'hamiltonian_info': None,
            'momentum_info': None,
            'corrections': {}
        }
        
        # Check Hamiltonian constraint
        if H is not None:
            H_norm = discrete_Linf_norm(H)
            if H_norm > self.hamiltonian_threshold:
                logger.info(f"Hamiltonian constraint {H_norm:.2e} exceeds threshold {self.hamiltonian_threshold:.2e}, solving elliptic problem")
                
                correction, info = self.solve_hamiltonian(H, step, t, use_mg)
                results['hamiltonian_solved'] = True
                results['hamiltonian_info'] = info
                results['corrections']['hamiltonian'] = correction
                
                logger.info(f"Hamiltonian elliptic solve completed, residual: {info['residual_after']:.2e}")
            else:
                logger.debug(f"Hamiltonian constraint {H_norm:.2e} below threshold, skipping solve")
        
        # Check momentum constraint
        if M is not None:
            M_norm = discrete_Linf_norm(M)
            if M_norm > self.momentum_threshold:
                logger.info(f"Momentum constraint {M_norm:.2e} exceeds threshold {self.momentum_threshold:.2e}, solving elliptic problem")
                
                correction, info = self.solve_momentum(M, step, t, use_mg)
                results['momentum_solved'] = True
                results['momentum_info'] = info
                results['corrections']['momentum'] = correction
                
                logger.info(f"Momentum elliptic solve completed, residual: {info['residual_after']:.2e}")
            else:
                logger.debug(f"Momentum constraint {M_norm:.2e} below threshold, skipping solve")
        
        return results


class GRConstraints:
    def __init__(self, fields, geometry, num_workers: int = None, aeonic_memory=None, memory_contract=None):
        self.fields = fields
        self.geometry = geometry
        self.num_workers = num_workers or min(mp.cpu_count(), 8)
        self.executor = ProcessPoolExecutor(max_workers=self.num_workers)
        
        # Elliptic solver for constraint cleanup
        self.cleanup_solver = ConstraintCleanupSolver(
            fields, geometry, aeonic_memory=aeonic_memory, memory_contract=memory_contract
        )
        
        # Threshold configuration
        self.hamiltonian_threshold = 1e-6
        self.momentum_threshold = 1e-6
        self.enable_threshold_cleanup = True
        
        # Periodic domain flag (MG requires periodic, GMRES works for any)
        self.periodic_domain = False

    def __getstate__(self):
        """Exclude the un-pickleable executor from the state."""
        state = self.__dict__.copy()
        del state['executor']
        del state['cleanup_solver']
        return state

    def __setstate__(self, state):
        """Restore the state and re-create the executor."""
        self.__dict__.update(state)
        self.executor = ProcessPoolExecutor(max_workers=self.num_workers)
        # Recreate cleanup_solver (aeonic_memory will need to be re-injected)
        self.cleanup_solver = ConstraintCleanupSolver(
            self.fields, self.geometry
        )

    def _compute_hamiltonian_chunk(self, slice_indices, R, gamma_sym6, K_sym6):
        """Compute Hamiltonian for a chunk of the grid."""
        gamma_inv = inv_sym6(gamma_sym6[slice_indices])
        K_trace = trace_sym6(K_sym6[slice_indices], gamma_inv)
        K_sq = norm2_sym6(K_sym6[slice_indices], gamma_inv)
        return R[slice_indices] + K_trace**2 - K_sq

    def compute_hamiltonian(self):
        """Compute Hamiltonian constraint \mathcal{H} = R + K^2 - K_{ij}K^{ij} - 2\Lambda using compiled function."""
        start_time = time.time()

        # Ensure geometry R is up to date
        if not hasattr(self.geometry, 'R') or self.geometry.R is None:
            self.geometry.compute_scalar_curvature()

        self.H = compute_hamiltonian_compiled(self.geometry.R, self.fields.gamma_sym6, self.fields.K_sym6, self.fields.Lambda)

        elapsed = time.time() - start_time
        logger.info(f"Hamiltonian constraint computation time: {elapsed:.4f}s (compiled)")




    def compute_momentum(self):
        """Compute momentum constraints \mathcal{M}^i = D_j (K^{ij} - \gamma^{ij} K)."""
        with Timer("compute_momentum") as timer:
            # Ensure geometry is up to date
            if not hasattr(self.geometry, 'christoffels') or self.geometry.christoffels is None:
                self.geometry.compute_christoffels()
            
            self.M = compute_momentum_compiled(
                self.fields.gamma_sym6,
                self.fields.K_sym6,
                self.geometry.christoffels,
                self.fields.dx, self.fields.dy, self.fields.dz,
                det_sym6  # Pass the det_sym6 function
            )
        
        logger.info(f"Momentum constraint computation time: {timer.elapsed_ms():.4f}ms (compiled)")

    def compute_residuals(self):
        """Compute residuals: eps_H (L2 H), eps_M (L2 sqrt(gamma^{ij} M_i M_j)), eps_proj (L2 aux constraints)."""
        # L2 norm: sqrt( sum field^2 dV ) using compiled function
        self.eps_H = discrete_L2_norm_compiled(self.H, self.fields.dx, self.fields.dy, self.fields.dz)
        self.eps_H_grid = self.H  # Grid of Hamiltonian constraint values

        # Compute invariant M_norm = sqrt( gamma^{ij} M_i M_j )
        gamma_inv = inv_sym6(self.fields.gamma_sym6)
        gamma_inv_full = sym6_to_mat33(gamma_inv)
        M_norm_squared = np.sum(gamma_inv_full * self.M[..., np.newaxis, :] * self.M[..., :, np.newaxis], axis=(-2, -1))
        M_norm = np.sqrt(M_norm_squared)
        self.eps_M = discrete_L2_norm_compiled(M_norm, self.fields.dx, self.fields.dy, self.fields.dz)

        # eps_proj: L2 of auxiliary constraints Z and Z_i (BSSN-Z4)
        # Assuming Z and Z_i are available in fields
        if hasattr(self.fields, 'Z') and hasattr(self.fields, 'Z_i'):
            Z_combined = np.concatenate([self.fields.Z[..., np.newaxis], self.fields.Z_i], axis=-1)
            self.eps_proj = discrete_L2_norm(Z_combined, self.fields.dx, self.fields.dy, self.fields.dz)
        else:
            self.eps_proj = 0.0

        # eps_clk will be computed in stepper, as it requires stage information
        self.eps_clk = 0.0  # Initialize to 0.0

        # Normalized versions (divide by initial if available, else 1)
        self.eps_H_norm = self.eps_H / max(1e-20, getattr(self, 'eps_H_initial', 1.0))
        self.eps_M_norm = self.eps_M / max(1e-20, getattr(self, 'eps_M_initial', 1.0))
        self.eps_proj_norm = self.eps_proj / max(1e-20, getattr(self, 'eps_proj_initial', 1.0))

        logger.debug("Computed constraint residuals", extra={
            "extra_data": {
                "eps_H": float(self.eps_H),
                "eps_M": float(self.eps_M),
                "eps_proj": float(self.eps_proj),
                "eps_H_norm": float(self.eps_H_norm),
                "eps_M_norm": float(self.eps_M_norm),
                "eps_proj_norm": float(self.eps_proj_norm),
                "H_stats": array_stats(self.H, "H"),
                "M_stats": array_stats(self.M, "M")
            }
        })

    def compute_stage_difference_Linf(self, Psi_used, Psi_auth):
        """Compute Linf norm of difference between Psi_used and Psi_auth states."""
        diffs = []
        for key in Psi_used:
            if key in Psi_auth:
                diff = np.max(np.abs(Psi_used[key] - Psi_auth[key]))
                diffs.append(diff)
        return max(diffs) if diffs else 0.0

    @staticmethod
    def compute_scaling_law(eps1, eps2, h1, h2):
        """Compute observed convergence order p_obs = log(eps1/eps2) / log(h1/h2)."""
        if eps1 <= 0 or eps2 <= 0 or h1 <= 0 or h2 <= 0:
            return float('nan')
        ratio_eps = eps1 / eps2
        ratio_h = h1 / h2
        if ratio_eps <= 0 or ratio_h <= 0 or ratio_h == 1:
            return float('nan')
        return np.log(ratio_eps) / np.log(ratio_h)

    def compute_all(self):
        """Compute all constraints and residuals."""
        self.compute_hamiltonian()
        self.compute_momentum()
        self.compute_residuals()

    def elliptic_cleanup(self, step: int = 0, t: float = 0.0) -> Dict[str, Any]:
        """
        Perform elliptic constraint cleanup using the cleanup solver.
        
        Uses MG for periodic domains, GMRES for non-periodic.
        
        Args:
            step: Current step number
            t: Current time
        
        Returns:
            Dict with cleanup results
        """
        # Use MG if periodic, GMRES otherwise
        use_mg = self.periodic_domain
        
        # Delegate to cleanup solver with H and M from constraints
        return self.cleanup_solver.threshold_triggered_cleanup(
            H=self.H if hasattr(self, 'H') else None,
            M=self.M if hasattr(self, 'M') else None,
            step=step, t=t, use_mg=use_mg
        )

    def set_aeonic_memory(self, aeonic_memory):
        """Set AEONIC memory for warm-start."""
        self.cleanup_solver.aeonic_memory = aeonic_memory
        self.cleanup_solver._init_solvers()

    def set_thresholds(self, hamiltonian: float = None, momentum: float = None):
        """Set cleanup thresholds."""
        if hamiltonian is not None:
            self.hamiltonian_threshold = hamiltonian
            self.cleanup_solver.hamiltonian_threshold = hamiltonian
        if momentum is not None:
            self.momentum_threshold = momentum
            self.cleanup_solver.momentum_threshold = momentum

    def get_cleanup_stats(self) -> Dict:
        """Get cleanup solver statistics."""
        return self.cleanup_solver.solve_stats.copy()


def compute_constraints_kernel(fields, geometry):
    """Standalone kernel for computing constraints."""
    constraints = GRConstraints(fields, geometry)
    constraints.compute_hamiltonian()
    constraints.compute_momentum()
    constraints.compute_residuals()
    return {
        'eps_H': constraints.eps_H,
        'eps_M': constraints.eps_M
    }
