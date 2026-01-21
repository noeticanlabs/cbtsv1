# Implements: LoC-5 clock coherence with multi-clock selection (CFL, gauge, coh, res); enforces LoC-4 discrete witness bound with eps_model residuals; logs LoC-6 representation fidelity via eta_rep and hashes.

# Lexicon declarations per canon v1.2
LEXICON_IMPORTS = [
    "LoC_axiom",
    "UFE_core",
    "GR_dyn",
    "CTL_time"
]
LEXICON_SYMBOLS = {
    "\\Psi": "UFE_state",
    "B": "UFE_baseline",
    "K": "UFE_coherence"
}

import numpy as np
import hashlib
import logging
import json
import os
import inspect
from datetime import datetime
from typing import Tuple, Dict, Any
try:
    from numba import jit
except ImportError:
    jit = lambda f=None, **kwargs: f if f else (lambda g: g)
from .logging_config import Timer, array_stats
from .gr_core_fields import inv_sym6, trace_sym6, sym6_to_mat33, mat33_to_sym6
from stepper_contract_memory import StepperContractWithMemory
from gr_solver.stepper_contract import StepperContract
from .gr_geometry import _sym6_to_mat33_jit, _inv_sym6_jit, _compute_christoffels_jit
from .gr_loc import GRLoC
from nsc_runtime_min import load_nscir, make_rhs_callable

logger = logging.getLogger('gr_solver.stepper')

@jit(nopython=True)
def _compute_gamma_tilde_rhs_jit(Nx, Ny, Nz, alpha, beta, phi, Gamma_tilde, A_sym6, gamma_tilde_sym6, dx, dy, dz, K_trace_scratch):
    """JIT-compiled computation of Gamma_tilde RHS."""
    # Precompute gamma_tilde_inv
    gamma_tilde_inv_full = np.zeros((Nx, Ny, Nz, 3, 3))
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                gamma_tilde_inv_full[i,j,k] = _sym6_to_mat33_jit(_inv_sym6_jit(gamma_tilde_sym6[i,j,k]))

    # Precompute A_tilde_uu (contravariant)
    A_tilde_uu = np.zeros((Nx, Ny, Nz, 3, 3))
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                A_full = _sym6_to_mat33_jit(A_sym6[i,j,k])
                A_tilde_uu[i,j,k] = gamma_tilde_inv_full[i,j,k] @ A_full @ gamma_tilde_inv_full[i,j,k]

    # Precompute Christoffel tildeGamma^i_{jk} - use the JIT christoffels
    christoffel_tilde_udd, _ = _compute_christoffels_jit(Nx, Ny, Nz, gamma_tilde_sym6, dx, dy, dz)

    # Gradients
    dalpha_x = np.gradient(alpha, dx, axis=0)
    dalpha_y = np.gradient(alpha, dy, axis=1)
    dalpha_z = np.gradient(alpha, dz, axis=2)
    dphi_x = np.gradient(phi, dx, axis=0)
    dphi_y = np.gradient(phi, dy, axis=1)
    dphi_z = np.gradient(phi, dz, axis=2)

    # Shift gradients
    dbeta = np.zeros((Nx, Ny, Nz, 3, 3))
    for k in range(3):
        dbeta[..., k, 0] = np.gradient(beta[..., k], dx, axis=0)
        dbeta[..., k, 1] = np.gradient(beta[..., k], dy, axis=1)
        dbeta[..., k, 2] = np.gradient(beta[..., k], dz, axis=2)

    # Second derivatives for C
    lap_beta = np.zeros((Nx, Ny, Nz, 3))
    for i in range(3):
        fxx = np.gradient(np.gradient(beta[..., i], dx, axis=0), dx, axis=0)
        fyy = np.gradient(np.gradient(beta[..., i], dy, axis=1), dy, axis=1)
        fzz = np.gradient(np.gradient(beta[..., i], dz, axis=2), dz, axis=2)
        lap_beta[..., i] = fxx + fyy + fzz

    div_beta = np.gradient(beta[..., 0], dx, axis=0) + np.gradient(beta[..., 1], dy, axis=1) + np.gradient(beta[..., 2], dz, axis=2)

    d_div_beta_x = np.gradient(div_beta, dx, axis=0)
    d_div_beta_y = np.gradient(div_beta, dy, axis=1)
    d_div_beta_z = np.gradient(div_beta, dz, axis=2)

    # Gamma gradients
    dGamma = np.zeros((Nx, Ny, Nz, 3, 3))
    for i in range(3):
        dGamma[..., i, 0] = np.gradient(Gamma_tilde[..., i], dx, axis=0)
        dGamma[..., i, 1] = np.gradient(Gamma_tilde[..., i], dy, axis=1)
        dGamma[..., i, 2] = np.gradient(Gamma_tilde[..., i], dz, axis=2)

    rhs_Gamma_tilde = np.zeros((Nx, Ny, Nz, 3))

    # A: advection
    rhs_Gamma_tilde += np.einsum('...k,...ik->...i', beta, dGamma)

    # B: stretching
    Gamma_dot_grad_beta = np.einsum('...k,...ik->...i', Gamma_tilde, dbeta)
    rhs_Gamma_tilde += -Gamma_dot_grad_beta + (2.0/3.0) * Gamma_tilde * div_beta[..., np.newaxis]

    # C: shift second-derivatives (approximated)
    d_div_beta = np.array([d_div_beta_x, d_div_beta_y, d_div_beta_z])
    d_div_beta = d_div_beta.transpose(1,2,3,0)
    rhs_Gamma_tilde += lap_beta + (1.0/3.0) * np.einsum('...ij,...j->...i', gamma_tilde_inv_full, d_div_beta)

    # D: lapse/curvature
    # -2 A^{ij} d_j alpha
    dalpha = np.array([dalpha_x, dalpha_y, dalpha_z]).transpose(1,2,3,0)
    rhs_Gamma_tilde += -2.0 * np.einsum('...ij,...j->...i', A_tilde_uu, dalpha)

    # 2 alpha (Gamma^i_{jk} A^{jk} + 6 A^{ij} d_j phi - (2/3) gamma^{ij} d_j K)
    GammaA = np.einsum('...ijk,...jk->...i', christoffel_tilde_udd, A_tilde_uu)

    dphi = np.array([dphi_x, dphi_y, dphi_z]).transpose(1,2,3,0)
    A_dphi = np.einsum('...ij,...j->...i', A_tilde_uu, dphi)

    dK_x = np.gradient(K_trace_scratch, dx, axis=0)
    dK_y = np.gradient(K_trace_scratch, dy, axis=1)
    dK_z = np.gradient(K_trace_scratch, dz, axis=2)
    dK = np.array([dK_x, dK_y, dK_z]).transpose(1,2,3,0)
    gamma_dK = np.einsum('...ij,...j->...i', gamma_tilde_inv_full, dK)

    rhs_Gamma_tilde += 2.0 * alpha[..., np.newaxis] * (GammaA + 6.0 * A_dphi - (2.0/3.0) * gamma_dK)

    return rhs_Gamma_tilde

class GRStepper(StepperContract):
    def __init__(self, fields, geometry, constraints, gauge, memory_contract=None, phaseloom=None, aeonic_mode=True, max_attempts=20, dt_floor=1e-10, temporal_system=None):
        super().__init__(max_attempts=max_attempts, dt_floor=dt_floor)
        self.fields = fields
        self.geometry = geometry
        self.constraints = constraints
        self.gauge = gauge
        self.dt_applied = 0.0
        self.damping_enabled = True
        self.lambda_val = 0.0
        self.dealiasing_enabled = True
        self.sources_func = None
        self.memory = StepperContractWithMemory(memory_contract, phaseloom, max_attempts=max_attempts, dt_floor=dt_floor) if memory_contract and phaseloom else None
        self.aeonic_mode = aeonic_mode
        self.loc_operator = GRLoC(fields, geometry, constraints, lambda_val=self.lambda_val, kappa_H=1.0, kappa_M=1.0)
        self.temporal_system = temporal_system

        # Load Hadamard RHS if available
        try:
            from nsc_runtime_min import make_rhs_callable
            self.rhs_func = make_rhs_callable('minkowski_rhs.nscir.json')
        except:
            self.rhs_func = None

        # Receipt chain for audit trail
        self.prev_receipt_hash = "0" * 64  # Initial hash for chain start
        self.receipts_file = "aeonic_receipts.jsonl"

        # Multi-rate stepping parameters
        self.slow_fields = ['phi', 'Z', 'Z_i']
        self.slow_rate = 5  # Update slow fields every 5 steps
        self.step_count = 0

        if self.aeonic_mode:
            # Preallocate RHS arrays
            self.rhs_gamma_sym6 = np.zeros_like(self.fields.gamma_sym6)
            self.rhs_K_sym6 = np.zeros_like(self.fields.K_sym6)
            self.rhs_phi = np.zeros_like(self.fields.phi)
            self.rhs_gamma_tilde_sym6 = np.zeros_like(self.fields.gamma_tilde_sym6)
            self.rhs_A_sym6 = np.zeros_like(self.fields.A_sym6)
            self.rhs_Gamma_tilde = np.zeros_like(self.fields.Gamma_tilde)
            self.rhs_Z = np.zeros_like(self.fields.Z)
            self.rhs_Z_i = np.zeros_like(self.fields.Z_i)

            # Sources arrays
            self.S_gamma_tilde_sym6 = np.zeros_like(self.fields.gamma_tilde_sym6)
            self.S_A_sym6 = np.zeros_like(self.fields.A_sym6)
            self.S_phi = np.zeros_like(self.fields.phi)
            self.S_Gamma_tilde = np.zeros_like(self.fields.Gamma_tilde)
            self.S_Z = np.zeros_like(self.fields.Z)
            self.S_Z_i = np.zeros_like(self.fields.Z_i)

            # Preallocate scratch buffers
            self.gamma_inv_scratch = np.zeros_like(self.fields.gamma_sym6)
            self.K_trace_scratch = np.zeros((self.fields.Nx, self.fields.Ny, self.fields.Nz))
            self.alpha_expanded_scratch = np.zeros((self.fields.Nx, self.fields.Ny, self.fields.Nz, 6))
            self.alpha_expanded_33_scratch = np.zeros((self.fields.Nx, self.fields.Ny, self.fields.Nz, 3, 3))
            self.lie_gamma_scratch = np.zeros_like(self.fields.gamma_sym6)
            self.DD_alpha_scratch = np.zeros((self.fields.Nx, self.fields.Ny, self.fields.Nz, 3, 3))
            self.DD_alpha_sym6_scratch = np.zeros_like(self.fields.gamma_sym6)
            self.ricci_sym6_scratch = np.zeros_like(self.fields.gamma_sym6)
            self.K_full_scratch = np.zeros((self.fields.Nx, self.fields.Ny, self.fields.Nz, 3, 3))
            self.gamma_inv_full_scratch = np.zeros((self.fields.Nx, self.fields.Ny, self.fields.Nz, 3, 3))
            self.K_contracted_full_scratch = np.zeros((self.fields.Nx, self.fields.Ny, self.fields.Nz, 3, 3))
            self.K_contracted_sym6_scratch = np.zeros_like(self.fields.gamma_sym6)
            self.lie_K_scratch = np.zeros_like(self.fields.gamma_sym6)
            self.lie_gamma_tilde_scratch = np.zeros_like(self.fields.gamma_sym6)
            self.psi_minus4_scratch = np.zeros((self.fields.Nx, self.fields.Ny, self.fields.Nz))
            self.psi_minus4_expanded_scratch = np.zeros((self.fields.Nx, self.fields.Ny, self.fields.Nz, 6))
            self.ricci_tf_sym6_scratch = np.zeros_like(self.fields.gamma_sym6)
            self.rhs_A_temp_scratch = np.zeros_like(self.fields.gamma_sym6)
            self.lie_A_scratch = np.zeros_like(self.fields.gamma_sym6)

    def check_tensor_layout_compliance(self):
        """Check tensor layout compliance against canonical shapes. Emit violation receipts."""
        expected_shapes = {
            'gamma_sym6': (self.fields.Nx, self.fields.Ny, self.fields.Nz, 6),
            'K_sym6': (self.fields.Nx, self.fields.Ny, self.fields.Nz, 6),
            'alpha': (self.fields.Nx, self.fields.Ny, self.fields.Nz),
            'beta': (self.fields.Nx, self.fields.Ny, self.fields.Nz, 3),
            'phi': (self.fields.Nx, self.fields.Ny, self.fields.Nz),
            'Z': (self.fields.Nx, self.fields.Ny, self.fields.Nz),
            'Z_i': (self.fields.Nx, self.fields.Ny, self.fields.Nz, 3),
            'gamma_tilde_sym6': (self.fields.Nx, self.fields.Ny, self.fields.Nz, 6),
            'A_sym6': (self.fields.Nx, self.fields.Ny, self.fields.Nz, 6),
            'Gamma_tilde': (self.fields.Nx, self.fields.Ny, self.fields.Nz, 3)
        }

        violations = []
        for field_name, expected_shape in expected_shapes.items():
            field = getattr(self.fields, field_name)
            actual_shape = field.shape
            if actual_shape != expected_shape:
                violations.append({
                    'field': field_name,
                    'expected_shape': expected_shape,
                    'actual_shape': actual_shape
                })

        layout_ok = len(violations) == 0

        if not layout_ok:
            # Emit violation receipt
            receipt = {
                'run_id': 'gr_solver_run_001',
                'step': getattr(self, 'current_step', 0),
                'event': 'LAYOUT_VIOLATION',
                't': getattr(self, 'current_t', 0.0),
                'dt': getattr(self, 'current_dt', 0.0),
                'stage': None,
                'grid': {
                    'Nx': self.fields.Nx,
                    'Ny': self.fields.Ny,
                    'Nz': self.fields.Nz,
                    'h': [self.fields.dx, self.fields.dy, self.fields.dz],
                    'domain': 'cartesian',
                    'periodic': False
                },
                'layout': {
                    'ok': False,
                    'violations': violations,
                    'first_bad_tensor': violations[0]['field'] if violations else None,
                    'got_shape': violations[0]['actual_shape'] if violations else None,
                    'expected_shape': violations[0]['expected_shape'] if violations else None
                },
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }

            # Hash this receipt
            receipt_str = json.dumps(receipt, sort_keys=True)
            receipt_hash = hashlib.sha256(receipt_str.encode()).hexdigest()
            receipt['receipt_hash'] = receipt_hash
            receipt['prev_receipt_hash'] = self.prev_receipt_hash
            self.prev_receipt_hash = receipt_hash

            # Write to JSONL file
            with open(self.receipts_file, 'a') as f:
                f.write(json.dumps(receipt) + '\n')

            raise ValueError(f"Tensor layout violations detected: {violations}")

        return layout_ok

    def emit_stage_rhs_receipt(self, stage, stage_time, rhs_norms):
        """Emit STAGE_RHS receipt after computing RHS for a stage."""
        # Compute operator hash (simplified - hash of compute_rhs method)
        operator_code = inspect.getsource(self.compute_rhs)
        operator_hash = hashlib.sha256(operator_code.encode()).hexdigest()

        # State hash (simplified fingerprint)
        state_str = f"{self.fields.alpha.sum():.6e}_{self.fields.gamma_sym6.sum():.6e}"
        state_hash = hashlib.sha256(state_str.encode()).hexdigest()

        receipt = {
            'run_id': 'gr_solver_run_001',
            'step': getattr(self, 'current_step', 0),
            'event': 'STAGE_RHS',
            't': getattr(self, 'current_t', 0.0),
            'dt': getattr(self, 'current_dt', 0.0),
            'stage': stage,
            'stage_time': stage_time,
            'grid': {
                'Nx': self.fields.Nx,
                'Ny': self.fields.Ny,
                'Nz': self.fields.Nz,
                'h': [self.fields.dx, self.fields.dy, self.fields.dz],
                'domain': 'cartesian',
                'periodic': False
            },
            'hash': {
                'state_before': state_hash,
                'rhs': operator_hash,
                'operators': operator_hash
            },
            'ledgers': {},  # Will be filled at ledger eval
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }

        # Add rhs norms
        receipt['rhs_norms'] = rhs_norms

        # Hash and chain
        receipt_str = json.dumps(receipt, sort_keys=True)
        receipt_hash = hashlib.sha256(receipt_str.encode()).hexdigest()
        receipt['receipt_hash'] = receipt_hash
        receipt['prev_receipt_hash'] = self.prev_receipt_hash
        self.prev_receipt_hash = receipt_hash

        # Write receipt
        with open(self.receipts_file, 'a') as f:
            f.write(json.dumps(receipt) + '\n')

    def compute_clocks(self, dt_candidate):
        """Compute all clock constraints and choose dt."""
        # CFL: dt < dx / c, where c is characteristic speed
        # Rough estimate: c ~ sqrt(alpha^2 + beta^2) for ADM
        c_max = np.sqrt(np.max(self.fields.alpha)**2 + np.max(np.linalg.norm(self.fields.beta, axis=-1))**2)
        h_min = min(self.fields.dx, self.fields.dy, self.fields.dz)
        dt_CFL = h_min / c_max if c_max > 0 else float('inf')

        # Gauge: dt < alpha * dx / |beta| or similar
        # Simplified: dt < alpha * h_min / (1 + |beta|)
        beta_norm = np.max(np.linalg.norm(self.fields.beta, axis=-1))
        dt_gauge = self.fields.alpha.max() * h_min / (1 + beta_norm)

        # Coherence (constraint damping): dt < 1 / lambda where lambda is damping rate
        dt_coh = 1.0 / max(self.lambda_val, 1e-6) if self.lambda_val > 0 else float('inf')

        # Resolution: dt < h_min / sqrt(K^2) or similar
        K_norm = np.max(np.linalg.norm(self.fields.K_sym6, axis=-1))
        dt_res = h_min / max(np.sqrt(K_norm), 1e-6)

        # Sigma (shock capturing or similar): placeholder
        dt_sigma = float('inf')  # Not implemented yet

        # Choose minimum
        dt_used = min(dt_candidate, dt_CFL, dt_gauge, dt_coh, dt_res, dt_sigma)

        clocks = {
            'dt_CFL': dt_CFL,
            'dt_gauge': dt_gauge,
            'dt_coh': dt_coh,
            'dt_res': dt_res,
            'dt_sigma': dt_sigma,
            'dt_used': dt_used
        }

        return clocks, dt_used

    def emit_clock_decision_receipt(self, clocks):
        """Emit CLOCK_DECISION receipt."""
        receipt = {
            'run_id': 'gr_solver_run_001',
            'step': self.current_step,
            'event': 'CLOCK_DECISION',
            't': self.current_t,
            'dt': self.current_dt,
            'stage': None,
            'grid': {
                'Nx': self.fields.Nx,
                'Ny': self.fields.Ny,
                'Nz': self.fields.Nz,
                'h': [self.fields.dx, self.fields.dy, self.fields.dz],
                'domain': 'cartesian',
                'periodic': False
            },
            'clocks': clocks,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }

        # Hash and chain
        receipt_str = json.dumps(receipt, sort_keys=True)
        receipt_hash = hashlib.sha256(receipt_str.encode()).hexdigest()
        receipt['receipt_hash'] = receipt_hash
        receipt['prev_receipt_hash'] = self.prev_receipt_hash
        self.prev_receipt_hash = receipt_hash

        # Write receipt
        with open(self.receipts_file, 'a') as f:
            f.write(json.dumps(receipt) + '\n')

    def check_gates_internal(self, rails_policy=None):
        """Check step gates. Return (accepted, hard_fail, penalty, reasons, margins)."""
        if rails_policy is None:
            rails_policy = {
                'eps_H_max': 1e-2,  # soft
                'eps_H_hard_max': 1e-1,
                'eps_M_max': 1e-2,
                'eps_M_hard_max': 1e-1,
                'eps_proj_max': 1e-2,
                'eps_proj_hard_max': 1e-1,
                'eps_clk_max': 1e-2,
                'eps_clk_hard_max': 1e-1,
                'spike_soft_max': 1e2,
                'spike_hard_max': 1e3
            }

        # Compute residuals
        eps_H = float(self.constraints.eps_H)
        eps_M = float(self.constraints.eps_M)
        eps_proj = float(self.constraints.eps_proj)
        eps_clk = float(self.constraints.eps_clk) if self.constraints.eps_clk is not None else 0.0

        eps_H_soft_max = rails_policy.get('eps_H_max', 1e-2)
        eps_H_hard_max = rails_policy.get('eps_H_hard_max', 1e-1)
        eps_M_soft_max = rails_policy.get('eps_M_max', 1e-2)
        eps_M_hard_max = rails_policy.get('eps_M_hard_max', 1e-1)
        eps_proj_soft_max = rails_policy.get('eps_proj_max', 1e-2)
        eps_proj_hard_max = rails_policy.get('eps_proj_hard_max', 1e-1)
        eps_clk_soft_max = rails_policy.get('eps_clk_max', 1e-2)
        eps_clk_hard_max = rails_policy.get('eps_clk_hard_max', 1e-1)
        spike_soft_max = rails_policy.get('spike_soft_max', 1e2)
        spike_hard_max = rails_policy.get('spike_hard_max', 1e3)

        # Check for hard fails: NaN or inf
        if np.isnan(eps_H) or np.isinf(eps_H) or np.isnan(eps_M) or np.isinf(eps_M) or np.isnan(eps_proj) or np.isinf(eps_proj) or np.isnan(eps_clk) or np.isinf(eps_clk):
            logger.error("Hard fail: NaN or infinite residuals in gates", extra={
                "extra_data": {
                    "eps_H": eps_H,
                    "eps_M": eps_M,
                    "eps_proj": eps_proj,
                    "eps_clk": eps_clk
                }
            })
            return False, True, float('inf'), ["NaN/inf residuals"], {}

        accepted = True
        hard_fail = False
        penalty = 0.0
        reasons = []
        margins = {}
        corrections = {}

        # Special handling for eps_H
        eps_H_warn = 7.5e-5
        eps_H_fail = 1e-4
        if eps_H > eps_H_fail:
            hard_fail = True
            accepted = False
            reasons.append(f"eps_H = {eps_H:.2e} > {eps_H_fail:.2e}")
            margins['eps_H'] = eps_H_fail - eps_H
        elif eps_H > eps_H_warn:
            penalty += (eps_H - eps_H_warn) / eps_H_warn
            reasons.append(f"Warn: eps_H = {eps_H:.2e} > {eps_H_warn:.2e}")
            margins['eps_H'] = eps_H_warn - eps_H
            corrections = {'reduce_dt': True, 'increase_kappa_budget': True, 'increase_projection_freq': True}

        # Check other eps
        for eps, name, soft, hard in [
            (eps_M, 'eps_M', eps_M_soft_max, eps_M_hard_max),
            (eps_proj, 'eps_proj', eps_proj_soft_max, eps_proj_hard_max),
            (eps_clk, 'eps_clk', eps_clk_soft_max, eps_clk_hard_max)
        ]:
            if eps > hard:
                hard_fail = True
                accepted = False
                reasons.append(f"{name} = {eps:.2e} > {hard:.2e}")
                margins[name] = hard - eps
            elif eps > soft:
                penalty += (eps - soft) / soft
                reasons.append(f"Soft: {name} = {eps:.2e} > {soft:.2e}")
                margins[name] = soft - eps

        # Spike norms
        spike_norms = {
            'alpha_spike': np.max(np.abs(np.gradient(self.fields.alpha))),
            'K_spike': np.max(np.abs(np.gradient(self.fields.K_sym6)))
        }
        for field, spike in spike_norms.items():
            if np.isnan(spike) or np.isinf(spike):
                hard_fail = True
                accepted = False
                reasons.append(f"Hard: {field} NaN/inf")
                margins[field] = float('-inf')
            elif spike > spike_hard_max:
                hard_fail = True
                accepted = False
                reasons.append(f"{field} spike = {spike:.2e} > {spike_hard_max:.2e}")
                margins[field] = spike_hard_max - spike
            elif spike > spike_soft_max:
                penalty += (spike - spike_soft_max) / spike_soft_max
                reasons.append(f"Soft: {field} spike = {spike:.2e} > {spike_soft_max:.2e}")
                margins[field] = spike_soft_max - spike

        return accepted, hard_fail, penalty, reasons, margins, corrections

    def check_gates(self, rails_policy) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Implement StepperContract.check_gates.

        Classifies violation types: dt-dependent (e.g., CFL), state-dependent (e.g., constraints), sem.
        """
        accepted, hard_fail, penalty, reasons, margins = self.check_gates(rails_policy)

        if hard_fail or not accepted:
            # Classify violation type
            violation_type = 'state'  # default
            for reason in reasons:
                if 'CFL' in reason or 'dt' in reason.lower() or 'timestep' in reason.lower():
                    violation_type = 'dt'
                    break
                elif 'NaN' in reason or 'inf' in reason or 'sem' in reason.upper():
                    violation_type = 'sem'
                    break
                # else state

            details = {
                'hard_fail': hard_fail,
                'penalty': penalty,
                'reasons': reasons,
                'margins': margins
            }
            return False, violation_type, details
        else:
            return True, '', {}

    def emit_ledger_eval_receipt(self, ledgers):
        """Emit LEDGER_EVAL receipt after step completion."""
        receipt = {
            'run_id': 'gr_solver_run_001',
            'step': self.current_step,
            'event': 'LEDGER_EVAL',
            't': self.current_t + self.current_dt,
            'dt': self.current_dt,
            'stage': None,
            'grid': {
                'Nx': self.fields.Nx,
                'Ny': self.fields.Ny,
                'Nz': self.fields.Nz,
                'h': [self.fields.dx, self.fields.dy, self.fields.dz],
                'domain': 'cartesian',
                'periodic': False
            },
            'ledgers': ledgers,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }

        # Hash and chain
        receipt_str = json.dumps(receipt, sort_keys=True)
        receipt_hash = hashlib.sha256(receipt_str.encode()).hexdigest()
        receipt['receipt_hash'] = receipt_hash
        receipt['prev_receipt_hash'] = self.prev_receipt_hash
        self.prev_receipt_hash = receipt_hash

        # Write receipt
        with open(self.receipts_file, 'a') as f:
            f.write(json.dumps(receipt) + '\n')

    def emit_step_receipt(self, accepted, rejection_reason=None, stage_eps_H=None, corrections_applied=None):
        """Emit STEP_ACCEPT or STEP_REJECT receipt."""
        event = 'STEP_ACCEPT' if accepted else 'STEP_REJECT'

        # Final ledgers
        ledgers = {
            'eps_H': float(self.constraints.eps_H),
            'eps_M': float(self.constraints.eps_M),
            'eps_proj': float(self.constraints.eps_proj),
            'eps_clk': float(self.constraints.eps_clk) if self.constraints.eps_clk is not None else 0.0,
            'eps_H_norm': float(self.constraints.eps_H_norm),
            'eps_M_norm': float(self.constraints.eps_M_norm),
            'eps_proj_norm': float(self.constraints.eps_proj_norm),
            'spikes': {
                'alpha_max_grad': float(np.max(np.abs(np.gradient(self.fields.alpha)))),
                'K_max_grad': float(np.max(np.abs(np.gradient(self.fields.K_sym6))))
            }
        }

        accepted_gates, hard_fail_gates, penalty_gates, reasons, margins, _ = self.check_gates_internal()

        receipt = {
            'run_id': 'gr_solver_run_001',
            'step': self.current_step,
            'event': event,
            't': self.current_t + self.current_dt if accepted else self.current_t,
            'dt': self.current_dt,
            'stage': None,
            'grid': {
                'Nx': self.fields.Nx,
                'Ny': self.fields.Ny,
                'Nz': self.fields.Nz,
                'h': [self.fields.dx, self.fields.dy, self.fields.dz],
                'domain': 'cartesian',
                'periodic': False
            },
            'ledgers': ledgers,
            'gates': {
                'pass': accepted_gates,
                'penalty': penalty_gates,
                'reasons': reasons,
                'margins': margins
            },
            'stage_eps_H': stage_eps_H or {},
            'rejection_reason': rejection_reason,
            'corrections_applied': corrections_applied or {},
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }

        # Hash and chain
        receipt_str = json.dumps(receipt, sort_keys=True)
        receipt_hash = hashlib.sha256(receipt_str.encode()).hexdigest()
        receipt['receipt_hash'] = receipt_hash
        receipt['prev_receipt_hash'] = self.prev_receipt_hash
        self.prev_receipt_hash = receipt_hash

        # Write receipt
        with open(self.receipts_file, 'a') as f:
            f.write(json.dumps(receipt) + '\n')

    def step_ufe(self, dt, t=0.0):
        """RK4 step for UFE."""
        with Timer("step_ufe") as timer:
            self.dt_applied = dt  # Set dt_applied as source of truth
            self.current_dt = dt
            self.current_t = t
            self.current_step = getattr(self, 'current_step', 0) + 1

            # Clock decision
            clocks, dt_used = self.compute_clocks(dt)
            if dt_used < dt:
                logger.warning(f"Clock constraint reduced dt from {dt} to {dt_used}")
                self.current_dt = dt_used
            self.emit_clock_decision_receipt(clocks)

            # Log initial constraint residuals before stepping
            self.constraints.compute_hamiltonian()
            self.constraints.compute_momentum()
            self.constraints.compute_residuals()
            # Set initial values for normalized residuals
            self.constraints.eps_H_initial = self.constraints.eps_H
            self.constraints.eps_M_initial = self.constraints.eps_M
            self.constraints.eps_proj_initial = self.constraints.eps_proj
            # Initialize eps_clk
            self.constraints.eps_clk = 0.0

            # Initialize stage eps_H logging
            stage_eps_H = {}
            stage_eps_H['eps_H_pre'] = float(self.constraints.eps_H)

            logger.debug("Starting UFE step", extra={
                "extra_data": {
                    "dt": dt,
                    "dt_used": dt_used,
                    "t": t,
                    "eps_H_initial": float(self.constraints.eps_H),
                    "eps_M_initial": float(self.constraints.eps_M),
                    "eps_proj_initial": float(self.constraints.eps_proj)
                }
            })

            slow_update = (self.step_count % self.slow_rate == 0)

            # Save initial state
            u0_gamma = self.fields.gamma_sym6.copy()
            u0_K = self.fields.K_sym6.copy()
            u0_phi = self.fields.phi.copy()
            u0_gamma_tilde = self.fields.gamma_tilde_sym6.copy()
            u0_A = self.fields.A_sym6.copy()
            u0_Gamma_tilde = self.fields.Gamma_tilde.copy()
            u0_Z = self.fields.Z.copy()
            u0_Z_i = self.fields.Z_i.copy()

            # Set authoritative state for LoC
            Psi_auth = {
                'gamma_sym6': u0_gamma,
                'K_sym6': u0_K,
                'phi': u0_phi,
                'gamma_tilde_sym6': u0_gamma_tilde,
                'A_sym6': u0_A,
                'Gamma_tilde': u0_Gamma_tilde,
                'Z': u0_Z,
                'Z_i': u0_Z_i
            }
            self.loc_operator.set_authoritative_state(Psi_auth)

            # Stage 1
            self.compute_rhs(t, slow_update)
            rhs_norms = {
                'gamma': np.linalg.norm(self.rhs_gamma_sym6),
                'K': np.linalg.norm(self.rhs_K_sym6),
                'phi': np.linalg.norm(self.rhs_phi),
                'gamma_tilde': np.linalg.norm(self.rhs_gamma_tilde_sym6),
                'A': np.linalg.norm(self.rhs_A_sym6),
                'Z': np.linalg.norm(self.rhs_Z),
                'Z_i': np.linalg.norm(self.rhs_Z_i)
            }
            self.emit_stage_rhs_receipt(1, t, rhs_norms)
            k1_gamma = self.rhs_gamma_sym6.copy()
            k1_K = self.rhs_K_sym6.copy()
            k1_phi = self.rhs_phi.copy()
            k1_gamma_tilde = self.rhs_gamma_tilde_sym6.copy()
            k1_A = self.rhs_A_sym6.copy()
            k1_Z = self.rhs_Z.copy()
            k1_Z_i = self.rhs_Z_i.copy()

            # Update eps_clk: Linf of Psi_used - Psi_auth after stage 1
            Psi_used_stage1 = {
                'gamma_sym6': u0_gamma + (dt/2) * k1_gamma,
                'K_sym6': u0_K + (dt/2) * k1_K,
                'phi': u0_phi + (dt/2) * k1_phi,
                'gamma_tilde_sym6': u0_gamma_tilde + (dt/2) * k1_gamma_tilde,
                'A_sym6': u0_A + (dt/2) * k1_A,
                'Gamma_tilde': u0_Gamma_tilde,  # Gamma_tilde not updated in stage
                'Z': u0_Z + (dt/2) * k1_Z,
                'Z_i': u0_Z_i + (dt/2) * k1_Z_i
            }
            stage_diff = self.constraints.compute_stage_difference_Linf(Psi_used_stage1, Psi_auth)
            self.constraints.eps_clk = max(self.constraints.eps_clk, stage_diff)

            # Stage 2: u + dt/2 * k1
            self.fields.gamma_sym6 = u0_gamma + (dt/2) * k1_gamma
            self.fields.K_sym6 = u0_K + (dt/2) * k1_K
            self.fields.phi = u0_phi + (dt/2) * k1_phi
            self.fields.gamma_tilde_sym6 = u0_gamma_tilde + (dt/2) * k1_gamma_tilde
            self.fields.A_sym6 = u0_A + (dt/2) * k1_A
            self.fields.Z = u0_Z + (dt/2) * k1_Z
            self.fields.Z_i = u0_Z_i + (dt/2) * k1_Z_i
            self.geometry.compute_christoffels()
            self.geometry.compute_ricci()
            self.geometry.compute_scalar_curvature()
            # Update eps_clk for stage 2
            Psi_used_stage2 = {
                'gamma_sym6': self.fields.gamma_sym6,
                'K_sym6': self.fields.K_sym6,
                'phi': self.fields.phi,
                'gamma_tilde_sym6': self.fields.gamma_tilde_sym6,
                'A_sym6': self.fields.A_sym6,
                'Gamma_tilde': self.fields.Gamma_tilde,
                'Z': self.fields.Z,
                'Z_i': self.fields.Z_i
            }
            stage_diff = self.constraints.compute_stage_difference_Linf(Psi_used_stage2, Psi_auth)
            self.constraints.eps_clk = max(self.constraints.eps_clk, stage_diff)
            self.compute_rhs(t + dt/2, slow_update)
            rhs_norms = {
                'gamma': np.linalg.norm(self.rhs_gamma_sym6),
                'K': np.linalg.norm(self.rhs_K_sym6),
                'phi': np.linalg.norm(self.rhs_phi),
                'gamma_tilde': np.linalg.norm(self.rhs_gamma_tilde_sym6),
                'A': np.linalg.norm(self.rhs_A_sym6),
                'Z': np.linalg.norm(self.rhs_Z),
                'Z_i': np.linalg.norm(self.rhs_Z_i)
            }
            self.emit_stage_rhs_receipt(2, t + dt/2, rhs_norms)
            k2_gamma = self.rhs_gamma_sym6.copy()
            k2_K = self.rhs_K_sym6.copy()
            k2_phi = self.rhs_phi.copy()
            k2_gamma_tilde = self.rhs_gamma_tilde_sym6.copy()
            k2_A = self.rhs_A_sym6.copy()
            k2_Z = self.rhs_Z.copy()
            k2_Z_i = self.rhs_Z_i.copy()

            # Stage 3: u + dt/2 * k2
            self.fields.gamma_sym6 = u0_gamma + (dt/2) * k2_gamma
            self.fields.K_sym6 = u0_K + (dt/2) * k2_K
            self.fields.phi = u0_phi + (dt/2) * k2_phi
            self.fields.gamma_tilde_sym6 = u0_gamma_tilde + (dt/2) * k2_gamma_tilde
            self.fields.A_sym6 = u0_A + (dt/2) * k2_A
            self.fields.Z = u0_Z + (dt/2) * k2_Z
            self.fields.Z_i = u0_Z_i + (dt/2) * k2_Z_i
            self.geometry.compute_christoffels()
            self.geometry.compute_ricci()
            self.geometry.compute_scalar_curvature()
            # Update eps_clk for stage 3
            Psi_used_stage3 = {
                'gamma_sym6': self.fields.gamma_sym6,
                'K_sym6': self.fields.K_sym6,
                'phi': self.fields.phi,
                'gamma_tilde_sym6': self.fields.gamma_tilde_sym6,
                'A_sym6': self.fields.A_sym6,
                'Gamma_tilde': self.fields.Gamma_tilde,
                'Z': self.fields.Z,
                'Z_i': self.fields.Z_i
            }
            stage_diff = self.constraints.compute_stage_difference_Linf(Psi_used_stage3, Psi_auth)
            self.constraints.eps_clk = max(self.constraints.eps_clk, stage_diff)
            self.compute_rhs(t + dt/2, slow_update)
            rhs_norms = {
                'gamma': np.linalg.norm(self.rhs_gamma_sym6),
                'K': np.linalg.norm(self.rhs_K_sym6),
                'phi': np.linalg.norm(self.rhs_phi),
                'gamma_tilde': np.linalg.norm(self.rhs_gamma_tilde_sym6),
                'A': np.linalg.norm(self.rhs_A_sym6),
                'Z': np.linalg.norm(self.rhs_Z),
                'Z_i': np.linalg.norm(self.rhs_Z_i)
            }
            self.emit_stage_rhs_receipt(3, t + dt/2, rhs_norms)
            k3_gamma = self.rhs_gamma_sym6.copy()
            k3_K = self.rhs_K_sym6.copy()
            k3_phi = self.rhs_phi.copy()
            k3_gamma_tilde = self.rhs_gamma_tilde_sym6.copy()
            k3_A = self.rhs_A_sym6.copy()
            k3_Z = self.rhs_Z.copy()
            k3_Z_i = self.rhs_Z_i.copy()

            # Stage 4: u + dt * k3
            self.fields.gamma_sym6 = u0_gamma + dt * k3_gamma
            self.fields.K_sym6 = u0_K + dt * k3_K
            self.fields.phi = u0_phi + dt * k3_phi
            self.fields.gamma_tilde_sym6 = u0_gamma_tilde + dt * k3_gamma_tilde
            self.fields.A_sym6 = u0_A + dt * k3_A
            self.fields.Z = u0_Z + dt * k3_Z
            self.fields.Z_i = u0_Z_i + dt * k3_Z_i
            self.geometry.compute_christoffels()
            self.geometry.compute_ricci()
            self.geometry.compute_scalar_curvature()
            # Update eps_clk for stage 4
            Psi_used_stage4 = {
                'gamma_sym6': self.fields.gamma_sym6,
                'K_sym6': self.fields.K_sym6,
                'phi': self.fields.phi,
                'gamma_tilde_sym6': self.fields.gamma_tilde_sym6,
                'A_sym6': self.fields.A_sym6,
                'Gamma_tilde': self.fields.Gamma_tilde,
                'Z': self.fields.Z,
                'Z_i': self.fields.Z_i
            }
            stage_diff = self.constraints.compute_stage_difference_Linf(Psi_used_stage4, Psi_auth)
            self.constraints.eps_clk = max(self.constraints.eps_clk, stage_diff)
            self.compute_rhs(t + dt, slow_update)
            rhs_norms = {
                'gamma': np.linalg.norm(self.rhs_gamma_sym6),
                'K': np.linalg.norm(self.rhs_K_sym6),
                'phi': np.linalg.norm(self.rhs_phi),
                'gamma_tilde': np.linalg.norm(self.rhs_gamma_tilde_sym6),
                'A': np.linalg.norm(self.rhs_A_sym6),
                'Z': np.linalg.norm(self.rhs_Z),
                'Z_i': np.linalg.norm(self.rhs_Z_i)
            }
            self.emit_stage_rhs_receipt(4, t + dt, rhs_norms)
            k4_gamma = self.rhs_gamma_sym6.copy()
            k4_K = self.rhs_K_sym6.copy()
            k4_phi = self.rhs_phi.copy()
            k4_gamma_tilde = self.rhs_gamma_tilde_sym6.copy()
            k4_A = self.rhs_A_sym6.copy()
            k4_Z = self.rhs_Z.copy()
            k4_Z_i = self.rhs_Z_i.copy()

            # Final update
            self.fields.gamma_sym6 = u0_gamma + (dt/6) * (k1_gamma + 2*k2_gamma + 2*k3_gamma + k4_gamma)
            self.fields.K_sym6 = u0_K + (dt/6) * (k1_K + 2*k2_K + 2*k3_K + k4_K)
            self.fields.phi = u0_phi + (dt/6) * (k1_phi + 2*k2_phi + 2*k3_phi + k4_phi)
            self.fields.gamma_tilde_sym6 = u0_gamma_tilde + (dt/6) * (k1_gamma_tilde + 2*k2_gamma_tilde + 2*k3_gamma_tilde + k4_gamma_tilde)
            self.fields.A_sym6 = u0_A + (dt/6) * (k1_A + 2*k2_A + 2*k3_A + k4_A)
            self.fields.Z = u0_Z + (dt/6) * (k1_Z + 2*k2_Z + 2*k3_Z + k4_Z)
            self.fields.Z_i = u0_Z_i + (dt/6) * (k1_Z_i + 2*k2_Z_i + 2*k3_Z_i + k4_Z_i)

            # Recompute geometry for final state
            self.geometry.compute_christoffels()
            self.geometry.compute_ricci()
            self.geometry.compute_scalar_curvature()

            # Apply constraint damping (projection effects)
            self.apply_damping()
            self.constraints.compute_residuals()
            stage_eps_H['eps_H_post_phys'] = float(self.constraints.eps_H)
            stage_eps_H['eps_H_post_cons'] = float(self.constraints.eps_H)
            # Evolve gauge
            self.gauge.evolve_lapse(dt)
            self.gauge.evolve_shift(dt)
            # Recompute geometry and constraints after gauge evolution
            self.geometry.compute_christoffels()
            self.geometry.compute_ricci()
            self.geometry.compute_scalar_curvature()
            self.constraints.compute_hamiltonian()
            self.constraints.compute_momentum()
            self.constraints.compute_residuals()
            stage_eps_H['eps_H_post_gauge'] = float(self.constraints.eps_H)
            stage_eps_H['eps_H_post_filter'] = float(self.constraints.eps_H)

            # Emit LEDGER_EVAL receipt
            final_ledgers = {
                'eps_H': float(self.constraints.eps_H),
                'eps_M': float(self.constraints.eps_M),
                'eps_proj': float(self.constraints.eps_proj),
                'eps_clk': float(self.constraints.eps_clk) if self.constraints.eps_clk is not None else 0.0,
                'eps_H_norm': float(self.constraints.eps_H_norm),
                'eps_M_norm': float(self.constraints.eps_M_norm),
                'eps_proj_norm': float(self.constraints.eps_proj_norm),
                'spikes': {
                    'alpha_max_grad': float(np.max(np.abs(np.gradient(self.fields.alpha)))),
                    'K_max_grad': float(np.max(np.abs(np.gradient(self.fields.K_sym6))))
                }
            }
            self.emit_ledger_eval_receipt(final_ledgers)

            # Check gates
            accepted_gates, hard_fail_gates, penalty_gates, reasons, margins, corrections = self.check_gates_internal()
            if corrections:
                self.apply_corrections(corrections)

            accepted = accepted_gates
            hard_fail = hard_fail_gates
            penalty = penalty_gates

            if not hard_fail:
                # Log soft violations
                if penalty > 0:
                    logger.warning("UFE step accepted with soft violations", extra={
                        "extra_data": {
                            "penalty": penalty,
                            "reasons": reasons
                        }
                    })
                # Accept step
                # Increment step count for multi-rate
                self.step_count += 1
                logger.debug("UFE step accepted", extra={
                    "extra_data": {
                        "execution_time_ms": timer.elapsed_ms(),
                        "eps_H_final": float(self.constraints.eps_H),
                        "eps_M_final": float(self.constraints.eps_M),
                        "field_stats": {
                            "gamma_sym6": array_stats(self.fields.gamma_sym6, "gamma_sym6"),
                            "K_sym6": array_stats(self.fields.K_sym6, "K_sym6"),
                            "alpha": array_stats(self.fields.alpha, "alpha")
                        }
                    }
                })
                self.emit_step_receipt(accepted=True, stage_eps_H=stage_eps_H, corrections_applied=corrections)
                # Increment step count for multi-rate
                self.step_count += 1
                return True, None, self.current_dt, None, stage_eps_H  # accepted, state unchanged, dt_used, no reason, stage_eps_H
            else:
                # Reject step: rollback to initial state
                self.fields.gamma_sym6 = u0_gamma
                self.fields.K_sym6 = u0_K
                self.fields.phi = u0_phi
                self.fields.gamma_tilde_sym6 = u0_gamma_tilde
                self.fields.A_sym6 = u0_A
                self.fields.Z = u0_Z
                self.fields.Z_i = u0_Z_i
                # Recompute geometry for rolled-back state
                self.geometry.compute_christoffels()
                self.geometry.compute_ricci()
                self.geometry.compute_scalar_curvature()

                rejection_reason = f"Gates failed: {', '.join(reasons)}"
                self.emit_step_receipt(accepted=False, rejection_reason=rejection_reason, stage_eps_H=stage_eps_H, corrections_applied=corrections)

                # Suggest smaller dt for retry
                dt_new = max(self.current_dt / 2.0, 1e-10)

                logger.warning("UFE step rejected - rolling back", extra={
                    "extra_data": {
                        "rejection_reason": rejection_reason,
                        "dt_new": dt_new,
                        "execution_time_ms": timer.elapsed_ms(),
                        "eps_H_final": float(self.constraints.eps_H),
                        "eps_M_final": float(self.constraints.eps_M)
                    }
                })
                # Note: step_count not incremented on rejection
                return False, None, dt_new, rejection_reason, stage_eps_H  # rejected, state rolled back, new dt, reason, stage_eps_H

    def step(self, X_n, t_n, dt_candidate, rails_policy, phaseloom_caps):
        """
        Implement StepperContract.step using step_ufe.
        This ensures consistency and removes code duplication.
        """
        # X_n is ignored as we operate on self.fields directly

        dt = dt_candidate
        rejection_reason = None

        # Attempt loop
        for attempt in range(self.max_attempts):
            # step_ufe performs one attempt
            # It handles: clock decision, RHS, update, gauge (at end), gates, receipts
            success, _, dt_new, reason, _ = self.step_ufe(dt, t_n)

            if success:
                return True, None, dt, None
            else:
                # Retry with new dt suggested by step_ufe
                dt = dt_new
                rejection_reason = reason

        return False, None, dt, rejection_reason

    def attempt_receipt(self, X_n, t_n, dt_attempted, attempt_number):
        """Emit attempt receipt for every attempt."""
        # Log residuals and dt
        eps_H = float(self.constraints.eps_H)
        eps_M = float(self.constraints.eps_M)
        logger.info("Attempt receipt", extra={
            "extra_data": {
                "attempt_number": attempt_number,
                "t_n": t_n,
                "dt_attempted": dt_attempted,
                "eps_H": eps_H,
                "eps_M": eps_M
            }
        })
        # Does not advance τ

    def step_receipt(self, X_next, t_next, dt_used):
        """Emit step receipt only on acceptance."""
        # Advance audit time τ
        if self.temporal_system:
            self.temporal_system.audit_time()
        logger.info("Step receipt: accepted", extra={
            "extra_data": {
                "t_next": t_next,
                "dt_used": dt_used,
                "tau": self.temporal_system.tau if self.temporal_system else None
            }
        })

    def compute_rhs(self, t=0.0, slow_update=True):
        """Compute full ADM RHS B with spatial derivatives."""
        rhs_timer = Timer("compute_rhs")
        with rhs_timer:
            # Check tensor layout compliance before RHS computation
            self.check_tensor_layout_compliance()

            if self.rhs_func:
                # Use Hadamard VM via rhs_func
                fields_dict = {
                    'gamma_sym6': self.fields.gamma_sym6,
                    'K_sym6': self.fields.K_sym6,
                    'alpha': self.fields.alpha,
                    'beta': self.fields.beta,
                    'phi': self.fields.phi,
                    'gamma_tilde_sym6': self.fields.gamma_tilde_sym6,
                    'A_sym6': self.fields.A_sym6,
                    'Gamma_tilde': self.fields.Gamma_tilde,
                    'Z': self.fields.Z,
                    'Z_i': self.fields.Z_i,
                    'dx': self.fields.dx, 'dy': self.fields.dy, 'dz': self.fields.dz
                }
                rhs_result = self.rhs_func(fields_dict, lambda_val=self.lambda_val, sources_enabled=self.sources_func is not None)
                # Map to self.rhs_*
                self.rhs_gamma_sym6[:] = rhs_result['rhs_gamma_sym6']
                self.rhs_K_sym6[:] = rhs_result['rhs_K_sym6']
                self.rhs_phi[:] = rhs_result['rhs_phi']
                self.rhs_gamma_tilde_sym6[:] = rhs_result['rhs_gamma_tilde_sym6']
                self.rhs_A_sym6[:] = rhs_result['rhs_A_sym6']
                self.rhs_Gamma_tilde[:] = rhs_result['rhs_Gamma_tilde']
                self.rhs_Z[:] = rhs_result['rhs_Z']
                self.rhs_Z_i[:] = rhs_result['rhs_Z_i']
                return

        if self.aeonic_mode:
            rhs_gamma_sym6 = self.rhs_gamma_sym6
            rhs_K_sym6 = self.rhs_K_sym6
            rhs_phi = self.rhs_phi
            rhs_gamma_tilde_sym6 = self.rhs_gamma_tilde_sym6
            rhs_A_sym6 = self.rhs_A_sym6
            rhs_Gamma_tilde = self.rhs_Gamma_tilde
            rhs_Z = self.rhs_Z
            rhs_Z_i = self.rhs_Z_i
            S_gamma_tilde_sym6 = self.S_gamma_tilde_sym6
            S_A_sym6 = self.S_A_sym6
            S_phi = self.S_phi
            S_Gamma_tilde = self.S_Gamma_tilde
            S_Z = self.S_Z
            S_Z_i = self.S_Z_i
            gamma_inv_scratch = self.gamma_inv_scratch
            K_trace_scratch = self.K_trace_scratch
            alpha_expanded_scratch = self.alpha_expanded_scratch
            alpha_expanded_33_scratch = self.alpha_expanded_33_scratch
            lie_gamma_scratch = self.lie_gamma_scratch
            DD_alpha_scratch = self.DD_alpha_scratch
            DD_alpha_sym6_scratch = self.DD_alpha_sym6_scratch
            ricci_sym6_scratch = self.ricci_sym6_scratch
            K_full_scratch = self.K_full_scratch
            gamma_inv_full_scratch = self.gamma_inv_full_scratch
            K_contracted_full_scratch = self.K_contracted_full_scratch
            K_contracted_sym6_scratch = self.K_contracted_sym6_scratch
            lie_K_scratch = self.lie_K_scratch
            lie_gamma_tilde_scratch = self.lie_gamma_tilde_scratch
            psi_minus4_scratch = self.psi_minus4_scratch
            psi_minus4_expanded_scratch = self.psi_minus4_expanded_scratch
            ricci_tf_sym6_scratch = self.ricci_tf_sym6_scratch
            rhs_A_temp_scratch = self.rhs_A_temp_scratch
            lie_A_scratch = self.lie_A_scratch
        else:
            rhs_gamma_sym6 = np.zeros_like(self.fields.gamma_sym6)
            rhs_K_sym6 = np.zeros_like(self.fields.K_sym6)
            rhs_phi = np.zeros_like(self.fields.phi)
            rhs_gamma_tilde_sym6 = np.zeros_like(self.fields.gamma_tilde_sym6)
            rhs_A_sym6 = np.zeros_like(self.fields.A_sym6)
            rhs_Gamma_tilde = np.zeros_like(self.fields.Gamma_tilde)
            rhs_Z = np.zeros_like(self.fields.Z)
            rhs_Z_i = np.zeros_like(self.fields.Z_i)
            S_gamma_tilde_sym6 = np.zeros_like(self.fields.gamma_tilde_sym6)
            S_A_sym6 = np.zeros_like(self.fields.A_sym6)
            S_phi = np.zeros_like(self.fields.phi)
            S_Gamma_tilde = np.zeros_like(self.fields.Gamma_tilde)
            S_Z = np.zeros_like(self.fields.Z)
            S_Z_i = np.zeros_like(self.fields.Z_i)
            gamma_inv_scratch = np.zeros_like(self.fields.gamma_sym6)
            K_trace_scratch = np.zeros((self.fields.Nx, self.fields.Ny, self.fields.Nz))
            alpha_expanded_scratch = np.zeros((self.fields.Nx, self.fields.Ny, self.fields.Nz, 6))
            alpha_expanded_33_scratch = np.zeros((self.fields.Nx, self.fields.Ny, self.fields.Nz, 3, 3))
            lie_gamma_scratch = np.zeros_like(self.fields.gamma_sym6)
            DD_alpha_scratch = np.zeros((self.fields.Nx, self.fields.Ny, self.fields.Nz, 3, 3))
            DD_alpha_sym6_scratch = np.zeros_like(self.fields.gamma_sym6)
            ricci_sym6_scratch = np.zeros_like(self.fields.gamma_sym6)
            K_full_scratch = np.zeros((self.fields.Nx, self.fields.Ny, self.fields.Nz, 3, 3))
            gamma_inv_full_scratch = np.zeros((self.fields.Nx, self.fields.Ny, self.fields.Nz, 3, 3))
            K_contracted_full_scratch = np.zeros((self.fields.Nx, self.fields.Ny, self.fields.Nz, 3, 3))
            K_contracted_sym6_scratch = np.zeros_like(self.fields.gamma_sym6)
            lie_K_scratch = np.zeros_like(self.fields.gamma_sym6)
            lie_gamma_tilde_scratch = np.zeros_like(self.fields.gamma_sym6)
            psi_minus4_scratch = np.zeros((self.fields.Nx, self.fields.Ny, self.fields.Nz))
            psi_minus4_expanded_scratch = np.zeros((self.fields.Nx, self.fields.Ny, self.fields.Nz, 6))
            ricci_tf_sym6_scratch = np.zeros_like(self.fields.gamma_sym6)
            rhs_A_temp_scratch = np.zeros_like(self.fields.gamma_sym6)
            lie_A_scratch = np.zeros_like(self.fields.gamma_sym6)

        # Set sources if func provided
        if self.sources_func is not None:
            sources = self.sources_func(t)
            S_gamma_tilde_sym6[:] = sources['S_gamma_tilde_sym6']
            S_A_sym6[:] = sources['S_A_sym6']
            S_phi[:] = sources['S_phi']
            S_Gamma_tilde[:] = sources['S_Gamma_tilde']
            S_Z[:] = sources['S_Z']
            S_Z_i[:] = sources['S_Z_i']

        # Ensure geometry is computed
        if not hasattr(self.geometry, 'ricci') or self.geometry.ricci is None:
            self.geometry.compute_ricci()

        # Compute K = gamma^{ij} K_ij
        gamma_inv_scratch[:] = inv_sym6(self.fields.gamma_sym6)
        K_trace_scratch[:] = trace_sym6(self.fields.K_sym6, gamma_inv_scratch)

        alpha_expanded_scratch[:] = self.fields.alpha[..., np.newaxis]  # (Nx,Ny,Nz,1) -> (Nx,Ny,Nz,6)
        alpha_expanded_33_scratch[:] = self.fields.alpha[..., np.newaxis, np.newaxis]  # (Nx,Ny,Nz,1,1) -> (Nx,Ny,Nz,3,3)

        # ADM ∂t gamma_ij = -2 α K_ij + L_β γ_ij
        lie_gamma_scratch[:] = self.geometry.lie_derivative_gamma(self.fields.gamma_sym6, self.fields.beta)
        rhs_gamma_sym6[:] = -2.0 * alpha_expanded_scratch * self.fields.K_sym6 + lie_gamma_scratch

        # ADM ∂t K_ij = -D_i D_j α + α R_ij - 2 α K_ik γ^{kl} K_lj + α K K_ij + L_β K_ij
        # D_i D_j α
        DD_alpha_scratch[:] = self.geometry.second_covariant_derivative_scalar(self.fields.alpha)
        DD_alpha_sym6_scratch[:] = mat33_to_sym6(DD_alpha_scratch)

        # R_ij in sym6 form
        ricci_sym6_scratch[:] = mat33_to_sym6(self.geometry.ricci)

        # K_ik γ^{kl} K_lj = (K γ^{-1} K)_ij
        K_full_scratch[:] = sym6_to_mat33(self.fields.K_sym6)
        gamma_inv_full_scratch[:] = sym6_to_mat33(gamma_inv_scratch)

        # K^{kl} = γ^{ki} γ^{lj} K_ij, but for contraction K_ik γ^{kl} K_lj
        K_contracted_full_scratch[:] = np.einsum('...ij,...jk,...kl->...il', K_full_scratch, gamma_inv_full_scratch, K_full_scratch)

        K_contracted_sym6_scratch[:] = mat33_to_sym6(K_contracted_full_scratch)

        lie_K_scratch[:] = self.geometry.lie_derivative_K(self.fields.K_sym6, self.fields.beta)

        # Lambda term: + 2 \alpha \Lambda \gamma_{ij}
        lambda_term = 2.0 * alpha_expanded_scratch * self.fields.Lambda * self.fields.gamma_sym6

        rhs_K_sym6[:] = (-DD_alpha_sym6_scratch +
                              alpha_expanded_scratch * ricci_sym6_scratch +
                              -2.0 * alpha_expanded_scratch * K_contracted_sym6_scratch +
                              alpha_expanded_scratch * K_trace_scratch[..., np.newaxis] * self.fields.K_sym6 +
                              lambda_term +
                              lie_K_scratch)

        # BSSN ∂_0 φ = - (α/6) K + (1/6) ∂_k β^k
        # ∂_t φ = ∂_0 φ + β^k ∂_k φ
        if slow_update:
            div_beta = np.gradient(self.fields.beta[..., 0], self.fields.dx, axis=0) + \
                       np.gradient(self.fields.beta[..., 1], self.fields.dy, axis=1) + \
                       np.gradient(self.fields.beta[..., 2], self.fields.dz, axis=2)
            rhs_phi[:] = - (self.fields.alpha / 6.0) * K_trace_scratch + (1.0 / 6.0) * div_beta
            # Add advection term β^k ∂_k φ
            grad_phi = np.array([np.gradient(self.fields.phi, self.fields.dx, axis=0),
                                 np.gradient(self.fields.phi, self.fields.dy, axis=1),
                                 np.gradient(self.fields.phi, self.fields.dz, axis=2)])
            advection_phi = np.sum(self.fields.beta * grad_phi.transpose(1,2,3,0), axis=-1)
            rhs_phi[:] += advection_phi
            rhs_phi += S_phi
        else:
            rhs_phi[:] = 0.0

        # Full BSSN Gamma_tilde evolution: ∂_t \tilde Γ^i = A + B + C + D


        rhs_Gamma_tilde = _compute_gamma_tilde_rhs_jit(self.fields.Nx, self.fields.Ny, self.fields.Nz,
                                                        self.fields.alpha, self.fields.beta, self.fields.phi,
                                                        self.fields.Gamma_tilde, self.fields.A_sym6,
                                                        self.fields.gamma_tilde_sym6, self.fields.dx, self.fields.dy, self.fields.dz,
                                                        K_trace_scratch)

        # BSSN ∂t γ̃_ij = -2 α A_ij + L_β γ̃_ij
        lie_gamma_tilde_scratch[:] = self.geometry.lie_derivative_gamma(self.fields.gamma_tilde_sym6, self.fields.beta)
        rhs_gamma_tilde_sym6[:] = -2.0 * alpha_expanded_scratch * self.fields.A_sym6 + lie_gamma_tilde_scratch

        # BSSN ∂_0 A_ij = e^{-4φ} [-D_i D_j α + α R_ij]^TF + α (K A_ij - 2 A_il A^l_j) + 2 A_k(i ∂_j) β^k - (2/3) A_ij ∂_k β^k
        # Simplified version, ignoring matter for now
        psi_minus4_scratch[:] = np.exp(-4 * self.fields.phi)
        psi_minus4_expanded_scratch[:] = psi_minus4_scratch[..., np.newaxis]
        ricci_tf_sym6_scratch[:] = ricci_sym6_scratch - (1/3) * self.fields.gamma_sym6 * (self.geometry.R[..., np.newaxis] if hasattr(self.geometry, 'R') else 0)  # Approximate

        rhs_A_temp_scratch[:] = psi_minus4_expanded_scratch * alpha_expanded_scratch * ricci_tf_sym6_scratch + alpha_expanded_scratch * K_trace_scratch[..., np.newaxis] * self.fields.A_sym6

        # Add shift terms for A_ij
        # A_full = sym6_to_mat33(self.fields.A_sym6)
        # shift_term_A = np.zeros((self.fields.Nx, self.fields.Ny, self.fields.Nz, 3, 3))
        # for i in range(3):
        #     for j in range(3):
        #         for k in range(3):
        #             shift_term_A[..., i, j] += A_full[..., k, i] * grad_beta[..., k, j] + A_full[..., k, j] * grad_beta[..., k, i]
        # gauge_term_A = - (2/3.0) * A_full * div_beta[..., np.newaxis, np.newaxis]
        # rhs_A_temp_scratch += mat33_to_sym6(shift_term_A + gauge_term_A)

        lie_A_scratch[:] = self.geometry.lie_derivative_K(self.fields.A_sym6, self.fields.beta)
        rhs_A_sym6[:] = rhs_A_temp_scratch + lie_A_scratch

        # RHS for constraint damping
        # For Z: ∂t Z = - kappa alpha H
        if slow_update:
            self.constraints.compute_hamiltonian()
            kappa = 1.0
            rhs_Z[:] = -kappa * self.fields.alpha * self.constraints.H
            # For Z_i: ∂t Z_i = - kappa alpha M_i
            self.constraints.compute_momentum()
            rhs_Z_i[:] = -kappa * self.fields.alpha[:, :, :, np.newaxis] * self.constraints.M
            rhs_Z += S_Z
            rhs_Z_i += S_Z_i
        else:
            rhs_Z[:] = 0.0
            rhs_Z_i[:] = 0.0

        # Add sources
        rhs_gamma_tilde_sym6 += S_gamma_tilde_sym6
        rhs_A_sym6 += S_A_sym6
        rhs_Gamma_tilde += S_Gamma_tilde

        # Add LoC augmentation
        if self.lambda_val > 0:
            K_LoC_scaled = self.loc_operator.get_K_LoC_for_rhs()
            rhs_gamma_sym6 += K_LoC_scaled['gamma_sym6']
            rhs_K_sym6 += K_LoC_scaled['K_sym6']
            rhs_gamma_tilde_sym6 += K_LoC_scaled['gamma_tilde_sym6']
            rhs_A_sym6 += K_LoC_scaled['A_sym6']
            rhs_Gamma_tilde += K_LoC_scaled['Gamma_tilde']
            if slow_update:
                rhs_phi += K_LoC_scaled['phi']
                rhs_Z += K_LoC_scaled['Z']
                rhs_Z_i += K_LoC_scaled['Z_i']

        logger.info("compute_rhs timing", extra={
            "extra_data": {
                "compute_rhs_ms": rhs_timer.elapsed_ms()
            }
        })

        if not self.aeonic_mode:
            self.rhs_gamma_sym6 = rhs_gamma_sym6
            self.rhs_K_sym6 = rhs_K_sym6
            self.rhs_phi = rhs_phi
            self.rhs_gamma_tilde_sym6 = rhs_gamma_tilde_sym6
            self.rhs_A_sym6 = rhs_A_sym6
            self.rhs_Gamma_tilde = rhs_Gamma_tilde
            self.rhs_Z = rhs_Z
            self.rhs_Z_i = rhs_Z_i

    def apply_projection(self):
        """Apply constraint projection (K_proj). Placeholder for projection operators."""
        # TODO: Implement projection to enforce constraints
        pass

    def apply_boundary_conditions(self):
        """Apply boundary conditions (K_bc). Placeholder."""
        # TODO: Implement boundary condition enforcement
        pass

    def compute_dominance(self):
        """Compute dominance D_lambda = |lambda| |K| / (|B| + eps)."""
        # Estimate |B| as norm of baseline RHS components
        B_norm = (np.linalg.norm(self.rhs_gamma_sym6) + np.linalg.norm(self.rhs_K_sym6) +
                  np.linalg.norm(self.rhs_phi) + np.linalg.norm(self.rhs_gamma_tilde_sym6) +
                  np.linalg.norm(self.rhs_A_sym6) + np.linalg.norm(self.rhs_Z) + np.linalg.norm(self.rhs_Z_i))
        # Estimate |K| as coherence contribution, rough: lambda * B_norm
        K_norm = abs(self.lambda_val) * B_norm
        eps = 1e-10
        D_lambda = abs(self.lambda_val) * K_norm / (B_norm + eps)
        return D_lambda

    def apply_damping(self):
        """Apply constraint damping: reduce constraint violations."""
        if not self.damping_enabled:
            logger.debug("Damping disabled")
            return
        # Simple damping: decay the extrinsic curvature to reduce H
        # Note: For high-frequency gauge pulses, this is insufficient (eps_H ~ 2.5e-3). Full Z4 needed.
        decay_factor = np.exp(-self.lambda_val)  # Use lambda_val as damping rate
        old_max_K = np.max(np.abs(self.fields.K_sym6))
        self.fields.K_sym6 *= decay_factor
        new_max_K = np.max(np.abs(self.fields.K_sym6))
        logger.debug("Applied constraint damping", extra={
            "extra_data": {
                "lambda_val": self.lambda_val,
                "decay_factor": decay_factor,
                "K_max_before": old_max_K,
                "K_max_after": new_max_K
            }
        })

    def apply_corrections(self, corrections):
        """Apply bounded corrective actions for warn level violations."""
        if 'reduce_dt' in corrections:
            if hasattr(self, 'current_dt') and self.current_dt > 0:
                old_dt = self.current_dt
                self.current_dt = max(1e-6, 0.8 * self.current_dt)
                logger.info("Corrective action: dt reduction", extra={
                    "extra_data": {
                        "action": "dt_reduction",
                        "before": old_dt,
                        "after": self.current_dt
                    }
                })
        if 'increase_kappa_budget' in corrections:
            old_kappa_H = self.loc_operator.kappa_H
            old_kappa_M = self.loc_operator.kappa_M
            self.loc_operator.kappa_H = min(1.0, self.loc_operator.kappa_H + 0.1)
            self.loc_operator.kappa_M = min(1.0, self.loc_operator.kappa_M + 0.1)
            logger.info("Corrective action: kappa increase", extra={
                "extra_data": {
                    "action": "kappa_increase",
                    "kappa_H_before": old_kappa_H,
                    "kappa_H_after": self.loc_operator.kappa_H,
                    "kappa_M_before": old_kappa_M,
                    "kappa_M_after": self.loc_operator.kappa_M
                }
            })
        if 'increase_projection_freq' in corrections:
            old_lambda = self.lambda_val
            self.lambda_val = min(10.0, self.lambda_val * 2.0)
            logger.info("Corrective action: projection frequency boost", extra={
                "extra_data": {
                    "action": "projection_freq_boost",
                    "lambda_before": old_lambda,
                    "lambda_after": self.lambda_val
                }
            })
