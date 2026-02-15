# gr_stepper.py
# =============================================================================
# GR Evolution Stepper with RK4 Time Integration and Multi-Rate Clock Selection
# =============================================================================
# 
# This module implements the core time-stepping logic for GR evolution using the
# BSSN (Baumgarte-Shapiro-Shibata-Nakamura) formulation. It provides the GRStepper
# class which orchestrates the complete evolution cycle including:
# 
# 1. **RK4 Integration**: Fourth-order Runge-Kutta time integration for all ADM/BSSN 
#    fields (gamma_ij, K_ij, phi, gamma_tilde_ij, A_ij, Gamma_tilde^i, Z, Z_i)
# 
# 2. **Multi-Rate Clock Selection**: Adaptive timestep selection based on multiple
#    clock sources (CFL, gauge, coherence, resolution constraints)
# 
# 3. **Constraint Monitoring**: Real-time Hamiltonian and momentum constraint 
#    evaluation with gate checking for step acceptance
# 
# 4. **LoC Coherence Control**: Integration with the Limit of Coherence (LoC) 
#    operator for damping constraint violations
# 
# 5. **Gauge Evolution**: Lapse (alpha) and shift (beta) evolution after physics
# 
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
import logging
from typing import Tuple, Dict, Any
try:
    from numba import jit
except ImportError:
    jit = lambda f=None, **kwargs: f if f else (lambda g: g)
from .logging_config import Timer, array_stats
from .gr_core_fields import inv_sym6, trace_sym6, sym6_to_mat33, mat33_to_sym6
from src.contracts.stepper_contract import StepperContract
from .gr_loc import GRLoC
from .gr_rhs import GRRhs
from .gr_ledger import GRLedger
from .gr_scheduler import GRScheduler
from .gr_gates import GateChecker
from .gr_receipts import compute_debt_from_residuals
from .hpc_kernels import fused_evolution_kernel
from .theorem_validators import TheoremValidator
from .gate_policy import GatePolicy
from .hard_invariants import HardInvariantChecker

logger = logging.getLogger('gr_solver.stepper')


class RetryCounter:
    """Enforces retry bounds per Theorem Lemma 2 (Bounded Work Inflation).
    
    Lemma 2 constraint: attempts ≤ (1 + N_retry) where N_retry is the max number
    of retries allowed per accepted step.
    
    This class maintains a counter and enforces hard failure when the limit is exceeded.
    """
    
    def __init__(self, max_retries: int = 3):
        """Initialize retry counter.
        
        Args:
            max_retries: N_retry from Lemma 2 (maximum number of retry attempts)
                        Defaults to 3
        """
        self.max_retries = max_retries
        self.attempt = 0
        self.max_attempts = 1 + max_retries
    
    def increment(self):
        """Increment attempt counter and check for hard failure.
        
        Raises:
            RuntimeError: If attempts exceed max_attempts limit
        """
        self.attempt += 1
        if self.attempt > self.max_attempts:
            raise RuntimeError(
                f"Retry limit exceeded: {self.attempt} > {self.max_attempts} "
                f"(max_retries={self.max_retries})"
            )
    
    def reset(self):
        """Reset counter for next step."""
        self.attempt = 0
    
    def can_retry(self) -> bool:
        """Check if another retry attempt is allowed.
        
        Returns:
            bool: True if attempt < max_attempts, False otherwise
        """
        return self.attempt < self.max_attempts


class GRStepper(StepperContract):
    def __init__(self, fields, geometry, constraints, gauge, memory_contract=None, phaseloom=None, aeonic_mode=True, max_attempts=20, dt_floor=1e-8, temporal_system=None, analysis_mode=False):
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
        self.aeonic_mode = aeonic_mode
        self.analysis_mode = analysis_mode
        self.loc_operator = GRLoC(fields, geometry, constraints, lambda_val=self.lambda_val, kappa_H=1.0, kappa_M=1.0)
        self.temporal_system = temporal_system
        
        self.rhs_computer = GRRhs(fields, geometry, constraints, self.loc_operator, self.lambda_val, self.sources_func, aeonic_mode)

        self.ledger = GRLedger()
        self.scheduler = GRScheduler(fields, Lambda=fields.Lambda)
        self.gatekeeper = GateChecker(constraints, analysis_mode=analysis_mode)

        # Initialize theorem validator with config from GatePolicy
        gate_policy = GatePolicy()
        self.policy = gate_policy  # Store policy for retry_policy access
        tv_config = gate_policy.theorem_validation
        if tv_config.get('enabled', True):
            self.theorem_validator = TheoremValidator(
                gamma=tv_config.get('gamma', 0.8),
                b=tv_config.get('b', 1e-4),
                enable_halt_on_violation=tv_config.get('halt_on_violation', False)
            )
        else:
            self.theorem_validator = None
        
        # Initialize hard invariant checker per Theorem Lemma 1
        hi_config = gate_policy.hard_invariants
        self.invariant_checker = HardInvariantChecker(
            tolerance=hi_config.get('tolerance', 1e-14)
        )
        
        # Track debt across steps for contraction validation
        self.last_accepted_debt = 0.0

        # Multi-rate stepping parameters
        self.slow_fields = ['phi', 'Z', 'Z_i']
        self.slow_rate = 5  # Update slow fields every 5 steps
        self.step_count = 0

        # Pre-allocated buffers for fused evolution kernel (64-byte aligned)
        self._allocate_fused_buffers()

    @property
    def receipts(self):
        return self.ledger.receipts

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
            self.ledger.emit_layout_violation_receipt(
                getattr(self, 'current_step', 0),
                getattr(self, 'current_t', 0.0),
                getattr(self, 'current_dt', 0.0),
                self.fields,
                violations
            )

            raise ValueError(f"Tensor layout violations detected: {violations}")

        return layout_ok

    def _allocate_fused_buffers(self):
        """Allocate aligned buffers for fused evolution kernel."""
        Nx, Ny, Nz = self.fields.Nx, self.fields.Ny, self.fields.Nz
        
        # Allocate with 64-byte alignment for SIMD efficiency
        alignment = 64
        
        # Gamma buffer (6 components)
        self._fused_gamma = np.zeros((Nx, Ny, Nz, 6), dtype=np.float64)
        if self._fused_gamma.ctypes.data % alignment != 0:
            # Reallocate with proper alignment
            self._fused_gamma = np.ascontiguousarray(self._fused_gamma)
        
        # K buffer (6 components)
        self._fused_K = np.zeros((Nx, Ny, Nz, 6), dtype=np.float64)
        if self._fused_K.ctypes.data % alignment != 0:
            self._fused_K = np.ascontiguousarray(self._fused_K)
        
        # Alpha buffer (scalar)
        self._fused_alpha = np.zeros((Nx, Ny, Nz), dtype=np.float64)
        if self._fused_alpha.ctypes.data % alignment != 0:
            self._fused_alpha = np.ascontiguousarray(self._fused_alpha)
        
        # Beta buffer (3 components)
        self._fused_beta = np.zeros((Nx, Ny, Nz, 3), dtype=np.float64)
        if self._fused_beta.ctypes.data % alignment != 0:
            self._fused_beta = np.ascontiguousarray(self._fused_beta)
        
        logger.debug("Fused evolution buffers allocated", extra={
            "extra_data": {
                "gamma_aligned": self._fused_gamma.ctypes.data % alignment == 0,
                "K_aligned": self._fused_K.ctypes.data % alignment == 0,
                "alpha_aligned": self._fused_alpha.ctypes.data % alignment == 0,
                "beta_aligned": self._fused_beta.ctypes.data % alignment == 0
            }
        })

    def _check_invariants_pre(self):
        """Check pre-update invariants: fields must be finite and bounded."""
        violations = []
        
        # Check gamma_sym6
        if not np.isfinite(self.fields.gamma_sym6).all():
            violations.append("gamma_sym6 contains NaN/Inf")
        
        # Check K_sym6
        if not np.isfinite(self.fields.K_sym6).all():
            violations.append("K_sym6 contains NaN/Inf")
        
        # Check alpha (lapse, must be positive)
        if not np.isfinite(self.fields.alpha).all():
            violations.append("alpha contains NaN/Inf")
        if np.any(self.fields.alpha <= 0):
            violations.append("alpha contains non-positive values")
        
        # Check beta (shift)
        if not np.isfinite(self.fields.beta).all():
            violations.append("beta contains NaN/Inf")
        
        # Check magnitude bounds for metric (should be close to flat space)
        gamma_mean = np.mean(self.fields.gamma_sym6)
        if abs(gamma_mean - 1.0) > 0.5:  # Allow 50% deviation from flat space
            violations.append(f"gamma_sym6 mean {gamma_mean} far from 1.0")
        
        return violations

    def _check_invariants_post(self, norm_dgamma, norm_dK):
        """Check post-update invariants: update norms finite, fields valid."""
        violations = []
        
        # Check update norms are finite
        if not np.isfinite(norm_dgamma):
            violations.append(f"norm_dgamma is not finite: {norm_dgamma}")
        if not np.isfinite(norm_dK):
            violations.append(f"norm_dK is not finite: {norm_dK}")
        
        # Check update norms are reasonable (not NaN/Inf exploding)
        if norm_dgamma > 1e10:
            violations.append(f"norm_dgamma too large: {norm_dgamma}")
        if norm_dK > 1e10:
            violations.append(f"norm_dK too large: {norm_dK}")
        
        # Check resulting fields are finite
        if not np.isfinite(self.fields.gamma_sym6).all():
            violations.append("updated gamma_sym6 contains NaN/Inf")
        if not np.isfinite(self.fields.K_sym6).all():
            violations.append("updated K_sym6 contains NaN/Inf")
        if not np.isfinite(self.fields.alpha).all():
            violations.append("updated alpha contains NaN/Inf")
        if not np.isfinite(self.fields.beta).all():
            violations.append("updated beta contains NaN/Inf")
        
        return violations


    def check_gates_internal(self, rails_policy=None):
        """Check step gates. Return (accepted, hard_fail, penalty, reasons, margins, corrections, debt_decomposition)."""
        return self.gatekeeper.check_gates_internal(rails_policy)

    def check_gates(self, rails_policy) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Implement StepperContract.check_gates.
        Delegates to gatekeeper.
        """
        return self.gatekeeper.check_gates(rails_policy)

    def step_ufe(self, dt, t=0.0, rails_policy=None):
        """
        Perform a single RK4 step for the Unified Field Evolution (UFE) system.
        
        This is the core time-stepping method implementing 4th-order Runge-Kutta 
        integration for all BSSN/ADM variables. The method:
        
        1. **Clock Decision**: Queries the scheduler to determine if dt should be 
           reduced based on multi-clock constraints (CFL, gauge, coherence, resolution)
        
        2. **Pre-Update Ledger**: Records initial constraint residuals (Hamiltonian H,
           Momentum M, projection Z, Z_i) as reference for normalized residuals
        
        3. **RK4 Integration**:
           - Stage 1: Compute RHS at t using current state u_0
           - Stage 2: u_1 = u_0 + (dt/2)*k1, compute RHS at t + dt/2
           - Stage 3: u_2 = u_0 + (dt/2)*k2, compute RHS at t + dt/2
           - Stage 4: u_3 = u_0 + dt*k3, compute RHS at t + dt
           - Combined: u_{n+1} = u_0 + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        
        4. **Clock Error Tracking**: Computes epsilon_clk = max_Linf(Psi_used - Psi_auth)
           at each stage to quantify discretization coherence error
        
        5. **Gauge Evolution**: Evolves lapse (alpha) and shift (beta) after physics
           update using appropriate gauge conditions
        
        6. **Post-Update Processing**:
           - Constraint damping via LoC operator
           - Algebraic constraint enforcement (det(gamma_tilde)=1, tr(A)=0)
           - Geometry recomputation
        
        7. **Gate Checking**: Validates step against constraint thresholds and
           coherence gates for acceptance/rejection
        
        8. **Rollback on Failure**: If hard gates fail, state rolls back to u_0
           and reduced dt is suggested
        
        Args:
            dt: Proposed timestep (may be reduced by clock constraints)
            t: Current physical time
            rails_policy: Optional policy for rail constraints
            
        Returns:
            Tuple: (accepted, state_unchanged, dt_used, rejection_reason, stage_eps_H)
                   - accepted: bool indicating step acceptance
                   - state_unchanged: None (state modified in-place)
                   - dt_used: actual timestep applied
                   - rejection_reason: str if rejected, None if accepted
                   - stage_eps_H: dict of constraint evolution through stages
        """
        with Timer("step_ufe") as timer:
            self.dt_applied = dt  # Set dt_applied as source of truth
            self.current_dt = dt
            self.current_t = t
            self.current_step = getattr(self, 'current_step', 0) + 1

            # Clock decision
            clocks, dt_used = self.scheduler.compute_clocks(dt, self.lambda_val)
            if dt_used < dt:
                logger.warning(f"Clock constraint reduced dt from {dt} to {dt_used}")
                self.current_dt = dt_used
            self.ledger.emit_clock_decision_receipt(self.current_step, self.current_t, self.current_dt, self.fields, clocks)

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



            
            # ========================================================================
            # STAGE 1: k1 = RHS(t, u0)
            # ========================================================================
            # Compute initial RHS at time t using state u0
            self.check_tensor_layout_compliance()
            self.rhs_computer.compute_rhs(t, slow_update)
            rhs_norms = {
                'gamma': np.linalg.norm(self.rhs_computer.rhs_gamma_sym6),
                'K': np.linalg.norm(self.rhs_computer.rhs_K_sym6),
                'phi': np.linalg.norm(self.rhs_computer.rhs_phi),
                'gamma_tilde': np.linalg.norm(self.rhs_computer.rhs_gamma_tilde_sym6),
                'A': np.linalg.norm(self.rhs_computer.rhs_A_sym6),
                'Z': np.linalg.norm(self.rhs_computer.rhs_Z),
                'Z_i': np.linalg.norm(self.rhs_computer.rhs_Z_i)
            }
            self.ledger.emit_stage_rhs_receipt(self.current_step, self.current_t, self.current_dt, 1, t, rhs_norms, self.fields, self.rhs_computer.compute_rhs)
            k1_gamma = self.rhs_computer.rhs_gamma_sym6.copy()
            k1_K = self.rhs_computer.rhs_K_sym6.copy()
            k1_phi = self.rhs_computer.rhs_phi.copy()
            k1_gamma_tilde = self.rhs_computer.rhs_gamma_tilde_sym6.copy()
            k1_A = self.rhs_computer.rhs_A_sym6.copy()
            k1_Z = self.rhs_computer.rhs_Z.copy()
            k1_Z_i = self.rhs_computer.rhs_Z_i.copy()

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

            # ========================================================================
            # STAGE 2: u2 = u0 + (dt/2)*k1, compute RHS at t + dt/2
            # ========================================================================
            # Update fields to intermediate state for stage 2 evaluation
            self.fields.gamma_sym6 = u0_gamma + (dt/2) * k1_gamma
            self.fields.K_sym6 = u0_K + (dt/2) * k1_K
            self.fields.phi = u0_phi + (dt/2) * k1_phi
            self.fields.gamma_tilde_sym6 = u0_gamma_tilde + (dt/2) * k1_gamma_tilde
            self.fields.A_sym6 = u0_A + (dt/2) * k1_A
            self.fields.Z = u0_Z + (dt/2) * k1_Z
            self.fields.Z_i = u0_Z_i + (dt/2) * k1_Z_i
            
            # Recompute geometry-dependent quantities for the intermediate state
            # This is required because Christoffel symbols and Ricci tensor depend
            # on the metric, which changes at each RK stage
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
            self.check_tensor_layout_compliance()
            self.rhs_computer.compute_rhs(t + dt/2, slow_update)
            rhs_norms = {
                'gamma': np.linalg.norm(self.rhs_computer.rhs_gamma_sym6),
                'K': np.linalg.norm(self.rhs_computer.rhs_K_sym6),
                'phi': np.linalg.norm(self.rhs_computer.rhs_phi),
                'gamma_tilde': np.linalg.norm(self.rhs_computer.rhs_gamma_tilde_sym6),
                'A': np.linalg.norm(self.rhs_computer.rhs_A_sym6),
                'Z': np.linalg.norm(self.rhs_computer.rhs_Z),
                'Z_i': np.linalg.norm(self.rhs_computer.rhs_Z_i)
            }
            self.ledger.emit_stage_rhs_receipt(self.current_step, self.current_t, self.current_dt, 2, t + dt/2, rhs_norms, self.fields, self.rhs_computer.compute_rhs)
            k2_gamma = self.rhs_computer.rhs_gamma_sym6.copy()
            k2_K = self.rhs_computer.rhs_K_sym6.copy()
            k2_phi = self.rhs_computer.rhs_phi.copy()
            k2_gamma_tilde = self.rhs_computer.rhs_gamma_tilde_sym6.copy()
            k2_A = self.rhs_computer.rhs_A_sym6.copy()
            k2_Z = self.rhs_computer.rhs_Z.copy()
            k2_Z_i = self.rhs_computer.rhs_Z_i.copy()

            # ========================================================================
            # STAGE 3: u3 = u0 + (dt/2)*k2, compute RHS at t + dt/2
            # ========================================================================
            # Second midpoint evaluation - uses k2 instead of k1 for update
            self.fields.gamma_sym6 = u0_gamma + (dt/2) * k2_gamma
            self.fields.K_sym6 = u0_K + (dt/2) * k2_K
            self.fields.phi = u0_phi + (dt/2) * k2_phi
            self.fields.gamma_tilde_sym6 = u0_gamma_tilde + (dt/2) * k2_gamma_tilde
            self.fields.A_sym6 = u0_A + (dt/2) * k2_A
            self.fields.Z = u0_Z + (dt/2) * k2_Z
            self.fields.Z_i = u0_Z_i + (dt/2) * k2_Z_i
            
            # Geometry recomputation for stage 3 state
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
            self.check_tensor_layout_compliance()
            self.rhs_computer.compute_rhs(t + dt/2, slow_update)
            rhs_norms = {
                'gamma': np.linalg.norm(self.rhs_computer.rhs_gamma_sym6),
                'K': np.linalg.norm(self.rhs_computer.rhs_K_sym6),
                'phi': np.linalg.norm(self.rhs_computer.rhs_phi),
                'gamma_tilde': np.linalg.norm(self.rhs_computer.rhs_gamma_tilde_sym6),
                'A': np.linalg.norm(self.rhs_computer.rhs_A_sym6),
                'Z': np.linalg.norm(self.rhs_computer.rhs_Z),
                'Z_i': np.linalg.norm(self.rhs_computer.rhs_Z_i)
            }
            self.ledger.emit_stage_rhs_receipt(self.current_step, self.current_t, self.current_dt, 3, t + dt/2, rhs_norms, self.fields, self.rhs_computer.compute_rhs)
            k3_gamma = self.rhs_computer.rhs_gamma_sym6.copy()
            k3_K = self.rhs_computer.rhs_K_sym6.copy()
            k3_phi = self.rhs_computer.rhs_phi.copy()
            k3_gamma_tilde = self.rhs_computer.rhs_gamma_tilde_sym6.copy()
            k3_A = self.rhs_computer.rhs_A_sym6.copy()
            k3_Z = self.rhs_computer.rhs_Z.copy()
            k3_Z_i = self.rhs_computer.rhs_Z_i.copy()

            # ========================================================================
            # STAGE 4: u4 = u0 + dt*k3, compute RHS at t + dt
            # ========================================================================
            # Full-step evaluation - uses k3 for forward jump
            self.fields.gamma_sym6 = u0_gamma + dt * k3_gamma
            self.fields.K_sym6 = u0_K + dt * k3_K
            self.fields.phi = u0_phi + dt * k3_phi
            self.fields.gamma_tilde_sym6 = u0_gamma_tilde + dt * k3_gamma_tilde
            self.fields.A_sym6 = u0_A + dt * k3_A
            self.fields.Z = u0_Z + dt * k3_Z
            self.fields.Z_i = u0_Z_i + dt * k3_Z_i
            
            # Geometry recomputation for stage 4 state
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
            self.check_tensor_layout_compliance()
            self.rhs_computer.compute_rhs(t + dt, slow_update)
            rhs_norms = {
                'gamma': np.linalg.norm(self.rhs_computer.rhs_gamma_sym6),
                'K': np.linalg.norm(self.rhs_computer.rhs_K_sym6),
                'phi': np.linalg.norm(self.rhs_computer.rhs_phi),
                'gamma_tilde': np.linalg.norm(self.rhs_computer.rhs_gamma_tilde_sym6),
                'A': np.linalg.norm(self.rhs_computer.rhs_A_sym6),
                'Z': np.linalg.norm(self.rhs_computer.rhs_Z),
                'Z_i': np.linalg.norm(self.rhs_computer.rhs_Z_i)
            }
            self.ledger.emit_stage_rhs_receipt(self.current_step, self.current_t, self.current_dt, 4, t + dt, rhs_norms, self.fields, self.rhs_computer.compute_rhs)
            k4_gamma = self.rhs_computer.rhs_gamma_sym6.copy()
            k4_K = self.rhs_computer.rhs_K_sym6.copy()
            k4_phi = self.rhs_computer.rhs_phi.copy()
            k4_gamma_tilde = self.rhs_computer.rhs_gamma_tilde_sym6.copy()
            k4_A = self.rhs_computer.rhs_A_sym6.copy()
            k4_Z = self.rhs_computer.rhs_Z.copy()
            k4_Z_i = self.rhs_computer.rhs_Z_i.copy()

            # ========================================================================
            # RK4 COMBINATION: u_{n+1} = u_0 + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
            # ========================================================================
            # The RK4 coefficients (1/6, 2/6, 2/6, 1/6) ensure 4th-order accuracy.
            # This combines all four stage derivatives with appropriate weights.
            
            # Compute combined RHS for fused kernel (gamma, K, alpha, beta evolution)
            # Using RK4 coefficients: (k1 + 2*k2 + 2*k3 + k4) / 6
            rhs_gamma_combined = (k1_gamma + 2*k2_gamma + 2*k3_gamma + k4_gamma) / 6.0
            rhs_K_combined = (k1_K + 2*k2_K + 2*k3_K + k4_K) / 6.0
            
            # Gauge fields (alpha, beta) are evolved AFTER physics, so RHS is zero for now
            # The gauge.evolve_* methods modify alpha and beta in place
            rhs_alpha_combined = np.zeros_like(self.fields.alpha)
            rhs_beta_combined = np.zeros_like(self.fields.beta)
            
            # Pre-update invariants check
            pre_violations = self._check_invariants_pre()
            if pre_violations:
                logger.error("Pre-update invariants violated - rejecting step", extra={"extra_data": {"violations": pre_violations}})
                # Reject the step due to pre-update invariant violation
                self.fields.gamma_sym6 = u0_gamma
                self.fields.K_sym6 = u0_K
                rejection_reason = f"Pre-update invariants violated: {pre_violations}"
                self.ledger.emit_step_receipt(self.current_step, t, self.current_dt, self.fields,
                                             accepted=False, ledgers={}, gates={},
                                             rejection_reason=rejection_reason, stage_eps_H=None)
                return False, None, self.current_dt * 0.5, rejection_reason, stage_eps_H
            
            # Final update using fused evolution kernel
            norm_dgamma, norm_dK = fused_evolution_kernel(
                self.fields.gamma_sym6,
                self.fields.K_sym6,
                self.fields.alpha,
                self.fields.beta,
                rhs_gamma_combined,
                rhs_K_combined,
                rhs_alpha_combined,
                rhs_beta_combined,
                self.current_dt,
                self._fused_gamma,  # Use pre-allocated buffer
                self._fused_K,      # Use pre-allocated buffer
                self._fused_alpha,  # Use pre-allocated buffer
                self._fused_beta    # Use pre-allocated buffer
            )
            
            # Copy results back to fields (in-place modification)
            self.fields.gamma_sym6[...] = self._fused_gamma[...]
            self.fields.K_sym6[...] = self._fused_K[...]
            self.fields.alpha[...] = self._fused_alpha[...]
            self.fields.beta[...] = self._fused_beta[...]
            
            # Post-update invariants check
            post_violations = self._check_invariants_post(norm_dgamma, norm_dK)
            if post_violations:
                logger.error("Post-update invariants violated - rejecting step", extra={
                    "extra_data": {
                        "violations": post_violations,
                        "norm_dgamma": norm_dgamma,
                        "norm_dK": norm_dK
                    }
                })
                # Rollback and reject the step due to post-update invariant violation
                self.fields.gamma_sym6 = u0_gamma
                self.fields.K_sym6 = u0_K
                rejection_reason = f"Post-update invariants violated: {post_violations}"
                self.ledger.emit_step_receipt(self.current_step, t + dt, dt, self.fields,
                                             accepted=False, ledgers={}, gates={},
                                             rejection_reason=rejection_reason, stage_eps_H=None)
                return False, None, dt * 0.5, rejection_reason, stage_eps_H
            
            # Log update norms
            logger.debug("Fused evolution completed", extra={
                "extra_data": {
                    "norm_dgamma": norm_dgamma,
                    "norm_dK": norm_dK
                }
            })
            
            # Continue with slow fields using separate update (non-fused for now)
            self.fields.phi = u0_phi + (dt/6) * (k1_phi + 2*k2_phi + 2*k3_phi + k4_phi)
            self.fields.gamma_tilde_sym6 = u0_gamma_tilde + (dt/6) * (k1_gamma_tilde + 2*k2_gamma_tilde + 2*k3_gamma_tilde + k4_gamma_tilde)
            self.fields.A_sym6 = u0_A + (dt/6) * (k1_A + 2*k2_A + 2*k3_A + k4_A)
            self.fields.Z = u0_Z + (dt/6) * (k1_Z + 2*k2_Z + 2*k3_Z + k4_Z)
            self.fields.Z_i = u0_Z_i + (dt/6) * (k1_Z_i + 2*k2_Z_i + 2*k3_Z_i + k4_Z_i)

            # Recompute geometry for final state
            self.geometry.compute_christoffels()
            self.geometry.compute_ricci()
            self.geometry.compute_scalar_curvature()

            # ========================================================================
            # POST-PHYSICS PHASE
            # ========================================================================
            
            # Compute eps_H after physics update (RK)
            self.constraints.compute_residuals()
            stage_eps_H['eps_H_post_phys'] = float(self.constraints.eps_H)

            # ========================================================================
            # GAUGE EVOLUTION: Evolve lapse (alpha) and shift (beta)
            # ========================================================================
            # Gauge evolution is performed AFTER physics to ensure:
            # 1. Constraints are evaluated on the evolved spacetime geometry
            # 2. Gauge conditions depend on the final state, not intermediate stages
            # Common gauge conditions: 1+log slicing for alpha, Gamma-driver for beta
            self.gauge.evolve_lapse(dt)
            self.gauge.evolve_shift(dt)
            
            # Recompute geometry after gauge evolution (lapse/shift affect Christoffels)
            self.geometry.compute_christoffels()
            self.geometry.compute_ricci()
            self.geometry.compute_scalar_curvature()
            self.constraints.compute_residuals()
            stage_eps_H['eps_H_post_gauge'] = float(self.constraints.eps_H)

            # ========================================================================
            # CONSTRAINT DAMPING PHASE
            # ========================================================================
            # Apply constraint damping via LoC operator to reduce constraint violations.
            # The damping term K_damp = -kappa * gradient(constraint_energy) pushes
            # the system back toward the constraint surface.
            self.apply_damping()
            
            # ========================================================================
            # ALGEBRAIC CONSTRAINT ENFORCEMENT
            # ========================================================================
            # Re-enforce algebraic constraints after damping/filter:
            # 1. det(gamma_tilde) = 1 (conformal metric determinant)
            # 2. tr(A) = 0 (tracelessness of A_ij)
            self.geometry.enforce_det_gamma_tilde()
            self.geometry.enforce_traceless_A()
            
            # Recompute derived geometry after constraint enforcement
            self.geometry.compute_christoffels()
            self.geometry.compute_ricci()
            self.geometry.compute_scalar_curvature()
            self.constraints.compute_residuals()
            stage_eps_H['eps_H_post_cons'] = float(self.constraints.eps_H)
            stage_eps_H['eps_H_post_filter'] = float(self.constraints.eps_H)

            # Compute stage jumps Delta_epsilon_H
            stage_eps_H['Delta_eps_H_phys'] = stage_eps_H['eps_H_post_phys'] - stage_eps_H['eps_H_pre']
            stage_eps_H['Delta_eps_H_gauge'] = stage_eps_H['eps_H_post_gauge'] - stage_eps_H['eps_H_pre']
            stage_eps_H['Delta_eps_H_cons'] = stage_eps_H['eps_H_post_cons'] - stage_eps_H['eps_H_pre']
            stage_eps_H['Delta_eps_H_filter'] = stage_eps_H['eps_H_post_filter'] - stage_eps_H['eps_H_pre']

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
            self.ledger.emit_ledger_eval_receipt(self.current_step, self.current_t + self.current_dt, self.current_dt, self.fields, final_ledgers, debt_decomposition=None)

            # ========================================================================
            # GATE CHECKING PHASE
            # ========================================================================
            # Check all coherence gates G1-G4 for step acceptance:
            # G1: Constraint residuals within bounds (eps_H <= H_max, eps_M <= M_max)
            # G2: Clock coherence error within tolerance (eps_clk <= clk_max)
            # G3: Projection error within tolerance (eps_proj <= proj_max)
            # G4: Damage monotonicity (D <= D_prev + budget)
            #
            # Returns:
            #   accepted_gates: bool - all gates passed
            #   hard_fail_gates: bool - critical failure (rollback required)
            #   penalty_gates: bool - soft violations (warning but accept)
            #   reasons: list of violation descriptions
            #   margins: float - how close to thresholds
            #   corrections: dict - recommended parameter adjustments
            
            accepted_gates, hard_fail_gates, penalty_gates, reasons, margins, corrections, debt_decomposition = self.check_gates_internal(rails_policy)
            if corrections:
                self.apply_corrections(corrections)

            # ========================================================================
            # HARD INVARIANT VALIDATION (Theorem Lemma 1)
            # ========================================================================
            # Enforce hard invariants before accepting step:
            # If hard invariants hold in initial state and accepted states must
            # satisfy hard invariants, then every accepted state satisfies them.
            
            hi_config = self.policy.hard_invariants
            if hi_config.get('check_before_acceptance', True):
                logger.debug(
                    f"Checking hard invariants at step {self.current_step}",
                    extra={"extra_data": {"step": self.current_step, "t": self.current_t}}
                )
                
                is_valid, violations, margins_inv = self.invariant_checker.check_hard_invariants(self.fields)
                
                if not is_valid:
                    logger.warning(
                        f"Hard invariant violation at step {self.current_step}: {violations}",
                        extra={
                            "extra_data": {
                                "step": self.current_step,
                                "violations": violations,
                                "margins": margins_inv
                            }
                        }
                    )
                    # Reject step due to hard invariant violation
                    accepted_gates = False
                    hard_fail_gates = True
                    reasons.append(f"Hard invariant violation: {', '.join(violations)}")
                else:
                    logger.debug(
                        f"Hard invariants satisfied at step {self.current_step}",
                        extra={"extra_data": {"margins": margins_inv}}
                    )

            accepted = accepted_gates
            hard_fail = hard_fail_gates
            penalty = penalty_gates

            # Prepare final ledgers for step receipt
            ledgers_for_receipt = {
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
            gates_for_receipt = {'pass': accepted_gates, 'penalty': penalty_gates, 'reasons': reasons, 'margins': margins}

            if not hard_fail:
                # ========================================================================
                # STEP ACCEPTED
                # ========================================================================
                # Log soft violations if any
                if penalty > 0:
                    logger.warning("UFE step accepted with soft violations", extra={
                        "extra_data": {
                            "penalty": penalty,
                            "reasons": reasons
                        }
                    })
                
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
                
                # Emit step receipt for accepted step
                self.ledger.emit_step_receipt(self.current_step, self.current_t, self.current_dt, self.fields,
                                              accepted=True, ledgers=ledgers_for_receipt, gates=gates_for_receipt,
                                              stage_eps_H=stage_eps_H, corrections_applied=corrections, debt_decomposition=debt_decomposition)
                
                # Validate Lemma 3 (Debt Boundedness Under Contractive Repair)
                if self.theorem_validator is not None:
                    debt_after = debt_decomposition.get('total_debt', 0.0)
                    is_valid, margin, msg = self.theorem_validator.validate_contraction(
                        debt_before=self.last_accepted_debt,
                        debt_after=debt_after,
                        step_num=self.current_step
                    )
                    # Update debt tracking for next step
                    self.last_accepted_debt = debt_after
                
                # Increment step count for multi-rate stepping
                self.step_count += 1
                
                return True, None, self.current_dt, None, stage_eps_H
            else:
                # ========================================================================
                # STEP REJECTED - ROLLBACK
                # ========================================================================
                # Restore all fields to their initial state u_0.
                # This ensures the state remains physically valid for retry with smaller dt.
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
                
                # Emit step receipt for rejected step
                self.ledger.emit_step_receipt(self.current_step, self.current_t, self.current_dt, self.fields,
                                              accepted=False, ledgers=ledgers_for_receipt, gates=gates_for_receipt,
                                              rejection_reason=rejection_reason, stage_eps_H=stage_eps_H,
                                              corrections_applied=corrections, debt_decomposition=debt_decomposition)

                # Suggest smaller timestep for retry
                # Exponential backoff with floor at dt_floor (typically 1e-8)
                dt_new = max(self.current_dt / 2.0, 1e-8)

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
        Implement StepperContract.step using step_ufe with bounded retry enforcement.
        
        This method enforces Theorem Lemma 2 (Bounded Work Inflation) by limiting
        the number of retry attempts to max_retries, ensuring the total number of
        attempts does not exceed 1 + N_retry.
        
        Raises:
            RuntimeError: If retry limit is exceeded
        """
        # X_n is ignored as we operate on self.fields directly

        dt = dt_candidate
        rejection_reason = None
        
        # Create RetryCounter per Theorem Lemma 2
        max_retries = self.policy.retry_policy.get('max_retries', 3)
        retry_counter = RetryCounter(max_retries=max_retries)

        # Bounded retry loop with hard failure at limit
        while True:
            try:
                retry_counter.increment()
            except RuntimeError as e:
                # Hard fail when max attempts exceeded
                logger.error(
                    f"Step hard fail - retry limit exceeded: {e}",
                    extra={
                        "extra_data": {
                            "attempt": retry_counter.attempt,
                            "max_attempts": retry_counter.max_attempts,
                            "max_retries": retry_counter.max_retries,
                            "rejection_reason": rejection_reason
                        }
                    }
                )
                raise
            
            # Log retry attempt if not the first attempt
            if retry_counter.attempt > 1:
                logger.info(
                    f"Retry attempt {retry_counter.attempt}/{retry_counter.max_attempts}",
                    extra={
                        "extra_data": {
                            "attempt": retry_counter.attempt,
                            "max_attempts": retry_counter.max_attempts,
                            "dt": dt,
                            "previous_rejection": rejection_reason
                        }
                    }
                )
            
            # step_ufe performs one attempt
            # It handles: clock decision, RHS, update, gauge (at end), gates, receipts
            success, _, dt_new, reason, stage_eps_H = self.step_ufe(dt, t_n, rails_policy)

            # Prepare final ledgers for step receipt (copied from step_ufe for consistency)
            ledgers_for_receipt = {
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
            accepted_gates, hard_fail_gates, penalty_gates, reasons_gates, margins_gates, corrections_gates, debt_decomposition_gates = self.check_gates_internal(rails_policy)
            gates_for_receipt = {'pass': accepted_gates, 'penalty': penalty_gates, 'reasons': reasons_gates, 'margins': margins_gates}

            if success:
                # Step accepted - log completion with attempt count
                if retry_counter.attempt > 1:
                    logger.info(
                        f"Step accepted after {retry_counter.attempt} attempts",
                        extra={
                            "extra_data": {
                                "total_attempts": retry_counter.attempt,
                                "max_attempts": retry_counter.max_attempts,
                                "eps_H": float(self.constraints.eps_H)
                            }
                        }
                    )
                
                self.ledger.emit_step_receipt(self.current_step, t_n + dt, dt, self.fields, accepted=True, ledgers=ledgers_for_receipt, gates=gates_for_receipt, stage_eps_H=stage_eps_H, corrections_applied=corrections_gates, debt_decomposition=debt_decomposition_gates)
                return True, None, dt, None
            else:
                # Step rejected - check if we can retry
                rejection_reason = reason
                
                if not retry_counter.can_retry():
                    # No more retries allowed - hard fail
                    logger.error(
                        f"Step hard fail - all {retry_counter.max_attempts} attempts exhausted",
                        extra={
                            "extra_data": {
                                "total_attempts": retry_counter.attempt,
                                "max_attempts": retry_counter.max_attempts,
                                "max_retries": retry_counter.max_retries,
                                "final_rejection": rejection_reason
                            }
                        }
                    )
                    self.ledger.emit_step_receipt(self.current_step, t_n + dt, dt, self.fields, accepted=False, ledgers=ledgers_for_receipt, gates=gates_for_receipt, rejection_reason=rejection_reason, stage_eps_H=stage_eps_H, corrections_applied=corrections_gates, debt_decomposition=debt_decomposition_gates)
                    raise RuntimeError(
                        f"Step failed after {retry_counter.max_attempts} attempts - hard fail"
                    )
                
                # Prepare for retry with reduced dt
                self.ledger.emit_step_receipt(self.current_step, t_n + dt, dt, self.fields, accepted=False, ledgers=ledgers_for_receipt, gates=gates_for_receipt, rejection_reason=reason, stage_eps_H=stage_eps_H, corrections_applied=corrections_gates, debt_decomposition=debt_decomposition_gates)
                dt = dt_new  # Use new dt suggested by step_ufe

    def attempt_receipt(self, X_n, t_n, dt_candidate, rails_policy, phaseloom_caps) -> Tuple[bool, Any, float, Any]:
        """
        Attempt a step and return success status, new state (if successful), new dt (if rejected), and rejection reason.
        This method will call the 'step' method internally.
        """
        success, X_np1, dt_new, rejection_reason = self.step(X_n, t_n, dt_candidate, rails_policy, phaseloom_caps)
        return success, X_np1, dt_new, rejection_reason

    def step_receipt(self, current_step: int, t_end: float, dt: float, current_state: Any, accepted: bool, ledgers: dict, gates: dict, rejection_reason: str = None, stage_eps_H: dict = None, corrections_applied: dict = None):
        """
        Emit a step receipt using the internal ledger.
        This method proxies to self.ledger.emit_step_receipt.
        """
        self.ledger.emit_step_receipt(current_step, t_end, dt, current_state, accepted, ledgers, gates, rejection_reason, stage_eps_H, corrections_applied)


    def apply_damping(self):
        """Apply constraint damping: reduce constraint violations."""
        self.gatekeeper.apply_damping(self.lambda_val, self.damping_enabled)

    def apply_corrections(self, corrections):
        """Apply bounded corrective actions for warn level violations."""
        self.current_dt, self.lambda_val = self.gatekeeper.apply_corrections(corrections, self.current_dt, self.lambda_val)
        # Sync components
        self.rhs_computer.lambda_val = self.lambda_val
        self.loc_operator.lambda_val = self.lambda_val
    
    def get_theorem_validation_report(self) -> Dict[str, Any]:
        """Retrieve theorem validation report from the validator.
        
        Returns:
            Dictionary containing validation results:
                - 'num_violations': Count of contractions violated
                - 'violations': List of violation tuples (step, debt_before, debt_after, threshold)
                - 'status': Status message
                
            Returns None if theorem_validator is not enabled.
        """
        if self.theorem_validator is None:
            logger.info("Theorem validation disabled")
            return {'status': 'Theorem validation disabled'}
        
        return self.theorem_validator.get_violation_report()