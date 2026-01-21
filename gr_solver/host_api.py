import numpy as np
import hashlib
import pickle
from typing import Dict, Any
import logging
logger = logging.getLogger('gr_solver.host_api')

class GRHostAPI:
    """
    Thin host interface for Noetica/NSC integration with PhaseLoom GR/NR solver.
    Wraps the orchestrator without touching internal physics logic.
    """

    def __init__(self, fields, geometry, constraints, gauge, stepper, orchestrator):
        self.fields = fields
        self.geometry = geometry
        self.constraints = constraints
        self.gauge = gauge
        self.stepper = stepper
        self.orchestrator = orchestrator
        self.state_snapshot = None
        self.accepted = False

    def get_state_hash(self) -> str:
        """Return SHA256 hash of canonical state serialization."""
        # Canonical serialization: flatten arrays in order
        state_data = {
            'gamma': self.fields.gamma_sym6.tobytes(),
            'K': self.fields.K_sym6.tobytes(),
            'alpha': self.fields.alpha.tobytes(),
            'beta': self.fields.beta.tobytes(),
            'phi': self.fields.phi.tobytes(),
            't': self.orchestrator.t,
            'step': self.orchestrator.step
        }
        canonical_bytes = pickle.dumps(state_data, protocol=pickle.HIGHEST_PROTOCOL)
        return hashlib.sha256(canonical_bytes).hexdigest()

    def snapshot(self) -> bytes:
        """Capture current state for rollback."""
        state_data = {
            'gamma': self.fields.gamma_sym6.copy(),
            'K': self.fields.K_sym6.copy(),
            'alpha': self.fields.alpha.copy(),
            'beta': self.fields.beta.copy(),
            'phi': self.fields.phi.copy(),
            'gamma_tilde_sym6': self.fields.gamma_tilde_sym6.copy(),
            'A_sym6': self.fields.A_sym6.copy(),
            'Gamma_tilde': self.fields.Gamma_tilde.copy(),
            'Z': self.fields.Z.copy(),
            'Z_i': self.fields.Z_i.copy(),
            't': self.orchestrator.t,
            'step': self.orchestrator.step,
            'eps_H_prev': self.orchestrator.eps_H_prev,
            'eps_M_prev': self.orchestrator.eps_M_prev,
            'm_det_prev': self.orchestrator.m_det_prev,
            'dt_prev': self.orchestrator.dt_prev,
            'prev_K': self.orchestrator.prev_K.copy() if self.orchestrator.prev_K is not None else None,
            'prev_gamma': self.orchestrator.prev_gamma.copy() if self.orchestrator.prev_gamma is not None else None
        }
        self.state_snapshot = pickle.dumps(state_data, protocol=pickle.HIGHEST_PROTOCOL)
        return self.state_snapshot

    def restore(self, snapshot: bytes) -> None:
        """Restore state from snapshot."""
        state_data = pickle.loads(snapshot)
        self.fields.gamma_sym6[:] = state_data['gamma']
        self.fields.K_sym6[:] = state_data['K']
        self.fields.alpha[:] = state_data['alpha']
        self.fields.beta[:] = state_data['beta']
        self.fields.phi[:] = state_data['phi']
        self.fields.gamma_tilde_sym6[:] = state_data['gamma_tilde_sym6']
        self.fields.A_sym6[:] = state_data['A_sym6']
        self.fields.Gamma_tilde[:] = state_data['Gamma_tilde']
        self.fields.Z[:] = state_data['Z']
        self.fields.Z_i[:] = state_data['Z_i']
        self.orchestrator.t = state_data['t']
        self.orchestrator.step = state_data['step']
        self.orchestrator.eps_H_prev = state_data['eps_H_prev']
        self.orchestrator.eps_M_prev = state_data['eps_M_prev']
        self.orchestrator.m_det_prev = state_data['m_det_prev']
        self.orchestrator.dt_prev = state_data['dt_prev']
        self.orchestrator.prev_K = state_data['prev_K']
        self.orchestrator.prev_gamma = state_data['prev_gamma']
        self.accepted = False
        logger.debug("State restored from snapshot")

    def step(self, dt: float, stage: int) -> None:
        """Advance one solver stage."""
        if stage == 0:
            # First stage: UFE step
            self.stepper.step_ufe(dt, self.orchestrator.t)
        elif stage == 1:
            # Gauge evolution
            self.gauge.evolve_lapse(dt)
            self.gauge.evolve_shift(dt)
        else:
            raise ValueError(f"Unknown stage: {stage}")
        self.accepted = False

    def compute_constraints(self) -> Dict[str, Any]:
        """Compute and return constraint residuals and coherence indicator."""
        # Recompute geometry if needed
        self.geometry.compute_christoffels()
        self.geometry.compute_ricci()
        self.geometry.compute_scalar_curvature()

        self.constraints.compute_hamiltonian()
        self.constraints.compute_momentum()
        self.constraints.compute_residuals()

        eps_H = float(self.constraints.eps_H)
        eps_M = float(self.constraints.eps_M)
        # Use max R as coherence indicator
        R = float(np.max(self.geometry.R) if hasattr(self.geometry, 'R') and self.geometry.R is not None else 0.0)

        return {
            'eps_H': eps_H,
            'eps_M': eps_M,
            'R': R
        }

    def energy_metrics(self) -> Dict[str, Any]:
        """Return energy/hamiltonian metrics."""
        # H is eps_H, dH is step drift (change in H over dt)
        H = float(self.constraints.eps_H) if hasattr(self.constraints, 'eps_H') else 0.0
        dH = 0.0  # Placeholder: would compute (H - prev_H) / dt
        if self.orchestrator.eps_H_prev is not None and self.orchestrator.dt_prev is not None and self.orchestrator.dt_prev > 0:
            dH = (H - self.orchestrator.eps_H_prev) / self.orchestrator.dt_prev

        return {
            'H': H,
            'dH': dH
        }

    def apply_gauge(self, dt: float) -> None:
        """Apply gauge enforcement."""
        self.gauge.evolve_lapse(dt)
        self.gauge.evolve_shift(dt)

    def apply_dissipation(self, level: int) -> None:
        """Apply dissipation at specified level (≥j tail control)."""
        # Increase Kreiss-Oliger dissipation for higher levels
        # Assumes stepper has dissipation control
        if hasattr(self.stepper, 'dissipation_level'):
            self.stepper.dissipation_level = max(self.stepper.dissipation_level, level)
        else:
            # Fallback: apply to gauge or fields
            # Simple numerical dissipation: apply slight exponential decay to damp high frequencies
            decay_factor = 1.0 - level * 0.001
            self.fields.gamma_sym6 *= decay_factor
            self.fields.K_sym6 *= decay_factor
            self.fields.alpha *= decay_factor
            self.fields.beta *= decay_factor
            self.fields.phi *= decay_factor
        logger.debug(f"Dissipation applied at level {level}")

    def accept_step(self) -> None:
        """Commit the current step."""
        self.accepted = True
        self.orchestrator.accepted_step_count += 1
        # Update prev values
        self.orchestrator.eps_H_prev = self.constraints.eps_H
        self.orchestrator.eps_M_prev = self.constraints.eps_M
        self.orchestrator.dt_prev = getattr(self.orchestrator, 'last_dt', None)
        self.orchestrator.prev_K = self.fields.K_sym6[..., 0].copy()
        self.orchestrator.prev_gamma = self.fields.gamma_sym6[..., 0].copy()
        logger.debug("Step accepted")

    def reject_step(self) -> None:
        """Reject the current step (rollback handled separately)."""
        self.accepted = False
        self.orchestrator.rollback_count += 1
        logger.debug("Step rejected")