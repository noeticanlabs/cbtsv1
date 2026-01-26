import numpy as np
import hashlib
import copy

class GRHostAPI:
    """
    Interface between the GR Solver (Physics Kernel) and the Noetica/PhaseLoom Pilot.
    Exposes fine-grained control over evolution, constraints, and state management.
    """
    def __init__(self, fields, geometry, constraints, gauge, stepper, orchestrator):
        self.fields = fields
        self.geometry = geometry
        self.constraints = constraints
        self.gauge = gauge
        self.stepper = stepper
        self.orchestrator = orchestrator
        self._snapshot_buffer = None

    def step(self, dt, stage=0):
        """
        Perform a single physics step (or stage) of size dt.
        """
        # Map to the stepper's UFE update.
        # Note: We pass the current time t, but do not auto-increment it here
        # to allow the host (pilot) to manage the time coordinate during retries.
        t = self.orchestrator.t
        self.stepper.step_ufe(dt, t)
        
    def apply_gauge(self, dt):
        """
        Apply gauge evolution for duration dt.
        """
        self.gauge.evolve_lapse(dt)
        self.gauge.evolve_shift(dt)

    def apply_dissipation(self, strength=1.0):
        """
        Apply dissipation/filtering to current state.
        """
        if hasattr(self.stepper, 'apply_kreiss_oliger'):
            self.stepper.apply_kreiss_oliger(strength)

    def compute_constraints(self):
        """
        Compute and return constraint residuals.
        """
        self.geometry.compute_all() # Ensure geometry is fresh
        self.constraints.compute_all()
        
        # Return dictionary matching receipt schema expectations
        return {
            'eps_H': float(self.constraints.eps_H),
            'eps_M': float(self.constraints.eps_M),
            'R': float(np.max(np.abs(self.geometry.R))) if self.geometry.R is not None else 0.0
        }

    def energy_metrics(self):
        """
        Return Hamiltonian and its time derivative proxy.
        """
        # Assumes constraints have been computed recently
        H_grid = self.constraints.H
        # Approximate L2 squared integral
        vol_elem = self.fields.dx * self.fields.dy * self.fields.dz
        H_int = np.sum(H_grid**2) * vol_elem
        
        return {
            'H': float(H_int),
            'dH': 0.0 # Placeholder: requires history to compute dH/dt
        }

    def snapshot(self):
        """
        Create a deep copy of the current physical state.
        """
        state = {
            'gamma': self.fields.gamma_sym6.copy(),
            'K': self.fields.K_sym6.copy(),
            'alpha': self.fields.alpha.copy(),
            'beta': self.fields.beta.copy(),
            'phi': self.fields.phi.copy(),
            't': self.orchestrator.t,
            'step': self.orchestrator.step
        }
        return state

    def restore(self, snapshot):
        """
        Restore physical state from snapshot.
        """
        np.copyto(self.fields.gamma_sym6, snapshot['gamma'])
        np.copyto(self.fields.K_sym6, snapshot['K'])
        np.copyto(self.fields.alpha, snapshot['alpha'])
        np.copyto(self.fields.beta, snapshot['beta'])
        np.copyto(self.fields.phi, snapshot['phi'])
        self.orchestrator.t = snapshot['t']
        self.orchestrator.step = snapshot['step']
        
        # Invalidate derived geometry to force recomputation
        if hasattr(self.geometry, 'invalidate'):
            self.geometry.invalidate()
        else:
            self.geometry.compute_all()

    def accept_step(self):
        """Commit the current step (advance counters)."""
        self.orchestrator.step += 1

    def reject_step(self):
        """Explicit rejection signal."""
        pass

    def get_state_hash(self):
        """Compute a hash of the current state for receipts."""
        # Hash central value of gamma_xx for quick verification
        center = self.fields.Nx // 2
        val = self.fields.gamma_sym6[center, center, center, 0]
        return hashlib.sha256(f"{val:.16f}".encode()).hexdigest()[:16]