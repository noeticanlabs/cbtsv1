# host_api.py
# =============================================================================
# GR Host API Shim (PhaseLoom ↔ GR Integration)
# =============================================================================
#
# This module provides the host interface layer between the Noetica/PhaseLoom
# orchestrator and the GR solver physics kernel. It exposes fine-grained control
# over evolution, constraints, state management, and step acceptance/rejection.
#
# **Interface Methods:**
#   - get_state_hash() -> str: SHA-256 hash of canonical state serialization
#   - snapshot() -> bytes: JSON-serialized state snapshot for rollback
#   - restore(snapshot: bytes) -> None: Restore state from snapshot
#   - step(dt: float, stage: int) -> None: Perform one solver stage
#   - compute_constraints() -> dict: Returns eps_H, eps_M, R
#   - energy_metrics() -> dict: Returns H, dH
#   - apply_gauge(dt: float) -> None: Evolve gauge fields
#   - apply_dissipation(level: int) -> None: Apply dissipation operators
#   - accept_step() -> None: Commit the current step
#   - reject_step() -> None: Signal step rejection for rollback
#
# **Integration:**
#   - Binds to existing GR solver components (fields, geometry, constraints, gauge, stepper)
#   - Compatible with PhaseLoom 27-thread orchestration
#   - Supports Aeonic memory and receipt generation
#
# Lexicon declarations per canon v1.2
LEXICON_IMPORTS = [
    "LoC axioma",
    "UFE_core",
    "GR_dyn",
    "CTL_time"
]
LEXICON_SYMBOLS = {
    "\\Psi": "UFE_state",
    "\\Hash": "GR_host_api.state_hash"
}

import numpy as np
import hashlib
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger('gr_solver.host_api')


class GRHostAPI:
    """
    Interface between the GR Solver (Physics Kernel) and the Noetica/PhaseLoom Pilot.
    
    This class provides a stable, minimal API surface for the orchestrator to:
    - Query solver state (hash, constraints, energy metrics)
    - Control evolution (step, gauge, dissipation)
    - Manage checkpoints (snapshot/restore for rollback)
    - Handle step acceptance/rejection for audit trails
    
    The API is designed to be "socket-compatible" - the orchestrator never touches
    solver internals directly, only through these well-defined methods.
    """
    
    def __init__(self, fields, geometry, constraints, gauge, stepper, orchestrator):
        """
        Initialize the Host API with references to GR solver components.
        
        Args:
            fields: GRCoreFields - The physical fields (gamma, K, alpha, beta, etc.)
            geometry: GRGeometry - Geometry objects (Christoffels, Ricci, curvature)
            constraints: GRConstraints - Constraint computation and monitoring
            gauge: GRGauge - Gauge evolution (lapse, shift)
            stepper: GRStepper - Time stepping and RK integration
            orchestrator: GRPhaseLoomOrchestrator - Orchestration state (t, step)
        """
        self.fields = fields
        self.geometry = geometry
        self.constraints = constraints
        self.gauge = gauge
        self.stepper = stepper
        self.orchestrator = orchestrator
        self._snapshot_buffer = None
        
        # Track energy for dH computation
        self._last_H: Optional[float] = None
        self._last_t: Optional[float] = None
        
        logger.info("GRHostAPI initialized", extra={
            "extra_data": {
                "grid_size": [fields.Nx, fields.Ny, fields.Nz],
                "component_refs": ["fields", "geometry", "constraints", "gauge", "stepper", "orchestrator"]
            }
        })
    
    def get_state_hash(self) -> str:
        """
        Compute SHA-256 hash of the canonical state serialization.
        
        This hash provides a deterministic fingerprint of the solver state
        for receipt chaining and state verification. The hash covers all
        core fields (gamma_sym6, K_sym6, alpha, beta) at the grid center
        where physical violations would be most apparent.
        
        Returns:
            Hex string (64 characters) representing the SHA-256 hash.
        """
        # Use central values for stable hash (avoids numerical noise at boundaries)
        cx = self.fields.Nx // 2
        cy = self.fields.Ny // 2
        cz = self.fields.Nz // 2
        
        # Hash central values of all core fields
        state_values = {
            'gamma_xx': float(self.fields.gamma_sym6[cx, cy, cz, 0]),
            'gamma_yy': float(self.fields.gamma_sym6[cx, cy, cz, 1]),
            'gamma_zz': float(self.fields.gamma_sym6[cx, cy, cz, 2]),
            'K_xx': float(self.fields.K_sym6[cx, cy, cz, 0]),
            'K_yy': float(self.fields.K_sym6[cx, cy, cz, 1]),
            'K_zz': float(self.fields.K_sym6[cx, cy, cz, 2]),
            'alpha': float(self.fields.alpha[cx, cy, cz]),
            'beta_x': float(self.fields.beta[cx, cy, cz, 0]),
            'beta_y': float(self.fields.beta[cx, cy, cz, 1]),
            'beta_z': float(self.fields.beta[cx, cy, cz, 2]),
            'phi': float(self.fields.phi[cx, cy, cz]),
            't': float(self.orchestrator.t),
            'step': int(self.orchestrator.step)
        }
        
        state_str = json.dumps(state_values, sort_keys=True)
        return hashlib.sha256(state_str.encode()).hexdigest()
    
    def snapshot(self) -> bytes:
        """
        Create a JSON-serialized snapshot of the current physical state.
        
        This snapshot captures all fields and orchestrator state needed for
        potential rollback. It is serialized as JSON bytes for portability.
        
        The snapshot includes:
        - All core BSSN fields (gamma_sym6, K_sym6, alpha, beta, phi, etc.)
        - Conformal fields (gamma_tilde, A, Gamma_tilde, Z, Z_i)
        - Orchestrator state (t, step)
        
        Returns:
            JSON-encoded bytes representing the complete state snapshot.
        """
        state = {
            'version': '1.0',
            'timestamp': self.orchestrator.t,
            'step': self.orchestrator.step,
            'fields': {
                'gamma_sym6': self.fields.gamma_sym6.tolist(),
                'K_sym6': self.fields.K_sym6.tolist(),
                'alpha': self.fields.alpha.tolist(),
                'beta': self.fields.beta.tolist(),
                'phi': self.fields.phi.tolist(),
                'gamma_tilde_sym6': self.fields.gamma_tilde_sym6.tolist(),
                'A_sym6': self.fields.A_sym6.tolist(),
                'Gamma_tilde': self.fields.Gamma_tilde.tolist(),
                'Z': self.fields.Z.tolist(),
                'Z_i': self.fields.Z_i.tolist()
            },
            'grid': {
                'Nx': self.fields.Nx,
                'Ny': self.fields.Ny,
                'Nz': self.fields.Nz,
                'dx': self.fields.dx,
                'dy': self.fields.dy,
                'dz': self.fields.dz,
                'Lambda': self.fields.Lambda
            }
        }
        
        snapshot_bytes = json.dumps(state, indent=2).encode('utf-8')
        self._snapshot_buffer = snapshot_bytes
        
        logger.debug("State snapshot created", extra={
            "extra_data": {
                "snapshot_size_bytes": len(snapshot_bytes),
                "t": self.orchestrator.t,
                "step": self.orchestrator.step
            }
        })
        
        return snapshot_bytes
    
    def restore(self, snapshot: bytes) -> None:
        """
        Restore physical state from a JSON-serialized snapshot.
        
        This method reverses the snapshot() operation, restoring all fields
        and orchestrator state to the checkpoint values. After restoration,
        derived geometry quantities are invalidated to force recomputation.
        
        Args:
            snapshot: JSON-encoded bytes from a previous snapshot() call.
        
        Raises:
            ValueError: If snapshot format is invalid or incompatible.
        """
        if self._snapshot_buffer is not None:
            # Verify snapshot matches if we have a buffer
            if snapshot != self._snapshot_buffer:
                logger.warning("Restoring from different snapshot than buffered")
        
        # Decode and parse snapshot
        state = json.loads(snapshot.decode('utf-8'))
        
        # Validate version
        if state.get('version') != '1.0':
            raise ValueError(f"Incompatible snapshot version: {state.get('version')}")
        
        # Restore fields from lists back to numpy arrays
        fields_data = state['fields']
        self.fields.gamma_sym6 = np.array(fields_data['gamma_sym6'], dtype=np.float64)
        self.fields.K_sym6 = np.array(fields_data['K_sym6'], dtype=np.float64)
        self.fields.alpha = np.array(fields_data['alpha'], dtype=np.float64)
        self.fields.beta = np.array(fields_data['beta'], dtype=np.float64)
        self.fields.phi = np.array(fields_data['phi'], dtype=np.float64)
        self.fields.gamma_tilde_sym6 = np.array(fields_data['gamma_tilde_sym6'], dtype=np.float64)
        self.fields.A_sym6 = np.array(fields_data['A_sym6'], dtype=np.float64)
        self.fields.Gamma_tilde = np.array(fields_data['Gamma_tilde'], dtype=np.float64)
        self.fields.Z = np.array(fields_data['Z'], dtype=np.float64)
        self.fields.Z_i = np.array(fields_data['Z_i'], dtype=np.float64)
        
        # Restore orchestrator state
        self.orchestrator.t = state['timestamp']
        self.orchestrator.step = state['step']
        
        # Invalidate derived geometry to force recomputation
        self._invalidate_geometry()
        
        # Reset energy tracking
        self._last_H = None
        self._last_t = None
        
        logger.info("State restored from snapshot", extra={
            "extra_data": {
                "restored_t": self.orchestrator.t,
                "restored_step": self.orchestrator.step
            }
        })
    
    def _invalidate_geometry(self) -> None:
        """Invalidate cached geometry quantities to force recomputation."""
        # Reset geometry caches
        self.geometry.christoffels = None
        self.geometry.Gamma = None
        self.geometry.ricci = None
        self.geometry.R = None
        self.geometry.R_scalar = None
        
        # Reset constraint caches
        self.constraints.H = None
        self.constraints.M = None
        self.constraints.eps_H = None
        self.constraints.eps_M = None
    
    def step(self, dt: float, stage: int) -> None:
        """
        Perform a single physics step (or RK stage) of size dt.
        
        This method delegates to the stepper's UFE (Unified Field Evolution)
        update. The orchestrator manages the time coordinate and stage tracking;
        this method only performs the physics update.
        
        Note: The orchestrator is responsible for advancing t and step counters.
        This method performs the in-place field update.
        
        Args:
            dt: Time step size for this evolution stage.
            stage: RK stage index (0-3 for RK4, or 4 for final combination).
        """
        t = self.orchestrator.t
        
        logger.debug("Executing step", extra={
            "extra_data": {
                "dt": dt,
                "stage": stage,
                "t": t
            }
        })
        
        # Delegate to stepper's UFE update
        self.stepper.step_ufe(dt, t)
    
    def compute_constraints(self) -> Dict[str, float]:
        """
        Compute and return constraint residuals.
        
        This method ensures geometry and constraints are up-to-date, then
        returns the key constraint monitoring quantities:
        
        - eps_H: L2 norm of Hamiltonian constraint violation
        - eps_M: L2 norm of momentum constraint violation
        - R: Maximum absolute scalar curvature (coherence indicator)
        
        Returns:
            Dictionary with keys 'eps_H', 'eps_M', 'R'.
        """
        # Ensure geometry is fresh
        self.geometry.compute_all()
        
        # Compute all constraints
        self.constraints.compute_all()
        
        # Get R (scalar curvature) for coherence indicator
        R_max = float(np.max(np.abs(self.geometry.R))) if self.geometry.R is not None else 0.0
        
        result = {
            'eps_H': float(self.constraints.eps_H),
            'eps_M': float(self.constraints.eps_M),
            'R': R_max
        }
        
        logger.debug("Constraints computed", extra={
            "extra_data": result
        })
        
        return result
    
    def energy_metrics(self) -> Dict[str, float]:
        """
        Return Hamiltonian and its time derivative proxy.
        
        Computes the Hamiltonian-like quantity from constraint fields and
        estimates the energy change dH from the previous call.
        
        Note: dH is an approximation based on constraint violation energy,
        not the true ADM Hamiltonian (which would require boundary integrals).
        
        Returns:
            Dictionary with keys:
            - 'H': Integrated Hamiltonian-like quantity (L2 norm of H)
            - 'dH': Change in H since last call (energy drift indicator)
        """
        # Ensure constraints are computed
        if not hasattr(self.constraints, 'H') or self.constraints.H is None:
            self.constraints.compute_all()
        
        # Compute integrated H (L2-like measure of constraint energy)
        H_grid = self.constraints.H
        vol_elem = self.fields.dx * self.fields.dy * self.fields.dz
        H_int = np.sum(H_grid**2) * vol_elem
        
        # Compute dH (energy drift since last call)
        if self._last_H is not None:
            dH = abs(H_int - self._last_H)
        else:
            dH = 0.0
        
        # Store for next call
        self._last_H = H_int
        self._last_t = self.orchestrator.t
        
        result = {
            'H': float(H_int),
            'dH': float(dH)
        }
        
        logger.debug("Energy metrics computed", extra={
            "extra_data": result
        })
        
        return result
    
    def apply_gauge(self, dt: float) -> None:
        """
        Apply gauge evolution for duration dt.
        
        Evolves the lapse function (alpha) and shift vector (beta) according
        to the configured gauge conditions (1+log slicing and gamma-driver).
        
        Args:
            dt: Time step for gauge evolution.
        """
        logger.debug("Applying gauge evolution", extra={
            "extra_data": {"dt": dt}
        })
        
        self.gauge.evolve_lapse(dt)
        self.gauge.evolve_shift(dt)
    
    def apply_dissipation(self, level: int) -> None:
        """
        Apply dissipation/filtering to current state.
        
        Applies Kreiss-Oliger dissipation or similar high-wavenumber
        filtering to suppress numerical noise at the grid resolution limit.
        
        Args:
            level: Dissipation strength level (0=off, 1=standard, 2=strong).
        """
        logger.debug("Applying dissipation", extra={
            "extra_data": {"level": level}
        })
        
        if hasattr(self.stepper, 'apply_kreiss_oliger'):
            self.stepper.apply_kreiss_oliger(level)
        else:
            logger.warning("apply_dissipation called but stepper lacks Kreiss-Oliger implementation")
    
    def accept_step(self) -> None:
        """
        Commit the current step and advance counters.
        
        This method is called after successful gate checking to confirm
        the step is accepted. It increments the step counter in the
        orchestrator.
        """
        self.orchestrator.step += 1
        
        logger.debug("Step accepted", extra={
            "extra_data": {
                "new_step": self.orchestrator.step,
                "t": self.orchestrator.t
            }
        })
    
    def reject_step(self) -> None:
        """
        Signal step rejection for rollback.
        
        This method is a no-op for the solver itself (rollback is handled
        by the orchestrator via restore()). It exists for API completeness
        and audit trail logging.
        """
        logger.info("Step rejected", extra={
            "extra_data": {
                "step": self.orchestrator.step,
                "t": self.orchestrator.t
            }
        })


def create_host_api(solver) -> GRHostAPI:
    """
    Factory function to create a GRHostAPI from a GRSolver instance.
    
    This convenience function extracts the required components from a
    configured solver instance and returns a ready-to-use Host API.
    
    Args:
        solver: GRSolver instance with fields, geometry, constraints, gauge, stepper, orchestrator.
    
    Returns:
        GRHostAPI instance bound to the solver components.
    """
    return GRHostAPI(
        fields=solver.fields,
        geometry=solver.geometry,
        constraints=solver.constraints,
        gauge=solver.gauge,
        stepper=solver.stepper,
        orchestrator=solver.orchestrator
    )
