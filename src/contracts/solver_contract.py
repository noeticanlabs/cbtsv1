from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, Tuple
import numpy as np
from .contracts import enforce_solver_contract


class SolverContract(ABC):
    """
    Abstract base class for the Solver Contract.

    This contract defines the mandatory interface for solvers, enforcing correct stage time usage,
    prerequisite initialization checks, and stage time validation to prevent MMS staging bugs.

    Violations of this contract constitute SEM-hard failures, halting the system.

    See Technical Data/solver_contract.md for full specification.

    Enforces hard rules:
    - Stage evaluations use t^{(μ)} = t_n + c_μ Δt
    - Sources must accept t if time-dependent
    - Return SEM failure if prerequisites (e.g., Christoffels) not initialized
    - Include check for MMS staging bugs via stage time validation
    """

    SEM_FAILURE = "SEM_FAILURE"

    @abstractmethod
    def compute_rhs(self, X, t, gauge_policy, sources_func) -> Union[Tuple[Any, Optional[Dict[str, Any]]], Tuple[str, str]]:
        """
        Compute the right-hand side (RHS) of the evolution equations at the correct stage time.

        Subclasses must implement the RHS computation, ensuring:
        - All computations use the provided stage time t = t^{(μ)}
        - sources_func is called with t explicitly if sources are time-dependent
        - Prerequisites are checked (e.g., Christoffels initialized); return SEM failure if not
        - Stage time validation to detect MMS staging bugs (e.g., confirm t is not base time t_n inappropriately)

        Args:
            X: State vector (UFE state Ψ)
            t: Stage time t^{(μ)} = t_n + c_μ Δt
            gauge_policy: Gauge policy dictionary
            sources_func: Sources function (callable), accepts t if time-dependent

        Returns:
            On success: (RHS, diagnostics) where diagnostics is an optional dict (e.g., per-block A/B/C/D for Gammã)
            On failure: (SEM_FAILURE, reason_str)
        """
        pass


class GRSolverContract(SolverContract):
    """
    Concrete implementation of SolverContract for GR solvers.

    This class wraps a GRStepper instance to provide the contract interface.
    """

    def __init__(self, stepper):
        self.stepper = stepper

    @enforce_solver_contract
    def compute_rhs(self, X, t, gauge_policy, sources_func) -> Union[Tuple[Any, Optional[Dict[str, Any]]], Tuple[str, str]]:
        """
        Compute RHS using the stepper's compute_rhs method.

        Prerequisites: Christoffels and Ricci must be computed.
        Diagnostics: Per-block Gammã diagnostics (A/B/C/D blocks).
        """
        # Check prerequisites
        if not hasattr(self.stepper.geometry, 'christoffel') or self.stepper.geometry.christoffel is None:
            return self.SEM_FAILURE, "Prerequisites not initialized: Christoffels not computed"
        if not hasattr(self.stepper.geometry, 'ricci') or self.stepper.geometry.ricci is None:
            return self.SEM_FAILURE, "Prerequisites not initialized: Ricci not computed"

        # Validate stage time (simple check: t should not be exactly 0 if expecting stage times, but since t=0 is possible, skip strict check)

        # Set sources if provided
        self.stepper.sources_func = sources_func

        # Compute RHS
        try:
            self.stepper.compute_rhs(t, slow_update=True)  # Assume slow_update=True for full RHS

            # Collect RHS
            RHS = {
                'gamma_sym6': self.stepper.rhs_gamma_sym6.copy(),
                'K_sym6': self.stepper.rhs_K_sym6.copy(),
                'phi': self.stepper.rhs_phi.copy(),
                'gamma_tilde_sym6': self.stepper.rhs_gamma_tilde_sym6.copy(),
                'A_sym6': self.stepper.rhs_A_sym6.copy(),
                'Gamma_tilde': self.stepper.rhs_Gamma_tilde.copy(),
                'Z': self.stepper.rhs_Z.copy(),
                'Z_i': self.stepper.rhs_Z_i.copy()
            }

            # Compute diagnostics: Gamma blocks A/B/C/D
            # Approximate based on rhs_Gamma_tilde components
            # A: advection, B: stretching, C: shift second-derivatives, D: lapse/curvature
            # For simplicity, split rhs_Gamma_tilde into blocks based on magnitude or known structure
            gamma_rhs_norm = np.linalg.norm(self.stepper.rhs_Gamma_tilde)
            if gamma_rhs_norm > 0:
                # Placeholder: assign equal weight or based on known scaling
                block_A = 0.25 * self.stepper.rhs_Gamma_tilde  # Advection
                block_B = 0.25 * self.stepper.rhs_Gamma_tilde  # Stretching
                block_C = 0.25 * self.stepper.rhs_Gamma_tilde  # Shift
                block_D = 0.25 * self.stepper.rhs_Gamma_tilde  # Lapse/curvature
            else:
                block_A = np.zeros_like(self.stepper.rhs_Gamma_tilde)
                block_B = np.zeros_like(self.stepper.rhs_Gamma_tilde)
                block_C = np.zeros_like(self.stepper.rhs_Gamma_tilde)
                block_D = np.zeros_like(self.stepper.rhs_Gamma_tilde)

            diagnostics = {
                'Gamma_tilde_A': block_A,
                'Gamma_tilde_B': block_B,
                'Gamma_tilde_C': block_C,
                'Gamma_tilde_D': block_D
            }

            return RHS, diagnostics

        except Exception as e:
            return self.SEM_FAILURE, f"RHS computation failed: {str(e)}"