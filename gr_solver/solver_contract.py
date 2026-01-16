from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, Tuple


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