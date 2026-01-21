from abc import ABC, abstractmethod
from typing import Tuple, Any, Dict, List


class PhaseLoomContract(ABC):
    """
    Abstract base class defining the PhaseLoom Contract for computing dt caps and action suggestions.

    Hard Rules Enforcement:
    - SEM hard failure forces immediate reject/abort: Implementations must check for SEM failures and raise appropriate exceptions or abort computation.
    - Thresholds are part of policy hash: Policy thresholds are immutable within a computation window; changes require versioning to avoid mid-window SEM inconsistencies.
    - Residuals normalized or unit-typed: Input residuals_r must be normalized or have unit types to ensure consistency; non-compliant inputs should be rejected.
    - Action suggestions per tier R0/R1/R2: Action suggestions must be organized by tier (R0, R1, R2) for prioritization.
    """

    @abstractmethod
    def compute_caps(
        self,
        residuals_r: Any,  # Normalized residuals or unit-typed
        stiffness_sigmas: Any,  # Stiffness measures
        policy_thresholds: Any  # Immutable policy thresholds
    ) -> Tuple[Any, str, Dict[str, List[str]]]:
        """
        Compute dt caps, determine dominant thread, and provide action suggestions.

        Args:
            residuals_r: Normalized residuals or unit-typed residuals.
            stiffness_sigmas: Measures of stiffness for threads.
            policy_thresholds: Immutable thresholds from policy hash.

        Returns:
            Tuple of (dt_caps, dominant_thread, action_suggestions)
            - dt_caps: Computed dt capacity limits (type depends on implementation, e.g., dict or list).
            - dominant_thread: String identifier of the dominant thread.
            - action_suggestions: Dict with keys 'R0', 'R1', 'R2' containing lists of suggested actions.
        """
        pass