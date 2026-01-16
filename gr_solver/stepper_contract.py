from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple


class StepperContract(ABC):
    """
    Abstract base class for the Stepper Contract.

    This contract defines the mandatory interface for steppers, enforcing coherence and rollback safety.
    Violations are SEM-hard failures.

    See Technical Data/stepper_contract.md for full specification.

    Enforces distinguishing dt-dependent vs state-dependent violations,
    max_attempts, dt_floor, and prohibits continuing without changes on rejection.
    """

    def __init__(self, max_attempts: int, dt_floor: float):
        self.max_attempts = max_attempts
        self.dt_floor = dt_floor
        self.check_violations()

    @abstractmethod
    def step(self, X_n, t_n, dt_candidate, rails_policy, phaseloom_caps) -> Tuple[bool, Any, Optional[float], Optional[str]]:
        """
        Perform a step attempt, potentially with multiple retries.

        Subclasses must implement the stepper court logic: attempt loop with emission of attempt_receipt on each attempt,
        violation checks distinguishing dt-dependent (e.g., CFL) vs state-dependent (e.g., pre-step constraints violated),
        dt adjustment for dt-dependent (halve dt), enforcement of max_attempts and dt_floor,
        emission of step_receipt only on acceptance, and return of rolled-back state and new dt on rejection.

        Prohibits continuation without changes: must rollback to X_n and adjust dt on rejection.

        Returns: (accepted: bool, X_next_or_rolled: Any, dt_used_or_new: float | None, rejection_reason: str | None)
        """
        pass

    @abstractmethod
    def attempt_receipt(self, X_n, t_n, dt_attempted, attempt_number):
        """
        Emit attempt receipt for every attempt (accepted or rejected).

        Logs residuals, dt used, violation checks. Does not advance audit time τ.
        """
        pass

    @abstractmethod
    def step_receipt(self, X_next, t_next, dt_used):
        """
        Emit step receipt only on acceptance.

        Advances audit time τ and contributes to immutable history.
        """
        pass

    def check_violations(self):
        """
        Checks for SEM-hard violations in configuration.

        Raises ValueError if any violation is detected.
        """
        if self.max_attempts <= 0:
            raise ValueError("SEM-hard violation: max_attempts must be positive")
        if self.dt_floor <= 0:
            raise ValueError("SEM-hard violation: dt_floor must be positive")