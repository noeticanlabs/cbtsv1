from abc import ABC, abstractmethod

class TemporalSystemContract(ABC):
    """
    Abstract base class for the Temporal System Contract.

    This contract enforces the mandatory rules for time management in the system.
    Violations constitute SEM-hard failures.

    Attributes:
        t (float): Physical time (coordinate time).
        tau (int): Audit/coherence time.
        o (int): Orchestration index.
        s (int): Stage index.
        mu (int): Micro-step index.
        attempt_id (int): Attempt identifier.

    Abstract Methods:
        accepted_step(dt): Increments stage index s and physical time t by dt.
        audit_time(): Increments audit time τ.
        attempt(): Increments attempt_id.

    Methods:
        check_violations(): Checks for basic violations and raises ValueError if found.
    """

    def __init__(self):
        self.t = 0.0
        self.tau = 0
        self.o = 0
        self.s = 0
        self.mu = 0
        self.attempt_id = 0

    @abstractmethod
    def accepted_step(self, dt: float):
        """
        Advances the system on an accepted step.

        Increments the stage index s and physical time t by dt.
        """
        self.s += 1
        self.t += dt

    @abstractmethod
    def audit_time(self):
        """
        Advances the audit/coherence time.

        Increments τ.
        """
        self.tau += 1

    @abstractmethod
    def attempt(self):
        """
        Records an attempt.

        Increments attempt_id.
        """
        self.attempt_id += 1

    def check_violations(self):
        """
        Checks for SEM-hard violations.

        Raises ValueError if any violation is detected.
        """
        if self.t < 0:
            raise ValueError("SEM-hard violation: physical time t cannot be negative")
        if self.tau < 0:
            raise ValueError("SEM-hard violation: audit time τ cannot be negative")
        if self.o < 0:
            raise ValueError("SEM-hard violation: orchestration index o cannot be negative")
        if self.s < 0:
            raise ValueError("SEM-hard violation: stage index s cannot be negative")
        if self.mu < 0:
            raise ValueError("SEM-hard violation: micro-step index μ cannot be negative")
        if self.attempt_id < 0:
            raise ValueError("SEM-hard violation: attempt_id cannot be negative")