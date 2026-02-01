"""
VM Error Classes for NSC-M3L Virtual Machine
"""


class VMError(Exception):
    """VM execution error."""
    
    def __init__(self, message: str, ip: int = -1, instruction = None):
        self.message = message
        self.ip = ip
        self.instruction = instruction
        super().__init__(f"VM Error at IP={ip}: {message}")


class StackUnderflowError(VMError):
    """Stack underflow error."""
    
    def __init__(self, ip: int = -1):
        super().__init__("Stack underflow", ip)


class StackOverflowError(VMError):
    """Stack overflow error."""
    
    def __init__(self, ip: int = -1):
        super().__init__("Stack overflow", ip)


class DivisionByZeroError(VMError):
    """Division by zero error."""
    
    def __init__(self, ip: int = -1):
        super().__init__("Division by zero", ip)


class GateViolationError(VMError):
    """Gate condition violation."""
    
    def __init__(self, gate_id: int, value: float, threshold: float, ip: int = -1):
        self.gate_id = gate_id
        self.value = value
        self.threshold = threshold
        super().__init__(f"Gate {gate_id} violated: {value} >= {threshold}", ip)


class InvariantViolationError(VMError):
    """Invariant violation."""
    
    def __init__(self, invariant_id: str, value: float, expected: float, ip: int = -1):
        self.invariant_id = invariant_id
        self.value = value
        self.expected = expected
        super().__init__(f"Invariant '{invariant_id}' violated: {value} != {expected}", ip)
