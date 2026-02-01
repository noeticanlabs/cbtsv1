"""
NSC-M3L Virtual Machine Interpreter Package

Modularized from exec_vm.py:
    - errors.py: VM error classes
    - physics_kernels.py: Numba-accelerated physics operations
    - vm.py: VirtualMachine core class

Backward-compatible re-exports from original module.
"""

# Error classes
from src.nsc.exec_vm.errors import (
    VMError,
    StackUnderflowError,
    StackOverflowError,
    DivisionByZeroError,
    GateViolationError,
    InvariantViolationError,
)

# Physics kernels
from src.nsc.exec_vm.physics_kernels import (
    NUMBA_AVAILABLE,
    _compute_gradient_jit,
    _compute_divergence_jit,
    _compute_laplacian_jit,
    _compute_curl_jit,
)

# VirtualMachine
from src.nsc.exec_vm.vm import VirtualMachine

# Re-export all public symbols
__all__ = [
    # Errors
    'VMError',
    'StackUnderflowError',
    'StackOverflowError', 
    'DivisionByZeroError',
    'GateViolationError',
    'InvariantViolationError',
    # Physics
    'NUMBA_AVAILABLE',
    '_compute_gradient_jit',
    '_compute_divergence_jit',
    '_compute_laplacian_jit',
    '_compute_curl_jit',
    # VM
    'VirtualMachine',
]
