"""
NSC-M3L Virtual Machine Interpreter

Implements the bytecode VM for executing NSC-M3L programs.
Supports stack-based execution with physics/geometry operations.

Semantic Domain Objects:
    - Bytecode/IR for VM execution
    - Deterministic ordering guarantees

Denotation: Program → Executable bytecode with execution semantics

This module has been modularized into src/nsc/exec_vm/ package.
For direct access to submodules, import from:
    - src.nsc.exec_vm.errors
    - src.nsc.exec_vm.physics_kernels
    - src.nsc.exec_vm.vm
"""

# Re-export all symbols from the modularized package for backward compatibility
from src.nsc.exec_vm import (
    # Errors
    VMError,
    StackUnderflowError,
    StackOverflowError,
    DivisionByZeroError,
    GateViolationError,
    InvariantViolationError,
    # Physics
    NUMBA_AVAILABLE,
    _compute_gradient_jit,
    _compute_divergence_jit,
    _compute_laplacian_jit,
    _compute_curl_jit,
    # VM
    VirtualMachine,
)

# Also expose these as if they were defined at module level
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
