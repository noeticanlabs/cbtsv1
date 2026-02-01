# NSC Models - Execution Model
# VM bytecode, scheduling, JIT compilation

"""
NSC_Exec - Execution/Runtime Model

This module provides VM execution support for NSC bytecode:
- Bytecode interpretation
- Stage-time scheduling
- JIT compilation
- Determinism verification

Supported Models:
- EXEC: Bytecode execution
"""

from .vm import NSC_VM, OpCode, ExecutionState

__all__ = ['NSC_VM', 'OpCode', 'ExecutionState']
