# NSC Execution Model - Virtual Machine
# Bytecode interpretation and scheduling

"""
NSC_VM - NSC Virtual Machine

Provides stack-based bytecode execution for NSC programs:
- Arithmetic and tensor operations
- Physics operators (grad, div, curl, laplacian)
- Geometry operators (christoffel, ricci)
- Control flow (loops, conditionals)
- Ledger operations (gates, receipts)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple
from enum import Enum
import numpy as np


# =============================================================================
# Opcode Definitions
# =============================================================================

class OpCode(Enum):
    """NSC bytecode opcodes."""
    # Control flow
    HALT = 0x00
    NOP = 0x01
    JMP = 0x02
    JZ = 0x03      # Jump if zero
    JNZ = 0x04     # Jump if not zero
    CALL = 0x05
    RET = 0x06
    
    # Stack operations
    PUSH = 0x10
    POP = 0x11
    DUP = 0x12
    SWAP = 0x13
    ROT = 0x14     # Rotate top 3
    
    # Arithmetic
    ADD = 0x20
    SUB = 0x21
    MUL = 0x22
    DIV = 0x23
    NEG = 0x24
    ABS = 0x25
    SQRT = 0x26
    
    # Tensor operations
    MAKE_SYM6 = 0x30      # Construct symmetric 6-tensor
    MAKE_VEC3 = 0x31      # Construct 3-vector
    INDEX_GET = 0x32
    INDEX_SET = 0x33
    TRACE = 0x34
    TRANSPOSE = 0x35
    
    # Calculus operators
    GRAD = 0x40           # Gradient
    DIV_OP = 0x41         # Divergence
    CURL = 0x42           # Curl
    LAPLACIAN = 0x43      # Laplacian
    DDT = 0x44            # Time derivative
    
    # Geometry operators
    CHRISTOFFEL = 0x50
    RICCI = 0x51
    RICCI_SCALAR = 0x52
    EINSTEIN = 0x53
    RIEMANN = 0x54
    
    # Ledger operations
    EMIT = 0x60           # Emit receipt
    CHECK_GATE = 0x61     # Check gate condition
    CHECK_INV = 0x62      # Check invariant
    SEAL = 0x63           # Seal step
    VERIFY = 0x64         # Verify receipt
    
    # Stage control
    STAGE_ENTER = 0x70
    STAGE_EXIT = 0x71
    STAGE_SYNC = 0x72
    CKPT = 0x73           # Checkpoint
    RESTORE = 0x74        # Restore
    
    # Physics constraints
    ENFORCE_H = 0x80      # Hamiltonian constraint
    ENFORCE_M = 0x81      # Momentum constraint
    PROJECT = 0x82        # Project to constraint surface
    
    # BSSN operators
    BSSN_CONFORM = 0x90   # Conformal transform
    BSSN_TRACE = 0x91     # Trace K
    BSSN_GAMMA = 0x92     # Gamma driver


# =============================================================================
# Instruction and Program
# =============================================================================

@dataclass
class Instruction:
    """Single bytecode instruction."""
    opcode: OpCode
    operand: Optional[int] = None
    immediate: Optional[float] = None
    label: Optional[str] = None


@dataclass
class BytecodeProgram:
    """Complete bytecode program."""
    name: str
    entry_point: int = 0
    instructions: List[Instruction] = field(default_factory=list)
    constants: List[Any] = field(default_factory=list)
    labels: Dict[str, int] = field(default_factory=dict)
    
    def add_instruction(self, inst: Instruction):
        """Add instruction to program."""
        inst_index = len(self.instructions)
        self.instructions.append(inst)
        if inst.label:
            self.labels[inst.label] = inst_index
    
    def resolve_labels(self):
        """Resolve label references to indices."""
        for inst in self.instructions:
            if inst.label and inst.label in self.labels:
                inst.operand = self.labels[inst.label]


# =============================================================================
# Execution State
# =============================================================================

@dataclass
class ExecutionState:
    """VM execution state."""
    ip: int = 0                    # Instruction pointer
    sp: int = -1                   # Stack pointer
    fp: int = -1                   # Frame pointer
    running: bool = False
    halted: bool = False
    error: Optional[str] = None
    step_count: int = 0
    stage: str = "INIT"


# =============================================================================
# Virtual Machine
# =============================================================================

@dataclass
class NSC_VM:
    """NSC Virtual Machine for bytecode execution.
    
    Attributes:
        program: Bytecode to execute
        stack: Operand stack
        memory: Variable storage
        registers: Fast access registers
        state: Execution state
    """
    program: BytecodeProgram
    stack: List[Any] = field(default_factory=list)
    memory: Dict[str, Any] = field(default_factory=dict)
    registers: List[Any] = field(default_factory=list)
    state: ExecutionState = field(default_factory=ExecutionState)
    
    def __post_init__(self):
        """Initialize VM."""
        self.registers = [0.0] * 64
    
    def run(self, max_steps: int = 10000) -> ExecutionState:
        """Execute program."""
        self.state.running = True
        self.state.ip = self.program.entry_point
        
        for _ in range(max_steps):
            if not self.state.running or self.state.halted:
                break
            
            if self.state.ip >= len(self.program.instructions):
                self.state.halted = True
                break
            
            inst = self.program.instructions[self.state.ip]
            self._execute(inst)
            self.state.step_count += 1
            self.state.ip += 1
        
        self.state.running = False
        return self.state
    
    def step(self) -> bool:
        """Execute single instruction."""
        if self.state.ip >= len(self.program.instructions):
            self.state.halted = True
            return False
        
        inst = self.program.instructions[self.state.ip]
        self._execute(inst)
        self.state.step_count += 1
        self.state.ip += 1
        return not self.state.halted
    
    def _execute(self, inst: Instruction):
        """Execute single instruction."""
        opcode = inst.opcode
        
        if opcode == OpCode.HALT:
            self.state.halted = True
        
        elif opcode == OpCode.NOP:
            pass
        
        elif opcode == OpCode.PUSH:
            self._push(inst.immediate if inst.immediate is not None else inst.operand)
        
        elif opcode == OpCode.POP:
            self._pop()
        
        elif opcode == OpCode.DUP:
            if self.stack:
                self.stack.append(self.stack[-1])
        
        elif opcode == OpCode.SWAP:
            if len(self.stack) >= 2:
                self.stack[-1], self.stack[-2] = self.stack[-2], self.stack[-1]
        
        elif opcode == OpCode.ADD:
            b = self._pop()
            a = self._pop()
            self._push(a + b)
        
        elif opcode == OpCode.SUB:
            b = self._pop()
            a = self._pop()
            self._push(a - b)
        
        elif opcode == OpCode.MUL:
            b = self._pop()
            a = self._pop()
            self._push(a * b)
        
        elif opcode == OpCode.DIV:
            b = self._pop()
            a = self._pop()
            self._push(a / b if b != 0 else float('inf'))
        
        elif opcode == OpCode.NEG:
            a = self._pop()
            self._push(-a)
        
        elif opcode == OpCode.GRAD:
            # Gradient computation using finite differences
            field = self._pop()
            if isinstance(field, np.ndarray):
                # Compute gradient via central differences
                grad = np.gradient(field)
                self._push(grad)
            else:
                self._push(np.zeros(3))
        
        elif opcode == OpCode.DIV_OP:
            # Divergence computation using finite differences
            vector = self._pop()
            if isinstance(vector, np.ndarray):
                # Compute divergence
                div = np.sum(np.gradient(vector, axis=0))
                self._push(div)
            else:
                self._push(0.0)
        
        elif opcode == OpCode.CURL:
            # Curl computation: ∇×v
            v = self._pop()
            if isinstance(v, np.ndarray) and v.ndim == 1 and len(v) == 3:
                # Compute curl numerically
                curl = np.array([
                    np.gradient(v[2], axis=1) - np.gradient(v[1], axis=0),
                    np.gradient(v[0], axis=0) - np.gradient(v[2], axis=1),
                    np.gradient(v[1], axis=0) - np.gradient(v[0], axis=1)
                ])
                self._push(curl)
            else:
                self._push(np.zeros(3))
        
        elif opcode == OpCode.LAPLACIAN:
            # Laplacian computation using finite differences
            field = self._pop()
            if isinstance(field, np.ndarray):
                laplacian = np.sum(np.array([np.gradient(np.gradient(field, axis=i), axis=i) for i in range(field.ndim)]), axis=0)
                self._push(laplacian)
            else:
                self._push(0.0)
        
        elif opcode == OpCode.CHRISTOFFEL:
            # Christoffel symbols from metric
            # Input: metric tensor (3x3)
            metric = self._pop()
            if isinstance(metric, np.ndarray) and metric.shape == (3, 3):
                inv_metric = np.linalg.inv(metric)
                christoffel = np.zeros((3, 3, 3))
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            christoffel[i, j, k] = 0.5 * sum(
                                inv_metric[i, l] * (
                                    np.gradient(metric[l, j], axis=0) +
                                    np.gradient(metric[l, k], axis=0) -
                                    np.gradient(metric[j, k], axis=0)
                                ) for l in range(3)
                            )
                self._push(christoffel)
            else:
                self._push(np.zeros((3, 3, 3)))
        
        elif opcode == OpCode.RICCI:
            # Ricci tensor from Christoffel symbols
            christoffel = self._pop()
            if isinstance(christoffel, np.ndarray) and christoffel.shape == (3, 3, 3):
                ricci = np.zeros((3, 3))
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            ricci[i, j] += np.gradient(christoffel[j, k, i], axis=0) - np.gradient(christoffel[i, k, j], axis=0)
                self._push(ricci)
            else:
                self._push(np.zeros((3, 3)))
        
        elif opcode == OpCode.EMIT:
            # Emit receipt - placeholder
            self._push("receipt_id")
        
        elif opcode == OpCode.CHECK_GATE:
            # Check gate - placeholder
            self._push(1)
        
        else:
            # Unknown opcode - treat as no-op for now
            pass
    
    def _push(self, value: Any):
        """Push value onto stack."""
        self.stack.append(value)
        self.state.sp = len(self.stack) - 1
    
    def _pop(self) -> Any:
        """Pop value from stack."""
        if not self.stack:
            self.state.error = "Stack underflow"
            self.state.halted = True
            return 0
        value = self.stack.pop()
        self.state.sp = len(self.stack) - 1
        return value
    
    def set_memory(self, name: str, value: Any):
        """Set memory variable."""
        self.memory[name] = value
    
    def get_memory(self, name: str) -> Any:
        """Get memory variable."""
        return self.memory.get(name)


# =============================================================================
# Program Builders
# =============================================================================

def build_gradient_program() -> BytecodeProgram:
    """Build bytecode for gradient computation."""
    program = BytecodeProgram(name="gradient")
    
    # Stack: field
    # Output: [df/dx, df/dy, df/dz]
    program.add_instruction(Instruction(OpCode.PUSH, label="start"))
    program.add_instruction(Instruction(OpCode.GRAD))
    program.add_instruction(Instruction(OpCode.HALT))
    
    program.resolve_labels()
    return program


def build_divergence_program() -> BytecodeProgram:
    """Build bytecode for divergence computation."""
    program = BytecodeProgram(name="divergence")
    
    # Stack: vector
    # Output: scalar
    program.add_instruction(Instruction(OpCode.PUSH, label="start"))
    program.add_instruction(Instruction(OpCode.DIV_OP))
    program.add_instruction(Instruction(OpCode.HALT))
    
    program.resolve_labels()
    return program


def build_curl_program() -> BytecodeProgram:
    """Build bytecode for curl computation."""
    program = BytecodeProgram(name="curl")
    
    # Stack: vector
    # Output: vector
    program.add_instruction(Instruction(OpCode.PUSH, label="start"))
    program.add_instruction(Instruction(OpCode.CURL))
    program.add_instruction(Instruction(OpCode.HALT))
    
    program.resolve_labels()
    return program


def build_laplacian_program() -> BytecodeProgram:
    """Build bytecode for Laplacian computation."""
    program = BytecodeProgram(name="laplacian")
    
    # Stack: scalar
    # Output: scalar
    program.add_instruction(Instruction(OpCode.PUSH, label="start"))
    program.add_instruction(Instruction(OpCode.LAPLACIAN))
    program.add_instruction(Instruction(OpCode.HALT))
    
    program.resolve_labels()
    return program
