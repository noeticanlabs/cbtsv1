"""
NSC-M3L Bytecode Format and Types

Defines the bytecode instruction format, opcodes, and serialization
for the NSC-M3L virtual machine execution model.

Semantic Domain Objects:
    - Bytecode/IR for VM execution
    - Deterministic ordering guarantees
    - Stage-time schedule for LoC-Time

Denotation: Program → Executable bytecode with execution semantics
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict, Union
from enum import Enum, IntEnum
import struct


# =============================================================================
# Opcode Definitions
# =============================================================================

class OpCode(IntEnum):
    """
    NSC-M3L bytecode opcodes (extends Hadamard with physics/geometry ops)
    
    Opcode ranges:
        - 0x00-0x0F: Control flow
        - 0x10-0x1F: Stack operations  
        - 0x20-0x2F: Memory operations
        - 0x30-0x3F: Math operations
        - 0x40-0x4F: NSC-M3L Physics Ops
        - 0x50-0x5F: NSC-M3L Geometry Ops
        - 0x60-0x6F: NSC-M3L Ledger/EXEC Ops
        - 0x70-0x7F: Extended ops
        - 0x80-0xFF: Reserved/custom
    """
    # Control flow (0x00-0x0F)
    NOP = 0x00
    HALT = 0x01
    JMP = 0x02
    BR = 0x03          # Conditional branch
    BR_TRUE = 0x04     # Branch if true
    BR_FALSE = 0x05    # Branch if false
    CALL = 0x06
    RET = 0x07
    YIELD = 0x08       # Yield for stage transition
    AWAIT = 0x09       # Await stage completion
    
    # Stack operations (0x10-0x1F)
    PUSH = 0x10
    POP = 0x11
    DUP = 0x12
    SWAP = 0x13
    ROT = 0x14         # Rotate top 3
    OVER = 0x15
    UNDER = 0x16
    PICK = 0x17        # Pick nth from top
    
    # Memory operations (0x20-0x2F)
    LOAD = 0x20
    STORE = 0x21
    ALLOC = 0x22
    FREE = 0x23
    LOAD_CONST = 0x24
    LOAD_FIELD = 0x25   # Load from named field
    STORE_FIELD = 0x26  # Store to named field
    
    # Math operations (0x30-0x3F)
    ADD = 0x30
    SUB = 0x31
    MUL = 0x32
    DIV = 0x33
    MOD = 0x34
    NEG = 0x35
    ABS = 0x36
    SQRT = 0x37
    POW = 0x38
    MIN = 0x39
    MAX = 0x3A
    
    # Comparison operations (0x3B-0x3F)
    EQ = 0x3B
    NE = 0x3C
    LT = 0x3D
    LE = 0x3E
    GT = 0x3F
    
    # NSC-M3L Physics Ops (0x40-0x4F)
    GRAD = 0x40        # ∇ gradient (computes ∂_i f)
    DIV_OP = 0x41      # div (computes ∂_i v^i)
    CURL = 0x42        # curl (computes ∇ × v)
    LAPLACIAN = 0x43   # Δ (computes ∇² f = ∂_i ∂^i f)
    DDT = 0x44         # ∂/∂t (time derivative)
    PARTIAL = 0x45     # ∂_i (partial derivative in direction i)
    COV_DERIV = 0x46   # ∇^g covariant derivative
    LIE_DERIV = 0x47   # Lie derivative L_v
    EXT_DERIV = 0x48   # d (exterior derivative on forms)
    
    # NSC-M3L Geometry Ops (0x50-0x5F)
    HODGE = 0x50        # ⋆ Hodge star operator
    INNER = 0x51        # ⟨·,·⟩ interior product
    WEDGE = 0x52        # ∧ wedge product
    CONTRACT = 0x53     # Contraction of tensors
    SYM = 0x54          # Symmetrize
    ANTISYM = 0x55      # Antisymmetrize
    TRACE = 0x56        # Metric trace
    LOWER = 0x57        # Lower indices with metric
    RAISE = 0x58        # Raise indices with metric
    
    # Tensor construction (0x59-0x5F)
    MAKE_SYM6 = 0x59    # Construct sym6 from components
    MAKE_VEC3 = 0x5A    # Construct vec3 from components
    INDEX_GET = 0x5B    # Get tensor component
    INDEX_SET = 0x5C    # Set tensor component
    
    # NSC-M3L Ledger/EXEC Ops (0x60-0x6F)
    EMIT = 0x60         # Emit receipt to ledger
    CHECK_GATE = 0x61   # Gate condition check
    CHECK_INV = 0x62    # Invariant check
    SEAL = 0x63         # Seal step (commit to ledger)
    VERIFY = 0x64       # Verify receipt
    GET_RECEIPT = 0x65  # Get receipt by ID
    
    # Constraint enforcement (0x66-0x6F)
    ENFORCE_H = 0x66    # Enforce Hamiltonian constraint
    ENFORCE_M = 0x67    # Enforce momentum constraint
    PROJECT = 0x68      # Project to constraint surface
    PHI_FUNC = 0x69     # PHI function for constraints
    
    # Stage control (0x70-0x7F)
    STAGE_ENTER = 0x70  # Enter new stage
    STAGE_EXIT = 0x71   # Exit current stage
    STAGE_SYNC = 0x72   # Synchronize across stages
    CKPT = 0x73         # Create checkpoint
    RESTORE = 0x74      # Restore from checkpoint
    
    # JIT/Hotspot markers (0x75-0x7F)
    JIT_START = 0x75    # Mark JIT compilation start
    JIT_END = 0x76      # Mark JIT compilation end
    PROFILE = 0x77      # Profile counter
    
    # GR-specific operations (0x80-0x8F)
    CHRISTOFFEL = 0x80  # Compute Christoffel symbols
    RICCI = 0x81        # Compute Ricci tensor
    RICCI_SCALAR = 0x82 # Compute Ricci scalar
    EINSTEIN = 0x83     # Compute Einstein tensor
    RIEMANN = 0x84      # Compute Riemann tensor
    
    # BSSN operations (0x85-0x8F)
    BSSN_CONFORM = 0x85  # Conformal transformation
    BSSN_TRACE = 0x86    # Trace of extrinsic curvature
    BSSN_GAMMA = 0x87    # Gamma driver condition
    BSSN_LAMBDA = 0x88   # Lambda driver condition
    
    # Gauge operations (0x90-0x9F)
    GAUGE_LAPSE = 0x90   # Lapse condition
    GAUGE_SHIFT = 0x91   # Shift condition
    GAUGE_SINGULARITY = 0x92  # Singularity handling


# =============================================================================
# Instruction Format
# =============================================================================

@dataclass
class Instruction:
    """
    Single bytecode instruction.
    
    Format: 8 bytes total
        - byte 0: opcode (OpCode)
        - byte 1: operand1 (immediate or register index)
        - byte 2: operand2 (immediate or register index) 
        - byte 3: operand3 (immediate or register index)
        - bytes 4-7: immediate value (int32) or extended data
    """
    opcode: OpCode
    operand1: int = 0
    operand2: int = 0
    operand3: int = 0
    immediate: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate operands are in valid range."""
        self.operand1 = self.operand1 & 0xFF
        self.operand2 = self.operand2 & 0xFF
        self.operand3 = self.operand3 & 0xFF
        # Handle float immediates by converting to int for bitwise ops
        if isinstance(self.immediate, float):
            self.immediate = int(self.immediate) & 0xFFFFFFFF
        else:
            self.immediate = self.immediate & 0xFFFFFFFF
    
    def to_bytes(self) -> bytes:
        """Serialize instruction to bytes (8 bytes)."""
        return struct.pack(
            'BBBBi',
            self.opcode,
            self.operand1,
            self.operand2,
            self.operand3,
            self.immediate
        )
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'Instruction':
        """Deserialize instruction from bytes."""
        if len(data) < 8:
            raise ValueError(f"Instruction requires at least 8 bytes, got {len(data)}")
        opcode, operand1, operand2, operand3, immediate = struct.unpack('BBBBi', data[:8])
        return cls(
            opcode=OpCode(opcode),
            operand1=operand1,
            operand2=operand2,
            operand3=operand3,
            immediate=immediate
        )
    
    def __repr__(self) -> str:
        return f"Instr({self.opcode.name}, operands=[{self.operand1}, {self.operand2}, {self.operand3}], imm={self.immediate})"


@dataclass 
class InstructionWithSource:
    """Instruction with source location mapping for debugging."""
    instruction: Instruction
    line_number: int
    column: int
    source_text: str = ""
    
    def to_bytes(self) -> bytes:
        """Serialize with source reference."""
        data = self.instruction.to_bytes()
        # Add source reference (4 bytes for line, 2 for column, variable for text)
        text_bytes = self.source_text.encode('utf-8')[:255]
        header = struct.pack('HI', self.line_number, len(text_bytes))
        return data + header + text_bytes


# =============================================================================
# Bytecode Program
# =============================================================================

@dataclass
class BytecodeProgram:
    """
    Complete bytecode program with metadata.
    
    File format:
        - 4 bytes: Magic number (b"NSCM")
        - 2 bytes: Version number
        - 4 bytes: Entry point offset
        - 2 bytes: Stack size
        - 4 bytes: Data section size
        - 4 bytes: Number of instructions
        - Variable: Instructions (8 bytes each)
        - Variable: Constants section
        - Variable: Source map
        - Variable: Metadata
    """
    # Header
    magic: bytes = b"NSCM"
    version: int = 1
    entry_point: int = 0
    stack_size: int = 1024
    data_size: int = 0
    
    # Program content
    instructions: List[Instruction] = field(default_factory=list)
    constants: List[Any] = field(default_factory=list)
    
    # Debugging/analysis
    source_map: Dict[int, str] = field(default_factory=dict)
    line_map: Dict[int, int] = field(default_factory=dict)  # IP -> line number
    
    # Program metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __len__(self) -> int:
        """Return total size in bytes."""
        return len(self.to_bytes())
    
    def to_bytes(self) -> bytes:
        """Serialize complete program to bytes."""
        result = bytearray()
        
        # Header
        result.extend(self.magic)
        result.extend(struct.pack('>H', self.version))
        result.extend(struct.pack('>I', self.entry_point))
        result.extend(struct.pack('>H', self.stack_size))
        result.extend(struct.pack('>I', self.data_size))
        result.extend(struct.pack('>I', len(self.instructions)))
        
        # Instructions
        for inst in self.instructions:
            result.extend(inst.to_bytes())
        
        # Constants section
        result.extend(struct.pack('>I', len(self.constants)))
        for const in self.constants:
            if isinstance(const, (int, float)):
                result.extend(struct.pack('>d', float(const)))
            elif isinstance(const, str):
                result.extend(struct.pack('>I', len(const)))
                result.extend(const.encode('utf-8'))
            # Add more types as needed
        
        # Source map (IP -> source text)
        result.extend(struct.pack('>I', len(self.source_map)))
        for ip, source in self.source_map.items():
            result.extend(struct.pack('>I', ip))
            result.extend(struct.pack('>I', len(source)))
            result.extend(source.encode('utf-8'))
        
        return bytes(result)
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'BytecodeProgram':
        """Deserialize program from bytes."""
        pos = 0
        
        # Header
        magic = data[pos:pos+4]
        if magic != b"NSCM":
            raise ValueError(f"Invalid magic number: {magic}")
        pos += 4
        
        version = struct.unpack('>H', data[pos:pos+2])[0]
        pos += 2
        
        entry_point = struct.unpack('>I', data[pos:pos+4])[0]
        pos += 4
        
        stack_size = struct.unpack('>H', data[pos:pos+2])[0]
        pos += 2
        
        data_size = struct.unpack('>I', data[pos:pos+4])[0]
        pos += 4
        
        num_instructions = struct.unpack('>I', data[pos:pos+4])[0]
        pos += 4
        
        # Instructions
        instructions = []
        for _ in range(num_instructions):
            instructions.append(Instruction.from_bytes(data[pos:pos+8]))
            pos += 8
        
        # Constants
        num_constants = struct.unpack('>I', data[pos:pos+4])[0]
        pos += 4
        constants = []
        for _ in range(num_constants):
            const_type = struct.unpack('>d', data[pos:pos+8])[0]
            # Simplified - in practice need type tag
            constants.append(const_type)
            pos += 8
        
        # Source map
        num_entries = struct.unpack('>I', data[pos:pos+4])[0]
        pos += 4
        source_map = {}
        for _ in range(num_entries):
            ip = struct.unpack('>I', data[pos:pos+4])[0]
            pos += 4
            length = struct.unpack('>I', data[pos:pos+4])[0]
            pos += 4
            source = data[pos:pos+length].decode('utf-8')
            source_map[ip] = source
            pos += length
        
        return cls(
            version=version,
            entry_point=entry_point,
            stack_size=stack_size,
            data_size=data_size,
            instructions=instructions,
            constants=constants,
            source_map=source_map
        )
    
    def add_instruction(self, opcode: OpCode, operand1: int = 0, 
                       operand2: int = 0, operand3: int = 0, 
                       immediate: int = 0) -> int:
        """
        Add instruction and return its position.
        
        Returns:
            Position of the added instruction
        """
        ip = len(self.instructions)
        self.instructions.append(Instruction(
            opcode=opcode,
            operand1=operand1,
            operand2=operand2,
            operand3=operand3,
            immediate=immediate
        ))
        return ip


# =============================================================================
# Value Types for Stack/Memory
# =============================================================================

class ValueType(Enum):
    """Runtime value types for the VM."""
    NIL = 0
    INT = 1
    FLOAT = 2
    BOOL = 3
    STRING = 4
    TENSOR = 5         # Generic tensor (shape in metadata)
    VEC3 = 6           # 3D vector
    SYM6 = 7           # Symmetric 3x3 (6 components)
    FIELD = 8          # Named field reference
    RECEIPT = 9        # Ledger receipt
    STAGE = 10         # Stage enum
    METRIC = 11        # Metric tensor
    CONNECTION = 12    # Connection coefficients


@dataclass
class RuntimeValue:
    """Runtime value with type information."""
    value: Any
    value_type: ValueType
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_scalar(self) -> bool:
        return self.value_type in (ValueType.INT, ValueType.FLOAT)
    
    def __repr__(self) -> str:
        return f"Value({self.value_type.name}, {self.value})"


# =============================================================================
# Execution State
# =============================================================================

@dataclass
class ExecutionState:
    """
    Complete VM execution state for checkpoint/restore.
    """
    ip: int = 0
    sp: int = 0
    stack: List[RuntimeValue] = field(default_factory=list)
    memory: Dict[int, RuntimeValue] = field(default_factory=dict)
    registers: Dict[int, RuntimeValue] = field(default_factory=dict)
    clocks: Dict[str, float] = field(default_factory=dict)
    stage: str = "INIT"
    step_number: int = 0
    receipts: List['Receipt'] = field(default_factory=list)
    
    def snapshot(self) -> bytes:
        """Serialize state for checkpoint."""
        # Simplified - full implementation would serialize all fields
        return struct.pack('>III', self.ip, self.sp, self.step_number)
    
    @classmethod
    def restore(cls, data: bytes) -> 'ExecutionState':
        """Restore state from checkpoint."""
        ip, sp, step_number = struct.unpack('>III', data)
        return cls(ip=ip, sp=sp, step_number=step_number)


# =============================================================================
# Ledger Receipt Types
# =============================================================================

@dataclass 
class Receipt:
    """Ledger receipt for execution verification."""
    receipt_id: str
    step_number: int
    stage: str
    operation: str
    input_hash: str
    output_hash: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Convenience Functions
# =============================================================================

def create_nop_instruction() -> Instruction:
    """Create a NOP (no-op) instruction."""
    return Instruction(opcode=OpCode.NOP)

def create_halt_instruction() -> Instruction:
    """Create a HALT instruction."""
    return Instruction(opcode=OpCode.HALT)

def create_push_instruction(value: int) -> Instruction:
    """Create a PUSH instruction with immediate value."""
    return Instruction(
        opcode=OpCode.PUSH,
        immediate=value
    )

def create_load_instruction(reg_index: int) -> Instruction:
    """Create a LOAD instruction from register."""
    return Instruction(
        opcode=OpCode.LOAD,
        operand1=reg_index
    )

def create_jmp_instruction(target_ip: int) -> Instruction:
    """Create an unconditional jump instruction."""
    return Instruction(
        opcode=OpCode.JMP,
        immediate=target_ip
    )

def create_br_instruction(condition_reg: int, target_ip: int) -> Instruction:
    """Create a conditional branch instruction."""
    return Instruction(
        opcode=OpCode.BR,
        operand1=condition_reg,
        immediate=target_ip
    )

def create_emit_instruction(receipt_type: int) -> Instruction:
    """Create an EMIT instruction for ledger receipts."""
    return Instruction(
        opcode=OpCode.EMIT,
        operand1=receipt_type
    )

def create_gate_check_instruction(gate_id: int) -> Instruction:
    """Create a gate checking instruction."""
    return Instruction(
        opcode=OpCode.CHECK_GATE,
        operand1=gate_id
    )

def create_invariant_check_instruction(inv_id: int) -> Instruction:
    """Create an invariant checking instruction."""
    return Instruction(
        opcode=OpCode.CHECK_INV,
        operand1=inv_id
    )

def create_grad_instruction(dim: int = 0) -> Instruction:
    """Create a gradient computation instruction."""
    return Instruction(
        opcode=OpCode.GRAD,
        operand1=dim
    )

def create_laplacian_instruction() -> Instruction:
    """Create a Laplacian computation instruction."""
    return Instruction(opcode=OpCode.LAPLACIAN)

def create_seal_instruction() -> Instruction:
    """Create a step sealing instruction."""
    return Instruction(opcode=OpCode.SEAL)
