from dataclasses import dataclass, field
from typing import List, Optional, Any, Union
from nllc.ast import Span

# Types
@dataclass
class Type:
    pass

@dataclass
class IntType(Type):
    pass

@dataclass
class FloatType(Type):
    pass

@dataclass
class BoolType(Type):
    pass

@dataclass
class StrType(Type):
    pass

@dataclass
class ArrayType(Type):
    element_type: Type

@dataclass
class ObjectType(Type):
    pass

@dataclass
class TensorType(Type):
    """Tensor type for GR computations.
    dims=0: scalar, dims=1: vector, dims=2: 2-tensor, etc.
    """
    dims: int

@dataclass
class FieldType(Type):
    """Grid field type for numerical relativity.
    Represents a quantity defined on a grid point.
    """
    pass

@dataclass
class MetricType(Type):
    """Spacetime metric type.
    Represents a 4-dimensional spacetime metric.
    """
    pass

@dataclass
class ClockType(Type):
    """Temporal clock type for time evolution.
    """
    pass

# NSC-M3L Physics Types
@dataclass
class VectorType(Type):
    """Vector type for physics fields.
    Components: number of vector components.
    """
    components: int = 1

@dataclass
class SymmetricTensorType(Type):
    """Symmetric tensor type for physics (e.g., stress-energy tensor).
    rank: tensor rank (2 for 2-tensor, etc.)
    """
    rank: int = 2

@dataclass
class AntiSymmetricTensorType(Type):
    """Antisymmetric tensor type for physics (e.g., electromagnetic field).
    rank: tensor rank
    """
    rank: int = 2

@dataclass
class DivergenceFreeType(Type):
    """Divergence-free constraint type for NS/YM constraints.
    Used to mark fields that satisfy div(F) = 0 constraint.
    """
    field_name: str = ""

# Trace information
@dataclass
class Trace:
    file: str
    span: Span
    ast_path: str

# Values in SSA
@dataclass
class Value:
    name: str  # e.g., "%0"
    ty: Type

# Instructions
@dataclass
class Instruction:
    trace: Trace

@dataclass
class ConstInst(Instruction):
    result: Value
    value: Any  # int, float, bool, str, or list for array

@dataclass
class BinOpInst(Instruction):
    result: Value
    left: Value
    op: str  # +, -, *, /, ==, !=, <, >, etc.
    right: Value

@dataclass
class CallInst(Instruction):
    result: Optional[Value]  # None for void calls
    func: str
    args: List[Value]

@dataclass
class IntrinsicCallInst(Instruction):
    result: Optional[Value]  # None for void calls
    func: str
    args: List[Value]

@dataclass
class LoadInst(Instruction):
    result: Value
    ptr: Value  # for arrays or variables

@dataclass
class StoreInst(Instruction):
    ptr: Value
    index: Optional[Value]
    value: Value

@dataclass
class AllocInst(Instruction):
    result: Value
    ty: Type

@dataclass
class GetElementInst(Instruction):
    result: Value
    array: Value
    index: Value

# Control flow
@dataclass
class BrInst(Instruction):
    cond: Optional[Value]  # None for unconditional
    true_block: str
    false_block: Optional[str] = None

@dataclass
class RetInst(Instruction):
    value: Optional[Value]

# Basic Block
@dataclass
class BasicBlock:
    name: str
    instructions: List[Instruction] = field(default_factory=list)
    # Terminator is the last instruction, like BrInst or RetInst

# Function
@dataclass
class Function:
    name: str
    params: List[Value]  # parameter values
    return_ty: Type
    blocks: List[BasicBlock] = field(default_factory=list)

# Module
@dataclass
class Module:
    functions: List[Function] = field(default_factory=list)