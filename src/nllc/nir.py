from dataclasses import dataclass, field
from typing import List, Optional, Any, Union
from src.nllc.ast import Span

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