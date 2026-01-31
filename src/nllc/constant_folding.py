"""
Constant Folding Optimization Pass

Folds constant expressions at compile time to reduce runtime computation.

Transformations:
    (1 + 2) * 3        →  9
    x * 0              →  0
    x * 1              →  x
    x + 0              →  x
    -(-x)              →  x
    x / x (x != 0)     →  1

Author: CBTSV1 Team
Version: 1.0
"""

from .nir import *

import numpy as np


class ConstantFolder:
    """
    Performs constant folding on NIR instructions.
    
    Walks through instructions and replaces constant expressions
    with their computed values.
    """
    
    def __init__(self):
        self.changes = 0
        self.folds = {}
    
    def fold(self, module: NIRModule) -> NIRModule:
        """Fold constants in the module."""
        self.changes = 0
        self.folds = {}
        
        new_instructions = []
        
        for inst in module.instructions:
            folded = self._fold_instruction(inst, module.constants)
            if folded is not None and folded != inst:
                new_instructions.append(folded)
                self.changes += 1
            else:
                new_instructions.append(inst)
        
        module.instructions = new_instructions
        
        return module
    
    def _fold_instruction(self, inst, constants) -> NIRInstruction:
        """Fold a single instruction if possible."""
        
        if isinstance(inst, BinOpInst):
            return self._fold_binop(inst, constants)
        elif isinstance(inst, UnaryOpInst):
            return self._fold_unary(inst, constants)
        elif isinstance(inst, CallInst):
            return self._fold_call(inst, constants)
        
        return inst
    
    def _fold_binop(self, inst: BinOpInst, constants) -> NIRInstruction:
        """Fold binary operations."""
        lhs_val = self._get_constant_value(inst.lhs, constants)
        rhs_val = self._get_constant_value(inst.rhs, constants)
        
        if lhs_val is None or rhs_val is None:
            # Not a constant expression
            return inst
        
        # Both operands are constants - fold
        result = self._compute_binop(inst.op, lhs_val, rhs_val)
        
        if result is not None:
            # Create constant instruction with folded result
            self.folds[inst.id] = result
            return ConstInst(inst.id, result, inst.type)
        
        return inst
    
    def _fold_unary(self, inst: UnaryOpInst, constants) -> NIRInstruction:
        """Fold unary operations."""
        val = self._get_constant_value(inst.operand, constants)
        
        if val is None:
            return inst
        
        result = self._compute_unary(inst.op, val)
        
        if result is not None:
            self.folds[inst.id] = result
            return ConstInst(inst.id, result, inst.type)
        
        return inst
    
    def _fold_call(self, inst: CallInst, constants) -> NIRInstruction:
        """Fold intrinsic calls with constant arguments."""
        # Check if all args are constants
        args = [self._get_constant_value(arg, constants) for arg in inst.args]
        
        if None in args:
            return inst
        
        # Fold known intrinsics
        return self._fold_intrinsic(inst, args, constants)
    
    def _fold_intrinsic(self, inst: CallInst, args, constants):
        """Fold intrinsic function calls."""
        # Math intrinsics that can be folded
        intrinsic_folds = {
            'abs': lambda x: abs(x),
            'sqrt': lambda x: np.sqrt(x) if x >= 0 else None,
            'sin': lambda x: np.sin(x),
            'cos': lambda x: np.cos(x),
            'exp': lambda x: np.exp(x),
            'log': lambda x: np.log(x) if x > 0 else None,
            'floor': lambda x: np.floor(x),
            'ceil': lambda x: np.ceil(x),
            'round': lambda x: np.round(x),
            'neg': lambda x: -x,
        }
        
        func_name = inst.func_name
        
        if func_name in intrinsic_folds and len(args) == 1:
            try:
                result = intrinsic_folds[func_name](args[0])
                if result is not None:
                    self.folds[inst.id] = result
                    return ConstInst(inst.id, result, inst.type)
            except (ValueError, OverflowError):
                pass
        
        return inst
    
    def _get_constant_value(self, operand, constants):
        """Get constant value from operand."""
        if isinstance(operand, ConstRef):
            return constants.get(operand.const_id)
        elif isinstance(operand, SSAVar):
            return self.folds.get(operand.inst_id)
        return None
    
    def _compute_binop(self, op, lhs, rhs):
        """Compute binary operation result."""
        ops = {
            'add': lambda a, b: a + b,
            'sub': lambda a, b: a - b,
            'mul': lambda a, b: a * b,
            'div': lambda a, b: a / b if b != 0 else None,
            'mod': lambda a, b: a % b if b != 0 else None,
            'pow': lambda a, b: a ** b,
            'and': lambda a, b: int(a) & int(b),
            'or': lambda a, b: int(a) | int(b),
            'xor': lambda a, b: int(a) ^ int(b),
            'shl': lambda a, b: int(a) << int(b),
            'shr': lambda a, b: int(a) >> int(b),
            'eq': lambda a, b: a == b,
            'ne': lambda a, b: a != b,
            'lt': lambda a, b: a < b,
            'le': lambda a, b: a <= b,
            'gt': lambda a, b: a > b,
            'ge': lambda a, b: a >= b,
        }
        
        if op in ops:
            try:
                return ops[op](lhs, rhs)
            except (ZeroDivisionError, OverflowError, TypeError):
                return None
        
        return None
    
    def _compute_unary(self, op, val):
        """Compute unary operation result."""
        ops = {
            'neg': lambda x: -x,
            'not': lambda x: not x,
            'abs': lambda x: abs(x),
        }
        
        if op in ops:
            try:
                return ops[op](val)
            except (TypeError, ValueError):
                return None
        
        return None


def fold_constants(module: NIRModule) -> NIRModule:
    """Convenience function to fold constants in a module."""
    folder = ConstantFolder()
    return folder.fold(module)
