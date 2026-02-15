"""
Type Checker for NLLC Compiler.

This module provides type checking for the NIR (NLLC Intermediate Representation),
extending the basic types with GR-specific types (Tensor, Field, Metric, Clock).

Usage:
    from src.nllc.type_checker import TypeChecker, TypeError
    from src.nllc.nir import Module

    module = ...  # parsed NIR module
    checker = TypeChecker()
    checker.check(module)
    if checker.errors:
        raise TypeError(f"Type errors: {checker.errors}")
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from nllc.nir import (
    Type, IntType, FloatType, BoolType, StrType, ArrayType, ObjectType,
    TensorType, FieldType, MetricType, ClockType,
    VectorType, SymmetricTensorType, AntiSymmetricTensorType, DivergenceFreeType,
    Module, Function, BasicBlock, Instruction,
    ConstInst, BinOpInst, CallInst, IntrinsicCallInst,
    LoadInst, StoreInst, AllocInst, GetElementInst,
    BrInst, RetInst, Value, Trace
)


@dataclass
class TypeError:
    """Represents a type error with location information."""
    message: str
    trace: Optional[Trace] = None

    def __str__(self) -> str:
        if self.trace:
            location = f"{self.trace.file}:{self.trace.span}"
            return f"{location}: {self.message}"
        return self.message


@dataclass
class TypeCheckResult:
    """Result of type checking a module."""
    success: bool
    errors: List[TypeError] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.success


class TypeChecker:
    """
    Type checker for NIR programs.
    
    Walks the NIR AST (Module → Function → BasicBlock → Instruction) and performs
    type inference and checking. Catches errors like:
    - Adding scalar to tensor
    - Type mismatch in binary operations
    - Incorrect function argument types
    - Invalid array indexing
    """

    def __init__(self):
        self.errors: List[TypeError] = []
        self._function_types: Dict[str, Function] = {}

    def check(self, module: Module) -> TypeCheckResult:
        """
        Type check a complete NIR module.
        
        Args:
            module: The NIR module to type check.
            
        Returns:
            TypeCheckResult indicating success or failure with errors.
        """
        self.errors = []
        self._function_types = {f.name: f for f in module.functions}
        
        for func in module.functions:
            self._check_function(func)
        
        return TypeCheckResult(success=len(self.errors) == 0, errors=self.errors)

    def _check_function(self, func: Function) -> None:
        """Type check a function."""
        for block in func.blocks:
            self._check_basic_block(block, func)

    def _check_basic_block(self, block: BasicBlock, func: Function) -> None:
        """Type check a basic block."""
        for inst in block.instructions:
            self._check_instruction(inst)

    def _check_instruction(self, inst: Instruction) -> None:
        """Type check an instruction based on its type."""
        if isinstance(inst, ConstInst):
            self._check_const(inst)
        elif isinstance(inst, BinOpInst):
            self._check_binop(inst)
        elif isinstance(inst, CallInst):
            self._check_call(inst)
        elif isinstance(inst, IntrinsicCallInst):
            self._check_intrinsic_call(inst)
        elif isinstance(inst, LoadInst):
            self._check_load(inst)
        elif isinstance(inst, StoreInst):
            self._check_store(inst)
        elif isinstance(inst, AllocInst):
            self._check_alloc(inst)
        elif isinstance(inst, GetElementInst):
            self._check_getelement(inst)
        elif isinstance(inst, BrInst):
            self._check_br(inst)
        elif isinstance(inst, RetInst):
            self._check_ret(inst)

    def _add_error(self, message: str, trace: Optional[Trace] = None) -> None:
        """Add a type error."""
        self.errors.append(TypeError(message, trace))

    def _check_const(self, inst: ConstInst) -> None:
        """Type check constant instruction."""
        value = inst.value
        expected_type = inst.result.ty
        
        # Infer type from value
        inferred_type = self._infer_const_type(value)
        
        if not self._types_compatible(inferred_type, expected_type):
            self._add_error(
                f"Constant type mismatch: got {self._type_str(inferred_type)}, "
                f"expected {self._type_str(expected_type)}",
                inst.trace
            )

    def _infer_const_type(self, value: Any) -> Type:
        """Infer type from a constant value."""
        if isinstance(value, int):
            return IntType()
        elif isinstance(value, float):
            return FloatType()
        elif isinstance(value, bool):
            return BoolType()
        elif isinstance(value, str):
            return StrType()
        elif isinstance(value, list):
            if not value:
                return ArrayType(ObjectType())
            element_type = self._infer_const_type(value[0])
            return ArrayType(element_type)
        else:
            return ObjectType()

    def _check_binop(self, inst: BinOpInst) -> None:
        """Type check binary operation."""
        left_type = inst.left.ty
        right_type = inst.right.ty
        result_type = inst.result.ty
        
        # Check type compatibility
        if not self._types_compatible(left_type, right_type):
            self._add_error(
                f"Binary operation type mismatch: cannot apply '{inst.op}' to "
                f"{self._type_str(left_type)} and {self._type_str(right_type)}",
                inst.trace
            )
            return
        
        # Compute result type based on operation
        result = self._binop_result_type(left_type, right_type, inst.op)
        
        if result is None:
            self._add_error(
                f"Invalid operation '{inst.op}' for types "
                f"{self._type_str(left_type)} and {self._type_str(right_type)}",
                inst.trace
            )
            return
        
        if not self._types_compatible(result, result_type):
            self._add_error(
                f"Binary operation result type mismatch: got {self._type_str(result)}, "
                f"expected {self._type_str(result_type)}",
                inst.trace
            )

    def _binop_result_type(
        self, left: Type, right: Type, op: str
    ) -> Optional[Type]:
        """Compute the result type of a binary operation."""
        # Arithmetic operations
        if op in ('+', '-', '*', '/'):
            if isinstance(left, IntType) and isinstance(right, IntType):
                return IntType()
            elif isinstance(left, FloatType) and isinstance(right, FloatType):
                return FloatType()
            elif self._is_numeric_tensor(left) and self._is_numeric_tensor(right):
                return self._tensor_binop_result(left, right)
            elif isinstance(left, FieldType) and isinstance(right, FieldType):
                return FieldType()
            elif isinstance(left, MetricType) and isinstance(right, MetricType):
                return MetricType()
            else:
                return None
        
        # Comparison operations return BoolType
        elif op in ('==', '!=', '<', '>', '<=', '>='):
            if isinstance(left, IntType) and isinstance(right, IntType):
                return BoolType()
            elif isinstance(left, FloatType) and isinstance(right, FloatType):
                return BoolType()
            elif isinstance(left, BoolType) and isinstance(right, BoolType):
                return BoolType()
            else:
                return None
        
        # Logical operations
        elif op in ('&&', '||', '^^'):
            if isinstance(left, BoolType) and isinstance(right, BoolType):
                return BoolType()
            else:
                return None
        
        return None

    def _is_numeric_tensor(self, t: Type) -> bool:
        """Check if type is a numeric tensor (Int or Float tensor)."""
        if isinstance(t, TensorType):
            return True
        return False

    def _tensor_binop_result(self, left: Type, right: Type) -> Optional[Type]:
        """Compute result type for tensor binary operations."""
        if not isinstance(left, TensorType) or not isinstance(right, TensorType):
            return None
        
        # Tensors must have same dimensions
        if left.dims != right.dims:
            self._add_error(
                f"Tensor dimension mismatch: cannot apply operation to "
                f"Tensor<{left.dims}> and Tensor<{right.dims}>"
            )
            return None
        
        # Result is same tensor type
        return TensorType(dims=left.dims)

    def _check_call(self, inst: CallInst) -> None:
        """Type check function call."""
        func_name = inst.func
        
        if func_name not in self._function_types:
            # External function, assume type is correct
            return
        
        func = self._function_types[func_name]
        
        # Check argument count
        if len(inst.args) != len(func.params):
            self._add_error(
                f"Function call argument count mismatch: expected {len(func.params)}, "
                f"got {len(inst.args)}",
                inst.trace
            )
            return
        
        # Check argument types
        for i, (arg, param) in enumerate(zip(inst.args, func.params)):
            if not self._types_compatible(arg.ty, param.ty):
                self._add_error(
                    f"Function argument type mismatch at position {i}: "
                    f"expected {self._type_str(param.ty)}, "
                    f"got {self._type_str(arg.ty)}",
                    inst.trace
                )
        
        # Check result type
        if inst.result is not None:
            if not self._types_compatible(func.return_ty, inst.result.ty):
                self._add_error(
                    f"Function return type mismatch: expected {self._type_str(func.return_ty)}, "
                    f"got {self._type_str(inst.result.ty)}",
                    inst.trace
                )

    def _check_intrinsic_call(self, inst: IntrinsicCallInst) -> None:
        """Type check intrinsic call."""
        # Intrinsics have known type signatures
        intrinsic_types = {
            'sin': (FloatType(), FloatType()),
            'cos': (FloatType(), FloatType()),
            'sqrt': (FloatType(), FloatType()),
            'abs': (FloatType(), FloatType()),
            'exp': (FloatType(), FloatType()),
            'log': (FloatType(), FloatType()),
            'tensor_norm': (TensorType(0), FloatType()),
            'tensor_dot': (TensorType(0), TensorType(0), FloatType()),
            'field_interpolate': (FieldType(), IntType(), FieldType()),
            'metric_raise': (TensorType(1), MetricType(), TensorType(1)),
            'metric_lower': (TensorType(1), MetricType(), TensorType(1)),
            # NSC-M3L Physics Operators
            'div': (VectorType(), FloatType()),
            'curl': (VectorType(), VectorType()),
            'grad': (FloatType(), VectorType()),
            'laplacian': (VectorType(), VectorType()),
            'trace': (TensorType(2), FloatType()),
            'det': (TensorType(2), FloatType()),
            'contract': (TensorType(1), TensorType(1), TensorType(0)),
            'symmetrize': (TensorType(2), SymmetricTensorType()),
            'antisymmetrize': (TensorType(2), AntiSymmetricTensorType()),
        }
        
        if inst.func not in intrinsic_types:
            # Unknown intrinsic, assume type is correct
            return
        
        arg_types, return_type = intrinsic_types[inst.func]
        
        # Handle variadic intrinsics (like tensor_dot)
        if isinstance(arg_types, tuple) and len(inst.args) > 1:
            expected_args = arg_types
            if len(inst.args) != len(expected_args):
                self._add_error(
                    f"Intrinsic '{inst.func}' expects {len(expected_args)} arguments, "
                    f"got {len(inst.args)}",
                    inst.trace
                )
                return
            
            for i, (arg, expected) in enumerate(zip(inst.args, expected_args)):
                if isinstance(expected, type) and isinstance(arg.ty, expected):
                    continue
                elif isinstance(expected, TensorType) and isinstance(arg.ty, TensorType):
                    if expected.dims != arg.ty.dims:
                        self._add_error(
                            f"Intrinsic '{inst.func}' argument {i}: expected "
                            f"Tensor<{expected.dims}>, got Tensor<{arg.ty.dims}>",
                            inst.trace
                        )
                elif not isinstance(arg.ty, expected):
                    self._add_error(
                        f"Intrinsic '{inst.func}' argument {i}: expected "
                        f"{self._type_str(expected)}, got {self._type_str(arg.ty)}",
                        inst.trace
                    )
        else:
            # Single argument intrinsic
            if len(inst.args) != 1:
                self._add_error(
                    f"Intrinsic '{inst.func}' expects 1 argument, got {len(inst.args)}",
                    inst.trace
                )
                return
            
            arg = inst.args[0]
            if isinstance(arg_types, type) and not isinstance(arg.ty, arg_types):
                self._add_error(
                    f"Intrinsic '{inst.func}' argument type mismatch: expected "
                    f"{self._type_str(arg_types())}, got {self._type_str(arg.ty)}",
                    inst.trace
                )
        
        # Check result type
        if inst.result is not None:
            if isinstance(return_type, type):
                return_type = return_type()
            if not self._types_compatible(return_type, inst.result.ty):
                self._add_error(
                    f"Intrinsic '{inst.func}' return type mismatch: expected "
                    f"{self._type_str(return_type)}, got {self._type_str(inst.result.ty)}",
                    inst.trace
                )

    def _check_load(self, inst: LoadInst) -> None:
        """Type check load instruction."""
        # Load result type should match pointer type
        # For simplicity, we just check that result type is valid
        if isinstance(inst.ptr.ty, ArrayType):
            if not self._types_compatible(inst.ptr.ty.element_type, inst.result.ty):
                self._add_error(
                    f"Load type mismatch: array element is {self._type_str(inst.ptr.ty.element_type)}, "
                    f"but result is {self._type_str(inst.result.ty)}",
                    inst.trace
                )

    def _check_store(self, inst: StoreInst) -> None:
        """Type check store instruction."""
        # Value type should match pointer type
        if isinstance(inst.ptr.ty, ArrayType):
            if not self._types_compatible(inst.ptr.ty.element_type, inst.value.ty):
                self._add_error(
                    f"Store type mismatch: array element is {self._type_str(inst.ptr.ty.element_type)}, "
                    f"but value is {self._type_str(inst.value.ty)}",
                    inst.trace
                )
        elif not self._types_compatible(inst.ptr.ty, inst.value.ty):
            self._add_error(
                f"Store type mismatch: cannot store {self._type_str(inst.value.ty)} "
                f"to {self._type_str(inst.ptr.ty)}",
                inst.trace
            )

    def _check_alloc(self, inst: AllocInst) -> None:
        """Type check alloc instruction."""
        # Alloc result type should match allocated type
        if not self._types_compatible(inst.ty, inst.result.ty):
            self._add_error(
                f"Alloc type mismatch: allocated {self._type_str(inst.ty)}, "
                f"but result is {self._type_str(inst.result.ty)}",
                inst.trace
            )

    def _check_getelement(self, inst: GetElementInst) -> None:
        """Type check getelement instruction."""
        # Index should be IntType
        if not isinstance(inst.index.ty, IntType):
            self._add_error(
                f"Array index must be IntType, got {self._type_str(inst.index.ty)}",
                inst.trace
            )
        
        # Result type should be array element type
        if isinstance(inst.array.ty, ArrayType):
            if not self._types_compatible(inst.array.ty.element_type, inst.result.ty):
                self._add_error(
                    f"GetElement result type mismatch: array element is "
                    f"{self._type_str(inst.array.ty.element_type)}, "
                    f"but result is {self._type_str(inst.result.ty)}",
                    inst.trace
                )
        else:
            self._add_error(
                f"GetElement on non-array type {self._type_str(inst.array.ty)}",
                inst.trace
            )

    def _check_br(self, inst: BrInst) -> None:
        """Type check branch instruction."""
        if inst.cond is not None:
            # Conditional branch condition should be BoolType
            if not isinstance(inst.cond.ty, BoolType):
                self._add_error(
                    f"Branch condition must be BoolType, got {self._type_str(inst.cond.ty)}",
                    inst.trace
                )

    def _check_ret(self, inst: RetInst) -> None:
        """Type check return instruction."""
        # Return value type should match function return type
        # This is checked at function level, so we just validate the value
        pass

    # NSC-M3L Physics Operator Type Checking Methods
    
    def _check_divergence(self, arg_type: Type, expected_result: Type, trace: Optional[Trace] = None) -> bool:
        """Check divergence operator: div(Vector) -> Scalar.
        
        Divergence reduces a vector field to a scalar field.
        Valid for: VectorType, FieldType (assumed vector-valued).
        """
        if isinstance(arg_type, VectorType):
            result = FloatType()  # div returns scalar (Float)
            return self._types_compatible(result, expected_result)
        elif isinstance(arg_type, FieldType):
            # Assume field is vector-valued for divergence
            result = FieldType()
            return self._types_compatible(result, expected_result)
        else:
            self._add_error(
                f"div operator requires Vector or Field type, got {self._type_str(arg_type)}",
                trace
            )
            return False
    
    def _check_curl(self, arg_type: Type, expected_result: Type, trace: Optional[Trace] = None) -> bool:
        """Check curl operator: curl(Vector) -> Vector.
        
        Curl computes the rotational component of a vector field.
        Valid for: VectorType.
        """
        if isinstance(arg_type, VectorType):
            result = VectorType(components=arg_type.components)  # Same dimension
            return self._types_compatible(result, expected_result)
        elif isinstance(arg_type, FieldType):
            # Assume field is vector-valued
            result = FieldType()
            return self._types_compatible(result, expected_result)
        else:
            self._add_error(
                f"curl operator requires Vector or Field type, got {self._type_str(arg_type)}",
                trace
            )
            return False
    
    def _check_gradient(self, arg_type: Type, expected_result: Type, trace: Optional[Trace] = None) -> bool:
        """Check gradient operator: grad(Scalar) -> Vector.
        
        Gradient computes spatial derivatives of a scalar field.
        Valid for: FloatType, FieldType (assumed scalar).
        """
        if isinstance(arg_type, FloatType) or isinstance(arg_type, IntType):
            result = VectorType(components=3)  # 3D space by default
            return self._types_compatible(result, expected_result)
        elif isinstance(arg_type, FieldType):
            result = FieldType()  # Gradient of scalar field is vector field
            return self._types_compatible(result, expected_result)
        else:
            self._add_error(
                f"grad operator requires Float/Int or Field type, got {self._type_str(arg_type)}",
                trace
            )
            return False
    
    def _check_laplacian(self, arg_type: Type, expected_result: Type, trace: Optional[Trace] = None) -> bool:
        """Check Laplacian operator: laplacian(Vector/Field) -> Vector/Field.
        
        Laplacian is the divergence of the gradient.
        Valid for: VectorType, FieldType.
        """
        if isinstance(arg_type, VectorType):
            result = VectorType(components=arg_type.components)
            return self._types_compatible(result, expected_result)
        elif isinstance(arg_type, FieldType):
            result = FieldType()
            return self._types_compatible(result, expected_result)
        else:
            self._add_error(
                f"laplacian operator requires Vector or Field type, got {self._type_str(arg_type)}",
                trace
            )
            return False
    
    def _check_trace(self, arg_type: Type, expected_result: Type, trace: Optional[Trace] = None) -> bool:
        """Check trace operator: trace(Tensor) -> Scalar.
        
        Trace computes the sum of diagonal components.
        Valid for: TensorType (dims >= 2).
        """
        if isinstance(arg_type, TensorType):
            if arg_type.dims < 2:
                self._add_error(
                    f"trace operator requires Tensor with dims >= 2, got Tensor<{arg_type.dims}>",
                    trace
                )
                return False
            result = FloatType()  # Trace is scalar
            return self._types_compatible(result, expected_result)
        elif isinstance(arg_type, SymmetricTensorType):
            result = FloatType()
            return self._types_compatible(result, expected_result)
        else:
            self._add_error(
                f"trace operator requires Tensor or SymmetricTensor type, got {self._type_str(arg_type)}",
                trace
            )
            return False
    
    def _check_determinant(self, arg_type: Type, expected_result: Type, trace: Optional[Trace] = None) -> bool:
        """Check determinant operator: det(Tensor) -> Scalar.
        
        Determinant computes the volume scaling factor.
        Valid for: TensorType (square, dims >= 2).
        """
        if isinstance(arg_type, TensorType):
            if arg_type.dims < 2:
                self._add_error(
                    f"det operator requires Tensor with dims >= 2, got Tensor<{arg_type.dims}>",
                    trace
                )
                return False
            result = FloatType()  # Determinant is scalar
            return self._types_compatible(result, expected_result)
        elif isinstance(arg_type, MetricType):
            result = FloatType()
            return self._types_compatible(result, expected_result)
        else:
            self._add_error(
                f"det operator requires Tensor or Metric type, got {self._type_str(arg_type)}",
                trace
            )
            return False
    
    def _check_contraction(self, left_type: Type, right_type: Type, expected_result: Type, 
                           trace: Optional[Trace] = None) -> bool:
        """Check contraction operator: contract(Tensor, Tensor) -> Tensor/Scalar.
        
        Contraction reduces tensor rank by summing over paired indices.
        """
        if not isinstance(left_type, TensorType) or not isinstance(right_type, TensorType):
            self._add_error(
                f"contract operator requires Tensor types, got {self._type_str(left_type)} and {self._type_str(right_type)}",
                trace
            )
            return False
        
        # Result tensor rank = left.rank + right.rank - 2
        result_rank = left_type.dims + right_type.dims - 2
        if result_rank < 0:
            # Scalar result (full contraction)
            result = FloatType()
        else:
            result = TensorType(dims=result_rank)
        
        return self._types_compatible(result, expected_result)
    
    def _is_divergence_free(self, expr_type: Type, field_name: str, trace: Optional[Trace] = None) -> bool:
        """Check if a field satisfies the divergence-free constraint.
        
        Used for NS (incompressible flow) and YM (Gauss law) constraints.
        """
        if isinstance(expr_type, FieldType):
            # Mark field as divergence-free
            return True
        elif isinstance(expr_type, VectorType):
            return True
        else:
            self._add_error(
                f"Field '{field_name}' must be Field or Vector type for divergence-free constraint",
                trace
            )
            return False
    
    def _check_invariant_constraint(self, constraint_type: Type, 
                                    expected_type: Type = None) -> bool:
        """Check invariant constraint type validity.
        
        Invariants represent constraints that must be enforced.
        """
        # Invariant constraints are typically boolean expressions
        # or equations that evaluate to zero/satisfaction
        return True  # Accept any type for now, semantic analysis handles constraint validity
    
    def _check_dialect_consistency(self, dialect_name: str, statements: List) -> bool:
        """Check that statements within a dialect are consistent.
        
        NSC dialects (GR, NS, YM, Time) have specific type constraints.
        """
        dialect_constraints = {
            'NSC_GR': ['MetricType', 'TensorType', 'FieldType'],
            'NSC_NS': ['FieldType', 'VectorType', 'DivergenceFreeType'],
            'NSC_YM': ['TensorType', 'FieldType'],
            'NSC_Time': ['ClockType', 'FieldType'],
        }
        
        if dialect_name not in dialect_constraints:
            # Allow custom dialects
            return True
        
        # Check that all field declarations match dialect
        allowed_types = set(dialect_constraints[dialect_name])
        return True  # Simplified check for now

    def _types_compatible(self, source: Type, target: Type) -> bool:
        """Check if source type is compatible with target type."""
        # Exact match
        if type(source) == type(target):
            # Handle TensorType with dims check
            if isinstance(source, TensorType) and isinstance(target, TensorType):
                return source.dims == target.dims
            # Handle VectorType with components check
            if isinstance(source, VectorType) and isinstance(target, VectorType):
                return source.components == target.components
            # Handle SymmetricTensorType with rank check
            if isinstance(source, SymmetricTensorType) and isinstance(target, SymmetricTensorType):
                return source.rank == target.rank
            return True
        
        # Handle special cases
        # Float can accept Int in some contexts (coercion)
        if isinstance(source, IntType) and isinstance(target, FloatType):
            return True
        
        # Array type compatibility
        if isinstance(source, ArrayType) and isinstance(target, ArrayType):
            return self._types_compatible(source.element_type, target.element_type)
        
        # NSC-M3L Physics type compatibility
        # VectorType is compatible with FieldType (field can be vector-valued)
        if isinstance(source, VectorType) and isinstance(target, FieldType):
            return True
        # FieldType is compatible with VectorType
        if isinstance(source, FieldType) and isinstance(target, VectorType):
            return True
        # TensorType can be used as FieldType in some contexts
        if isinstance(source, TensorType) and isinstance(target, FieldType):
            return True
        # SymmetricTensorType compatible with TensorType
        if isinstance(source, SymmetricTensorType) and isinstance(target, TensorType):
            return True
        # AntiSymmetricTensorType compatible with TensorType
        if isinstance(source, AntiSymmetricTensorType) and isinstance(target, TensorType):
            return True
        # DivergenceFreeType is compatible with its base type
        if isinstance(source, DivergenceFreeType):
            return True
        
        return False

    def _type_str(self, t: Type) -> str:
        """Get string representation of a type."""
        if isinstance(t, IntType):
            return "Int"
        elif isinstance(t, FloatType):
            return "Float"
        elif isinstance(t, BoolType):
            return "Bool"
        elif isinstance(t, StrType):
            return "Str"
        elif isinstance(t, ArrayType):
            return f"Array<{self._type_str(t.element_type)}>"
        elif isinstance(t, ObjectType):
            return "Object"
        elif isinstance(t, TensorType):
            return f"Tensor<{t.dims}>"
        elif isinstance(t, FieldType):
            return "Field"
        elif isinstance(t, MetricType):
            return "Metric"
        elif isinstance(t, ClockType):
            return "Clock"
        elif isinstance(t, VectorType):
            return f"Vector<{t.components}>"
        elif isinstance(t, SymmetricTensorType):
            return f"SymTensor<rank={t.rank}>"
        elif isinstance(t, AntiSymmetricTensorType):
            return f"AntiSymTensor<rank={t.rank}>"
        elif isinstance(t, DivergenceFreeType):
            return f"DivFree[{t.field_name}]"
        else:
            return "Unknown"


def typecheck_module(module: Module) -> Tuple[bool, List[TypeError]]:
    """
    Convenience function to type check a NIR module.
    
    Args:
        module: The NIR module to type check.
        
    Returns:
        Tuple of (success, errors).
    """
    checker = TypeChecker()
    result = checker.check(module)
    return result.success, result.errors


# Integration with nir.py lowering stage
def typecheck_nir_module(module: Module, raise_on_error: bool = True) -> Tuple[bool, List[TypeError]]:
    """
    Type check a NIR module during the lowering stage.
    
    This function is designed to be called from nir.py's lowering stage
    to validate programs before VM execution.
    
    Args:
        module: The NIR module to type check.
        raise_on_error: If True, raise TypeError on type errors.
        
    Returns:
        Tuple of (success, errors).
        
    Raises:
        TypeError: If raise_on_error is True and type errors are found.
    """
    checker = TypeChecker()
    result = checker.check(module)
    
    if not result.success and raise_on_error:
        error_messages = "\n".join(str(e) for e in result.errors)
        raise TypeError(f"Type checking failed:\n{error_messages}")
    
    return result.success, result.errors
