"""
NSC-M3L Type Checker

Implements static semantics per specifications/nsc_m3l_v1.md section 3.
"""

from typing import Union, List, Optional, Set, Dict, Tuple, Any
from dataclasses import dataclass, field

from .ast import (
    Node, Program, Statement, Decl, Equation, Functional, Constraint, Directive,
    Atom, Group, OpCall, BinaryOp, Expr,
    Type, ScalarType, VectorType, TensorType, FieldType, FormType, OperatorType,
    ManifoldType, MetricType, LieAlgebraType, ConnectionType,
    Meta, ModelSelector, InvariantList, GateSpec, TargetList,
    Model, SmoothnessClass, SemanticType as ASTSemanticType
)
from .types import (
    Scalar, Vector, Tensor, Operator, Functional as FuncType, Field as FieldType_,
    Form, BundleConnection, Metric, Manifold, LieAlgebra, Connection,
    SemanticType, Dimension, TimeMode, Effect, Tag,
    TypeError, GeometryPrerequisiteError, RegularityError, ModelCompatibilityError
)


@dataclass
class SymbolInfo:
    """Symbol table entry."""
    name: str
    declared_type: SemanticType
    smoothness: Optional[SmoothnessClass] = None
    models_used: Set[Model] = field(default_factory=set)
    invariants_required: List[str] = field(default_factory=list)
    units: Optional[Dimension] = None
    time_mode: TimeMode = TimeMode.PHYSICAL
    effects: Set[Effect] = field(default_factory=set)
    is_metric: bool = False
    is_connection: bool = False
    is_manifold: bool = False
    lie_algebra: Optional[str] = None  # Name if Lie algebra valued


@dataclass
class Scope:
    """Symbol scope with nesting support."""
    symbols: Dict[str, SymbolInfo] = field(default_factory=dict)
    parent: Optional['Scope'] = None
    models_in_scope: Set[Model] = field(default_factory=set)
    metrics_in_scope: Set[str] = field(default_factory=set)
    connections_in_scope: Set[str] = field(default_factory=set)
    manifolds_in_scope: Set[str] = field(default_factory=set)
    
    def add_symbol(self, name: str, info: SymbolInfo) -> None:
        self.symbols[name] = info
        if info.is_metric:
            self.metrics_in_scope.add(name)
        if info.is_connection:
            self.connections_in_scope.add(name)
        if info.is_manifold:
            self.manifolds_in_scope.add(name)
    
    def lookup(self, name: str) -> Optional[SymbolInfo]:
        """Look up symbol in current or parent scopes."""
        if name in self.symbols:
            return self.symbols[name]
        if self.parent:
            return self.parent.lookup(name)
        return None
    
    def enter_scope(self) -> 'Scope':
        """Create and enter a child scope."""
        child = Scope(parent=self, models_in_scope=self.models_in_scope.copy())
        child.metrics_in_scope = self.metrics_in_scope.copy()
        child.connections_in_scope = self.connections_in_scope.copy()
        child.manifolds_in_scope = self.manifolds_in_scope.copy()
        return child
    
    def exit_scope(self) -> 'Scope':
        """Exit to parent scope."""
        return self.parent if self.parent else self


class TypeChecker:
    """
    NSC-M3L Type Checker.
    
    Implements type checking per section 3 of the specification:
    - Type inference for expressions
    - Regularity constraint enforcement
    - Geometry prerequisite checking
    - Model compatibility validation
    - Error reporting with location information
    """
    
    def __init__(self, invariant_registry: Optional[Dict] = None):
        """Initialize type checker with optional invariant registry."""
        self.global_scope = Scope()
        self.current_scope = self.global_scope
        self.errors: List[TypeError] = []
        self.warnings: List[str] = []
        self.invariant_registry = invariant_registry or {}
        self.current_models: Set[Model] = set()
        
        # Register built-in types
        self._init_builtin_types()
    
    def _init_builtin_types(self) -> None:
        """Initialize built-in type definitions."""
        # No built-in variables, but we track models
        self.global_scope.models_in_scope = set(Model)
    
    def check_program(self, program: Program) -> Program:
        """
        Type-check entire program, annotate AST with types.
        
        Args:
            program: The parsed program AST
            
        Returns:
            The annotated program with type information
        """
        self.errors = []
        self.warnings = []
        
        # Reset scope
        self.global_scope = Scope()
        self.global_scope.models_in_scope = set(Model)
        self.current_scope = self.global_scope
        
        # Process each statement
        for stmt in program.statements:
            self.check_statement(stmt)
        
        # Transfer type info to program
        program.type = ASTSemanticType.FUNCTIONAL
        program.domains_used = self.current_scope.models_in_scope
        
        if self.errors:
            raise self.errors[0]
        
        return program
    
    def check_statement(self, stmt: Statement) -> Statement:
        """Type-check a statement."""
        if isinstance(stmt, Decl):
            return self._check_decl(stmt)
        elif isinstance(stmt, Equation):
            return self._check_equation(stmt)
        elif isinstance(stmt, Functional):
            return self._check_functional(stmt)
        elif isinstance(stmt, Constraint):
            return self._check_constraint(stmt)
        elif isinstance(stmt, Directive):
            return self._check_directive(stmt)
        else:
            return stmt
    
    def check_expression(self, expr: Expr) -> Expr:
        """Type-check expression, return with inferred type."""
        inferred_type = self._infer_expression_type(expr)
        expr.type = self._semantic_to_ast_type(inferred_type)
        return expr
    
    def check_type(self, type_node: Type) -> SemanticType:
        """Validate type specification and return semantic type."""
        if isinstance(type_node, ScalarType):
            return Scalar()
        elif isinstance(type_node, VectorType):
            return Vector(dim=type_node.dim)
        elif isinstance(type_node, TensorType):
            return Tensor(k=type_node.k, l=type_node.l)
        elif isinstance(type_node, FieldType):
            value_type = self.check_type(type_node.value_type)
            return FieldType_(value_type=value_type)
        elif isinstance(type_node, FormType):
            return Form(p=type_node.p)
        elif isinstance(type_node, OperatorType):
            domain = self.check_type(type_node.domain)
            codomain = self.check_type(type_node.codomain)
            return Operator(domain=domain, codomain=codomain)
        elif isinstance(type_node, ManifoldType):
            return Manifold(dim=type_node.dim, signature=type_node.signature)
        elif isinstance(type_node, MetricType):
            return Metric(signature=type_node.signature, dim=type_node.dim)
        elif isinstance(type_node, LieAlgebraType):
            return LieAlgebra(name=type_node.name)
        elif isinstance(type_node, ConnectionType):
            return Connection(
                metric_compatible=type_node.metric_compatible,
                torsion_free=type_node.torsion_free
            )
        else:
            return Scalar()  # Default
    
    # === Declaration Checking ===
    
    def _check_decl(self, decl: Decl) -> Decl:
        """Type-check variable declaration."""
        semantic_type = self.check_type(decl.decl_type)
        
        # Parse metadata
        smoothness = None
        models_used: Set[Model] = set()
        invariants_required: List[str] = []
        units = None
        time_mode = TimeMode.PHYSICAL
        effects: Set[Effect] = set()
        
        is_metric = False
        is_connection = False
        is_manifold = False
        lie_algebra = None
        
        if decl.meta:
            for key, value in decl.meta.pairs.items():
                if key == 'regularity':
                    smoothness = self._parse_smoothness(value)
                elif key == 'models':
                    models_used = self._parse_models(value)
                elif key == 'invariants':
                    invariants_required = self._parse_invariants(value)
                elif key == 'units':
                    units = self._parse_dimension(value)
                elif key == 'time':
                    time_mode = self._parse_time_mode(value)
                elif key == 'effect':
                    effects.add(self._parse_effect(value))
                elif key == 'metric':
                    is_metric = value.lower() == 'true'
                elif key == 'connection':
                    is_connection = value.lower() == 'true'
                elif key == 'manifold':
                    is_manifold = value.lower() == 'true'
                elif key == 'lie_algebra':
                    lie_algebra = value
        
        # Check for metric/connection/manifold declarations
        if isinstance(semantic_type, Metric):
            is_metric = True
        if isinstance(semantic_type, Connection):
            is_connection = True
        if isinstance(semantic_type, Manifold):
            is_manifold = True
        
        # Create symbol info
        symbol_info = SymbolInfo(
            name=decl.ident,
            declared_type=semantic_type,
            smoothness=smoothness,
            models_used=models_used,
            invariants_required=invariants_required,
            units=units,
            time_mode=time_mode,
            effects=effects,
            is_metric=is_metric,
            is_connection=is_connection,
            is_manifold=is_manifold,
            lie_algebra=lie_algebra
        )
        
        # Add to scope
        self.current_scope.add_symbol(decl.ident, symbol_info)
        
        # Annotate decl
        decl.type = self._semantic_to_ast_type(semantic_type)
        decl.domains_used = models_used
        decl.regularity = smoothness
        decl.units = units
        decl.time_mode = time_mode.value if time_mode else None
        decl.invariants_required = invariants_required
        decl.effects = effects
        decl.model_tags = {}
        
        return decl
    
    # === Equation Checking ===
    
    def _check_equation(self, eq: Equation) -> Equation:
        """Type-check equation statement."""
        # Check both sides
        lhs_type = self._infer_expression_type(eq.lhs)
        rhs_type = self._infer_expression_type(eq.rhs)
        
        # Check type compatibility
        if not self._types_compatible(lhs_type, rhs_type):
            self._add_type_error(
                "Type mismatch in equation",
                expected=lhs_type,
                found=rhs_type,
                location=self._node_location(eq)
            )
        
        # Check model compatibility
        models = self.current_scope.models_in_scope
        self._check_model_compatibility(eq.lhs, models)
        self._check_model_compatibility(eq.rhs, models)
        
        # Annotate equation
        eq.type = ASTSemanticType.FUNCTIONAL
        eq.domains_used = models
        
        # Parse metadata
        if eq.meta:
            eq.regularity = self._parse_smoothness(eq.meta.pairs.get('regularity', ''))
            eq.invariants_required = self._parse_invariants(eq.meta.pairs.get('invariants', ''))
        
        return eq
    
    # === Functional Checking ===
    
    def _check_functional(self, func: Functional) -> Functional:
        """Type-check functional definition."""
        # Create new scope for bindings
        func_scope = self.current_scope.enter_scope()
        self.current_scope = func_scope
        
        # Process bindings
        for binding in func.bindings:
            if binding.type:
                binding_type = self.check_type(binding.type)
            else:
                binding_type = Scalar()  # Default
            
            symbol_info = SymbolInfo(
                name=binding.ident,
                declared_type=binding_type
            )
            self.current_scope.add_symbol(binding.ident, symbol_info)
        
        # Check expression
        expr_type = self._infer_expression_type(func.expr)
        
        # Functional must return scalar
        if not isinstance(expr_type, Scalar):
            self._add_type_error(
                "Functional must return scalar type",
                expected=Scalar(),
                found=expr_type,
                location=self._node_location(func)
            )
        
        # Exit scope
        self.current_scope = self.current_scope.exit_scope()
        
        # Annotate functional
        func.type = ASTSemanticType.FUNCTIONAL
        func.domains_used = self.current_scope.models_in_scope
        
        return func
    
    # === Constraint Checking ===
    
    def _check_constraint(self, constraint: Constraint) -> Constraint:
        """Type-check constraint definition."""
        # Check predicate type (should be boolean/scalar)
        pred_type = self._infer_expression_type(constraint.predicate)
        
        constraint.type = ASTSemanticType.FUNCTIONAL
        constraint.domains_used = self.current_scope.models_in_scope
        
        return constraint
    
    # === Directive Checking ===
    
    def _check_directive(self, directive: Directive) -> Directive:
        """Type-check directive statement."""
        if directive.directive_type.value == 'model':
            # @model(ModelList)
            if directive.model_selector:
                self.current_models = directive.model_selector.models.copy()
                self.current_scope.models_in_scope = self.current_models.copy()
        
        elif directive.directive_type.value == 'inv':
            # @inv(InvariantList)
            if directive.invariant_list:
                # Validate invariant IDs against registry
                for inv_id in directive.invariant_list.invariants:
                    if inv_id not in self.invariant_registry:
                        self.warnings.append(f"Unknown invariant: {inv_id}")
        
        directive.domains_used = self.current_models
        
        return directive
    
    # === Expression Type Inference ===
    
    def _infer_expression_type(self, expr: Expr) -> SemanticType:
        """Infer the semantic type of an expression."""
        if isinstance(expr, Atom):
            return self._infer_atom_type(expr)
        elif isinstance(expr, Group):
            return self._infer_expression_type(expr.inner)
        elif isinstance(expr, OpCall):
            return self._infer_op_call(expr)
        elif isinstance(expr, BinaryOp):
            return self._infer_binary_op(expr)
        else:
            return Scalar()
    
    def _infer_atom_type(self, atom: Atom) -> SemanticType:
        """Infer type of atomic expression."""
        # Check if it's a known symbol
        symbol = self.current_scope.lookup(atom.value)
        if symbol:
            # Track model usage
            for model in symbol.models_used:
                atom.domains_used.add(model)
            return symbol.declared_type
        
        # Try to parse as numeric literal
        if atom.value.replace('.', '').replace('-', '').isdigit():
            return Scalar()
        
        # Default to unknown scalar
        return Scalar()
    
    def _infer_binary_op(self, op: BinaryOp) -> SemanticType:
        """Infer result type of binary operation."""
        left_type = self._infer_expression_type(op.left)
        right_type = self._infer_expression_type(op.right)
        
        # Track domains
        op.domains_used = left_type if hasattr(left_type, '__iter__') and isinstance(left_type, set) else set()
        op.domains_used.update(right_type if hasattr(right_type, '__iter__') and isinstance(right_type, set) else set())
        
        # Addition/Subtraction
        if op.op in ('+', '-'):
            if self._types_compatible(left_type, right_type):
                return left_type
            else:
                self._add_type_error(
                    f"Type mismatch in {op.op} operation",
                    expected=left_type,
                    found=right_type,
                    location=self._node_location(op)
                )
                return left_type
        
        # Multiplication/Division
        elif op.op in ('*', '/'):
            return self._infer_mul_type(left_type, right_type)
        
        # Inner product
        elif op.op == '⟨,⟩' or op.op == '<,>':
            if isinstance(left_type, Vector) and isinstance(right_type, Vector):
                return Scalar()
            else:
                self._add_type_error(
                    "Inner product requires vector operands",
                    expected=Vector(),
                    found=left_type,
                    location=self._node_location(op)
                )
                return Scalar()
        
        # Commutator
        elif op.op == '[,]':
            if isinstance(left_type, LieAlgebra) and isinstance(right_type, LieAlgebra):
                if left_type.name == right_type.name:
                    return left_type
                else:
                    self._add_type_error(
                        "Commutator requires same Lie algebra",
                        expected=left_type,
                        found=right_type,
                        location=self._node_location(op)
                    )
                    return left_type
            else:
                self._add_type_error(
                    "Commutator requires Lie algebra operands",
                    expected=LieAlgebra("?"),
                    found=left_type,
                    location=self._node_location(op)
                )
                return LieAlgebra("?")
        
        return Scalar()
    
    def _infer_mul_type(self, a: SemanticType, b: SemanticType) -> SemanticType:
        """Infer result type of multiplication."""
        # Scalar * anything -> anything
        if isinstance(a, Scalar):
            return b
        if isinstance(b, Scalar):
            return a
        
        # Vector * Scalar -> Vector (already handled)
        # Tensor * Tensor -> Tensor contraction or product
        if isinstance(a, Tensor) and isinstance(b, Tensor):
            # Return Tensor with combined indices
            return Tensor(k=a.k + b.k, l=a.l + b.l)
        
        # Vector dot Vector -> Scalar (special case)
        if isinstance(a, Vector) and isinstance(b, Vector):
            return Scalar()
        
        # Field * Scalar -> Field
        if isinstance(a, FieldType_) and isinstance(b, Scalar):
            return a
        if isinstance(b, FieldType_) and isinstance(a, Scalar):
            return b
        
        return Scalar()
    
    def _infer_op_call(self, op_call: OpCall) -> SemanticType:
        """Infer result type of operator call."""
        op = op_call.op
        arg_type = self._infer_expression_type(op_call.arg)
        
        # Track regularity requirements
        op_call.domains_used = set()
        
        # === Gradient/Nabla ===
        if op == '∇' or op == 'grad':
            # [RULE] No Hidden Regularity
            self._check_regularity(op_call.arg, SmoothnessClass.C1)
            
            if isinstance(arg_type, FieldType_):
                value_type = arg_type.value_type
                if isinstance(value_type, Scalar):
                    # ∇(Field[Scalar]) -> Field[Vector]
                    op_call.domains_used.add(Model.CALC)
                    return FieldType_(value_type=Vector())
                elif isinstance(value_type, Vector):
                    # ∇(Field[Vector]) -> Field[Tensor(1,1)] (requires GEO + Metric)
                    if Model.GEO in self.current_scope.models_in_scope:
                        self._check_geometry_prereqs(op_call, {'metric': True})
                        op_call.domains_used.add(Model.GEO)
                    return FieldType_(value_type=Tensor(k=1, l=1))
            
            self._add_type_error(
                f"Gradient requires Field[Scalar] or Field[Vector], got {arg_type}",
                expected=FieldType_(value_type=Scalar()),
                found=arg_type,
                location=self._node_location(op_call)
            )
            return FieldType_(value_type=Scalar())
        
        # === Laplacian ===
        elif op == 'Δ' or op == 'laplacian':
            # [RULE] No Hidden Regularity
            self._check_regularity(op_call.arg, SmoothnessClass.C2)
            
            if isinstance(arg_type, FieldType_):
                value_type = arg_type.value_type
                if isinstance(value_type, Scalar):
                    # Δ(Field[Scalar]) -> Field[Scalar]
                    return FieldType_(value_type=Scalar())
                elif isinstance(value_type, Vector):
                    # Δ(Field[Vector]) -> Field[Vector] (requires GEO + Connection)
                    if Model.GEO in self.current_scope.models_in_scope:
                        self._check_geometry_prereqs(op_call, {'connection': True})
                        op_call.domains_used.add(Model.GEO)
                    return FieldType_(value_type=Vector())
            
            self._add_type_error(
                f"Laplacian requires Field[Scalar] or Field[Vector], got {arg_type}",
                expected=FieldType_(value_type=Scalar()),
                found=arg_type,
                location=self._node_location(op_call)
            )
            return FieldType_(value_type=Scalar())
        
        # === Divergence ===
        elif op == 'div':
            if isinstance(arg_type, FieldType_):
                value_type = arg_type.value_type
                if isinstance(value_type, Vector):
                    # div(Field[Vector]) -> Field[Scalar]
                    op_call.domains_used.add(Model.CALC)
                    return FieldType_(value_type=Scalar())
            
            self._add_type_error(
                f"Divergence requires Field[Vector], got {arg_type}",
                expected=FieldType_(value_type=Vector()),
                found=arg_type,
                location=self._node_location(op_call)
            )
            return FieldType_(value_type=Scalar())
        
        # === Curl ===
        elif op == 'curl':
            if isinstance(arg_type, FieldType_):
                value_type = arg_type.value_type
                if isinstance(value_type, Vector):
                    # curl(Field[Vector]) -> Field[Vector]
                    op_call.domains_used.add(Model.CALC)
                    return FieldType_(value_type=Vector())
            
            self._add_type_error(
                f"Curl requires Field[Vector], got {arg_type}",
                expected=FieldType_(value_type=Vector()),
                found=arg_type,
                location=self._node_location(op_call)
            )
            return FieldType_(value_type=Scalar())
        
        # === Time Derivative ===
        elif op == 'd/dt' or op == '∂/∂t':
            op_call.domains_used.add(Model.CALC)
            op_call.domains_used.add(Model.EXEC)
            # d/dt preserves type
            return arg_type
        
        # === Partial Derivative ===
        elif op == '∂':
            op_call.domains_used.add(Model.CALC)
            if isinstance(arg_type, FieldType_):
                value_type = arg_type.value_type
                if isinstance(value_type, Scalar):
                    return FieldType_(value_type=Vector())
                elif isinstance(value_type, Vector):
                    return FieldType_(value_type=Tensor(k=1, l=1))
            return arg_type
        
        # === Integral ===
        elif op == '∫':
            op_call.domains_used.add(Model.CALC)
            if isinstance(arg_type, FieldType_):
                value_type = arg_type.value_type
                if isinstance(value_type, Scalar):
                    # ∫ Field[Scalar] dV -> Scalar
                    return Scalar()
            
            self._add_type_error(
                f"Integral requires Field[Scalar], got {arg_type}",
                expected=FieldType_(value_type=Scalar()),
                found=arg_type,
                location=self._node_location(op_call)
            )
            return Scalar()
        
        # === Trace ===
        elif op == 'trace':
            if isinstance(arg_type, Tensor) and arg_type.k == 1 and arg_type.l == 1:
                return Scalar()
            self._add_type_error(
                f"Trace requires Tensor(1,1), got {arg_type}",
                expected=Tensor(k=1, l=1),
                found=arg_type,
                location=self._node_location(op_call)
            )
            return Scalar()
        
        # === Determinant ===
        elif op == 'det':
            if isinstance(arg_type, Tensor) and arg_type.k == 0 and arg_type.l == 2:
                return Scalar()
            self._add_type_error(
                f"Determinant requires Tensor(0,2), got {arg_type}",
                expected=Tensor(k=0, l=2),
                found=arg_type,
                location=self._node_location(op_call)
            )
            return Scalar()
        
        # Default: return scalar
        return Scalar()
    
    # === Compatibility Checking ===
    
    def check_model_compatibility(self, expr: Expr, models: Set[Model]) -> bool:
        """Check expression is valid in specified models."""
        return self._check_model_compatibility(expr, models)
    
    def _check_model_compatibility(self, expr: Expr, models: Set[Model]) -> bool:
        """Internal model compatibility check."""
        expr_domains = getattr(expr, 'domains_used', set())
        
        # Check if required models are available
        required = expr_domains - models
        if required:
            self._add_error(
                f"Expression requires models not in scope: {required}",
                location=self._node_location(expr)
            )
            return False
        
        return True
    
    def check_regularity(self, expr: Expr, required: SmoothnessClass) -> bool:
        """Check expression has required smoothness."""
        return self._check_regularity(expr, required)
    
    def _check_regularity(self, expr: Expr, required: SmoothnessClass) -> bool:
        """Internal regularity check."""
        expr_regularity = getattr(expr, 'regularity', None)
        
        if expr_regularity is None:
            # No regularity specified, assume C0 (weaker)
            expr_regularity = SmoothnessClass.C0
        
        # Check if expr regularity satisfies required
        if not self._smoothness_satisfies(expr_regularity, required):
            self._add_regularity_error(
                operator=expr.op if isinstance(expr, OpCall) else 'expression',
                required=required.value,
                found=expr_regularity.value,
                location=self._node_location(expr)
            )
            return False
        
        return True
    
    def check_geometry_prereqs(self, expr: Expr, geo_primitives: Dict) -> bool:
        """Check required geometry objects are in scope."""
        return self._check_geometry_prereqs(expr, geo_primitives)
    
    def _check_geometry_prereqs(self, expr: Expr, geo_primitives: Dict) -> bool:
        """Internal geometry prerequisite check."""
        if geo_primitives.get('metric'):
            if not self.current_scope.metrics_in_scope:
                self._add_geometry_error(
                    operator=expr.op if isinstance(expr, OpCall) else 'expression',
                    required="Metric in scope",
                    location=self._node_location(expr)
                )
                return False
        
        if geo_primitives.get('connection'):
            if not self.current_scope.connections_in_scope:
                self._add_geometry_error(
                    operator=expr.op if isinstance(expr, OpCall) else 'expression',
                    required="Connection in scope",
                    location=self._node_location(expr)
                )
                return False
        
        return True
    
    # === Type Utilities ===
    
    def _types_compatible(self, a: SemanticType, b: SemanticType) -> bool:
        """Check if two types are compatible for assignment/comparison."""
        if a == b:
            return True
        # Scalar is compatible with components
        if isinstance(a, Scalar) and isinstance(b, (Scalar, Vector, Tensor)):
            return True
        if isinstance(b, Scalar) and isinstance(a, (Scalar, Vector, Tensor)):
            return True
        # Field types match if value types match
        if isinstance(a, FieldType_) and isinstance(b, FieldType_):
            return self._types_compatible(a.value_type, b.value_type)
        return False
    
    def _smoothness_satisfies(actual: SmoothnessClass, required: SmoothnessClass) -> bool:
        """Check if actual smoothness satisfies required."""
        order = {
            SmoothnessClass.C0: 0,
            SmoothnessClass.C1: 1,
            SmoothnessClass.C2: 2,
            SmoothnessClass.L2: 1,
            SmoothnessClass.H1: 2,
            SmoothnessClass.H2: 3
        }
        actual_order = order.get(actual, 0)
        required_order = order.get(required, 0)
        return actual_order >= required_order
    
    def _semantic_to_ast_type(self, semantic: SemanticType) -> ASTSemanticType:
        """Convert semantic type to AST semantic type enum."""
        if isinstance(semantic, Scalar):
            return ASTSemanticType.SCALAR
        elif isinstance(semantic, Vector):
            return ASTSemanticType.VECTOR
        elif isinstance(semantic, Tensor):
            return ASTSemanticType.TENSOR
        elif isinstance(semantic, FieldType_):
            return ASTSemanticType.FIELD
        elif isinstance(semantic, Form):
            return ASTSemanticType.FORM
        elif isinstance(semantic, (Operator, FuncType)):
            return ASTSemanticType.OPERATOR
        elif isinstance(semantic, Manifold):
            return ASTSemanticType.MANIFOLD
        elif isinstance(semantic, Metric):
            return ASTSemanticType.METRIC
        elif isinstance(semantic, Connection):
            return ASTSemanticType.CONNECTION
        elif isinstance(semantic, LieAlgebra):
            return ASTSemanticType.LIE_ALGEBRA
        else:
            return ASTSemanticType.SCALAR
    
    # === Parsing Utilities ===
    
    def _parse_smoothness(self, value: str) -> Optional[SmoothnessClass]:
        """Parse smoothness class from string."""
        if not value:
            return None
        try:
            return SmoothnessClass(value.upper())
        except ValueError:
            return None
    
    def _parse_models(self, value: str) -> Set[Model]:
        """Parse model set from comma-separated string."""
        models = set()
        for part in value.split(','):
            part = part.strip().upper()
            try:
                models.add(Model(part))
            except ValueError:
                pass
        return models
    
    def _parse_invariants(self, value: str) -> List[str]:
        """Parse invariant list from comma-separated string."""
        return [inv.strip() for inv in value.split(',') if inv.strip()]
    
    def _parse_dimension(self, value: str) -> Optional[Dimension]:
        """Parse dimension from string."""
        if not value:
            return None
        try:
            return Dimension(value.lower())
        except ValueError:
            return None
    
    def _parse_time_mode(self, value: str) -> TimeMode:
        """Parse time mode from string."""
        value = value.lower()
        if value == 'audit':
            return TimeMode.AUDIT
        elif value == 'both':
            return TimeMode.BOTH
        return TimeMode.PHYSICAL
    
    def _parse_effect(self, value: str) -> Effect:
        """Parse effect from string."""
        value = value.lower()
        if value == 'read_state':
            return Effect.READ_STATE
        elif value == 'write_state':
            return Effect.WRITE_STATE
        elif value == 'nonlocal':
            return Effect.NONLOCAL
        elif value == 'gauge_change':
            return Effect.GAUGE_CHANGE
        return Effect.READ_STATE
    
    # === Error Reporting ===
    
    def _add_type_error(self, message: str, 
                        expected: Optional[SemanticType] = None,
                        found: Optional[SemanticType] = None,
                        location: Optional[str] = None) -> None:
        """Add a type error."""
        self.errors.append(TypeError(
            message=message,
            expected=expected,
            found=found,
            location=location
        ))
    
    def _add_regularity_error(self, operator: str, required: str,
                               found: Optional[str] = None,
                               location: Optional[str] = None) -> None:
        """Add a regularity error."""
        self.errors.append(RegularityError(
            operator=operator,
            required=required,
            found=found,
            location=location
        ))
    
    def _add_geometry_error(self, operator: str, required: str,
                            location: Optional[str] = None) -> None:
        """Add a geometry prerequisite error."""
        self.errors.append(GeometryPrerequisiteError(
            operator=operator,
            required=required,
            location=location
        ))
    
    def _add_error(self, message: str, location: Optional[str] = None) -> None:
        """Add a general error."""
        self.errors.append(TypeError(message=message, location=location))
    
    def _node_location(self, node: Node) -> str:
        """Get location string for a node."""
        return f"line {node.start}"
    
    def get_errors(self) -> List[TypeError]:
        """Get all collected errors."""
        return self.errors
    
    def has_errors(self) -> bool:
        """Check if any errors were collected."""
        return len(self.errors) > 0


# === Convenience Functions ===

def type_check_program(program: Program, 
                       invariant_registry: Optional[Dict] = None) -> Program:
    """
    Type-check a complete program.
    
    Args:
        program: The parsed program AST
        invariant_registry: Optional registry of valid invariant IDs
        
    Returns:
        Annotated program with type information
        
    Raises:
        TypeError: If type checking fails
    """
    checker = TypeChecker(invariant_registry=invariant_registry)
    return checker.check_program(program)


def type_check_expression(expr: Expr, 
                          symbols: Optional[Dict[str, SymbolInfo]] = None,
                          models: Optional[Set[Model]] = None) -> Expr:
    """
    Type-check a single expression.
    
    Args:
        expr: The expression to type-check
        symbols: Optional symbol table
        models: Optional set of models in scope
        
    Returns:
        Annotated expression with type information
    """
    checker = TypeChecker()
    if symbols:
        for name, info in symbols.items():
            checker.current_scope.add_symbol(name, info)
    if models:
        checker.current_scope.models_in_scope = models
    
    return checker.check_expression(expr)
