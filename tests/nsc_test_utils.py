"""
NSC-M3L Integration Test Utilities

Shared utilities for testing the complete NSC compilation pipeline.
Provides convenience functions for parsing, type checking, and full compilation.
"""

import pytest
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Set
from dataclasses import dataclass, field

from src.nsc.lex import tokenize
from src.nsc.parse import parse_program, parse_string, ParseError
from src.nsc.type_checker import TypeChecker, type_check_program, SymbolInfo, Scope
from src.nsc.types import (
    Scalar, Vector, Tensor, Field as FieldType, Form, Metric, Manifold, LieAlgebra,
    SemanticType, Dimension, TimeMode, Effect, Tag,
    TypeError, GeometryPrerequisiteError, RegularityError, ModelCompatibilityError
)
from src.nsc.ast import (
    Program, Statement, Expr, Decl, Equation, Functional, Constraint, Directive,
    Atom, Group, BinaryOp, OpCall,
    ScalarType, VectorType, TensorType, FieldType as ASTFieldType, FormType, OperatorType,
    ManifoldType, MetricType, LieAlgebraType, ConnectionType,
    ModelSelector, InvariantList, GateSpec, TargetList,
    Model, DirectiveType, Binding, Meta, SmoothnessClass
)
from src.nsc.disc_types import (
    Grid, UnstructuredGrid, FEMSpace, FDSpace, LatticeSpace,
    Stencil, QuadratureRule, DiscreteField, DiscreteOperator,
    StencilType, BoundaryConditionType, FEMElementType, LatticeType,
    StabilityInfo
)
from src.nsc.disc_lower import DiscreteLowerer, LoweringContext, lower_to_fd
from src.nsc.quadrature import gauss_legendre_1, gauss_legendre_2, gauss_legendre_3


# Type variable for generic result types
T = TypeVar('T')


@dataclass
class CompilationResult:
    """Result of a full compilation."""
    source: str
    tokens: List[Tuple[str, str]]
    ast: Optional[Program]
    typechecked: Optional[Program]
    errors: List[str]
    warnings: List[str]
    models: Set[Model] = field(default_factory=set)
    disc_output: Optional[Dict] = None
    stability_info: Optional[StabilityInfo] = None
    symbols: Dict[str, SymbolInfo] = field(default_factory=dict)
    
    @property
    def success(self) -> bool:
        """Check if compilation succeeded."""
        return len(self.errors) == 0 and self.ast is not None
    
    @property
    def has_ast(self) -> bool:
        """Check if AST was produced."""
        return self.ast is not None
    
    @property
    def is_typechecked(self) -> bool:
        """Check if type checking passed."""
        return self.typechecked is not None and len(self.errors) == 0


@dataclass
class ParseResult:
    """Result of parsing operation."""
    source: str
    tokens: List[Tuple[str, str]]
    ast: Optional[Program]
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        """Check if parsing succeeded."""
        return self.ast is not None and self.error is None
    
    @property
    def statement_count(self) -> int:
        """Get number of statements."""
        return len(self.ast.statements) if self.ast else 0


@dataclass
class TypeCheckResult:
    """Result of type checking operation."""
    program: Program
    symbols: Dict[str, SymbolInfo]
    errors: List[str]
    warnings: List[str]
    scope: Scope
    
    @property
    def success(self) -> bool:
        """Check if type checking succeeded."""
        return len(self.errors) == 0
    
    def get_type(self, ident: str) -> Optional[SemanticType]:
        """Get type of identifier."""
        if ident in self.symbols:
            return self.symbols[ident].declared_type
        return None


# === Tokenization Utilities ===

def tokenize_source(source: str) -> List[Tuple[str, str]]:
    """Tokenize NSC source code.
    
    Args:
        source: NSC source code string
        
    Returns:
        List of (token_type, token_value) tuples
    """
    return tokenize(source)


def tokenize_safe(source: str) -> Tuple[bool, List[Tuple[str, str]], Optional[str]]:
    """Tokenize source with error handling.
    
    Args:
        source: NSC source code string
        
    Returns:
        Tuple of (success, tokens, error_message)
    """
    try:
        tokens = tokenize(source)
        return True, tokens, None
    except Exception as e:
        return False, [], str(e)


# === Parsing Utilities ===

def parse_source(source: str) -> Program:
    """Parse NSC source code to AST.
    
    Args:
        source: NSC source code string
        
    Returns:
        Program AST node
        
    Raises:
        ParseError: If parsing fails
    """
    return parse_string(source)


def parse_safe(source: str) -> ParseResult:
    """Parse source with error handling.
    
    Args:
        source: NSC source code string
        
    Returns:
        ParseResult with success status and any error
    """
    try:
        tokens = tokenize(source)
        ast = parse_program(tokens)
        return ParseResult(source=source, tokens=tokens, ast=ast, error=None)
    except ParseError as e:
        return ParseResult(
            source=source,
            tokens=[],
            ast=None,
            error=str(e)
        )
    except Exception as e:
        return ParseResult(
            source=source,
            tokens=[],
            ast=None,
            error=f"Unexpected error: {e}"
        )


def parse_with_positions(source: str) -> ParseResult:
    """Parse source with position tracking.
    
    Args:
        source: NSC source code string
        
    Returns:
        ParseResult with AST and position info
    """
    return parse_safe(source)


# === Type Checking Utilities ===

def type_check_program(program: Program, 
                       invariant_registry: Optional[Dict[str, Dict]] = None) -> TypeCheckResult:
    """Type check a parsed program.
    
    Args:
        program: Program AST node
        invariant_registry: Optional registry of known invariants
        
    Returns:
        TypeCheckResult with symbols, errors, and scope
    """
    registry = invariant_registry or {
        "N:INV.gr.hamiltonian_constraint": {"description": "GR Hamiltonian constraint"},
        "N:INV.gr.momentum_constraint": {"description": "GR momentum constraint"},
        "N:INV.ns.div_free": {"description": "Navier-Stokes divergence-free constraint"},
        "N:INV.ns.energy_nonincreasing": {"description": "Navier-Stokes energy dissipation"},
        "N:INV.pde.energy_nonincreasing": {"description": "PDE energy estimate"},
        "N:INV.ym.gauss_law": {"description": "Yang-Mills Gauss law constraint"},
    }
    
    checker = TypeChecker(invariant_registry=registry)
    checked_program = checker.check_program(program)
    
    return TypeCheckResult(
        program=checked_program,
        symbols=checker.global_scope.symbols,
        errors=checker.errors,
        warnings=checker.warnings,
        scope=checker.global_scope
    )


def type_check_safe(source: str) -> Tuple[bool, Optional[Program], List[str]]:
    """Type check source with error handling.
    
    Args:
        source: NSC source code string
        
    Returns:
        Tuple of (success, typechecked_program, errors)
    """
    try:
        parse_result = parse_safe(source)
        if not parse_result.success:
            return False, None, [parse_result.error or "Parse failed"]
        
        type_result = type_check_program(parse_result.ast)
        return type_result.success, type_result.program, type_result.errors
    except Exception as e:
        return False, None, [str(e)]


def parse_and_typecheck(source: str) -> Program:
    """Parse and type check in one step.
    
    Args:
        source: NSC source code string
        
    Returns:
        Type-checked Program AST node
        
    Raises:
        TypeError: If type checking fails
    """
    program = parse_source(source)
    result = type_check_program(program)
    if result.errors:
        raise TypeError(f"Type checking failed: {result.errors}")
    return result.program


# === Full Pipeline Utilities ===

def compile_nsc_source(source: str, 
                       target_models: Optional[Set[Model]] = None) -> CompilationResult:
    """Full compilation pipeline: tokenize → parse → type check → lower.
    
    Args:
        source: NSC source code string
        target_models: Optional set of target models for lowering
        
    Returns:
        CompilationResult with all intermediate and final results
    """
    result = CompilationResult(
        source=source,
        tokens=[],
        ast=None,
        typechecked=None,
        errors=[],
        warnings=[]
    )
    
    # Tokenize
    try:
        result.tokens = tokenize(source)
    except Exception as e:
        result.errors.append(f"Lexing error: {e}")
        return result
    
    # Parse
    try:
        result.ast = parse_program(result.tokens)
    except ParseError as e:
        result.errors.append(f"Parse error: {e}")
        return result
    except Exception as e:
        result.errors.append(f"Unexpected parse error: {e}")
        return result
    
    # Type check
    type_result = type_check_program(result.ast)
    result.errors.extend(type_result.errors)
    result.warnings.extend(type_result.warnings)
    result.symbols = type_result.symbols
    
    if type_result.success:
        result.typechecked = type_result.program
    else:
        # Still return AST even if type checking failed
        result.typechecked = result.ast
    
    # Extract models from directives
    if result.ast:
        for stmt in result.ast.statements:
            if isinstance(stmt, Directive):
                if stmt.directive_type == DirectiveType.MODEL and stmt.model_selector:
                    result.models.update(stmt.model_selector.models)
    
    # Lower to DISC if requested
    if target_models and Model.DISC in target_models and result.typechecked:
        try:
            grid = Grid(shape=(50, 50), spacing=(0.1, 0.1))
            quad = gauss_legendre_2()
            context = LoweringContext(grid=grid, quadrature=quad)
            lowerer = DiscreteLowerer(context)
            result.disc_output = lowerer.lower_to_disc(result.typechecked)
            result.stability_info = result.disc_output.get("stability_info")
        except Exception as e:
            result.errors.append(f"DISC lowering error: {e}")
    
    return result


def compile_nsc_file(filepath: str) -> CompilationResult:
    """Compile NSC source from file.
    
    Args:
        filepath: Path to NSC source file
        
    Returns:
        CompilationResult
    """
    with open(filepath, 'r') as f:
        source = f.read()
    return compile_nsc_source(source)


# === Grid and DISC Utilities ===

def create_test_grid(dim: int = 2, shape: Tuple[int, ...] = None) -> Grid:
    """Create a test grid.
    
    Args:
        dim: Spatial dimension
        shape: Grid shape (defaults to (50,)*dim)
        
    Returns:
        Grid instance
    """
    if shape is None:
        shape = tuple(50 for _ in range(dim))
    spacing = tuple(0.1 for _ in range(dim))
    return Grid(dim=dim, shape=shape, spacing=spacing)


def create_test_quadrature(degree: int = 2) -> QuadratureRule:
    """Create a test quadrature rule.
    
    Args:
        degree: Degree of quadrature (1, 2, or 3)
        
    Returns:
        QuadratureRule instance
    """
    if degree == 1:
        return gauss_legendre_1()
    elif degree == 2:
        return gauss_legendre_2()
    else:
        return gauss_legendre_3()


def lower_to_disc(source: str, 
                  shape: Tuple[int, ...] = None,
                  spacing: float = 0.1) -> Dict:
    """Lower NSC source to DISC representation.
    
    Args:
        source: NSC source code
        shape: Grid shape
        spacing: Grid spacing
        
    Returns:
        DISC representation dictionary
    """
    dim = len(shape) if shape else 2
    grid = create_test_grid(dim, shape)
    quad = create_test_quadrature(2)
    
    context = LoweringContext(grid=grid, quadrature=quad)
    lowerer = DiscreteLowerer(context)
    
    program = parse_source(source)
    type_result = type_check_program(program)
    
    if not type_result.success:
        raise TypeError(f"Type checking failed: {type_result.errors}")
    
    return lowerer.lower_to_disc(type_result.program)


# === Assertion Helpers ===

def assert_no_errors(result: CompilationResult) -> None:
    """Assert compilation result has no errors.
    
    Args:
        result: CompilationResult to check
        
    Raises:
        AssertionError: If there are errors
    """
    assert result.errors == [], f"Expected no errors, got: {result.errors}"


def assert_no_warnings(result: CompilationResult) -> None:
    """Assert compilation result has no warnings.
    
    Args:
        result: CompilationResult to check
        
    Raises:
        AssertionError: If there are warnings
    """
    assert result.warnings == [], f"Expected no warnings, got: {result.warnings}"


def assert_ast_not_empty(result: CompilationResult) -> None:
    """Assert AST was produced.
    
    Args:
        result: CompilationResult to check
        
    Raises:
        AssertionError: If AST is None or empty
    """
    assert result.ast is not None, "Expected AST to be produced"
    assert len(result.ast.statements) > 0, "Expected non-empty AST"


def assert_type(result: TypeCheckResult, expr_id: str, expected_type: Type) -> None:
    """Assert expression has expected type.
    
    Args:
        result: TypeCheckResult
        expr_id: Expression identifier
        expected_type: Expected semantic type
        
    Raises:
        AssertionError: If type doesn't match
    """
    actual_type = result.get_type(expr_id)
    assert actual_type == expected_type, \
        f"Expected {expr_id} to have type {expected_type}, got {actual_type}"


def assert_contains_model(result: CompilationResult, model: Model) -> None:
    """Assert result contains specified model.
    
    Args:
        result: CompilationResult
        model: Model to check for
        
    Raises:
        AssertionError: If model not found
    """
    assert model in result.models, \
        f"Expected model {model} in {result.models}"


def assert_disc_output(result: CompilationResult) -> None:
    """Assert DISC output was produced.
    
    Args:
        result: CompilationResult
        
    Raises:
        AssertionError: If DISC output is None
    """
    assert result.disc_output is not None, "Expected DISC output to be produced"
    assert "operators" in result.disc_output, "Expected operators in DISC output"


# === Sample Programs ===

def simple_scalar_program() -> str:
    """Return a simple scalar declaration program."""
    return """
    x :: Scalar;
    y :: Scalar;
    z = x + y;
    """


def simple_field_program() -> str:
    """Return a simple field declaration program."""
    return """
    @model(CALC);
    u :: Field[Scalar];
    v :: Field[Scalar];
    w = u + v;
    """


def laplacian_program() -> str:
    """Return a Laplacian equation program."""
    return """
    @model(GEO, CALC, DISC);
    u :: Field[Scalar];
    f :: Field[Scalar];
    
    Eq := Δ(u) = f;
    
    ⇒ (DISC);
    """


def gr_hamiltonian_program() -> str:
    """Return a GR Hamiltonian constraint program."""
    return """
    @model(GEO, CALC, LEDGER);
    
    M :: Manifold(3+1, lorentzian);
    g :: Metric on M;
    K :: Field[Tensor(0,2)] on (M, t);
    
    H := R(g) + tr(K^2) - K_ij*K^ij = 0;
    
    @inv(N:INV.gr.hamiltonian_constraint);
    ⇒ (LEDGER, CALC, GEO);
    """


def navier_stokes_program() -> str:
    """Return a Navier-Stokes program."""
    return """
    @model(CALC, LEDGER, EXEC);
    
    u :: Field[Vector] on (M, t);
    p :: Field[Scalar] on (M, t);
    ν :: Scalar;
    
    Eq1 := d/dt(u) + (u·∇)(u) + ∇p - ν*Δ(u) = 0;
    Eq2 := div(u) = 0;
    
    @inv(N:INV.ns.div_free, N:INV.ns.energy_nonincreasing);
    ⇒ (LEDGER, CALC, EXEC);
    """


def wave_equation_program() -> str:
    """Return a wave equation program."""
    return """
    @model(GEO, CALC, DISC);
    
    u :: Field[Scalar] on M;
    
    WaveEq := d²/dt²(u) - Δ(u) = 0;
    
    @inv(N:INV.pde.energy_nonincreasing);
    ⇒ (DISC);
    """


# === Exception Mapping ===

LEX_ERRORS = {
    'unexpected_character': 'Unexpected character in input',
    'unterminated_string': 'Unterminated string',
    'invalid_token': 'Invalid token',
}

PARSE_ERRORS = {
    'unexpected_token': 'Unexpected token',
    'unclosed_group': 'Unclosed group',
    'expected_token': 'Expected token',
    'trailing_input': 'Trailing input',
}

TYPE_ERRORS = {
    'type_mismatch': 'Type mismatch',
    'unknown_symbol': 'Unknown symbol',
    'geometry_prerequisite': 'Geometry prerequisite not satisfied',
    'model_incompatible': 'Model compatibility error',
    'regularity_violation': 'Regularity constraint violated',
}


def expect_lex_error(source: str, error_type: str = None) -> None:
    """Assert that source produces a lexer error.
    
    Args:
        source: NSC source code
        error_type: Optional specific error type to check
        
    Raises:
        AssertionError: If no lexer error occurs
    """
    try:
        tokenize(source)
        pytest.fail(f"Expected lexer error for: {source}")
    except Exception as e:
        if error_type:
            assert error_type.lower() in str(e).lower(), \
                f"Expected {error_type} error, got: {e}"


def expect_parse_error(source: str, error_type: str = None) -> None:
    """Assert that source produces a parse error.
    
    Args:
        source: NSC source code
        error_type: Optional specific error type to check
        
    Raises:
        AssertionError: If no parse error occurs
    """
    try:
        parse_source(source)
        pytest.fail(f"Expected parse error for: {source}")
    except ParseError as e:
        if error_type:
            assert error_type.lower() in str(e).lower(), \
                f"Expected {error_type} error, got: {e}"


def expect_type_error(source: str, error_type: str = None) -> None:
    """Assert that source produces a type error.
    
    Args:
        source: NSC source code
        error_type: Optional specific error type to check
        
    Raises:
        AssertionError: If no type error occurs
    """
    try:
        parse_and_typecheck(source)
        pytest.fail(f"Expected type error for: {source}")
    except TypeError as e:
        if error_type:
            assert error_type.lower() in str(e).lower(), \
                f"Expected {error_type} error, got: {e}"
