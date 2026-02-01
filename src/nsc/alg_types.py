"""
NSC-M3L ALG Model: Algebraic Types

Defines algebraic types per specifications/nsc_m3l_v1.md section 4.1.
Provides ring/field elements, matrices, Lie algebra types, tensors, and polynomials.
"""

from dataclasses import dataclass, field
from typing import Union, List, Optional, Tuple, Set, Dict
from enum import Enum
from abc import ABC, abstractmethod


class RingType(Enum):
    """Classification of ring types."""
    INTEGERS = "integers"  # ℤ
    RATIONALS = "rationals"  # ℚ
    REALS = "reals"  # ℝ
    COMPLEX = "complex"  # ℂ
    POLYNOMIAL = "polynomial"  # ℝ[x₁,...,xₙ]
    MATRIX = "matrix"  # Matrix ring


class FieldType(Enum):
    """Classification of field types."""
    RATIONALS = "rationals"  # ℚ
    REALS = "reals"  # ℝ
    COMPLEX = "complex"  # ℂ
    RATIONAL_FUNCTION = "rational_function"  # ℝ(x₁,...,xₙ)


# === Base Algebraic Expression ===

class AlgebraicExpr(ABC):
    """Base class for all algebraic expressions."""
    
    @abstractmethod
    def __str__(self) -> str:
        pass
    
    @abstractmethod
    def __eq__(self, other) -> bool:
        pass
    
    @abstractmethod
    def __hash__(self) -> int:
        pass
    
    @abstractmethod
    def simplify(self) -> 'AlgebraicExpr':
        """Simplify to normal form."""
        pass


# === Ring/Field Elements ===

@dataclass
class RingElement(AlgebraicExpr):
    """Element of a ring."""
    value: Union[int, float, complex, 'Polynomial', 'Matrix']
    base_ring: RingType = RingType.REALS
    
    def __str__(self) -> str:
        if isinstance(self.value, (int, float)):
            return str(self.value)
        return f"R({self.value})"
    
    def __eq__(self, other) -> bool:
        return isinstance(other, RingElement) and self.value == other.value
    
    def __hash__(self) -> int:
        return hash((self.value, self.base_ring))
    
    @property
    def ring(self) -> RingType:
        """Get the base ring (for compatibility)."""
        return self.base_ring
    
    def simplify(self) -> 'AlgebraicExpr':
        return self


@dataclass
class FieldElement(AlgebraicExpr):
    """Element of a field."""
    value: Union[int, float, complex, 'RationalFunction']
    base_field: FieldType = FieldType.REALS
    
    def __str__(self) -> str:
        if isinstance(self.value, (int, float, complex)):
            return str(self.value)
        return f"F({self.value})"
    
    def __eq__(self, other) -> bool:
        return isinstance(other, FieldElement) and self.value == other.value
    
    def __hash__(self) -> int:
        return hash((self.value, self.base_field))
    
    def simplify(self) -> 'AlgebraicExpr':
        return self


# === Matrix Types ===

@dataclass
class Matrix(AlgebraicExpr):
    """Matrix over a ring."""
    data: List[List[RingElement]]  # 2D array of ring elements
    rows: int = field(init=False)
    cols: int = field(init=False)
    ring: RingType = RingType.REALS
    
    def __post_init__(self):
        self.rows = len(self.data)
        self.cols = len(self.data[0]) if self.data else 0
    
    def __str__(self) -> str:
        if self.rows == 1:
            return f"[{' '.join(str(x) for x in self.data[0])}]"
        elif self.rows == 2:
            lines = []
            for row in self.data:
                lines.append(f"[{' '.join(str(x) for x in row)}]")
            return "[\n  " + "\n  ".join(lines) + "\n]"
        return f"Matrix({self.rows}x{self.cols})"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Matrix):
            return False
        if self.rows != other.rows or self.cols != other.cols:
            return False
        return self.data == other.data
    
    def __hash__(self) -> int:
        return hash((tuple(tuple(row) for row in self.data), self.ring))
    
    def simplify(self) -> 'AlgebraicExpr':
        return self
    
    def is_square(self) -> bool:
        """Check if matrix is square."""
        return self.rows == self.cols


@dataclass
class Vector(AlgebraicExpr):
    """Vector over a ring."""
    components: List[RingElement]
    size: int = field(init=False)
    ring: RingType = RingType.REALS
    
    def __post_init__(self):
        self.size = len(self.components)
    
    def __str__(self) -> str:
        return f"[{' '.join(str(x) for x in self.components)}]"
    
    def __eq__(self, other) -> bool:
        return isinstance(other, Vector) and self.components == other.components
    
    def __hash__(self) -> int:
        return hash((tuple(self.components), self.ring))
    
    def simplify(self) -> 'AlgebraicExpr':
        return self


@dataclass
class LinearMap(AlgebraicExpr):
    """Linear transformation between vector spaces."""
    domain_dim: int
    codomain_dim: int
    matrix: Matrix  # Representation in standard basis
    
    def __str__(self) -> str:
        return f"LinearMap(ℝ^{self.domain_dim} → ℝ^{self.codomain_dim})"
    
    def __eq__(self, other) -> bool:
        return (isinstance(other, LinearMap) and 
                self.domain_dim == other.domain_dim and
                self.codomain_dim == other.codomain_dim and
                self.matrix == other.matrix)
    
    def __hash__(self) -> int:
        return hash((self.domain_dim, self.cod.matrix))
    
    def simplify(self) -> 'AlgebraicExpr':
        return self


@dataclass
class SparseMatrix(AlgebraicExpr):
    """Sparse matrix representation using coordinate storage."""
    entries: Dict[Tuple[int, int], RingElement]  # (i, j) -> value
    rows: int
    cols: int
    ring: RingType = RingType.REALS
    
    def __str__(self) -> str:
        return f"SparseMatrix({self.rows}x{self.cols}, {len(self.entries)} entries)"
    
    def __eq__(self, other) -> bool:
        return (isinstance(other, SparseMatrix) and 
                self.rows == other.rows and 
                self.cols == other.cols and
                self.entries == other.entries)
    
    def __hash__(self) -> int:
        return hash((frozenset(self.entries.items()), self.rows, self.cols))
    
    def simplify(self) -> 'AlgebraicExpr':
        return self


# === Lie Algebra Types ===

@dataclass
class LieAlgebraElement(AlgebraicExpr):
    """Element of a Lie algebra."""
    components: List[RingElement]  # Coefficients in some basis
    algebra: 'LieAlgebraType'  # Reference to parent algebra
    
    def __str__(self) -> str:
        if len(self.components) <= 3:
            return f"[{' '.join(str(x) for x in self.components)}]"
        return f"LieAlgElem(dim={len(self.components)})"
    
    def __eq__(self, other) -> bool:
        return (isinstance(other, LieAlgebraElement) and 
                self.algebra == other.algebra and
                self.components == other.components)
    
    def __hash__(self) -> int:
        return hash((tuple(self.components), self.algebra))
    
    def simplify(self) -> 'AlgebraicExpr':
        return self


@dataclass
class LieAlgebraType:
    """Lie algebra type (reference type, not expression)."""
    name: str  # e.g., "su(2)", "so(3)", "u(1)"
    dimension: int
    structure_constants: Optional[List[List[List[float]]]] = None  # f^k_ij
    killing_form: Optional[Matrix] = None
    
    def __str__(self) -> str:
        return f"LieAlgebra({self.name})"
    
    def __eq__(self, other) -> bool:
        return isinstance(other, LieAlgebraType) and self.name == other.name


@dataclass
class LieGroup(AlgebraicExpr):
    """Lie group (exponential of Lie algebra)."""
    algebra: LieAlgebraType
    
    def __str__(self) -> str:
        return f"LieGroup({self.algebra.name})"
    
    def __eq__(self, other) -> bool:
        return isinstance(other, LieGroup) and self.algebra == other.algebra
    
    def __hash__(self) -> int:
        return hash(self.algebra)
    
    def simplify(self) -> 'AlgebraicExpr':
        return self


# === Tensor Types ===

@dataclass
class Tensor(AlgebraicExpr):
    """Multi-index tensor."""
    components: Dict[Tuple[int, ...], RingElement]  # Multi-index -> value
    shape: Tuple[int, ...]  # Shape of tensor
    rank: int = field(init=False)
    
    def __post_init__(self):
        self.rank = len(self.shape)
    
    def __str__(self) -> str:
        return f"Tensor(shape={self.shape}, rank={self.rank})"
    
    def __eq__(self, other) -> bool:
        return (isinstance(other, Tensor) and 
                self.shape == other.shape and
                self.components == other.components)
    
    def __hash__(self) -> int:
        return hash((frozenset(self.components.items()), self.shape))
    
    def simplify(self) -> 'AlgebraicExpr':
        return self


@dataclass
class WedgeProduct(AlgebraicExpr):
    """Exterior algebra element."""
    components: Dict[Tuple[int, ...], RingElement]  # Antisymmetric indices
    degree: int
    dimension: int  # Ambient dimension
    
    def __str__(self) -> str:
        return f"∧^{self.degree} element"
    
    def __eq__(self, other) -> bool:
        return (isinstance(other, WedgeProduct) and 
                self.degree == other.degree and
                self.dimension == other.dimension and
                self.components == other.components)
    
    def __hash__(self) -> int:
        return hash((frozenset(self.components.items()), self.degree, self.dimension))
    
    def simplify(self) -> 'AlgebraicExpr':
        return self


@dataclass
class SymTensor(AlgebraicExpr):
    """Symmetric tensor."""
    components: Dict[Tuple[int, ...], RingElement]  # Symmetric indices
    degree: int
    dimension: int
    
    def __str__(self) -> str:
        return f"Sym^{self.degree} element"
    
    def __eq__(self, other) -> bool:
        return (isinstance(other, SymTensor) and 
                self.degree == other.degree and
                self.dimension == other.dimension and
                self.components == other.components)
    
    def __hash__(self) -> int:
        return hash((frozenset(self.components.items()), self.degree, self.dimension))
    
    def simplify(self) -> 'AlgebraicExpr':
        return self


# === Polynomial Types ===

@dataclass
class Monomial(AlgebraicExpr):
    """Monomial x1^e1 * x2^e2 * ... * xn^en."""
    variables: List[str]
    exponents: List[int]
    coefficient: RingElement = RingElement(1)
    
    def __post_init__(self):
        assert len(self.variables) == len(self.exponents)
    
    def __str__(self) -> str:
        if not self.variables:
            return str(self.coefficient)
        parts = []
        if self.coefficient != RingElement(1):
            parts.append(str(self.coefficient))
        for v, e in zip(self.variables, self.exponents):
            if e == 1:
                parts.append(v)
            else:
                parts.append(f"{v}^{e}")
        return "".join(parts)
    
    def __eq__(self, other) -> bool:
        return (isinstance(other, Monomial) and 
                self.variables == other.variables and
                self.exponents == other.exponents and
                self.coefficient == other.coefficient)
    
    def __hash__(self) -> int:
        return hash((tuple(self.variables), tuple(self.exponents), self.coefficient))
    
    def simplify(self) -> 'AlgebraicExpr':
        return self
    
    def degree(self) -> int:
        """Total degree of monomial."""
        return sum(self.exponents)


@dataclass
class Polynomial(AlgebraicExpr):
    """Multivariate polynomial."""
    terms: List[Monomial]
    variables: List[str] = field(default_factory=list)  # All variables appearing in polynomial
    
    def __post_init__(self):
        # Collect all variables from terms if not provided
        if not self.variables:
            all_vars = set()
            for term in self.terms:
                all_vars.update(term.variables)
            self.variables = list(all_vars)
    
    def __str__(self) -> str:
        if not self.terms:
            return "0"
        terms_str = [str(t) for t in self.terms]
        return " + ".join(terms_str)
    
    def __eq__(self, other) -> bool:
        return isinstance(other, Polynomial) and self.terms == other.terms
    
    def __hash__(self) -> int:
        return hash((tuple(self.terms), tuple(self.variables)))
    
    def simplify(self) -> 'AlgebraicExpr':
        """Combine like terms."""
        # Group by variables and exponents
        combined: Dict[Tuple[Tuple[str, ...], Tuple[int, ...]], float] = {}
        for term in self.terms:
            key = (tuple(term.variables), tuple(term.exponents))
            coeff_val = term.coefficient.value if hasattr(term.coefficient, 'value') else term.coefficient
            combined[key] = combined.get(key, 0.0) + coeff_val
        
        # Rebuild with non-zero coefficients
        new_terms = []
        for (vars_, exps), coeff in combined.items():
            if abs(coeff) > 1e-10:  # Non-zero
                monomial = Monomial(
                    variables=list(vars_),
                    exponents=list(exps),
                    coefficient=RingElement(coeff)
                )
                new_terms.append(monomial)
        
        return Polynomial(terms=new_terms, variables=self.variables)
    
    def degree(self) -> int:
        """Total degree of polynomial."""
        return max((t.degree() for t in self.terms), default=0)


@dataclass
class RationalFunction(AlgebraicExpr):
    """Ratio of polynomials."""
    numerator: Polynomial
    denominator: Polynomial
    
    def __str__(self) -> str:
        return f"({self.numerator}) / ({self.denominator})"
    
    def __eq__(self, other) -> bool:
        return (isinstance(other, RationalFunction) and 
                self.numerator == other.numerator and
                self.denominator == other.denominator)
    
    def __hash__(self) -> int:
        return hash((self.numerator, self.denominator))
    
    def simplify(self) -> 'AlgebraicExpr':
        # Would compute GCD and simplify
        return self


# === Algebraic Operations ===

@dataclass
class Sum(AlgebraicExpr):
    """Sum of expressions."""
    terms: List[AlgebraicExpr]
    
    def __str__(self) -> str:
        return " + ".join(str(t) for t in self.terms)
    
    def __eq__(self, other) -> bool:
        return isinstance(other, Sum) and self.terms == other.terms
    
    def __hash__(self) -> int:
        return hash((tuple(self.terms)))
    
    def simplify(self) -> 'AlgebraicExpr':
        return self


@dataclass
class Product(AlgebraicExpr):
    """Product of expressions."""
    factors: List[AlgebraicExpr]
    
    def __str__(self) -> str:
        return " * ".join(str(f) for f in self.factors)
    
    def __eq__(self, other) -> bool:
        return isinstance(other, Product) and self.factors == other.factors
    
    def __hash__(self) -> int:
        return hash((tuple(self.factors)))
    
    def simplify(self) -> 'AlgebraicExpr':
        return self


@dataclass
class Power(AlgebraicExpr):
    """Power expression."""
    base: AlgebraicExpr
    exponent: int
    
    def __str__(self) -> str:
        return f"({self.base})^{self.exponent}"
    
    def __eq__(self, other) -> bool:
        return isinstance(other, Power) and self.base == other.base and self.exponent == other.exponent
    
    def __hash__(self) -> int:
        return hash((self.base, self.exponent))
    
    def simplify(self) -> 'AlgebraicExpr':
        if self.exponent == 0:
            return RingElement(1)
        if self.exponent == 1:
            return self.base
        return self


@dataclass
class Commutator(AlgebraicExpr):
    """Commutator [a, b] = ab - ba."""
    a: AlgebraicExpr
    b: AlgebraicExpr
    
    def __str__(self) -> str:
        return f"[{self.a}, {self.b}]"
    
    def __eq__(self, other) -> bool:
        return isinstance(other, Commutator) and self.a == other.a and self.b == other.b
    
    def __hash__(self) -> int:
        return hash((self.a, self.b))
    
    def simplify(self) -> 'AlgebraicExpr':
        return self


@dataclass
class Bracket(AlgebraicExpr):
    """General bracket operation (Lie, Poisson, etc.)."""
    a: AlgebraicExpr
    b: AlgebraicExpr
    bracket_type: str = "lie"  # "lie", "poisson", etc.
    
    def __str__(self) -> str:
        return f"{{{self.a}, {self.b}}}_{self.bracket_type}"
    
    def __eq__(self, other) -> bool:
        return (isinstance(other, Bracket) and 
                self.a == other.a and 
                self.b == other.b and
                self.bracket_type == other.bracket_type)
    
    def __hash__(self) -> int:
        return hash((self.a, self.b, self.bracket_type))
    
    def simplify(self) -> 'AlgebraicExpr':
        return self


# === Union Type for All Algebraic Expressions ===

AlgExpr = Union[
    RingElement, FieldElement, Matrix, Vector, LinearMap, SparseMatrix,
    LieAlgebraElement, LieGroup,
    Tensor, WedgeProduct, SymTensor,
    Polynomial, RationalFunction, Monomial,
    Sum, Product, Power, Commutator, Bracket
]


# === Type Utilities ===

def get_alg_type_name(expr: AlgebraicExpr) -> str:
    """Get the type name of an algebraic expression."""
    return type(expr).__name__


def is_scalar(expr: AlgebraicExpr) -> bool:
    """Check if expression is a scalar ring element."""
    return isinstance(expr, (RingElement, FieldElement)) and not isinstance(expr, (Matrix, Vector))


def is_matrix(expr: AlgebraicExpr) -> bool:
    """Check if expression is a matrix."""
    return isinstance(expr, Matrix)


def is_polynomial(expr: AlgebraicExpr) -> bool:
    """Check if expression is a polynomial."""
    return isinstance(expr, Polynomial)


def is_lie_algebra_element(expr: AlgebraicExpr) -> bool:
    """Check if expression is a Lie algebra element."""
    return isinstance(expr, LieAlgebraElement)


def is_tensor(expr: AlgebraicExpr) -> bool:
    """Check if expression is a tensor."""
    return isinstance(expr, (Tensor, WedgeProduct, SymTensor))
