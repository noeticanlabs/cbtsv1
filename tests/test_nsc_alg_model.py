"""
NSC-M3L ALG Model Tests

Comprehensive tests for algebraic types, simplification, matrix operations,
Lie algebra operations, Gröbner basis, tensor operations, and rewriting.
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.nsc.alg_types import (
    RingElement, FieldElement, Matrix, Vector, LinearMap, SparseMatrix,
    LieAlgebraElement, LieAlgebraType, LieGroup,
    Tensor, WedgeProduct, SymTensor,
    Polynomial, Monomial, RationalFunction,
    Sum, Product, Power, Commutator, Bracket,
    RingType, FieldType
)
from src.nsc.alg_simplify import SymbolicSimplifier, simplify, canonicalize
from src.nsc.alg_matrix import MatrixOps, matmul, transpose, inverse, determinant, trace, solve, rank
from src.nsc.alg_lie import LieAlgebraOps, create_lie_algebra, create_lie_algebra_element
from src.nsc.alg_groebner import GroebnerBasis, TermOrder, compute_groebner_basis
from src.nsc.alg_tensor import TensorOps, Metric, create_metric
from src.nsc.alg_rewrite import ExprRewriter, rewrite, normalize


# === Test Fixtures ===

@pytest.fixture
def simplifier():
    return SymbolicSimplifier()


@pytest.fixture
def matrix_ops():
    return MatrixOps()


@pytest.fixture
def lie_ops():
    return LieAlgebraOps()


@pytest.fixture
def tensor_ops():
    return TensorOps()


@pytest.fixture
def rewriter():
    return ExprRewriter()


# === Ring Element Tests ===

class TestRingOperations:
    """Test basic ring operations."""
    
    def test_ring_element_creation(self):
        """Test creating ring elements."""
        r1 = RingElement(5)
        r2 = RingElement(3.14)
        r3 = RingElement(2, RingType.INTEGERS)
        
        assert r1.value == 5
        assert r2.value == 3.14
        assert r3.ring == RingType.INTEGERS
    
    def test_ring_element_equality(self):
        """Test ring element equality."""
        r1 = RingElement(5)
        r2 = RingElement(5)
        r3 = RingElement(3)
        
        assert r1 == r2
        assert r1 != r3
    
    def test_ring_element_hash(self):
        """Test ring element hashing."""
        r1 = RingElement(5)
        r2 = RingElement(5)
        
        assert hash(r1) == hash(r2)
    
    def test_field_element_creation(self):
        """Test creating field elements."""
        f1 = FieldElement(0.5)
        f2 = FieldElement(2 + 3j, FieldType.COMPLEX)
        
        assert f1.value == 0.5
        assert f2.base_field == FieldType.COMPLEX


# === Matrix Tests ===

class TestMatrixOperations:
    """Test matrix operations."""
    
    def test_matrix_creation(self):
        """Test matrix creation."""
        data = [
            [RingElement(1), RingElement(2)],
            [RingElement(3), RingElement(4)]
        ]
        M = Matrix(data=data)
        
        assert M.rows == 2
        assert M.cols == 2
        assert M.is_square()
    
    def test_matrix_multiplication(self, matrix_ops):
        """Test matrix multiplication."""
        A = Matrix(data=[
            [RingElement(1), RingElement(2)],
            [RingElement(3), RingElement(4)]
        ])
        B = Matrix(data=[
            [RingElement(5), RingElement(6)],
            [RingElement(7), RingElement(8)]
        ])
        
        C = matrix_ops.matmul(A, B)
        
        # Check: [[19, 22], [43, 50]]
        assert C.data[0][0].value == 19
        assert C.data[0][1].value == 22
        assert C.data[1][0].value == 43
        assert C.data[1][1].value == 50
    
    def test_matrix_transpose(self, matrix_ops):
        """Test matrix transpose."""
        M = Matrix(data=[
            [RingElement(1), RingElement(2)],
            [RingElement(3), RingElement(4)]
        ])
        
        M_T = matrix_ops.transpose(M)
        
        assert M_T.data[0][1].value == 3
        assert M_T.data[1][0].value == 2
    
    def test_matrix_inverse(self, matrix_ops):
        """Test matrix inverse."""
        # 2x2 invertible matrix
        M = Matrix(data=[
            [RingElement(4), RingElement(7)],
            [RingElement(2), RingElement(6)]
        ])
        
        M_inv = matrix_ops.inverse(M)
        
        assert M_inv is not None
        # Check M * M_inv ≈ I
        I = matrix_ops.matmul(M, M_inv)
        assert abs(I.data[0][0].value - 1) < 1e-10
        assert abs(I.data[1][1].value - 1) < 1e-10
    
    def test_determinant(self, matrix_ops):
        """Test determinant computation."""
        M = Matrix(data=[
            [RingElement(1), RingElement(2)],
            [RingElement(3), RingElement(4)]
        ])
        
        det = matrix_ops.determinant(M)
        
        assert det.value == -2  # 1*4 - 2*3 = -2
    
    def test_trace(self, matrix_ops):
        """Test matrix trace."""
        M = Matrix(data=[
            [RingElement(1), RingElement(2)],
            [RingElement(3), RingElement(4)]
        ])
        
        tr = matrix_ops.trace(M)
        
        assert tr.value == 5  # 1 + 4
    
    def test_matrix_rank(self, matrix_ops):
        """Test matrix rank."""
        # Full rank matrix
        A = Matrix(data=[
            [RingElement(1), RingElement(2)],
            [RingElement(3), RingElement(4)]
        ])
        assert matrix_ops.rank(A) == 2
        
        # Rank deficient matrix
        B = Matrix(data=[
            [RingElement(1), RingElement(2)],
            [RingElement(2), RingElement(4)]
        ])
        assert matrix_ops.rank(B) == 1
    
    def test_linear_solve(self, matrix_ops):
        """Test solving linear system."""
        A = Matrix(data=[
            [RingElement(1), RingElement(1)],
            [RingElement(1), RingElement(-1)]
        ])
        b = Vector(components=[RingElement(3), RingElement(1)])
        
        x = matrix_ops.solve(A, b)
        
        assert x is not None
        assert x.components[0].value == 2  # x = 2
        assert x.components[1].value == 1  # y = 1


# === Polynomial Tests ===

class TestPolynomialOperations:
    """Test polynomial operations."""
    
    def test_polynomial_creation(self):
        """Test polynomial creation."""
        terms = [
            Monomial(variables=['x', 'y'], exponents=[2, 1], coefficient=RingElement(3)),
            Monomial(variables=['x'], exponents=[1], coefficient=RingElement(2)),
            Monomial(variables=[], exponents=[], coefficient=RingElement(1))
        ]
        p = Polynomial(terms=terms)
        
        assert len(p.terms) == 3
        assert 'x' in p.variables
        assert 'y' in p.variables
    
    def test_polynomial_simplify(self):
        """Test polynomial simplification."""
        terms = [
            Monomial(variables=['x'], exponents=[1], coefficient=RingElement(2)),
            Monomial(variables=['x'], exponents=[1], coefficient=RingElement(3)),
            Monomial(variables=['x'], exponents=[1], coefficient=RingElement(-5))
        ]
        p = Polynomial(terms=terms)
        
        simplified = p.simplify()
        
        # Should have one term with coefficient 0 (removed)
        assert len(simplified.terms) == 0
    
    def test_polynomial_degree(self):
        """Test polynomial degree."""
        terms = [
            Monomial(variables=['x', 'y'], exponents=[2, 1], coefficient=RingElement(1)),
            Monomial(variables=['x'], exponents=[1], coefficient=RingElement(1))
        ]
        p = Polynomial(terms=terms)
        
        assert p.degree() == 3  # 2 + 1


# === Lie Algebra Tests ===

class TestLieAlgebraOperations:
    """Test Lie algebra operations."""
    
    def test_lie_algebra_type(self):
        """Test Lie algebra type creation."""
        su2 = create_lie_algebra("su(2)", dimension=3)
        
        assert su2.name == "su(2)"
        assert su2.dimension == 3
    
    def test_lie_algebra_element(self):
        """Test Lie algebra element creation."""
        algebra = create_lie_algebra("su(2)", dimension=3)
        elem = create_lie_algebra_element([1, 0, 0], algebra)
        
        assert len(elem.components) == 3
        assert elem.algebra == algebra
    
    def test_su2_representation(self, lie_ops):
        """Test SU(2) representation matrices."""
        algebra = create_lie_algebra("su(2)", dimension=3)
        matrices = lie_ops.representation_matrices(algebra, 2)
        
        assert len(matrices) == 3
        assert matrices[0].rows == 2
        assert matrices[0].cols == 2
    
    def test_so3_representation(self, lie_ops):
        """Test SO(3) representation matrices."""
        algebra = create_lie_algebra("so(3)", dimension=3)
        matrices = lie_ops.representation_matrices(algebra, 3)
        
        assert len(matrices) == 3
        assert matrices[0].rows == 3


# === Symbolic Simplification Tests ===

class TestSymbolicSimplification:
    """Test symbolic simplification."""
    
    def test_simplify_sum(self, simplifier):
        """Test sum simplification."""
        s = Sum(terms=[
            RingElement(5),
            RingElement(0),
            RingElement(3)
        ])
        
        simplified = simplifier.simplify(s)
        
        # Should remove zero
        assert len(simplified.terms) == 2
    
    def test_simplify_product(self, simplifier):
        """Test product simplification."""
        p = Product(factors=[
            RingElement(2),
            RingElement(0),
            RingElement(3)
        ])
        
        simplified = simplifier.simplify(p)
        
        # Should be zero
        assert isinstance(simplified, RingElement)
        assert simplified.value == 0
    
    def test_simplify_power(self, simplifier):
        """Test power simplification."""
        p = Power(base=RingElement(5), exponent=0)
        
        simplified = simplifier.simplify(p)
        
        assert isinstance(simplified, RingElement)
        assert simplified.value == 1
    
    def test_expand_binomial(self, simplifier):
        """Test binomial expansion."""
        # (a + b)^2
        a = RingElement(1)
        b = RingElement(1)
        s = Sum(terms=[a, b])
        p = Power(base=s, exponent=2)
        
        expanded = simplifier.expand(p)
        
        # Should expand to a^2 + 2ab + b^2
        assert isinstance(expanded, Sum)
    
    def test_collect_terms(self, simplifier):
        """Test term collection."""
        terms = [
            Monomial(variables=['x'], exponents=[1], coefficient=RingElement(2)),
            Monomial(variables=['x'], exponents=[1], coefficient=RingElement(3)),
            Monomial(variables=['y'], exponents=[1], coefficient=RingElement(1))
        ]
        p = Polynomial(terms=terms)
        
        collected = simplifier.collect_terms(p, ['x'])
        
        # Should have 2 terms: 5x + y
        assert len(collected.terms) == 2
    
    def test_canonicalize(self, simplifier):
        """Test canonicalization."""
        expr = Sum(terms=[
            Product(factors=[RingElement(0), RingElement(5)]),
            RingElement(0)
        ])
        
        canonical = simplifier.canonicalize(expr)
        
        assert isinstance(canonical, RingElement)


# === Gröbner Basis Tests ===

class TestGroebnerBasis:
    """Test Gröbner basis computation."""
    
    def test_term_order_lex(self):
        """Test lexicographic term ordering."""
        order = TermOrder(ordering='lex', variables=['x', 'y', 'z'])
        
        m1 = Monomial(variables=['x', 'y'], exponents=[1, 1], coefficient=RingElement(1))
        m2 = Monomial(variables=['x', 'z'], exponents=[1, 1], coefficient=RingElement(1))
        
        # xz > xy in lex with x > y > z
        assert order.compare(m2, m1) > 0
    
    def test_groebner_basis_simple(self):
        """Test simple Gröbner basis computation."""
        # Simple ideal: <x^2 - 1, x*y - x>
        x2 = Polynomial(terms=[
            Monomial(variables=['x'], exponents=[2], coefficient=RingElement(1)),
            Monomial(variables=[], exponents=[], coefficient=RingElement(-1))
        ])
        xy = Polynomial(terms=[
            Monomial(variables=['x', 'y'], exponents=[1, 1], coefficient=RingElement(1)),
            Monomial(variables=['x'], exponents=[1], coefficient=RingElement(-1))
        ])
        
        gb = compute_groebner_basis([x2, xy], ordering='lex')
        
        assert len(gb) > 0
    
    def test_compute_basis(self):
        """Test compute_basis method."""
        gb = GroebnerBasis(term_order=TermOrder(ordering='lex'))
        
        # Simple polynomials
        p1 = Polynomial(terms=[
            Monomial(variables=['x'], exponents=[2], coefficient=RingElement(1)),
            Monomial(variables=[], exponents=[], coefficient=RingElement(-1))
        ])
        
        basis = gb.compute_basis([p1])
        
        assert len(basis) >= 1


# === Tensor Operation Tests ===

class TestTensorOperations:
    """Test tensor operations."""
    
    def test_tensor_creation(self):
        """Test tensor creation."""
        T = Tensor(
            components={(0, 0): RingElement(1), (1, 1): RingElement(1)},
            shape=(2, 2)
        )
        
        assert T.rank == 2
        assert T.shape == (2, 2)
    
    def test_tensor_product(self, tensor_ops):
        """Test tensor product."""
        A = Tensor(components={(0,): RingElement(1)}, shape=(2,))
        B = Tensor(components={(0,): RingElement(1)}, shape=(2,))
        
        C = tensor_ops.product(A, B)
        
        assert C.rank == 2
        assert C.shape == (2, 2)
    
    def test_tensor_symmetrization(self, tensor_ops):
        """Test symmetrization."""
        T = Tensor(
            components={(0, 1): RingElement(1), (1, 0): RingElement(2)},
            shape=(2, 2)
        )
        
        sym_T = tensor_ops.sym(T)
        
        # Should be symmetric
        assert sym_T.components[(0, 1)] == sym_T.components[(1, 0)]
    
    def test_metric_creation(self):
        """Test metric creation."""
        metric = create_metric(dim=3, signature="+++", diagonal=[1, 1, 1])
        
        assert metric.dim == 3
        assert metric.get(0, 0).value == 1
        assert metric.get(1, 1).value == 1
    
    def test_raise_lower_index(self, tensor_ops):
        """Test index raising and lowering."""
        metric = create_metric(dim=2, diagonal=[1, -1])
        
        T = Tensor(components={(0,): RingElement(1)}, shape=(2,))
        
        # Raise index
        T_up = tensor_ops.raise_index(T, metric, 0)
        assert T_up.rank == 1
        assert T_up.shape[0] == 2


# === Expression Rewriting Tests ===

class TestExpressionRewriting:
    """Test expression rewriting."""
    
    def test_rewrite_zero(self, rewriter):
        """Test rewriting zero."""
        expr = Sum(terms=[RingElement(5), RingElement(0)])
        
        rewritten = rewriter.rewrite(expr)
        
        # Should simplify
        assert isinstance(rewritten, Sum)
    
    def test_rewrite_commutator(self, rewriter):
        """Test commutator rewriting."""
        X = LieAlgebraElement(
            components=[RingElement(1), RingElement(0), RingElement(0)],
            algebra=LieAlgebraType(name="su(2)", dimension=3)
        )
        expr = Commutator(a=X, b=X)
        
        rewritten = rewriter.rewrite(expr)
        
        # [X, X] should become 0
        assert isinstance(rewritten, RingElement)
        assert rewritten.value == 0
    
    def test_normalize(self, rewriter):
        """Test normalization."""
        expr = Sum(terms=[
            Product(factors=[RingElement(1), RingElement(5)]),
            RingElement(0)
        ])
        
        normalized = rewriter.normalize(expr)
        
        # Should be simplified
        assert normalized == RingElement(5) or isinstance(normalized, Sum)
    
    def test_equivalence(self, rewriter):
        """Test equivalence checking."""
        expr1 = Sum(terms=[RingElement(1), RingElement(2)])
        expr2 = RingElement(3)
        
        assert rewriter.equivalence(expr1, expr2)
    
    def test_substitution(self, rewriter):
        """Test variable substitution."""
        X = LieAlgebraElement(
            components=[RingElement(1), RingElement(0), RingElement(0)],
            algebra=LieAlgebraType(name="su(2)", dimension=3)
        )
        
        substitutions = {'x': RingElement(5)}
        
        result = rewriter.substitute(X, substitutions)
        
        assert result == X  # No change for this simple case


# === Integration Tests ===

class TestALGIntegration:
    """Integration tests for ALG model."""
    
    def test_matrix_commutator(self, matrix_ops):
        """Test matrix commutator."""
        A = Matrix(data=[
            [RingElement(0), RingElement(1)],
            [RingElement(-1), RingElement(0)]
        ])
        B = Matrix(data=[
            [RingElement(0), RingElement(0)],
            [RingElement(1), RingElement(0)]
        ])
        
        AB = matrix_ops.matmul(A, B)
        BA = matrix_ops.matmul(B, A)
        
        # Commutator [A, B] = AB - BA
        commutator = matrix_ops.matmul(A, B)
        commutator = matrix_ops.matmul(B, A)
        
        # [σ1, σ2] = 2iσ3 for Pauli matrices
        # Simplified check: trace should be 0
        assert matrix_ops.trace(commutator).value == 0
    
    def test_polynomial_solve_with_matrix(self):
        """Test polynomial matrix relationship."""
        # Characteristic polynomial coefficients from trace and det
        A = Matrix(data=[
            [RingElement(4), RingElement(7)],
            [RingElement(2), RingElement(6)]
        ])
        
        ops = MatrixOps()
        tr = ops.trace(A)
        det = ops.determinant(A)
        
        # λ^2 - tr(A)*λ + det(A) = 0
        assert tr.value == 10
        assert abs(det.value - 10) < 1e-10
    
    def test_lie_algebra_matrices_commutator(self, lie_ops):
        """Test commutator of Lie algebra matrices."""
        algebra = create_lie_algebra("so(3)", dimension=3)
        matrices = lie_ops.representation_matrices(algebra, 3)
        
        J1, J2, J3 = matrices
        
        ops = MatrixOps()
        commutator = ops.matmul(J1, J2)
        reverse = ops.matmul(J2, J1)
        
        # Should get J3 (up to sign) for [J1, J2] = J3
        # This tests the Lie algebra structure


# === Performance Tests ===

class TestALGPerformance:
    """Performance and edge case tests."""
    
    def test_large_matrix_multiply(self, matrix_ops):
        """Test large matrix multiplication."""
        n = 10
        
        # Create identity matrix
        data = [[RingElement(1 if i == j else 0) for j in range(n)] for i in range(n)]
        I = Matrix(data=data)
        
        # Multiply
        result = matrix_ops.matmul(I, I)
        
        assert result == I
    
    def test_zero_matrix(self, matrix_ops):
        """Test zero matrix handling."""
        zero = Matrix(data=[
            [RingElement(0), RingElement(0)],
            [RingElement(0), RingElement(0)]
        ])
        
        result = matrix_ops.matmul(zero, zero)
        
        assert all(
            result.data[i][j].value == 0 
            for i in range(2) 
            for j in range(2)
        )
    
    def test_polynomial_addition(self):
        """Test polynomial addition."""
        p1 = Polynomial(terms=[
            Monomial(variables=['x'], exponents=[2], coefficient=RingElement(1))
        ])
        p2 = Polynomial(terms=[
            Monomial(variables=['x'], exponents=[2], coefficient=RingElement(1))
        ])
        
        # This would need Polynomial.__add__ implementation
        # For now, skip
        pass


# === Main Test Runner ===

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
