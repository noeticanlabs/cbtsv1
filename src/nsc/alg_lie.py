"""
NSC-M3L ALG Model: Lie Algebra Operations

Implements Lie algebra operations including commutators, structure constants,
Casimir invariants, and Killing form.
"""

from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass

from .alg_types import (
    LieAlgebraElement, LieAlgebraType, LieGroup,
    Matrix, Vector, RingElement, Tensor,
    AlgebraicExpr, Commutator, Bracket
)
from .alg_matrix import MatrixOps


class LieAlgebraOps:
    """Lie algebra operations."""
    
    def __init__(self):
        self.matrix_ops = MatrixOps()
    
    def commutator(self, X: LieAlgebraElement, Y: LieAlgebraElement) -> LieAlgebraElement:
        """
        Compute commutator [X, Y] = XY - YX.
        
        For matrix representations: [X, Y] = X*Y - Y*X
        For abstract basis: [e_i, e_j] = f^k_ij e_k
        """
        if X.algebra != Y.algebra:
            raise ValueError("Commutator requires elements of the same Lie algebra")
        
        algebra = X.algebra
        
        # If we have structure constants, use them
        if algebra.structure_constants is not None:
            return self._commutator_basis(X, Y, algebra)
        
        # Otherwise, try matrix representation
        # This would use representation matrices if available
        raise NotImplementedError(
            "Commutator requires either structure constants or representation matrices"
        )
    
    def _commutator_basis(
        self, 
        X: LieAlgebraElement, 
        Y: LieAlgebraElement,
        algebra: LieAlgebraType
    ) -> LieAlgebraElement:
        """Compute commutator using structure constants."""
        f = algebra.structure_constants  # f^k_ij
        
        # [X, Y]^k = f^k_ij * X^i * Y^j
        dim = algebra.dimension
        result_components = [RingElement(0) for _ in range(dim)]
        
        for k in range(dim):
            for i in range(dim):
                for j in range(dim):
                    f_ijk = f[k][i][j] if f else 0
                    result_components[k] = RingElement(
                        result_components[k].value + 
                        f_ijk * X.components[i].value * Y.components[j].value
                    )
        
        return LieAlgebraElement(
            components=result_components,
            algebra=algebra
        )
    
    def ad(self, X: LieAlgebraElement) -> Matrix:
        """
        Compute ad(X)(Y) = [X, Y].
        
        Returns matrix representation of adjoint action.
        """
        algebra = X.algebra
        dim = algebra.dimension
        
        if algebra.structure_constants is None:
            raise ValueError("ad(X) requires structure constants")
        
        f = algebra.structure_constants
        
        # Build matrix where (ad(X))^k_j = f^k_ij * X^i
        data = []
        for k in range(dim):
            row = []
            for j in range(dim):
                val = 0
                for i in range(dim):
                    val += f[k][i][j] * X.components[i].value
                row.append(RingElement(val))
            data.append(row)
        
        return Matrix(data=data, ring=algebra.dimension)
    
    def ad_exp(self, X: LieAlgebraElement, n: int) -> Matrix:
        """Compute ad(X)^n using repeated application."""
        ad_X = self.ad(X)
        result = self._identity_matrix(ad_X.rows)
        
        for _ in range(n):
            result = self.matrix_ops.matmul(result, ad_X)
        
        return result
    
    def _identity_matrix(self, n: int) -> Matrix:
        """Create n x n identity matrix."""
        data = []
        for i in range(n):
            row = []
            for j in range(n):
                row.append(RingElement(1 if i == j else 0))
            data.append(row)
        return Matrix(data=data)
    
    def structure_constants(
        self, 
        basis: List[LieAlgebraElement]
    ) -> List[List[List[float]]]:
        """
        Compute structure constants f^k_ij from basis.
        
        [e_i, e_j] = f^k_ij e_k
        """
        if len(basis) < 2:
            raise ValueError("Need at least 2 basis elements")
        
        dim = len(basis)
        f: List[List[List[float]]] = [[[0.0] * dim for _ in range(dim)] for _ in range(dim)]
        
        for i in range(dim):
            for j in range(i + 1, dim):  # Use antisymmetry
                commutator = self.commutator(basis[i], basis[j])
                # Express result in basis
                for k in range(dim):
                    f[k][i][j] = commutator.components[k].value
                    f[k][j][i] = -f[k][i][j]  # Antisymmetric
        
        return f
    
    def casimir_invariant(self, algebra: LieAlgebraType) -> AlgebraicExpr:
        """
        Compute quadratic Casimir invariant.
        
        C = sum_k (ad(X^k))^2
        or in basis: C = g^{ij} X_i X_j where g is Killing form
        """
        dim = algebra.dimension
        
        # Build Killing form matrix
        if algebra.killing_form is None:
            # Compute Killing form from structure constants
            K = self._compute_killing_form(algebra)
        else:
            K = algebra.killing_form
        
        # C = trace(ad(X)^2) for arbitrary X
        # In a basis, this becomes sum_{i,j} K^{ij} X_i X_j
        from .alg_types import Polynomial, Monomial, Sum, Product
        
        terms = []
        for i in range(dim):
            for j in range(dim):
                # Create term K^{ij} * X_i * X_j
                if i != j:
                    xi = LieAlgebraElement(
                        components=[RingElement(1 if k == i else 0) for k in range(dim)],
                        algebra=algebra
                    )
                    xj = LieAlgebraElement(
                        components=[RingElement(1 if k == j else 0) for k in range(dim)],
                        algebra=algebra
                    )
                    term = Product(factors=[xi, xj])
                    terms.append(term)
        
        return Sum(terms=terms) if terms else RingElement(0)
    
    def _compute_killing_form(self, algebra: LieAlgebraType) -> Matrix:
        """Compute Killing form from structure constants."""
        dim = algebra.dimension
        f = algebra.structure_constants
        
        if f is None:
            raise ValueError("Killing form computation requires structure constants")
        
        # K_{ij} = tr(ad(e_i) ad(e_j)) = f^k_il f^l_jk
        K_data = []
        for i in range(dim):
            row = []
            for j in range(dim):
                val = 0.0
                for k in range(dim):
                    for l in range(dim):
                        val += f[k][i][l] * f[l][j][k]
                row.append(RingElement(val))
            K_data.append(row)
        
        return Matrix(data=K_data)
    
    def Killing_form(
        self, 
        X: LieAlgebraElement, 
        Y: LieAlgebraElement
    ) -> RingElement:
        """Compute Killing form K(X, Y) = tr(ad(X) ad(Y))."""
        algebra = X.algebra
        
        # Get ad matrices
        ad_X = self.ad(X)
        ad_Y = self.ad(Y)
        
        # K(X,Y) = trace(ad(X) * ad(Y))
        product = self.matrix_ops.matmul(ad_X, ad_Y)
        
        return self.matrix_ops.trace(product)
    
    def representation_matrices(
        self, 
        algebra: LieAlgebraType, 
        dim: int
    ) -> List[Matrix]:
        """
        Get faithful representation matrices for the Lie algebra.
        
        For su(2): Pauli matrices (times i)
        For so(3): Standard rotation generators
        """
        name = algebra.name.lower()
        
        if name == "su(2)" or name == "su2":
            return self._su2_representation()
        elif name == "so(3)" or name == "so3":
            return self._so3_representation()
        elif name == "u(1)" or name == "u1":
            return self._u1_representation()
        else:
            raise NotImplementedError(
                f"No built-in representation for {algebra.name}"
            )
    
    def _su2_representation(self) -> List[Matrix]:
        """SU(2) representation using Pauli matrices (imaginary unit included)."""
        # T_a = -i/2 * sigma_a where sigma_a are Pauli matrices
        # This gives generators with correct commutation relations
        
        # T_1 = [[0, -i/2], [i/2, 0]] = -i/2 * sigma_1
        # T_2 = [[0, -1/2], [-1/2, 0]] = -i/2 * sigma_2  
        # T_3 = [[-i/2, 0], [0, i/2]] = -i/2 * sigma_3
        
        T1 = Matrix(data=[
            [RingElement(0), RingElement(-0.5)],
            [RingElement(0.5), RingElement(0)]
        ])
        
        T2 = Matrix(data=[
            [RingElement(0), RingElement(-0.5)],
            [RingElement(-0.5), RingElement(0)]
        ])
        
        T3 = Matrix(data=[
            [RingElement(-0.5), RingElement(0)],
            [RingElement(0), RingElement(0.5)]
        ])
        
        return [T1, T2, T3]
    
    def _so3_representation(self) -> List[Matrix]:
        """SO(3) representation using rotation generators."""
        # Standard basis for so(3): infinitesimal rotations
        
        J1 = Matrix(data=[
            [RingElement(0), RingElement(0), RingElement(0)],
            [RingElement(0), RingElement(0), RingElement(-1)],
            [RingElement(0), RingElement(1), RingElement(0)]
        ])
        
        J2 = Matrix(data=[
            [RingElement(0), RingElement(0), RingElement(1)],
            [RingElement(0), RingElement(0), RingElement(0)],
            [RingElement(-1), RingElement(0), RingElement(0)]
        ])
        
        J3 = Matrix(data=[
            [RingElement(0), RingElement(-1), RingElement(0)],
            [RingElement(1), RingElement(0), RingElement(0)],
            [RingElement(0), RingElement(0), RingElement(0)]
        ])
        
        return [J1, J2, J3]
    
    def _u1_representation(self) -> List[Matrix]:
        """U(1) representation - single generator."""
        # U(1) has one generator: the identity (for charge)
        return [Matrix(data=[[RingElement(1)]])]
    
    def exponential_map(self, X: LieAlgebraElement) -> Matrix:
        """Compute exp(X) - matrix exponential."""
        import math
        
        algebra = X.algebra
        dim = algebra.dimension
        
        # If we have a representation, use matrix exponential
        try:
            matrices = self.representation_matrices(algebra, dim)
            # Find which basis element X is
            rep_matrix = None
            for i, comp in enumerate(X.components):
                if comp.value != 0:
                    if rep_matrix is None:
                        rep_matrix = matrices[i] * comp.value
                    else:
                        # Add scaled matrix
                        for r in range(dim):
                            for c in range(dim):
                                rep_matrix.data[r][c] = RingElement(
                                    rep_matrix.data[r][c].value + 
                                    matrices[i].data[r][c].value * comp.value
                                )
            
            if rep_matrix is None:
                return self._identity_matrix(dim)
            
            return self._matrix_exp(rep_matrix)
        except NotImplementedError:
            # Fall back to scalar exponential
            total = sum(comp.value for comp in X.components)
            return Matrix(data=[[RingElement(math.exp(total))]])
    
    def _matrix_exp(self, A: Matrix) -> Matrix:
        """Compute matrix exponential using Taylor series."""
        import math
        
        n = A.rows
        result = self._identity_matrix(n)
        term = self._identity_matrix(n)
        factorial = 1
        
        for k in range(1, 20):  # Truncate after 20 terms
            factorial *= k
            term = self.matrix_ops.matmul(term, A)
            # Add term / k!
            for i in range(n):
                for j in range(n):
                    result.data[i][j] = RingElement(
                        result.data[i][j].value + term.data[i][j].value / factorial
                    )
        
        return result
    
    def bracket_to_commutator(self, bracket: 'Bracket') -> Commutator:
        """Convert Bracket operation to Commutator."""
        return Commutator(a=bracket.a, b=bracket.b)


def create_lie_algebra(
    name: str,
    dimension: int,
    structure_constants: Optional[List[List[List[float]]]] = None,
    killing_form: Optional[Matrix] = None
) -> LieAlgebraType:
    """Create a Lie algebra type."""
    return LieAlgebraType(
        name=name,
        dimension=dimension,
        structure_constants=structure_constants,
        killing_form=killing_form
    )


def create_lie_algebra_element(
    components: List[float],
    algebra: LieAlgebraType
) -> LieAlgebraElement:
    """Create a Lie algebra element from components."""
    return LieAlgebraElement(
        components=[RingElement(c) for c in components],
        algebra=algebra
    )
