"""
NSC-M3L ALG Model: Matrix Operations

Implements matrix algebra and linear algebra operations.
"""

from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
from math import sqrt

from .alg_types import (
    Matrix, Vector, RingElement, LinearMap, SparseMatrix, RingType
)
from .alg_simplify import SymbolicSimplifier


class MatrixOps:
    """Matrix operations and linear algebra."""
    
    def __init__(self):
        self.simplifier = SymbolicSimplifier()
    
    def matmul(self, A: Matrix, B: Matrix) -> Matrix:
        """Matrix multiplication: C = A * B."""
        if A.cols != B.rows:
            raise ValueError(f"Dimension mismatch: {A.cols} != {B.rows}")
        
        # Compute C = A * B
        C_data = []
        for i in range(A.rows):
            row = []
            for j in range(B.cols):
                # Compute dot product of A[i,:] and B[:,j]
                val = RingElement(0)
                for k in range(A.cols):
                    val = RingElement(val.value + self._mult_elements(A.data[i][k], B.data[k][j]))
                row.append(val)
            C_data.append(row)
        
        return Matrix(data=C_data, ring=A.ring)
    
    def _mult_elements(self, a: RingElement, b: RingElement) -> Union[int, float]:
        """Multiply two ring elements and return scalar value."""
        if isinstance(a.value, (int, float)) and isinstance(b.value, (int, float)):
            return a.value * b.value
        return 0  # Placeholder for complex ring elements
    
    def transpose(self, A: Matrix) -> Matrix:
        """Matrix transpose: A^T."""
        A_T_data = []
        for j in range(A.cols):
            row = []
            for i in range(A.rows):
                row.append(A.data[i][j])
            A_T_data.append(row)
        
        return Matrix(data=A_T_data, ring=A.ring)
    
    def inverse(self, A: Matrix) -> Optional[Matrix]:
        """Matrix inverse using Gaussian elimination."""
        if not A.is_square():
            raise ValueError("Cannot invert non-square matrix")
        
        n = A.rows
        
        # Create augmented matrix [A | I]
        aug = []
        for i in range(n):
            row = []
            for j in range(n):
                row.append(A.data[i][j])
            # Identity matrix
            for j in range(n):
                row.append(RingElement(1 if i == j else 0))
            aug.append(row)
        
        # Gaussian elimination
        for col in range(n):
            # Find pivot
            pivot_row = None
            for row in range(col, n):
                if aug[row][col].value != 0:
                    pivot_row = row
                    break
            
            if pivot_row is None:
                return None  # Singular matrix
            
            # Swap rows if needed
            if pivot_row != col:
                aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
            
            # Scale pivot row
            pivot_val = aug[col][col].value
            for j in range(2 * n):
                aug[col][j] = RingElement(aug[col][j].value / pivot_val)
            
            # Eliminate column
            for row in range(n):
                if row != col:
                    factor = aug[row][col].value
                    if factor != 0:
                        for j in range(2 * n):
                            aug[row][j] = RingElement(aug[row][j].value - factor * aug[col][j].value)
        
        # Extract inverse
        inv_data = []
        for i in range(n):
            row = []
            for j in range(n, 2 * n):
                row.append(aug[i][j])
            inv_data.append(row)
        
        return Matrix(data=inv_data, ring=A.ring)
    
    def determinant(self, A: Matrix) -> RingElement:
        """Matrix determinant using Laplace expansion."""
        if not A.is_square():
            raise ValueError("Cannot compute determinant of non-square matrix")
        
        n = A.rows
        
        if n == 1:
            return A.data[0][0]
        if n == 2:
            a, b = A.data[0]
            c, d = A.data[1]
            return RingElement(a.value * d.value - b.value * c.value)
        
        # Laplace expansion for larger matrices
        det = RingElement(0)
        for j in range(n):
            minor = self._minor(A, 0, j)
            cofactor = RingElement((-1) ** j * self.determinant(minor).value)
            det = RingElement(det.value + A.data[0][j].value * cofactor.value)
        
        return det
    
    def _minor(self, A: Matrix, row: int, col: int) -> Matrix:
        """Compute minor by removing row and col."""
        minor_data = []
        for i in range(A.rows):
            if i == row:
                continue
            minor_row = []
            for j in range(A.cols):
                if j == col:
                    continue
                minor_row.append(A.data[i][j])
            if minor_row:
                minor_data.append(minor_row)
        
        return Matrix(data=minor_data, ring=A.ring)
    
    def trace(self, A: Matrix) -> RingElement:
        """Matrix trace: sum of diagonal elements."""
        if not A.is_square():
            raise ValueError("Trace requires square matrix")
        
        tr = RingElement(0)
        for i in range(A.rows):
            tr = RingElement(tr.value + A.data[i][i].value)
        return tr
    
    def eigenvalues(self, A: Matrix) -> List[Union[int, float, complex]]:
        """Compute eigenvalues (placeholder - uses numpy if available)."""
        if not A.is_square():
            raise ValueError("Eigenvalues require square matrix")
        
        # Try to use numpy if available
        try:
            import numpy as np
            n = A.rows
            # Convert to numpy array
            numpy_A = np.array([[A.data[i][j].value for j in range(n)] for i in range(n)])
            eigvals = np.linalg.eigvals(numpy_A)
            return [complex(v) for v in eigvals]
        except ImportError:
            # Fallback: 2x2 closed form
            n = A.rows
            if n == 2:
                a, b = A.data[0]
                c, d = A.data[1]
                tr = a.value + d.value
                det = a.value * d.value - b.value * c.value
                discriminant = tr * tr - 4 * det
                return [
                    (tr + sqrt(discriminant)) / 2,
                    (tr - sqrt(discriminant)) / 2
                ]
            # For larger matrices without numpy, return empty
            return []
    
    def characteristic_poly(self, A: Matrix) -> 'Polynomial':
        """Characteristic polynomial det(λI - A)."""
        # This is a placeholder - would need polynomial ring implementation
        from .alg_types import Polynomial, Monomial
        return Polynomial(terms=[])
    
    def rank(self, A: Matrix) -> int:
        """Matrix rank using row reduction."""
        if A.rows == 0 or A.cols == 0:
            return 0
        
        # Create working copy
        M = [row[:] for row in A.data]
        m, n = len(M), len(M[0])
        
        rank = 0
        row = 0
        
        for col in range(n):
            # Find pivot
            pivot_row = None
            for r in range(row, m):
                if M[r][col].value != 0:
                    pivot_row = r
                    break
            
            if pivot_row is None:
                continue
            
            # Swap rows
            M[row], M[pivot_row] = M[pivot_row], M[row]
            
            # Scale pivot
            pivot_val = M[row][col].value
            for j in range(col, n):
                M[row][j] = RingElement(M[row][j].value / pivot_val)
            
            # Eliminate column
            for r in range(m):
                if r != row and M[r][col].value != 0:
                    factor = M[r][col].value
                    for j in range(col, n):
                        M[r][j] = RingElement(M[r][j].value - factor * M[row][j].value)
            
            row += 1
            rank += 1
            if row >= m:
                break
        
        return rank
    
    def nullity(self, A: Matrix) -> int:
        """Nullity = dim(kernel) = n - rank."""
        return A.cols - self.rank(A)
    
    def solve(self, A: Matrix, b: Vector) -> Optional[Vector]:
        """Solve linear system Ax = b."""
        if not A.is_square() or A.rows != b.size:
            raise ValueError("Invalid dimensions for linear system")
        
        n = A.rows
        
        # Create augmented matrix [A | b]
        aug = []
        for i in range(n):
            row = A.data[i] + [b.components[i]]
            aug.append(row)
        
        # Gaussian elimination with back substitution
        for col in range(n):
            # Find pivot
            pivot_row = None
            for row in range(col, n):
                if aug[row][col].value != 0:
                    pivot_row = row
                    break
            
            if pivot_row is None:
                return None  # No unique solution
            
            # Swap rows
            if pivot_row != col:
                aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
            
            # Scale pivot row
            pivot_val = aug[col][col].value
            for j in range(col, n + 1):
                aug[col][j] = RingElement(aug[col][j].value / pivot_val)
            
            # Eliminate column
            for row in range(n):
                if row != col:
                    factor = aug[row][col].value
                    if factor != 0:
                        for j in range(col, n + 1):
                            aug[row][j] = RingElement(aug[row][j].value - factor * aug[col][j].value)
        
        # Extract solution
        x = []
        for i in range(n):
            x.append(aug[i][n])
        
        return Vector(components=x, ring=A.ring)
    
    def dot_product(self, v: Vector, w: Vector) -> RingElement:
        """Dot product of two vectors."""
        if v.size != w.size:
            raise ValueError("Vector sizes must match")
        
        result = RingElement(0)
        for i in range(v.size):
            result = RingElement(result.value + v.components[i].value * w.components[i].value)
        return result
    
    def cross_product(self, v: Vector, w: Vector) -> Vector:
        """Cross product of two 3D vectors."""
        if v.size != 3 or w.size != 3:
            raise ValueError("Cross product requires 3D vectors")
        
        components = [
            RingElement(v.components[1].value * w.components[2].value - v.components[2].value * w.components[1].value),
            RingElement(v.components[2].value * w.components[0].value - v.components[0].value * w.components[2].value),
            RingElement(v.components[0].value * w.components[1].value - v.components[1].value * w.components[0].value)
        ]
        return Vector(components=components, ring=v.ring)
    
    def outer_product(self, v: Vector, w: Vector) -> Matrix:
        """Outer product v ⊗ w."""
        data = []
        for i in range(v.size):
            row = []
            for j in range(w.size):
                row.append(RingElement(v.components[i].value * w.components[j].value))
            data.append(row)
        return Matrix(data=data, ring=v.ring)
    
    def norm(self, v: Vector) -> RingElement:
        """Euclidean norm of vector."""
        return RingElement(sqrt(self.dot_product(v, v).value))


# Convenience functions
def matmul(A: Matrix, B: Matrix) -> Matrix:
    """Convenience function for matrix multiplication."""
    ops = MatrixOps()
    return ops.matmul(A, B)


def transpose(A: Matrix) -> Matrix:
    """Convenience function for matrix transpose."""
    ops = MatrixOps()
    return ops.transpose(A)


def inverse(A: Matrix) -> Optional[Matrix]:
    """Convenience function for matrix inverse."""
    ops = MatrixOps()
    return ops.inverse(A)


def determinant(A: Matrix) -> RingElement:
    """Convenience function for matrix determinant."""
    ops = MatrixOps()
    return ops.determinant(A)


def trace(A: Matrix) -> RingElement:
    """Convenience function for matrix trace."""
    ops = MatrixOps()
    return ops.trace(A)


def solve(A: Matrix, b: Vector) -> Optional[Vector]:
    """Convenience function for solving linear systems."""
    ops = MatrixOps()
    return ops.solve(A, b)


def rank(A: Matrix) -> int:
    """Convenience function for matrix rank."""
    ops = MatrixOps()
    return ops.rank(A)
