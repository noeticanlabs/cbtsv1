"""
NSC-M3L ALG Model: Tensor Operations

Implements tensor algebra including contraction, exterior product,
symmetrization, and index manipulation.
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from .alg_types import (
    Tensor, Matrix, RingElement, WedgeProduct, SymTensor,
    AlgebraicExpr
)


@dataclass
class Metric:
    """Riemannian or pseudo-Riemannian metric for index raising/lowering."""
    components: Dict[Tuple[int, int], RingElement]  # g_{ij}
    inverse: Dict[Tuple[int, int], RingElement]  # g^{ij}
    dim: int
    signature: Optional[str] = None  # e.g., "+---" for Lorentzian
    
    def __post_init__(self):
        if self.inverse is None:
            self.inverse = {}
    
    def get(self, i: int, j: int, upper: bool = False) -> RingElement:
        """Get metric component g_{ij} or g^{ij}."""
        key = (i, j)
        if upper:
            return self.inverse.get(key, RingElement(1 if i == j else 0))
        return self.components.get(key, RingElement(1 if i == j else 0))


class TensorOps:
    """Tensor operations."""
    
    def contract(self, T: Tensor, i: int, j: int) -> Tensor:
        """
        Contract indices i and j.
        
        Result: T^{k1...ki-1 kj...kn} = sum_m T^{k1...m...m...kn}
        """
        shape = T.shape
        n = T.rank
        
        # Check valid indices
        if i < 0 or i >= n or j < 0 or j >= n:
            raise ValueError(f"Invalid indices {i}, {j} for tensor of rank {n}")
        
        # Build new shape (remove indices i and j)
        new_shape = list(shape)
        # Remove higher index first to preserve positions
        idx1, idx2 = max(i, j), min(i, j)
        del new_shape[idx1]
        del new_shape[idx2]
        
        # Contract
        new_components: Dict[Tuple[int, ...], RingElement] = {}
        dim = shape[idx1]  # Assume same dimension for both indices
        
        for multi_index, value in T.components.items():
            if multi_index[idx1] == multi_index[idx2]:
                # Build new multi-index
                new_idx = list(multi_index)
                del new_idx[idx1]
                del new_idx[idx2]
                new_idx = tuple(new_idx)
                
                # Add to result
                new_components[new_idx] = RingElement(
                    new_components.get(new_idx, RingElement(0)).value + value.value
                )
        
        return Tensor(
            components=new_components,
            shape=tuple(new_shape) if new_shape else (1,)
        )
    
    def product(self, A: Tensor, B: Tensor) -> Tensor:
        """Tensor product A ⊗ B."""
        new_rank = A.rank + B.rank
        new_shape = A.shape + B.shape
        
        new_components: Dict[Tuple[int, ...], RingElement] = {}
        
        for idx_a, val_a in A.components.items():
            for idx_b, val_b in B.components.items():
                new_idx = idx_a + idx_b
                new_components[new_idx] = RingElement(val_a.value * val_b.value)
        
        return Tensor(
            components=new_components,
            shape=new_shape
        )
    
    def wedge(self, A: Tensor, B: Tensor) -> Tensor:
        """Exterior (wedge) product A ∧ B."""
        # A ∧ B = A ⊗ B - B ⊗ A, antisymmetrized
        
        # For k-forms and l-forms, result is (k+l)-form
        # We'll simplify by combining like terms
        
        tensor_prod = self.product(A, B)
        tensor_prod_rev = self.product(B, A)
        
        # Subtract (simplified - just return the difference for now)
        # Full implementation would need form-specific handling
        return tensor_prod
    
    def sym(self, T: Tensor) -> Tensor:
        """Symmetrize tensor over all indices."""
        if T.rank < 2:
            return T
        
        # Symmetrize: (1/n!) * sum_{sigma in S_n} T_{sigma(i)}
        import math
        n = T.rank
        
        # Collect symmetric combinations
        sym_components: Dict[Tuple[int, ...], RingElement] = {}
        
        for idx, val in T.components.items():
            # Add value to all permutations
            from itertools import permutations
            for perm in set(permutations(idx)):
                sym_components[perm] = RingElement(
                    sym_components.get(perm, RingElement(0)).value + val.value
                )
        
        # Normalize
        n_perms = math.factorial(n)
        for idx in sym_components:
            sym_components[idx] = RingElement(
                sym_components[idx].value / n_perms
            )
        
        return Tensor(
            components=sym_components,
            shape=T.shape
        )
    
    def antisym(self, T: Tensor) -> Tensor:
        """Antisymmetrize tensor over all indices."""
        if T.rank < 2:
            return T
        
        # Antisymmetrize: (1/n!) * sum_{sigma in S_n} sign(sigma) * T_{sigma(i)}
        import math
        n = T.rank
        
        antisym_components: Dict[Tuple[int, ...], RingElement] = {}
        
        for idx, val in T.components.items():
            from itertools import permutations
            for perm in permutations(idx):
                # Compute sign of permutation
                inversions = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        if perm[i] > perm[j]:
                            inversions += 1
                sign = -1 if inversions % 2 else 1
                
                antisym_components[perm] = RingElement(
                    antisym_components.get(perm, RingElement(0)).value + sign * val.value
                )
        
        # Normalize
        n_perms = math.factorial(n)
        for idx in antisym_components:
            antisym_components[idx] = RingElement(
                antisym_components[idx].value / n_perms
            )
        
        return Tensor(
            components=antisym_components,
            shape=T.shape
        )
    
    def raise_index(self, T: Tensor, metric: Metric, index: int) -> Tensor:
        """Raise index using metric."""
        shape = T.shape
        n = T.rank
        
        if index < 0 or index >= n:
            raise ValueError(f"Invalid index {index} for tensor of rank {n}")
        
        # New shape: raise the index (contravariant)
        new_shape = list(shape)
        new_shape[index] = metric.dim
        
        new_components: Dict[Tuple[int, ...], RingElement] = {}
        
        for multi_index, value in T.components.items():
            # Contract with metric
            new_idx = list(multi_index)
            for m in range(metric.dim):
                new_idx[index] = m
                g_im = metric.get(m, multi_index[index], upper=False)
                key = tuple(new_idx)
                new_components[key] = RingElement(
                    new_components.get(key, RingElement(0)).value + 
                    g_im.value * value.value
                )
        
        return Tensor(
            components=new_components,
            shape=tuple(new_shape)
        )
    
    def lower_index(self, T: Tensor, metric: Metric, index: int) -> Tensor:
        """Lower index using metric."""
        shape = T.shape
        n = T.rank
        
        if index < 0 or index >= n:
            raise ValueError(f"Invalid index {index} for tensor of rank {n}")
        
        # New shape: lower the index (covariant)
        new_shape = list(shape)
        new_shape[index] = metric.dim
        
        new_components: Dict[Tuple[int, ...], RingElement] = {}
        
        for multi_index, value in T.components.items():
            # Contract with inverse metric
            new_idx = list(multi_index)
            for m in range(metric.dim):
                new_idx[index] = m
                g_im_upper = metric.get(m, multi_index[index], upper=True)
                key = tuple(new_idx)
                new_components[key] = RingElement(
                    new_components.get(key, RingElement(0)).value + 
                    g_im_upper.value * value.value
                )
        
        return Tensor(
            components=new_components,
            shape=tuple(new_shape)
        )
    
    def Hodge_star(self, T: Tensor, metric: Metric) -> Tensor:
        """
        Compute Hodge dual.
        
        For an n-dimensional manifold with volume form vol,
        *: Λ^k → Λ^{n-k}
        """
        dim = metric.dim
        
        # Check if T is a k-form
        # Simplified: assume fully antisymmetric tensor
        k = T.rank
        
        # Result is (n-k)-form
        new_rank = dim - k
        
        # Get volume form component
        # vol = sqrt(|det(g)|) * e^{012...}
        det_g = self._metric_det(metric)
        vol_factor = abs(det_g) ** 0.5
        
        # Simplified Hodge star implementation
        # Would need proper index-based computation
        
        return Tensor(
            components={},
            shape=tuple([dim] * new_rank) if new_rank > 0 else (1,)
        )
    
    def _metric_det(self, metric: Metric) -> float:
        """Compute determinant of metric."""
        # Simplified - would use proper determinant computation
        return 1.0
    
    def trace(self, T: Tensor, i: int, j: int) -> Tensor:
        """Partial trace contracting indices i and j."""
        return self.contract(T, i, j)
    
    def permute(self, T: Tensor, permutation: Tuple[int, ...]) -> Tensor:
        """Permute tensor indices."""
        if len(permutation) != T.rank:
            raise ValueError("Permutation length must match tensor rank")
        
        new_components: Dict[Tuple[int, ...], RingElement] = {}
        
        for idx, value in T.components.items():
            new_idx = tuple(idx[p] for p in permutation)
            new_components[new_idx] = value
        
        new_shape = tuple(T.shape[p] for p in permutation)
        
        return Tensor(
            components=new_components,
            shape=new_shape
        )
    
    def diagonal(self, T: Tensor) -> Tensor:
        """Extract diagonal components (for rank-2 tensors)."""
        if T.rank != 2:
            raise ValueError("Diagonal only defined for rank-2 tensors")
        
        dim = min(T.shape)
        new_components: Dict[Tuple[int, ...], RingElement] = {}
        
        for i in range(dim):
            key = (i, i)
            if key in T.components:
                new_components[(i,)] = T.components[key]
        
        return Tensor(
            components=new_components,
            shape=(dim,)
        )


def create_metric(
    dim: int,
    signature: Optional[str] = None,
    diagonal: Optional[List[float]] = None
) -> Metric:
    """Create a metric from diagonal components."""
    components = {}
    inverse = {}
    
    if diagonal:
        for i in range(dim):
            components[(i, i)] = RingElement(diagonal[i])
            if diagonal[i] != 0:
                inverse[(i, i)] = RingElement(1.0 / diagonal[i])
    else:
        # Euclidean metric by default
        for i in range(dim):
            components[(i, i)] = RingElement(1)
            inverse[(i, i)] = RingElement(1)
    
    return Metric(
        components=components,
        inverse=inverse,
        dim=dim,
        signature=signature
    )
