# coherence_framework.coherence.core
# =============================================================================
# Canonical Coherence Core Module
# =============================================================================
# 
# This module implements the Defined Coherence canon per the specification:
#     𝔠(x) = ⟨r̃(x), W r̃(x)⟩,  where r̃ = S⁻¹r
#
# This is a VENDOR STUB - when coherence-framework is published, replace
# this entire vendor directory with the real package.
#
# Lexicon declarations per canon v1.2
LEXICON_IMPORTS = [
    "LoC_axiom",
    "UFE_core", 
    "GR_dyn",
    "CTL_time"
]
LEXICON_SYMBOLS = {
    "\\mathfrak c": "coherence_functional",
    "\\tilde r": "scaled_residual",
    "W": "weight_matrix"
}

import hashlib
import numpy as np
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ResidualBlock:
    """
    Canonical residual block for coherence computation.
    
    Attributes:
        name: Identifier for this block (e.g., "hamiltonian", "momentum")
        vector: Raw residual vector (numpy array)
        scale: Block-specific scale factor (S diagonal entry)
        weight: Block-specific weight (W diagonal entry)
        dim: Vector dimension (computed property)
        l2: L2 norm of scaled vector (computed property)
        linf: L-infinity norm (computed property)
        hash: SHA256 hash of canonical vector bytes (computed property)
    """
    name: str
    vector: np.ndarray
    scale: float = 1.0
    weight: float = 1.0
    
    # Computed properties (lazy-computed on first access)
    _dim: int = field(init=False, default=0)
    _l2: float = field(init=False, default=0.0)
    _linf: float = field(init=False, default=0.0)
    _hash: str = field(init=False, default="")
    _computed: bool = field(init=False, default=False)
    
    def _compute_properties(self) -> None:
        """Lazy-compute dimension, norms, and hash on first access."""
        if self._computed:
            return
            
        # Dimension
        self._dim = self.vector.size
        
        # Scaled vector: r̃ = scale * r
        scaled = self.scale * self.vector
        
        # L2 norm: ||r̃||_2
        self._l2 = float(np.sqrt(np.sum(scaled ** 2)))
        
        # L-infinity norm: ||r̃||_inf
        self._linf = float(np.max(np.abs(scaled)))
        
        # Hash: SHA256 of canonical representation
        # Use scaled vector bytes for canonical hash
        self._hash = hashlib.sha256(scaled.tobytes()).hexdigest()
        
        self._computed = True
    
    @property
    def dim(self) -> int:
        self._compute_properties()
        return self._dim
    
    @property
    def l2(self) -> float:
        self._compute_properties()
        return self._l2
    
    @property
    def linf(self) -> float:
        self._compute_properties()
        return self._linf
    
    @property
    def hash(self) -> str:
        self._compute_properties()
        return self._hash
    
    def to_dict(self) -> dict:
        """Convert to audit-friendly dictionary (no arrays)."""
        self._compute_properties()
        return {
            "name": self.name,
            "dim": self.dim,
            "l2": self.l2,
            "linf": self.linf,
            "hash": self.hash,
            "scale": self.scale,
            "weight": self.weight
        }


@dataclass
class CoherenceResult:
    """
    Result of canonical coherence computation.
    
    Attributes:
        coherence_value: The scalar coherence 𝔠 = ⟨r̃, W r̃⟩
        blocks: Dictionary of ResidualBlock objects used in computation
        covariance_model: String identifier for covariance model used
    """
    coherence_value: float
    blocks: Dict[str, ResidualBlock]
    covariance_model: str = "diag"
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "coherence_value": self.coherence_value,
            "covariance_model": self.covariance_model,
            "blocks": {name: block.to_dict() for name, block in self.blocks.items()}
        }


def compute_coherence(blocks: Dict[str, ResidualBlock]) -> CoherenceResult:
    """
    Compute canonical coherence functional.
    
    The coherence functional is defined as:
        𝔠 = Σ_i weight_i * ||scale_i * vector_i||²
    
    This implements the diagonal covariance model where:
        - S (scale matrix) is diagonal with entries = block.scale
        - W (weight matrix) is diagonal with entries = block.weight
    
    Args:
        blocks: Dictionary mapping block names to ResidualBlock objects
        
    Returns:
        CoherenceResult with coherence_value and block summaries
        
    Example:
        >>> hamiltonian = ResidualBlock(
        ...     name="hamiltonian",
        ...     vector=np.array([0.1, 0.2, 0.3]),
        ...     scale=1.0,
        ...     weight=1.0
        ... )
        >>> result = compute_coherence({"hamiltonian": hamiltonian})
        >>> print(result.coherence_value)
        0.14  # 0.1² + 0.2² + 0.3² = 0.01 + 0.04 + 0.09 = 0.14
    """
    if not blocks:
        return CoherenceResult(coherence_value=0.0, blocks={}, covariance_model="diag")
    
    total_coherence = 0.0
    
    for name, block in blocks.items():
        # Compute scaled residual: r̃ = scale * r
        scaled = block.scale * block.vector
        
        # Compute weighted L2 squared: weight * ||r̃||²
        l2_squared = np.sum(scaled ** 2)
        contribution = block.weight * l2_squared
        
        total_coherence += contribution
    
    return CoherenceResult(
        coherence_value=float(total_coherence),
        blocks=blocks,
        covariance_model="diag"
    )


def compute_coherence_with_covariance(
    blocks: Dict[str, ResidualBlock],
    covariance_matrix: np.ndarray
) -> CoherenceResult:
    """
    Compute coherence with explicit covariance matrix.
    
    This variant uses a full covariance matrix rather than diagonal weights.
    Useful for correlated residuals.
    
    Args:
        blocks: Dictionary of ResidualBlock objects
        covariance_matrix: Full covariance matrix W (must be square, size matching total dimension)
        
    Returns:
        CoherenceResult with full covariance computation
    """
    if not blocks:
        return CoherenceResult(coherence_value=0.0, blocks={}, covariance_model="full")
    
    # Concatenate all scaled vectors into single state vector
    scaled_vectors = []
    for name, block in blocks.items():
        scaled = block.scale * block.vector
        scaled_vectors.append(scaled)
    
    r_tilde = np.concatenate(scaled_vectors)
    
    # Compute quadratic form: r̃^T * W * r̃
    coherence_value = float(r_tilde @ covariance_matrix @ r_tilde)
    
    return CoherenceResult(
        coherence_value=coherence_value,
        blocks=blocks,
        covariance_model="full"
    )


# Convenience function for creating blocks from arrays
def create_residual_block(
    name: str,
    vector: np.ndarray,
    scale: float = 1.0,
    weight: float = 1.0
) -> ResidualBlock:
    """
    Factory function to create a ResidualBlock with proper flattening.
    
    Args:
        name: Block identifier
        vector: Residual array (any shape)
        scale: Scale factor
        weight: Weight factor
        
    Returns:
        ResidualBlock with flattened vector
    """
    return ResidualBlock(
        name=name,
        vector=vector.flatten(),
        scale=scale,
        weight=weight
    )
