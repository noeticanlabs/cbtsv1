# coherence_integration.py
# =============================================================================
# Coherence Integration Layer
# =============================================================================
#
# This module replaces solver-local coherence computation with canonical
# compute_coherence() from the coherence-framework.
#
# IMPORTANT: cbtsv1 should NEVER compute coherence directly. All coherence
# values must flow through this integration layer.
#
# Lexicon declarations per canon v1.2
LEXICON_IMPORTS = [
    "LoC_axiom",
    "UFE_core",
    "GR_dyn",
    "CTL_time"
]
LEXICON_SYMBOLS = {
    "\\mathfrak c": "coherence_functional"
}

import logging
from typing import Dict, Any, Optional

# Import canonical compute_coherence from vendor
from cbtsv1.vendor.coherence_framework.coherence.core import compute_coherence

# Import the GR block adapter
from .defined_coherence_blocks import build_residual_blocks, summarize_blocks

logger = logging.getLogger('gr_solver.coherence_integration')


def compute_gr_coherence(
    fields: Any,
    constraints: Any,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compute canonical coherence for GR state.
    
    This function is the main integration point between cbtsv1's GR solver
    and the canonical coherence framework. It:
    1. Builds residual blocks from GR constraints
    2. Calls canonical compute_coherence()
    3. Returns both the coherence value and block summaries
    
    Args:
        fields: GRCoreFields instance
        constraints: GRConstraints instance (with H, M computed)
        config: Coherence configuration dict with 'blocks' key
        
    Returns:
        dict with:
            - coherence_value: float - the canonical coherence 𝔠 = ⟨r̃, W r̃⟩
            - blocks: dict - block summaries (dim, l2, linf, hash)
            - raw_result: CoherenceResult - full result object (for debugging)
    
    Example:
        >>> from cbtsv1.solvers.gr.coherence_integration import compute_gr_coherence
        >>> result = compute_gr_coherence(solver.fields, solver.constraints, config)
        >>> print(f"Coherence: {result['coherence_value']:.6e}")
        Coherence: 1.234567e-08
    """
    # Build canonical residual blocks from GR state
    blocks = build_residual_blocks(fields, constraints, config)
    
    # Compute canonical coherence
    raw_result = compute_coherence(blocks)
    
    # Create audit-friendly summary
    block_summary = summarize_blocks(blocks)
    
    logger.debug("Computed canonical coherence", extra={
        "extra_data": {
            "coherence_value": raw_result.coherence_value,
            "hamiltonian_l2": block_summary["hamiltonian"]["l2"],
            "momentum_l2": block_summary["momentum"]["l2"],
            "hamiltonian_linf": block_summary["hamiltonian"]["linf"],
            "momentum_linf": block_summary["momentum"]["linf"]
        }
    })
    
    return {
        "coherence_value": raw_result.coherence_value,
        "blocks": block_summary,
        "raw_result": raw_result
    }


def compute_gr_coherence_with_projection(
    fields: Any,
    constraints: Any,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compute canonical coherence including BSSN-Z4 projection constraints.
    
    This variant includes optional projection constraints (Z, Z_i) if present
    in the configuration and available in the fields.
    
    Args:
        fields: GRCoreFields instance
        constraints: GRConstraints instance
        config: Coherence configuration
        
    Returns:
        dict with coherence_value and extended block summaries
    """
    from .defined_coherence_blocks import build_residual_blocks_with_projection
    
    blocks = build_residual_blocks_with_projection(fields, constraints, config)
    raw_result = compute_coherence(blocks)
    block_summary = summarize_blocks(blocks)
    
    return {
        "coherence_value": raw_result.coherence_value,
        "blocks": block_summary,
        "raw_result": raw_result
    }


class CoherenceTracker:
    """
    Track coherence history over solver evolution.
    
    This class maintains a history of coherence values and block summaries
    for analysis and debugging.
    """
    
    def __init__(self):
        self.history = []
    
    def record(self, coherence_result: Dict[str, Any], step: int, t: float):
        """
        Record a coherence computation result.
        
        Args:
            coherence_result: Result from compute_gr_coherence()
            step: Current solver step
            t: Current time
        """
        self.history.append({
            "step": step,
            "t": t,
            "coherence_value": coherence_result["coherence_value"],
            "blocks": coherence_result["blocks"]
        })
    
    def get_history(self) -> list:
        """Return full history."""
        return self.history
    
    def get_coherence_series(self) -> list:
        """Return list of coherence values over time."""
        return [h["coherence_value"] for h in self.history]
    
    def clear(self):
        """Clear history."""
        self.history = []


def create_default_coherence_config() -> Dict[str, Any]:
    """
    Create default coherence configuration.
    
    Returns a minimal configuration with identity scales and weights.
    This is useful for testing or when no config file is available.
    
    Returns:
        dict with default block configuration
    """
    return {
        "version": "1.0.0",
        "covariance_model": "diag",
        "blocks": {
            "hamiltonian": {
                "scale": 1.0,
                "weight": 1.0,
                "description": "Hamiltonian constraint H = R + K² - K_ij K^ij - 2Λ"
            },
            "momentum": {
                "scale": 1.0,
                "weight": 1.0,
                "description": "Momentum constraint M^i = D_j(K^ij - γ^ij K)"
            }
        }
    }
