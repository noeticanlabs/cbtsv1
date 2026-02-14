# defined_coherence_blocks.py
# =============================================================================
# GR Residual Block Adapter
# =============================================================================
# 
# This module converts cbtsv1 GR state into canonical ResidualBlocks
# for coherence computation per the Defined Coherence canon.
#
# The ONLY place where cbtsv1 produces residual blocks for coherence.
#
# Lexicon declarations per canon v1.2
LEXICON_IMPORTS = [
    "LoC_axiom",
    "UFE_core",
    "GR_dyn", 
    "CTL_time"
]
LEXICON_SYMBOLS = {
    "\\mathcal{H}": "GR_constraint.hamiltonian",
    "\\mathcal{M}^i": "GR_constraint.momentum"
}

import numpy as np
from typing import Dict, Any, Optional

# Import canonical ResidualBlock from vendor
from cbtsv1.vendor.coherence_framework.coherence.core import ResidualBlock


def build_residual_blocks(
    fields: Any,
    constraints: Any,
    config: Dict[str, Any]
) -> Dict[str, ResidualBlock]:
    """
    Build canonical residual blocks from GR constraints.
    
    This function extracts the constraint residuals from cbtsv1's GRConstraints
    and packages them into canonical ResidualBlocks for coherence computation.
    
    Args:
        fields: GRCoreFields instance (required for grid spacing info if needed)
        constraints: GRConstraints instance (must have H and M computed)
        config: dict with 'blocks' containing scale/weight for each block
                e.g., {"blocks": {"hamiltonian": {"scale": 1.0, "weight": 1.0},
                                  "momentum": {"scale": 1.0, "weight": 1.0}}}
    
    Returns:
        dict[str, ResidualBlock] with keys: "hamiltonian", "momentum"
    
    Raises:
        ValueError: If constraints.H or constraints.M are not computed
    
    Example:
        >>> from cbtsv1.solvers.gr.defined_coherence_blocks import build_residual_blocks
        >>> blocks = build_residual_blocks(solver.fields, solver.constraints, config)
        >>> print(blocks['hamiltonian'].l2)  # L2 norm of Hamiltonian residual
    """
    blocks = {}
    
    # Get block configuration with defaults
    blocks_cfg = config.get("blocks", {})
    
    hamiltonian_cfg = blocks_cfg.get("hamiltonian", {"scale": 1.0, "weight": 1.0})
    momentum_cfg = blocks_cfg.get("momentum", {"scale": 1.0, "weight": 1.0})
    
    # Validate constraints are computed
    if not hasattr(constraints, 'H') or constraints.H is None:
        raise ValueError("Hamiltonian constraint (H) not computed. Call constraints.compute_hamiltonian() first.")
    
    if not hasattr(constraints, 'M') or constraints.M is None:
        raise ValueError("Momentum constraint (M) not computed. Call constraints.compute_momentum() first.")
    
    # ========================================================================
    # Hamiltonian Block
    # ========================================================================
    # H has shape (Nx, Ny, Nz) - scalar constraint
    hamiltonian_vector = constraints.H.flatten()
    
    blocks["hamiltonian"] = ResidualBlock(
        name="hamiltonian",
        vector=hamiltonian_vector,
        scale=hamiltonian_cfg.get("scale", 1.0),
        weight=hamiltonian_cfg.get("weight", 1.0)
    )
    
    # ========================================================================
    # Momentum Block
    # ========================================================================
    # M has shape (Nx, Ny, Nz, 3) - vector constraint
    # Note: The momentum constraint M^i is trace-adjusted in the constraints module
    # We use the raw M array (already trace-adjusted by GRConstraints)
    momentum_vector = constraints.M.flatten()
    
    blocks["momentum"] = ResidualBlock(
        name="momentum",
        vector=momentum_vector,
        scale=momentum_cfg.get("scale", 1.0),
        weight=momentum_cfg.get("weight", 1.0)
    )
    
    return blocks


def summarize_blocks(blocks: Dict[str, ResidualBlock]) -> Dict[str, Dict]:
    """
    Create audit-friendly summary of blocks (no large arrays).
    
    This produces a summary suitable for storage in receipts/ledgers,
    containing only metadata (dimension, norms, hashes) without the 
    actual residual vectors.
    
    Args:
        blocks: Dictionary of ResidualBlock objects
        
    Returns:
        dict with per-block: dim, l2, linf, hash, scale, weight
    
    Example:
        >>> blocks = build_residual_blocks(fields, constraints, config)
        >>> summary = summarize_blocks(blocks)
        >>> print(summary['hamiltonian']['l2'])
        0.00012345
    """
    summary = {}
    for name, block in blocks.items():
        summary[name] = {
            "dim": block.dim,
            "l2": block.l2,
            "linf": block.linf,
            "hash": block.hash,
            "scale": block.scale,
            "weight": block.weight
        }
    return summary


def build_residual_blocks_with_projection(
    fields: Any,
    constraints: Any,
    config: Dict[str, Any]
) -> Dict[str, ResidualBlock]:
    """
    Build residual blocks including BSSN-Z4 projection constraints.
    
    This variant includes optional projection constraints (Z, Z_i) from
    the BSSN-Z4 formulation if available.
    
    Args:
        fields: GRCoreFields instance
        constraints: GRConstraints instance
        config: Configuration dict with block settings
        
    Returns:
        dict with keys: "hamiltonian", "momentum", optionally "projection"
    """
    blocks = build_residual_blocks(fields, constraints, config)
    
    # Add projection constraint block if available
    projection_cfg = config.get("blocks", {}).get("projection", None)
    if projection_cfg is not None:
        if hasattr(fields, 'Z') and fields.Z is not None:
            # Z is the conformal factor constraint
            z_vector = fields.Z.flatten()
            blocks["projection_conformal"] = ResidualBlock(
                name="projection_conformal",
                vector=z_vector,
                scale=projection_cfg.get("scale", 1.0),
                weight=projection_cfg.get("weight", 1.0)
            )
        
        if hasattr(fields, 'Z_i') and fields.Z_i is not None:
            # Z_i is the vector projection constraint
            z_i_vector = fields.Z_i.flatten()
            blocks["projection_vector"] = ResidualBlock(
                name="projection_vector",
                vector=z_i_vector,
                scale=projection_cfg.get("scale", 1.0),
                weight=projection_cfg.get("weight", 1.0)
            )
    
    return blocks


def load_coherence_config(config_path: str) -> Dict[str, Any]:
    """
    Load coherence configuration from JSON file.
    
    Args:
        config_path: Path to JSON config file
        
    Returns:
        Configuration dictionary
    """
    import json
    with open(config_path, 'r') as f:
        return json.load(f)
