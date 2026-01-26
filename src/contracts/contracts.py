"""
Contract decorators for GR Solver components.

Enforces input/output rules, prerequisites, and exception safety boundaries
as defined in Technical Data/solver_contract.md and other contract specifications.
"""

import functools
import logging
import numpy as np
from typing import Any, Tuple, Dict, Optional, Union

logger = logging.getLogger('gr_solver.contracts')

def enforce_solver_contract(func):
    """
    Decorator to enforce the Solver Contract input/output rules.

    Enforces:
    1.  **Prerequisites**: Checks that geometry (Christoffels and Ricci) is initialized.
    2.  **Input Types**: Validates that `t` is a scalar.
    3.  **Output Format**: Ensures return value is `(RHS, diagnostics)` or `("SEM_FAILURE", reason)`.
    4.  **Exception Safety**: Catches unhandled exceptions and converts them to SEM failures.

    Usage:
        @enforce_solver_contract
        def compute_rhs(self, X, t, gauge_policy, sources_func):
            ...
    """
    @functools.wraps(func)
    def wrapper(self, X, t, gauge_policy, sources_func, *args, **kwargs) -> Union[Tuple[Dict[str, Any], Optional[Dict[str, Any]]], Tuple[str, str]]:
        # 1. Check Prerequisites
        # Attempt to find geometry object on self or self.stepper
        geom = getattr(self, 'geometry', None)
        if geom is None and hasattr(self, 'stepper'):
            geom = getattr(self.stepper, 'geometry', None)

        if geom:
            # Check Christoffels (support both singular and plural naming if ambiguous)
            if (not hasattr(geom, 'christoffels') or geom.christoffels is None) and \
               (not hasattr(geom, 'christoffel') or geom.christoffel is None):
                return "SEM_FAILURE", "Prerequisites not initialized: Christoffels not computed"
            
            # Check Ricci
            if not hasattr(geom, 'ricci') or geom.ricci is None:
                return "SEM_FAILURE", "Prerequisites not initialized: Ricci not computed"

        # 2. Validate Inputs
        if not (isinstance(t, (int, float)) or np.isscalar(t)):
             return "SEM_FAILURE", f"Contract violation: t must be scalar, got {type(t)}"

        # 3. Execute
        try:
            result = func(self, X, t, gauge_policy, sources_func, *args, **kwargs)
        except Exception as e:
            logger.error(f"Solver contract exception in {func.__name__}: {e}", exc_info=True)
            return "SEM_FAILURE", f"Unhandled exception: {str(e)}"

        # 4. Validate Outputs
        if not isinstance(result, tuple) or len(result) != 2:
            return "SEM_FAILURE", f"Contract violation: Expected tuple(RHS, diagnostics), got {type(result)}"

        lhs, rhs = result
        if lhs == "SEM_FAILURE":
            if not isinstance(rhs, str):
                 return "SEM_FAILURE", "Contract violation: Failure reason must be string"
            return result
        
        # Success case validation
        if not isinstance(lhs, dict):
             return "SEM_FAILURE", f"Contract violation: RHS must be dict, got {type(lhs)}"
        
        if rhs is not None and not isinstance(rhs, dict):
             return "SEM_FAILURE", f"Contract violation: Diagnostics must be dict or None, got {type(rhs)}"

        return result

    return wrapper