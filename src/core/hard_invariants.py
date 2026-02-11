# hard_invariants.py
# =============================================================================
# Hard Invariant Validation per Theorem Lemma 1
# =============================================================================
#
# This module implements validation of hard invariants required by Theorem Lemma 1:
# "If hard invariants hold in initial state and accepted states must satisfy 
#  hard invariants, then every accepted state satisfies hard invariants."
#
# Hard invariants validated:
# 1. All metric components finite (not NaN/inf)
# 2. Lapse alpha > 0 (positive everywhere)
# 3. Conformal metric gamma_ij > 0 (positive definite)
# 4. det(gamma_ij) > 0 (determinant positive)
# 5. K_ij symmetric (extrinsic curvature) - guaranteed by storage format
# 6. All components well-defined

import numpy as np
import logging
from typing import Tuple, List, Dict, Any

logger = logging.getLogger('gr_solver.hard_invariants')


class HardInvariantChecker:
    """Validates hard invariants per Theorem Lemma 1.
    
    Enforces that every accepted state satisfies the hard invariants:
    - All metric components finite (not NaN/inf)
    - Lapse alpha > 0 (positive everywhere)
    - Conformal metric gamma_ij positive definite
    - det(gamma_ij) > 0 (determinant positive)
    - K_ij symmetric (by storage format)
    - All components well-defined
    """
    
    def __init__(self, tolerance: float = 1e-14):
        """Initialize hard invariant checker.
        
        Args:
            tolerance: Numerical tolerance for comparisons
        """
        self.tolerance = tolerance
        self.violations = []
    
    def check_hard_invariants(self, fields: Any) -> Tuple[bool, List[str], Dict[str, Any]]:
        """Verify I_hard(x) = true for fields.
        
        Validates all required hard invariants per Theorem Lemma 1.
        
        Args:
            fields: GRCoreFields object with:
                - alpha: Lapse field (shape: spatial grid)
                - gamma_sym6: Conformal metric as sym6 (shape: 6 x spatial grid)
                - K_sym6: Extrinsic curvature as sym6 (shape: 6 x spatial grid)
        
        Returns:
            (is_valid, violations_list, margin_dict) where:
            - is_valid (bool): True if all hard invariants satisfied
            - violations_list (List[str]): List of violation descriptions
            - margin_dict (Dict[str, float]): Safety margins for each check
        """
        violations = []
        margins = {}
        
        # Check 1: Lapse alpha > 0 (positive everywhere)
        if np.any(fields.alpha <= 0):
            violations.append("Lapse alpha not positive everywhere")
            margins['alpha_min'] = float(np.min(fields.alpha))
        else:
            margins['alpha_min'] = float(np.min(fields.alpha))
        
        # Check 2: Lapse alpha finite
        if np.any(~np.isfinite(fields.alpha)):
            violations.append("Lapse alpha contains NaN/inf")
            margins['alpha_has_nan'] = True
        else:
            margins['alpha_has_nan'] = False
        
        # Check 3: Metric gamma_sym6 finite
        if np.any(~np.isfinite(fields.gamma_sym6)):
            violations.append("Conformal metric gamma contains NaN/inf")
            margins['gamma_has_nan'] = True
        else:
            margins['gamma_has_nan'] = False
        
        # Check 4: Extrinsic curvature K_sym6 finite
        if np.any(~np.isfinite(fields.K_sym6)):
            violations.append("Extrinsic curvature K contains NaN/inf")
            margins['K_has_nan'] = True
        else:
            margins['K_has_nan'] = False
        
        # Check 5: Positive definite metric (eigenvalues > 0)
        # gamma_sym6 is stored as 6-vector: [gamma_11, gamma_12, gamma_13, gamma_22, gamma_23, gamma_33]
        gamma_valid, min_eig = self._check_metric_positive_definite(fields.gamma_sym6)
        if not gamma_valid:
            violations.append("Conformal metric not positive definite")
            margins['metric_eigenvalue_min'] = float(min_eig)
        else:
            margins['metric_eigenvalue_min'] = float(min_eig)
        
        # Check 6: Positive determinant of metric
        det_gamma = self._determinant_sym6(fields.gamma_sym6)
        if np.any(det_gamma <= 0):
            violations.append("Metric determinant not positive")
            margins['metric_det_min'] = float(np.min(det_gamma))
        else:
            margins['metric_det_min'] = float(np.min(det_gamma))
        
        # Check 7: K_ij symmetric (guaranteed by storage format, but verify)
        # Since stored as symmetric, this is always true, but we can log it
        margins['K_symmetric'] = True
        
        is_valid = len(violations) == 0
        if not is_valid:
            self.violations.append({
                'violations': violations,
                'margins': margins
            })
            logger.warning(
                f"Hard invariant violations detected: {violations}",
                extra={"extra_data": {"margins": margins}}
            )
        
        return is_valid, violations, margins
    
    def _check_metric_positive_definite(self, gamma_sym6: np.ndarray) -> Tuple[bool, float]:
        """Check that metric is positive definite.
        
        gamma_sym6 format: [11, 12, 13, 22, 23, 33] at each grid point
        
        Args:
            gamma_sym6: Metric as sym6 array (shape: 6 x ...)
        
        Returns:
            (is_positive_definite, min_eigenvalue)
        """
        # Reshape to iterate over grid points
        # gamma_sym6.shape is (6, nx, ny, nz) or (6, n_cells)
        original_shape = gamma_sym6.shape
        n_components = original_shape[0]
        
        if n_components != 6:
            logger.error(f"Expected gamma_sym6 to have 6 components, got {n_components}")
            return False, -np.inf
        
        # Flatten grid dimensions
        gamma_flat = gamma_sym6.reshape(6, -1)
        n_points = gamma_flat.shape[1]
        
        min_eigenvalue = np.inf
        
        # Check eigenvalues at each grid point
        for i in range(n_points):
            # Extract symmetric 3x3 matrix from sym6 components
            g = np.array([
                [gamma_flat[0, i], gamma_flat[1, i], gamma_flat[2, i]],
                [gamma_flat[1, i], gamma_flat[3, i], gamma_flat[4, i]],
                [gamma_flat[2, i], gamma_flat[4, i], gamma_flat[5, i]]
            ])
            
            # Compute eigenvalues
            try:
                eigs = np.linalg.eigvals(g)
                min_eig_here = np.min(eigs)
                min_eigenvalue = min(min_eigenvalue, min_eig_here)
                
                if min_eig_here <= 0:
                    return False, min_eigenvalue
            except np.linalg.LinAlgError:
                logger.error(f"Failed to compute eigenvalues at grid point {i}")
                return False, -np.inf
        
        return True, min_eigenvalue
    
    def _determinant_sym6(self, gamma_sym6: np.ndarray) -> np.ndarray:
        """Compute determinant of symmetric 3x3 matrices stored as sym6.
        
        For a symmetric 3x3 matrix:
        [[a, b, c],
         [b, d, e],
         [c, e, f]]
        
        det = a(df - e²) - b(bf - ce) + c(be - cd)
            = adf - ae² - b²f + bce + bce - c²d
            = adf - ae² - b²f + 2bce - c²d
        
        Args:
            gamma_sym6: Array of shape (6, ...) where components are [a, b, c, d, e, f]
        
        Returns:
            Determinant array of shape (...)
        """
        a = gamma_sym6[0]  # gamma_11
        b = gamma_sym6[1]  # gamma_12
        c = gamma_sym6[2]  # gamma_13
        d = gamma_sym6[3]  # gamma_22
        e = gamma_sym6[4]  # gamma_23
        f = gamma_sym6[5]  # gamma_33
        
        # det = a(df - e²) - b(bf - ce) + c(be - cd)
        det = a * (d * f - e**2) - b * (b * f - c * e) + c * (b * e - c * d)
        
        return det
    
    def _min_eigenvalue(self, gamma_sym6: np.ndarray) -> float:
        """Return minimum eigenvalue of metric across all grid points.
        
        Args:
            gamma_sym6: Metric as sym6 array
        
        Returns:
            Minimum eigenvalue value
        """
        _, min_eig = self._check_metric_positive_definite(gamma_sym6)
        return min_eig
    
    def get_report(self) -> Dict[str, Any]:
        """Return summary of hard invariant violations.
        
        Returns:
            Dictionary with violation summary or success message
        """
        if not self.violations:
            return {
                'status': 'success',
                'message': 'All hard invariants satisfied ✓',
                'num_violations': 0
            }
        
        return {
            'status': 'failed',
            'num_violations': len(self.violations),
            'violations': self.violations
        }
    
    def reset(self) -> None:
        """Reset violation history."""
        self.violations = []
