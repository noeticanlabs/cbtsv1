"""Theorem validators for stability theorems per coherence_math_spine/06_stability_theorems.md

This module implements validation of key stability theorems that must be respected by the solver.
Currently validates Lemma 3 (Debt Boundedness Under Contractive Repair).
"""

import logging
from typing import Tuple, List, Optional, Dict, Any

logger = logging.getLogger('gr_solver.theorem_validator')


class TheoremValidator:
    """Validates stability theorems per coherence_math_spine/06_stability_theorems.md
    
    This validator ensures that the solver respects post-step contractive repair properties.
    Key theorem: Lemma 3 (Debt Boundedness)
        𝔉𝔠(x^(k+1)) ≤ γ·𝔉𝔠(x^(k)) + b, where γ ∈ (0,1)
    
    This must be verified post-step to ensure the solver respects the contractive repair property.
    """
    
    def __init__(self, gamma: float = 0.8, b: float = 1e-4, enable_halt_on_violation: bool = False):
        """Initialize TheoremValidator with contraction parameters.
        
        Args:
            gamma: Contraction coefficient (default 0.8, must be in (0,1))
            b: Affine offset (default 1e-4)
            enable_halt_on_violation: If True, raise on first violation (strict mode)
            
        Raises:
            AssertionError: If gamma is not in (0,1)
        """
        assert 0 < gamma < 1, f"gamma must be in (0,1), got {gamma}"
        self.gamma = gamma
        self.b = b
        self.enable_halt_on_violation = enable_halt_on_violation
        self.violations = []  # Track all violations for post-run analysis
        
        logger.info(f"TheoremValidator initialized with gamma={gamma}, b={b}, halt_on_violation={enable_halt_on_violation}")
    
    def validate_contraction(self, debt_before: float, debt_after: float, step_num: Optional[int] = None) -> Tuple[bool, float, str]:
        """Validate Lemma 3: debt_after ≤ γ·debt_before + b
        
        Lemma 3 (Debt Boundedness Under Contractive Repair):
        The debt function 𝔉𝔠 satisfies a contraction property:
            𝔉𝔠(x^(k+1)) ≤ γ·𝔉𝔠(x^(k)) + b
        where γ ∈ (0,1) is the contraction coefficient and b is an affine offset.
        
        Args:
            debt_before: Debt value at step k (before repair)
            debt_after: Debt value at step k+1 (after repair)
            step_num: Step number for logging (optional)
            
        Returns:
            Tuple[bool, float, str]:
                - is_valid: True if contraction condition is satisfied
                - margin: Threshold - debt_after (margin to threshold, negative if violated)
                - violation_msg: Descriptive message for logging
        """
        threshold = self.gamma * debt_before + self.b
        margin = threshold - debt_after
        is_valid = debt_after <= threshold
        
        if not is_valid:
            msg = (f"Step {step_num}: Contraction violated - "
                   f"debt {debt_after:.6e} > bound {threshold:.6e} "
                   f"(margin {margin:.6e})")
            self.violations.append((step_num, debt_before, debt_after, threshold))
            
            if self.enable_halt_on_violation:
                logger.error(msg)
                raise RuntimeError(msg)
            else:
                logger.warning(msg)
            
            return False, margin, msg
        
        msg = (f"Step {step_num}: Contraction ok - "
               f"debt {debt_after:.6e} ≤ {threshold:.6e} "
               f"(margin {margin:.6e})")
        logger.debug(msg)
        
        return True, margin, msg
    
    def get_violation_report(self) -> Dict[str, Any]:
        """Return summary of all contraction violations.
        
        Returns:
            Dict with keys:
                - 'num_violations': Count of violations
                - 'violations': List of violation tuples (step, debt_before, debt_after, threshold)
                - 'status': Status message for logging
            
            If no violations, returns dict with status "No contractions violated ✓"
        """
        if not self.violations:
            report = {
                'num_violations': 0,
                'violations': [],
                'status': 'No contractions violated ✓'
            }
            logger.info("Theorem validation complete: " + report['status'])
            return report
        
        report = {
            'num_violations': len(self.violations),
            'first_violation_step': self.violations[0][0],
            'violations': self.violations,
            'status': f'{len(self.violations)} contraction violations detected'
        }
        
        logger.warning(f"Theorem validation summary: {report['status']}")
        for i, (step, debt_before, debt_after, threshold) in enumerate(self.violations, 1):
            logger.warning(
                f"  Violation {i}: Step {step} - "
                f"debt {debt_after:.6e} > bound {threshold:.6e} "
                f"(before={debt_before:.6e})"
            )
        
        return report
    
    def reset(self) -> None:
        """Reset violation tracking for a new run.
        
        Clears all recorded violations, allowing the validator to be reused
        for multiple solver runs.
        """
        self.violations = []
        logger.debug("TheoremValidator reset - violations cleared")
