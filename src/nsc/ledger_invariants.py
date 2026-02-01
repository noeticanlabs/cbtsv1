"""Invariant checking logic for LEDGER receipt validation.

This module provides the InvariantChecker class for verifying mathematical
invariants against computational residuals and metrics.
"""

from typing import Dict, Any, Tuple, List, Optional
from .ledger_types import InvariantSpec, InvariantResult


class InvariantChecker:
    """Check invariants against residuals and metrics.
    
    Invariants define mathematical constraints that must hold throughout
    computation. The checker supports both absolute and relative tolerance
    checking.
    
    Attributes:
        invariants: Dictionary mapping invariant IDs to their specifications.
        history: History of values for each invariant.
    """
    
    DEFAULT_ABS_TOLERANCE = 1e-10
    DEFAULT_REL_TOLERANCE = 1e-8
    
    def __init__(self, invariant_specs: Dict[str, InvariantSpec]):
        """Initialize the invariant checker.
        
        Args:
            invariant_specs: Dictionary mapping invariant IDs to InvariantSpec.
        """
        self.invariants = invariant_specs
        self.history: Dict[str, List[float]] = {}
    
    def check(self, 
              invariant_id: str,
              residuals: Dict[str, float],
              metrics: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Check a single invariant.
        
        Args:
            invariant_id: The ID of the invariant to check.
            residuals: Dictionary of residual values from computation.
            metrics: Dictionary of computed metrics.
            
        Returns:
            Tuple of (passed, result_dict).
        """
        spec = self.invariants.get(invariant_id)
        
        # If no invariant is defined, consider it passed
        if not spec:
            return True, {
                "status": "unknown_invariant",
                "invariant_id": invariant_id,
            }
        
        # Get the source value from residuals or metrics
        value = residuals.get(spec.source_metric)
        if value is None:
            value = metrics.get(spec.source_metric)
        
        if value is None:
            return False, {
                "status": "missing_data",
                "invariant_id": invariant_id,
                "source_metric": spec.source_metric,
            }
        
        # Build result
        result: Dict[str, Any] = {
            "invariant_id": invariant_id,
            "value": value,
            "source_metric": spec.source_metric,
            "tolerance_abs": spec.tolerance_abs,
            "tolerance_rel": spec.tolerance_rel,
        }
        
        # Check absolute tolerance
        abs_passed = abs(value) < spec.tolerance_abs
        
        if abs_passed:
            result["status"] = "pass"
            result["reason"] = "within_absolute_tolerance"
            result["passed"] = True
            return True, result
        
        # Check relative tolerance if specified
        if spec.tolerance_rel > 0:
            # For relative tolerance, we need a reference value
            # Use the absolute value as reference if value is non-zero
            ref_value = abs(value)
            if ref_value > 0:
                rel_error = abs(value) / ref_value
                rel_passed = rel_error < spec.tolerance_rel
                
                if rel_passed:
                    result["status"] = "pass"
                    result["reason"] = "within_relative_tolerance"
                    result["relative_error"] = rel_error
                    result["passed"] = True
                    return True, result
        
        # Failed both tolerances
        result["status"] = "fail"
        result["reason"] = "outside_tolerance"
        result["passed"] = False
        return False, result
    
    def check_with_value(self,
                         invariant_id: str,
                         value: float) -> Tuple[bool, Dict[str, Any]]:
        """Check an invariant with a direct value.
        
        Args:
            invariant_id: The ID of the invariant to check.
            value: The value to check against tolerance.
            
        Returns:
            Tuple of (passed, result_dict).
        """
        spec = self.invariants.get(invariant_id)
        
        if not spec:
            return True, {
                "status": "unknown_invariant",
                "invariant_id": invariant_id,
            }
        
        result: Dict[str, Any] = {
            "invariant_id": invariant_id,
            "value": value,
            "tolerance_abs": spec.tolerance_abs,
            "tolerance_rel": spec.tolerance_rel,
        }
        
        # Check absolute tolerance
        abs_passed = abs(value) < spec.tolerance_abs
        
        if abs_passed:
            result["status"] = "pass"
            result["reason"] = "within_absolute_tolerance"
            result["passed"] = True
            return True, result
        
        # Check relative tolerance if specified
        if spec.tolerance_rel > 0 and abs(value) > 0:
            rel_error = abs(value) / abs(value)
            rel_passed = rel_error < spec.tolerance_rel
            
            if rel_passed:
                result["status"] = "pass"
                result["reason"] = "within_relative_tolerance"
                result["relative_error"] = rel_error
                result["passed"] = True
                return True, result
        
        result["status"] = "fail"
        result["reason"] = "outside_tolerance"
        result["passed"] = False
        return False, result
    
    def check_all(self,
                  invariant_ids: List[str],
                  residuals: Dict[str, float],
                  metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Check multiple invariants.
        
        Args:
            invariant_ids: List of invariant IDs to check.
            residuals: Dictionary of residual values.
            metrics: Dictionary of computed metrics.
            
        Returns:
            Dictionary mapping invariant IDs to their check results.
        """
        results = {}
        all_passed = True
        
        for inv_id in invariant_ids:
            passed, result = self.check(inv_id, residuals, metrics)
            results[inv_id] = {
                **result,
                "invariant_id": inv_id,
                "passed": passed,
            }
            if not passed:
                all_passed = False
        
        return results, all_passed
    
    def compute_residual(self,
                         invariant_id: str,
                         actual: Any,
                         expected: Any) -> float:
        """Compute residual for an invariant check.
        
        The residual is the difference between actual and expected values.
        
        Args:
            invariant_id: The invariant being checked.
            actual: The actual computed value.
            expected: The expected/theoretical value.
            
        Returns:
            The absolute residual value.
        """
        spec = self.invariants.get(invariant_id)
        
        if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
            return abs(float(actual) - float(expected))
        elif isinstance(actual, complex) and isinstance(expected, complex):
            return abs(actual - expected)
        else:
            # For non-numeric types, return 0 if equal, 1 if not
            return 0.0 if actual == expected else 1.0
    
    def check_residual(self,
                       invariant_id: str,
                       residual: float) -> Tuple[bool, Dict[str, Any]]:
        """Check a residual against an invariant's tolerance.
        
        Args:
            invariant_id: The invariant to check against.
            residual: The residual value to check.
            
        Returns:
            Tuple of (passed, result_dict).
        """
        spec = self.invariants.get(invariant_id)
        
        if not spec:
            return True, {
                "status": "unknown_invariant",
                "invariant_id": invariant_id,
            }
        
        result: Dict[str, Any] = {
            "invariant_id": invariant_id,
            "residual": residual,
            "tolerance_abs": spec.tolerance_abs,
            "tolerance_rel": spec.tolerance_rel,
        }
        
        # Check absolute tolerance
        passed = residual < spec.tolerance_abs
        
        result["status"] = "pass" if passed else "fail"
        result["passed"] = passed
        
        return passed, result
    
    def check_with_history(self,
                           invariant_id: str,
                           value: float,
                           record: bool = True) -> Tuple[bool, Dict[str, Any]]:
        """Check an invariant with history tracking.
        
        Args:
            invariant_id: The invariant to check.
            value: The value to check.
            record: Whether to record this value in history.
            
        Returns:
            Tuple of (passed, result_dict).
        """
        if record:
            if invariant_id not in self.history:
                self.history[invariant_id] = []
            self.history[invariant_id].append(value)
        
        return self.check_with_value(invariant_id, value)
    
    def check_all_with_history(self,
                               invariant_ids: List[str],
                               residuals: Dict[str, float],
                               metrics: Dict[str, Any],
                               record: bool = True) -> Tuple[Dict[str, Any], bool]:
        """Check multiple invariants with history tracking.
        
        Args:
            invariant_ids: List of invariant IDs to check.
            residuals: Dictionary of residual values.
            metrics: Dictionary of computed metrics.
            record: Whether to record values in history.
            
        Returns:
            Tuple of (results_dict, all_passed).
        """
        results = {}
        all_passed = True
        
        for inv_id in invariant_ids:
            value = residuals.get(self.invariants[inv_id].source_metric, 0.0)
            
            if record:
                passed, result = self.check_with_history(inv_id, value)
            else:
                passed, result = self.check(inv_id, residuals, metrics)
            
            results[inv_id] = {
                **result,
                "invariant_id": inv_id,
                "passed": passed,
            }
            
            if not passed:
                all_passed = False
        
        return results, all_passed
    
    def get_history(self, invariant_id: str) -> List[float]:
        """Get the history of values for an invariant.
        
        Args:
            invariant_id: The invariant ID to get history for.
            
        Returns:
            List of historical values.
        """
        return self.history.get(invariant_id, [])
    
    def clear_history(self, invariant_id: Optional[str] = None) -> None:
        """Clear history for a specific invariant or all invariants.
        
        Args:
            invariant_id: Specific invariant to clear (clears all if None).
        """
        if invariant_id:
            self.history[invariant_id] = []
        else:
            self.history = {}
    
    def get_spec(self, invariant_id: str) -> Optional[InvariantSpec]:
        """Get the specification for an invariant.
        
        Args:
            invariant_id: The invariant ID to look up.
            
        Returns:
            The InvariantSpec or None if not found.
        """
        return self.invariants.get(invariant_id)
    
    def add_spec(self, spec: InvariantSpec) -> None:
        """Add or update an invariant specification.
        
        Args:
            spec: The InvariantSpec to add.
        """
        self.invariants[spec.invariant_id] = spec
    
    def add_specs(self, specs: List[InvariantSpec]) -> None:
        """Add or update multiple invariant specifications.
        
        Args:
            specs: List of InvariantSpec objects to add.
        """
        for spec in specs:
            self.add_spec(spec)
    
    def batch_check(self,
                    residuals: Dict[str, float],
                    metrics: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        """Check all registered invariants against residuals and metrics.
        
        Args:
            residuals: Dictionary of residual values.
            metrics: Dictionary of computed metrics.
            
        Returns:
            Tuple of (results_dict, all_passed).
        """
        return self.check_all(
            list(self.invariants.keys()),
            residuals,
            metrics
        )
