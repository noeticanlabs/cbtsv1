"""Gate evaluation logic for LEDGER receipt validation.

This module provides the GateEvaluator class for evaluating gate conditions
against computational residuals, including threshold checking and hysteresis.
"""

from typing import Dict, Any, Tuple, List, Optional
from .ledger_types import GateSpec, GateResult


class GateEvaluator:
    """Evaluate gate conditions against computational residuals.
    
    Gates define acceptance conditions for computational steps, with
    configurable thresholds, hysteresis bands, and comparison operators.
    
    Attributes:
        gate_specs: Dictionary mapping gate IDs to their specifications.
        history: History of values for each gate (for window calculations).
    """
    
    COMPARISON_OPS = {
        "le": lambda a, b: a <= b,   # Less than or equal
        "ge": lambda a, b: a >= b,   # Greater than or equal
        "lt": lambda a, b: a < b,    # Less than
        "gt": lambda a, b: a > b,    # Greater than
        "eq": lambda a, b: abs(a - b) < 1e-15,  # Equal (with tolerance)
    }
    
    def __init__(self, gate_specs: Dict[str, GateSpec]):
        """Initialize the gate evaluator.
        
        Args:
            gate_specs: Dictionary mapping gate IDs to GateSpec objects.
        """
        self.gate_specs = gate_specs
        self.history: Dict[str, List[float]] = {}
    
    def evaluate(self, gate_id: str, value: float) -> Tuple[bool, Dict[str, Any]]:
        """Evaluate a single gate condition.
        
        Args:
            gate_id: The ID of the gate to evaluate.
            value: The value to check against the gate threshold.
            
        Returns:
            Tuple of (passed, result_dict).
        """
        spec = self.gate_specs.get(gate_id)
        
        # If no gate is defined, consider it passed
        if not spec:
            return True, {
                "status": "pass",
                "reason": "no_gate_defined",
                "gate_id": gate_id,
            }
        
        # Get the comparison operator
        op = self.COMPARISON_OPS.get(spec.comparison, self.COMPARISON_OPS["le"])
        
        # Check basic threshold
        passed = op(value, spec.threshold)
        
        # Build result
        result: Dict[str, Any] = {
            "gate_id": gate_id,
            "value": value,
            "threshold": spec.threshold,
            "comparison": spec.comparison,
            "status": "pass" if passed else "fail",
        }
        
        # Handle hysteresis for failed thresholds
        if not passed and spec.hysteresis > 0:
            # Check if within hysteresis band of threshold
            # For "le" gates, hysteresis means values just above threshold
            # might still be acceptable in certain contexts
            if spec.comparison == "le":
                if value <= spec.threshold + spec.hysteresis:
                    result["status"] = "review"
                    result["hysteresis_status"] = "within_upper_band"
                    passed = True  # Consider it passing in review mode
            elif spec.comparison == "ge":
                if value >= spec.threshold - spec.hysteresis:
                    result["status"] = "review"
                    result["hysteresis_status"] = "within_lower_band"
                    passed = True
            else:
                result["status"] = "review"
            
            result["hysteresis"] = spec.hysteresis
        
        return passed, result
    
    def evaluate_all(self, residuals: Dict[str, float]) -> Dict[str, Any]:
        """Evaluate all gates against provided residuals.
        
        Args:
            residuals: Dictionary mapping metric names to their residual values.
            
        Returns:
            Dictionary mapping gate IDs to their evaluation results.
        """
        results = {}
        
        # First, check gates defined in specs
        for gate_id, spec in self.gate_specs.items():
            passed, result = self.evaluate(gate_id, residuals.get(spec.gate_id, 0.0))
            results[gate_id] = {
                **result,
                "passed": passed,
            }
        
        # Also check any residuals that might have implicit gates
        for residual_name, value in residuals.items():
            if residual_name not in self.gate_specs:
                # No explicit gate, but still record the value
                passed, result = self.evaluate(residual_name, value)
                results[residual_name] = {
                    **result,
                    "passed": passed,
                    "implicit_gate": True,
                }
        
        return results
    
    def evaluate_with_history(self, 
                              gate_id: str, 
                              value: float,
                              record: bool = True) -> Tuple[bool, Dict[str, Any]]:
        """Evaluate a gate with history tracking.
        
        Records the value in history for window-based evaluations.
        
        Args:
            gate_id: The gate ID to evaluate.
            value: The value to check.
            record: Whether to record this value in history.
            
        Returns:
            Tuple of (passed, result_dict).
        """
        if record:
            if gate_id not in self.history:
                self.history[gate_id] = []
            self.history[gate_id].append(value)
        
        return self.evaluate(gate_id, value)
    
    def check_window(self, 
                     gate_id: str, 
                     values: Optional[List[float]] = None) -> Dict[str, Any]:
        """Check gate over a window of values.
        
        For gates with a window specification, computes the average over
        the window and checks against the threshold.
        
        Args:
            gate_id: The gate to check.
            values: Optional list of values (uses history if not provided).
            
        Returns:
            Dictionary with window check results.
        """
        spec = self.gate_specs.get(gate_id)
        
        if not spec or not spec.window:
            return {
                "status": "na",
                "reason": "no_window_specified",
                "gate_id": gate_id,
            }
        
        # Get values from history if not provided
        if values is None:
            values = self.history.get(gate_id, [])
        
        if len(values) < spec.window:
            return {
                "status": "insufficient_data",
                "gate_id": gate_id,
                "window_size": spec.window,
                "available_points": len(values),
            }
        
        # Get the window of values
        window = values[-spec.window:]
        avg = sum(window) / len(window)
        
        # Evaluate with the average
        op = self.COMPARISON_OPS.get(spec.comparison, self.COMPARISON_OPS["le"])
        passed = op(avg, spec.threshold)
        
        return {
            "status": "pass" if passed else "fail",
            "gate_id": gate_id,
            "window_average": avg,
            "window_size": spec.window,
            "threshold": spec.threshold,
            "comparison": spec.comparison,
            "passed": passed,
            "window_values": window,
        }
    
    def check_all_windows(self) -> Dict[str, Dict[str, Any]]:
        """Check all gates with window specifications.
        
        Returns:
            Dictionary mapping gate IDs to their window check results.
        """
        results = {}
        for gate_id in self.gate_specs:
            results[gate_id] = self.check_window(gate_id)
        return results
    
    def clear_history(self, gate_id: Optional[str] = None) -> None:
        """Clear history for a specific gate or all gates.
        
        Args:
            gate_id: Specific gate to clear (clears all if None).
        """
        if gate_id:
            self.history[gate_id] = []
        else:
            self.history = {}
    
    def get_history(self, gate_id: str) -> List[float]:
        """Get the history of values for a gate.
        
        Args:
            gate_id: The gate ID to get history for.
            
        Returns:
            List of historical values.
        """
        return self.history.get(gate_id, [])
    
    def get_spec(self, gate_id: str) -> Optional[GateSpec]:
        """Get the specification for a gate.
        
        Args:
            gate_id: The gate ID to look up.
            
        Returns:
            The GateSpec or None if not found.
        """
        return self.gate_specs.get(gate_id)
    
    def add_spec(self, spec: GateSpec) -> None:
        """Add or update a gate specification.
        
        Args:
            spec: The GateSpec to add.
        """
        self.gate_specs[spec.gate_id] = spec
    
    def add_specs(self, specs: List[GateSpec]) -> None:
        """Add or update multiple gate specifications.
        
        Args:
            specs: List of GateSpec objects to add.
        """
        for spec in specs:
            self.add_spec(spec)
    
    def batch_evaluate(self, 
                       residuals: Dict[str, float],
                       record_history: bool = True) -> Tuple[bool, Dict[str, Any]]:
        """Evaluate all gates against residuals with optional history recording.
        
        Args:
            residuals: Dictionary of residual values.
            record_history: Whether to record values in history.
            
        Returns:
            Tuple of (all_passed, results_dict).
        """
        results = {}
        all_passed = True
        
        for gate_id in self.gate_specs:
            value = residuals.get(gate_id, 0.0)
            if record_history:
                passed, result = self.evaluate_with_history(gate_id, value)
            else:
                passed, result = self.evaluate(gate_id, value)
            
            results[gate_id] = {
                **result,
                "passed": passed,
            }
            
            if not passed:
                all_passed = False
        
        return all_passed, results
