"""Ledger validation for receipt and program verification.

This module provides the LedgerValidator class for validating complete
ledgers including hash chain integrity, gate outcomes, and invariant satisfaction.
"""

from typing import Dict, Any, Tuple, List, Optional
from .ledger_types import (
    Receipt, ReceiptType, LedgerSpec, GateSpec, InvariantSpec
)
from .ledger_hash import HashChain
from .ledger_gates import GateEvaluator
from .ledger_invariants import InvariantChecker


class LedgerValidator:
    """Validate complete ledger including receipts and chain integrity.
    
    The validator checks:
    1. Hash chain continuity and integrity
    2. Required receipts are present
    3. All gate conditions are satisfied
    4. All invariants are satisfied
    
    Attributes:
        ledger_spec: The ledger specification to validate against.
        invariant_checker: InvariantChecker instance.
        gate_evaluator: GateEvaluator instance.
    """
    
    def __init__(self, ledger_spec: LedgerSpec):
        """Initialize the ledger validator.
        
        Args:
            ledger_spec: The LedgerSpec defining validation requirements.
        """
        self.ledger_spec = ledger_spec
        
        # Initialize checkers with specs from ledger
        self.invariant_checker = InvariantChecker({
            inv.invariant_id: inv for inv in ledger_spec.invariants
        })
        
        self.gate_evaluator = GateEvaluator({
            gate.gate_id: gate for gate in ledger_spec.gates
        })
    
    def validate_full_ledger(self, 
                             receipts: List[Receipt]) -> Tuple[bool, Dict[str, Any]]:
        """Validate a complete ledger.
        
        Performs comprehensive validation of all ledger aspects.
        
        Args:
            receipts: List of receipts to validate.
            
        Returns:
            Tuple of (is_valid, results_dict).
        """
        results: Dict[str, Any] = {
            "chain_valid": False,
            "all_gates_passed": False,
            "all_invariants_satisfied": False,
            "required_receipts_present": False,
            "proof_obligations_fulfilled": False,
            "errors": [],
            "warnings": [],
            "receipt_summary": {},
            "gate_results": {},
            "invariant_results": {},
        }
        
        # 1. Validate hash chain
        hash_chain = HashChain(self.ledger_spec.hash_algorithm)
        chain_valid, chain_errors = hash_chain.validate_chain(receipts)
        results["chain_valid"] = chain_valid
        if not chain_valid:
            results["errors"].extend(chain_errors)
        
        # 2. Check required receipts are present
        required = set(self.ledger_spec.required_receipts)
        present = set(rcpt.receipt_type for rcpt in receipts)
        missing = required - present
        
        if missing:
            results["errors"].append(
                f"Missing required receipts: {[r.value for r in missing]}"
            )
        else:
            results["required_receipts_present"] = True
        
        # Build receipt summary
        for rcpt in receipts:
            rcpt_type = rcpt.receipt_type.value
            results["receipt_summary"][rcpt_type] = \
                results["receipt_summary"].get(rcpt_type, 0) + 1
        
        # 3. Check gate outcomes
        gate_results = self._extract_gate_results(receipts)
        results["gate_results"] = gate_results
        
        all_gates_passed = all(
            r.get("status") == "pass" or r.get("status") == "review"
            for r in gate_results.values()
        )
        results["all_gates_passed"] = all_gates_passed
        
        if not all_gates_passed:
            failed_gates = [
                k for k, v in gate_results.items() 
                if v.get("status") == "fail"
            ]
            results["warnings"].append(
                f"Failed gates: {failed_gates}"
            )
        
        # 4. Check invariants
        invariant_results = self._extract_invariant_results(receipts)
        results["invariant_results"] = invariant_results
        
        all_invariants_passed = all(
            r.get("passed", False) for r in invariant_results.values()
        )
        results["all_invariants_satisfied"] = all_invariants_passed
        
        if not all_invariants_passed:
            failed_invariants = [
                k for k, v in invariant_results.items() 
                if not v.get("passed", True)
            ]
            results["errors"].append(
                f"Failed invariants: {failed_invariants}"
            )
        
        # 5. Check proof obligations
        proof_results = self._check_proof_obligations(receipts)
        results["proof_obligations"] = proof_results
        results["proof_obligations_fulfilled"] = all(proof_results.values())
        
        # 6. Compute overall validity
        results["valid"] = (
            results["chain_valid"] and
            results["all_gates_passed"] and
            results["all_invariants_satisfied"] and
            results["required_receipts_present"]
        )
        
        return results["valid"], results
    
    def _extract_gate_results(self, receipts: List[Receipt]) -> Dict[str, Any]:
        """Extract gate results from receipts.
        
        Args:
            receipts: List of receipts to extract from.
            
        Returns:
            Dictionary mapping gate IDs to their results.
        """
        results = {}
        
        for rcpt in receipts:
            if rcpt.gates:
                for gate_id, gate_data in rcpt.gates.items():
                    if gate_id not in results:
                        results[gate_id] = {}
                    results[gate_id].update(gate_data)
            
            # Also check for gate pass/fail receipts
            if rcpt.receipt_type == ReceiptType.GATE_PASS:
                results[rcpt.intent_id] = {
                    "status": "pass",
                    "value": rcpt.residuals.get("value"),
                    "threshold": rcpt.residuals.get("threshold"),
                }
            elif rcpt.receipt_type == ReceiptType.GATE_FAIL:
                results[rcpt.intent_id] = {
                    "status": "fail",
                    "value": rcpt.residuals.get("value"),
                    "threshold": rcpt.residuals.get("threshold"),
                    "reason": rcpt.metrics.get("reason", "threshold_exceeded"),
                }
        
        return results
    
    def _extract_invariant_results(self, receipts: List[Receipt]) -> Dict[str, Any]:
        """Extract invariant results from receipts.
        
        Args:
            receipts: List of receipts to extract from.
            
        Returns:
            Dictionary mapping invariant IDs to their results.
        """
        results = {}
        
        for rcpt in receipts:
            if rcpt.receipt_type == ReceiptType.CHECK_INVARIANT:
                inv_id = rcpt.intent_id
                passed = rcpt.status == "pass"
                results[inv_id] = {
                    "passed": passed,
                    "value": rcpt.residuals.get("value"),
                    "tolerance": rcpt.residuals.get("tolerance"),
                    "status": rcpt.status,
                }
        
        # Also extract from metrics if present
        for rcpt in receipts:
            if hasattr(rcpt, 'metrics') and rcpt.metrics:
                for key, value in rcpt.metrics.items():
                    if key.startswith("invariant_"):
                        inv_id = key[len("invariant_"):]
                        results[inv_id] = results.get(inv_id, {})
                        results[inv_id]["metric_result"] = value
        
        return results
    
    def _check_proof_obligations(self, 
                                  receipts: List[Receipt]) -> Dict[str, bool]:
        """Check proof obligations from receipts.
        
        Args:
            receipts: List of receipts to check.
            
        Returns:
            Dictionary mapping obligation IDs to their fulfillment status.
        """
        results = {}
        
        for obligation in self.ledger_spec.proof_obligations:
            results[obligation] = False
        
        # Check if run summary is present and complete
        for rcpt in receipts:
            if rcpt.receipt_type == ReceiptType.RUN_SUMMARY:
                if rcpt.status == "complete":
                    for obligation in self.ledger_spec.proof_obligations:
                        results[obligation] = True
        
        return results
    
    def validate_receipt_chain(self, 
                                receipts: List[Receipt]) -> Tuple[bool, List[str]]:
        """Validate only the hash chain of receipts.
        
        Args:
            receipts: List of receipts to validate.
            
        Returns:
            Tuple of (is_valid, list_of_errors).
        """
        hash_chain = HashChain(self.ledger_spec.hash_algorithm)
        return hash_chain.validate_chain(receipts)
    
    def validate_gates(self, 
                       residuals: Dict[str, float]) -> Tuple[bool, Dict[str, Any]]:
        """Validate gates against residuals.
        
        Args:
            residuals: Dictionary of residual values.
            
        Returns:
            Tuple of (all_passed, results_dict).
        """
        all_passed, results = self.gate_evaluator.batch_evaluate(residuals)
        return all_passed, results
    
    def validate_invariants(self,
                            residuals: Dict[str, float],
                            metrics: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Validate invariants against residuals and metrics.
        
        Args:
            residuals: Dictionary of residual values.
            metrics: Dictionary of computed metrics.
            
        Returns:
            Tuple of (all_passed, results_dict).
        """
        results, all_passed = self.invariant_checker.batch_check(
            residuals, metrics
        )
        return all_passed, results
    
    def check_receipt_required(self, 
                               receipt_type: ReceiptType,
                               receipts: List[Receipt]) -> bool:
        """Check if a required receipt type is present.
        
        Args:
            receipt_type: The receipt type to check.
            receipts: List of receipts to check.
            
        Returns:
            True if the receipt type is present.
        """
        return any(rcpt.receipt_type == receipt_type for rcpt in receipts)
    
    def get_missing_receipts(self, receipts: List[Receipt]) -> List[ReceiptType]:
        """Get list of required receipt types that are missing.
        
        Args:
            receipts: List of receipts to check.
            
        Returns:
            List of missing ReceiptType values.
        """
        required = set(self.ledger_spec.required_receipts)
        present = set(rcpt.receipt_type for rcpt in receipts)
        return list(required - present)
    
    def get_validation_summary(self, 
                               receipts: List[Receipt]) -> Dict[str, Any]:
        """Get a summary of validation status.
        
        Args:
            receipts: List of receipts to validate.
            
        Returns:
            Dictionary with validation summary.
        """
        valid, results = self.validate_full_ledger(receipts)
        
        return {
            "valid": valid,
            "chain_valid": results["chain_valid"],
            "gates_valid": results["all_gates_passed"],
            "invariants_valid": results["all_invariants_satisfied"],
            "receipts_complete": results["required_receipts_present"],
            "error_count": len(results["errors"]),
            "warning_count": len(results["warnings"]),
            "total_receipts": len(receipts),
            "receipt_types_present": list(results["receipt_summary"].keys()),
        }
    
    def add_invariant(self, spec: InvariantSpec) -> None:
        """Add an invariant specification.
        
        Args:
            spec: The InvariantSpec to add.
        """
        self.invariant_checker.add_spec(spec)
        self.ledger_spec.invariants.append(spec)
    
    def add_gate(self, spec: GateSpec) -> None:
        """Add a gate specification.
        
        Args:
            spec: The GateSpec to add.
        """
        self.gate_evaluator.add_spec(spec)
        self.ledger_spec.gates.append(spec)
    
    def set_required_receipts(self, 
                               receipt_types: List[ReceiptType]) -> None:
        """Set required receipt types.
        
        Args:
            receipt_types: List of required receipt types.
        """
        self.ledger_spec.required_receipts = receipt_types
