"""LEDGER lowering for NSC-M3L.

This module provides the LedgerLowerer class for lowering NSC-M3L programs
to the LEDGER model, extracting invariants, gates, and proof obligations.
"""

from typing import Dict, Any, List, Optional
from .ledger_types import (
    LedgerSpec, GateSpec, InvariantSpec, ReceiptType
)


class LedgerLowerer:
    """Lower NSC-M3L programs to LEDGER model.
    
    The lowerer extracts LEDGER specifications from program constructs,
    including @inv and @gate directives, and builds the complete LedgerSpec.
    
    Attributes:
        ledger_spec: The resulting ledger specification.
    """
    
    # Default gate thresholds for common residuals
    DEFAULT_GATES = {
        "residual_l2": GateSpec(
            gate_id="residual_l2",
            threshold=1e-10,
            hysteresis=0.0,
            comparison="le",
        ),
        "residual_max": GateSpec(
            gate_id="residual_max",
            threshold=1e-8,
            hysteresis=0.0,
            comparison="le",
        ),
        "cfl_number": GateSpec(
            gate_id="cfl_number",
            threshold=0.5,
            hysteresis=0.1,
            comparison="le",
        ),
        "energy_change": GateSpec(
            gate_id="energy_change",
            threshold=1e-6,
            hysteresis=0.0,
            comparison="le",
        ),
    }
    
    # Default invariants
    DEFAULT_INVARIANTS = {
        "mass_conservation": InvariantSpec(
            invariant_id="mass_conservation",
            tolerance_abs=1e-10,
            tolerance_rel=1e-8,
            gate_key="residual_l2",
            source_metric="mass",
        ),
        "energy_conservation": InvariantSpec(
            invariant_id="energy_conservation",
            tolerance_abs=1e-8,
            tolerance_rel=1e-6,
            gate_key="energy_change",
            source_metric="energy",
        ),
    }
    
    # Required receipts for typical runs
    DEFAULT_REQUIRED_RECEIPTS = [
        ReceiptType.STEP_PROPOSED,
        ReceiptType.STEP_ACCEPTED,
        ReceiptType.CHECK_INVARIANT,
        ReceiptType.RUN_SUMMARY,
    ]
    
    def __init__(self, ledger_spec: Optional[LedgerSpec] = None):
        """Initialize the ledger lowerer.
        
        Args:
            ledger_spec: Optional initial ledger specification.
        """
        self.ledger_spec = ledger_spec or LedgerSpec()
    
    def lower(self, program: Any) -> LedgerSpec:
        """Lower a program to a ledger specification.
        
        Extracts all LEDGER-related constructs from the program and
        builds the complete LedgerSpec.
        
        Args:
            program: The NSC-M3L program to lower.
            
        Returns:
            The generated LedgerSpec.
        """
        # Extract from directives in program statements
        for stmt in self._get_statements(program):
            # Process @inv directives
            inv_directives = self._extract_invariant_directives(stmt)
            for inv_id, inv_spec in inv_directives.items():
                if not self.ledger_spec.get_invariant(inv_id):
                    self.ledger_spec.invariants.append(inv_spec)
            
            # Process @gate directives
            gate_directives = self._extract_gate_directives(stmt)
            for gate_id, gate_spec in gate_directives.items():
                if not self.ledger_spec.get_gate(gate_id):
                    self.ledger_spec.gates.append(gate_spec)
            
            # Process other ledger-related directives
            proof_obligations = self._extract_proof_obligations(stmt)
            for obligation in proof_obligations:
                if obligation not in self.ledger_spec.proof_obligations:
                    self.ledger_spec.proof_obligations.append(obligation)
        
        return self.ledger_spec
    
    def _get_statements(self, program: Any) -> List[Any]:
        """Get statements from a program.
        
        Args:
            program: The program object.
            
        Returns:
            List of statements.
        """
        if hasattr(program, 'statements'):
            return program.statements
        elif hasattr(program, 'body'):
            return program.body
        elif isinstance(program, list):
            return program
        else:
            return []
    
    def _extract_invariant_directives(self, stmt: Any) -> Dict[str, InvariantSpec]:
        """Extract invariant directives from a statement.
        
        Args:
            stmt: The statement to process.
            
        Returns:
            Dictionary mapping invariant IDs to their specs.
        """
        result = {}
        
        if not hasattr(stmt, 'directives'):
            return result
        
        for directive in stmt.directives:
            if directive.type == "inv":
                # Extract invariant information
                inv_id = getattr(directive, 'invariant_id', None) or \
                         getattr(directive, 'id', None) or \
                         getattr(directive, 'name', '')
                
                if inv_id:
                    spec = InvariantSpec(
                        invariant_id=inv_id,
                        tolerance_abs=getattr(directive, 'tolerance_abs', 1e-10),
                        tolerance_rel=getattr(directive, 'tolerance_rel', 1e-8),
                        gate_key=getattr(directive, 'gate_key', ''),
                        source_metric=getattr(directive, 'source_metric', ''),
                    )
                    result[inv_id] = spec
        
        return result
    
    def _extract_gate_directives(self, stmt: Any) -> Dict[str, GateSpec]:
        """Extract gate directives from a statement.
        
        Args:
            stmt: The statement to process.
            
        Returns:
            Dictionary mapping gate IDs to their specs.
        """
        result = {}
        
        if not hasattr(stmt, 'directives'):
            return result
        
        for directive in stmt.directives:
            if directive.type == "gate":
                # Extract gate information
                gate_id = getattr(directive, 'gate_id', None) or \
                          getattr(directive, 'id', None) or \
                          getattr(directive, 'name', '')
                
                if gate_id:
                    spec = GateSpec(
                        gate_id=gate_id,
                        threshold=getattr(directive, 'threshold', 1.0),
                        hysteresis=getattr(directive, 'hysteresis', 0.0),
                        comparison=getattr(directive, 'comparison', 'le'),
                        window=getattr(directive, 'window', None),
                    )
                    result[gate_id] = spec
        
        return result
    
    def _extract_proof_obligations(self, stmt: Any) -> List[str]:
        """Extract proof obligations from a statement.
        
        Args:
            stmt: The statement to process.
            
        Returns:
            List of proof obligation identifiers.
        """
        result = []
        
        if not hasattr(stmt, 'directives'):
            return result
        
        for directive in stmt.directives:
            if directive.type == "proof":
                obligation = getattr(directive, 'obligation_id', None) or \
                             getattr(directive, 'id', None) or \
                             getattr(directive, 'name', '')
                if obligation:
                    result.append(obligation)
        
        return result
    
    def extract_gate_requirements(self, program: Any) -> Dict[str, GateSpec]:
        """Extract gate requirements from a program.
        
        Args:
            program: The program to extract from.
            
        Returns:
            Dictionary mapping gate IDs to their specs.
        """
        gates = {}
        
        for stmt in self._get_statements(program):
            stmt_gates = self._extract_gate_directives(stmt)
            gates.update(stmt_gates)
        
        return gates
    
    def extract_invariant_requirements(self, program: Any) -> Dict[str, InvariantSpec]:
        """Extract invariant requirements from a program.
        
        Args:
            program: The program to extract from.
            
        Returns:
            Dictionary mapping invariant IDs to their specs.
        """
        invariants = {}
        
        for stmt in self._get_statements(program):
            stmt_invariants = self._extract_invariant_directives(stmt)
            invariants.update(stmt_invariants)
        
        return invariants
    
    def add_default_gates(self) -> None:
        """Add default gate specifications."""
        for gate_id, gate in self.DEFAULT_GATES.items():
            if not self.ledger_spec.get_gate(gate_id):
                self.ledger_spec.gates.append(gate)
    
    def add_default_invariants(self) -> None:
        """Add default invariant specifications."""
        for inv_id, inv in self.DEFAULT_INVARIANTS.items():
            if not self.ledger_spec.get_invariant(inv_id):
                self.ledger_spec.invariants.append(inv)
    
    def set_default_required_receipts(self) -> None:
        """Set default required receipt types."""
        self.ledger_spec.required_receipts = self.DEFAULT_REQUIRED_RECEIPTS.copy()
    
    def build_from_directives(self, 
                               inv_directives: List[Dict[str, Any]],
                               gate_directives: List[Dict[str, Any]]) -> LedgerSpec:
        """Build ledger spec from directive dictionaries.
        
        Args:
            inv_directives: List of invariant directive dicts.
            gate_directives: List of gate directive dicts.
            
        Returns:
            The built LedgerSpec.
        """
        for inv in inv_directives:
            spec = InvariantSpec(
                invariant_id=inv.get('invariant_id', ''),
                tolerance_abs=inv.get('tolerance_abs', 1e-10),
                tolerance_rel=inv.get('tolerance_rel', 1e-8),
                gate_key=inv.get('gate_key', ''),
                source_metric=inv.get('source_metric', ''),
            )
            self.ledger_spec.invariants.append(spec)
        
        for gate in gate_directives:
            spec = GateSpec(
                gate_id=gate.get('gate_id', ''),
                threshold=gate.get('threshold', 1.0),
                hysteresis=gate.get('hysteresis', 0.0),
                comparison=gate.get('comparison', 'le'),
                window=gate.get('window', None),
            )
            self.ledger_spec.gates.append(spec)
        
        return self.ledger_spec
    
    def get_ledger_spec(self) -> LedgerSpec:
        """Get the current ledger specification.
        
        Returns:
            The current LedgerSpec.
        """
        return self.ledger_spec
    
    def set_ledger_spec(self, spec: LedgerSpec) -> None:
        """Set the ledger specification.
        
        Args:
            spec: The LedgerSpec to set.
        """
        self.ledger_spec = spec
