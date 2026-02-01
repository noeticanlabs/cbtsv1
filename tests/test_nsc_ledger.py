"""Tests for NSC-M3L LEDGER Integration.

This module contains comprehensive tests for the LEDGER model including:
- Receipt generation and serialization
- Hash chain validation
- Gate evaluation
- Invariant checking
- Ledger specification creation
- Full ledger validation
"""

import pytest
import json
import time
from typing import Dict, Any, List

from src.nsc.ledger_types import (
    Receipt, ReceiptType, LedgerSpec, GateSpec, InvariantSpec, GateResult, InvariantResult
)
from src.nsc.ledger_hash import HashChain
from src.nsc.ledger_gates import GateEvaluator
from src.nsc.ledger_invariants import InvariantChecker
from src.nsc.ledger_receipts import ReceiptGenerator
from src.nsc.ledger_validator import LedgerValidator
from src.nsc.ledger_lower import LedgerLowerer


class TestReceiptTypes:
    """Tests for receipt type enumeration."""
    
    def test_receipt_type_values(self):
        """Test receipt type enum values."""
        assert ReceiptType.STEP_PROPOSED.value == "A:RCPT.step.proposed"
        assert ReceiptType.STEP_ACCEPTED.value == "A:RCPT.step.accepted"
        assert ReceiptType.STEP_REJECTED.value == "A:RCPT.step.rejected"
        assert ReceiptType.GATE_PASS.value == "A:RCPT.gate.pass"
        assert ReceiptType.GATE_FAIL.value == "A:RCPT.gate.fail"
        assert ReceiptType.CHECK_INVARIANT.value == "A:RCPT.check.invariant"
        assert ReceiptType.CKPT_CREATED.value == "A:RCPT.ckpt.created"
        assert ReceiptType.ROLLBACK_EXECUTED.value == "A:RCPT.rollback.executed"
        assert ReceiptType.RUN_SUMMARY.value == "A:RCPT.run.summary"
    
    def test_receipt_type_count(self):
        """Test total number of receipt types."""
        assert len(ReceiptType) == 9


class TestReceipt:
    """Tests for Receipt dataclass."""
    
    def test_receipt_creation(self):
        """Test basic receipt creation."""
        rcpt = Receipt(
            receipt_type=ReceiptType.STEP_PROPOSED,
            timestamp=1234567890.0,
            run_id="R-test123",
            step_id=1,
            intent_id="intent_1",
            ops=["op1", "op2"],
            residuals={"res1": 1e-10},
        )
        
        assert rcpt.receipt_type == ReceiptType.STEP_PROPOSED
        assert rcpt.run_id == "R-test123"
        assert rcpt.step_id == 1
        assert len(rcpt.ops) == 2
        assert rcpt.residuals["res1"] == 1e-10
    
    def test_receipt_to_dict(self):
        """Test receipt serialization to dictionary."""
        rcpt = Receipt(
            receipt_type=ReceiptType.STEP_ACCEPTED,
            timestamp=1234567890.0,
            run_id="R-test123",
            status="accepted",
        )
        
        data = rcpt.to_dict()
        
        assert data["receipt_type"] == "A:RCPT.step.accepted"
        assert data["run_id"] == "R-test123"
        assert data["status"] == "accepted"
    
    def test_receipt_from_dict(self):
        """Test receipt deserialization from dictionary."""
        data = {
            "receipt_type": "A:RCPT.gate.pass",
            "timestamp": 1234567890.0,
            "run_id": "R-test456",
            "status": "pass",
            "intent_id": "gate_1",
        }
        
        rcpt = Receipt.from_dict(data)
        
        assert rcpt.receipt_type == ReceiptType.GATE_PASS
        assert rcpt.run_id == "R-test456"
        assert rcpt.status == "pass"


class TestGateSpec:
    """Tests for GateSpec dataclass."""
    
    def test_gate_spec_creation(self):
        """Test gate specification creation."""
        gate = GateSpec(
            gate_id="residual_l2",
            threshold=1e-10,
            hysteresis=1e-12,
            comparison="le",
            window=10,
        )
        
        assert gate.gate_id == "residual_l2"
        assert gate.threshold == 1e-10
        assert gate.hysteresis == 1e-12
        assert gate.comparison == "le"
        assert gate.window == 10
    
    def test_gate_spec_serialization(self):
        """Test gate spec serialization."""
        gate = GateSpec(gate_id="test_gate", threshold=0.5, comparison="lt")
        
        data = gate.to_dict()
        restored = GateSpec.from_dict(data)
        
        assert restored.gate_id == gate.gate_id
        assert restored.threshold == gate.threshold


class TestInvariantSpec:
    """Tests for InvariantSpec dataclass."""
    
    def test_invariant_spec_creation(self):
        """Test invariant specification creation."""
        inv = InvariantSpec(
            invariant_id="mass_conservation",
            tolerance_abs=1e-10,
            tolerance_rel=1e-8,
            gate_key="residual_l2",
            source_metric="mass",
        )
        
        assert inv.invariant_id == "mass_conservation"
        assert inv.tolerance_abs == 1e-10
        assert inv.gate_key == "residual_l2"
    
    def test_invariant_spec_serialization(self):
        """Test invariant spec serialization."""
        inv = InvariantSpec(invariant_id="test_inv", tolerance_abs=1e-12)
        
        data = inv.to_dict()
        restored = InvariantSpec.from_dict(data)
        
        assert restored.invariant_id == inv.invariant_id
        assert restored.tolerance_abs == inv.tolerance_abs


class TestLedgerSpec:
    """Tests for LedgerSpec dataclass."""
    
    def test_ledger_spec_creation(self):
        """Test ledger specification creation."""
        spec = LedgerSpec(
            invariants=[
                InvariantSpec(invariant_id="inv1", tolerance_abs=1e-10)
            ],
            gates=[
                GateSpec(gate_id="gate1", threshold=0.5)
            ],
            required_receipts=[ReceiptType.STEP_PROPOSED],
            proof_obligations=["PO-001"],
        )
        
        assert len(spec.invariants) == 1
        assert len(spec.gates) == 1
        assert len(spec.required_receipts) == 1
    
    def test_get_gate(self):
        """Test getting a gate by ID."""
        spec = LedgerSpec(
            gates=[GateSpec(gate_id="test_gate", threshold=0.5)]
        )
        
        gate = spec.get_gate("test_gate")
        assert gate is not None
        assert gate.threshold == 0.5
        
        missing = spec.get_gate("missing_gate")
        assert missing is None
    
    def test_get_invariant(self):
        """Test getting an invariant by ID."""
        spec = LedgerSpec(
            invariants=[InvariantSpec(invariant_id="test_inv", tolerance_abs=1e-10)]
        )
        
        inv = spec.get_invariant("test_inv")
        assert inv is not None
        assert inv.tolerance_abs == 1e-10
    
    def test_ledger_spec_serialization(self):
        """Test ledger spec serialization."""
        spec = LedgerSpec(
            invariants=[InvariantSpec(invariant_id="inv1")],
            gates=[GateSpec(gate_id="gate1", threshold=1.0)],
            hash_algorithm="sha512",
        )
        
        data = spec.to_dict()
        restored = LedgerSpec.from_dict(data)
        
        assert len(restored.invariants) == 1
        assert len(restored.gates) == 1
        assert restored.hash_algorithm == "sha512"


class TestHashChain:
    """Tests for HashChain class."""
    
    def test_hash_chain_initialization(self):
        """Test hash chain initialization."""
        chain = HashChain()
        
        assert chain.algorithm == "sha256"
        assert chain.genesis_hash is not None
        assert len(chain.genesis_hash) == 64  # SHA-256 hex length
    
    def test_custom_algorithm(self):
        """Test hash chain with custom algorithm."""
        chain = HashChain(algorithm="sha512")
        
        assert chain.algorithm == "sha512"
        assert len(chain.genesis_hash) == 128  # SHA-512 hex length
    
    def test_invalid_algorithm(self):
        """Test hash chain with invalid algorithm."""
        with pytest.raises(ValueError):
            HashChain(algorithm="invalid")
    
    def test_compute_hash(self):
        """Test hash computation."""
        chain = HashChain()
        data = {"key": "value", "number": 42}
        
        hash1 = chain.compute_hash(data, None)
        hash2 = chain.compute_hash(data, None)
        
        # Same data should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 64
    
    def test_compute_hash_with_prev(self):
        """Test hash computation with previous hash."""
        chain = HashChain()
        data = {"key": "value"}
        
        hash1 = chain.compute_hash(data, "prevhash123")
        hash2 = chain.compute_hash(data, "differenthash")
        
        # Different prev hash should produce different hash
        assert hash1 != hash2
    
    def test_validate_chain_empty(self):
        """Test validating empty chain."""
        chain = HashChain()
        valid, errors = chain.validate_chain([])
        
        assert valid
        assert len(errors) == 0
    
    def test_validate_chain_valid(self):
        """Test validating a valid chain."""
        chain = HashChain()
        
        # Create actual Receipt objects for proper serialization
        receipts = []
        prev_hash = chain.genesis_hash
        
        for i in range(3):
            rcpt = Receipt(
                receipt_type=ReceiptType.STEP_PROPOSED,
                timestamp=1234567890.0 + i,
                run_id="R-test",
                step_id=i + 1,
                hash_prev=prev_hash,
            )
            # Compute hash using the chain
            data = chain.serialize_for_hash(rcpt)
            rcpt.hash = chain.compute_hash(data, rcpt.hash_prev)
            receipts.append(rcpt)
            prev_hash = rcpt.hash
        
        valid, errors = chain.validate_chain(receipts)
        
        assert valid, f"Chain validation failed: {errors}"
        assert len(errors) == 0
    
    def test_validate_chain_broken(self):
        """Test validating a broken chain."""
        chain = HashChain()
        
        class MockReceipt:
            def __init__(self, prev_hash, curr_hash):
                self.hash_prev = prev_hash
                self.hash = curr_hash
        
        receipts = [
            MockReceipt(chain.genesis_hash, "wronghash"),
        ]
        
        valid, errors = chain.validate_chain(receipts)
        
        assert not valid
        assert len(errors) > 0
    
    def test_serialize_for_hash(self):
        """Test serialization for hashing."""
        chain = HashChain()
        
        rcpt = Receipt(
            receipt_type=ReceiptType.STEP_PROPOSED,
            timestamp=1234567890.0,
            run_id="R-test",
            hash="should_be_excluded",
            hash_prev="should_also_be_excluded",
        )
        
        data = chain.serialize_for_hash(rcpt)
        
        assert "hash" not in data
        assert "hash_prev" not in data
        assert data["run_id"] == "R-test"


class TestGateEvaluator:
    """Tests for GateEvaluator class."""
    
    def test_evaluate_le_pass(self):
        """Test less-than-or-equal comparison that passes."""
        spec = GateSpec(gate_id="test", threshold=0.5, comparison="le")
        evaluator = GateEvaluator({"test": spec})
        
        passed, result = evaluator.evaluate("test", 0.3)
        
        assert passed
        assert result["status"] == "pass"
    
    def test_evaluate_le_fail(self):
        """Test less-than-or-equal comparison that fails."""
        spec = GateSpec(gate_id="test", threshold=0.5, comparison="le")
        evaluator = GateEvaluator({"test": spec})
        
        passed, result = evaluator.evaluate("test", 0.7)
        
        assert not passed
        assert result["status"] == "fail"
    
    def test_evaluate_ge_pass(self):
        """Test greater-than-or-equal comparison."""
        spec = GateSpec(gate_id="test", threshold=0.5, comparison="ge")
        evaluator = GateEvaluator({"test": spec})
        
        passed, result = evaluator.evaluate("test", 0.7)
        
        assert passed
        assert result["status"] == "pass"
    
    def test_evaluate_lt(self):
        """Test less-than comparison."""
        spec = GateSpec(gate_id="test", threshold=0.5, comparison="lt")
        evaluator = GateEvaluator({"test": spec})
        
        passed, _ = evaluator.evaluate("test", 0.3)
        assert passed
        
        passed, _ = evaluator.evaluate("test", 0.5)
        assert not passed
    
    def test_evaluate_gt(self):
        """Test greater-than comparison."""
        spec = GateSpec(gate_id="test", threshold=0.5, comparison="gt")
        evaluator = GateEvaluator({"test": spec})
        
        passed, _ = evaluator.evaluate("test", 0.7)
        assert passed
        
        passed, _ = evaluator.evaluate("test", 0.5)
        assert not passed
    
    def test_evaluate_no_gate(self):
        """Test evaluation when no gate is defined."""
        evaluator = GateEvaluator({})
        
        passed, result = evaluator.evaluate("unknown", 0.5)
        
        assert passed
        assert result["reason"] == "no_gate_defined"
    
    def test_evaluate_with_hysteresis(self):
        """Test evaluation with hysteresis band."""
        spec = GateSpec(
            gate_id="test",
            threshold=0.5,
            hysteresis=0.1,
            comparison="le"
        )
        evaluator = GateEvaluator({"test": spec})
        
        # Value just above threshold but within hysteresis
        passed, result = evaluator.evaluate("test", 0.55)
        
        assert result["status"] == "review"
        assert result["hysteresis"] == 0.1
    
    def test_evaluate_all(self):
        """Test evaluating all gates."""
        spec1 = GateSpec(gate_id="gate1", threshold=0.5, comparison="le")
        spec2 = GateSpec(gate_id="gate2", threshold=0.3, comparison="le")
        evaluator = GateEvaluator({"gate1": spec1, "gate2": spec2})
        
        residuals = {"gate1": 0.2, "gate2": 0.5}
        results = evaluator.evaluate_all(residuals)
        
        assert results["gate1"]["passed"]
        assert not results["gate2"]["passed"]
    
    def test_check_window(self):
        """Test window-based evaluation."""
        spec = GateSpec(gate_id="test", threshold=0.5, comparison="le", window=3)
        evaluator = GateEvaluator({"test": spec})
        
        # Add history
        evaluator.history["test"] = [0.2, 0.3, 0.4]
        
        result = evaluator.check_window("test")
        
        assert result["status"] == "pass"
        assert result["window_average"] == pytest.approx(0.3)
    
    def test_check_window_insufficient_data(self):
        """Test window check with insufficient data."""
        spec = GateSpec(gate_id="test", threshold=0.5, comparison="le", window=5)
        evaluator = GateEvaluator({"test": spec})
        
        evaluator.history["test"] = [0.2, 0.3, 0.4]
        
        result = evaluator.check_window("test")
        
        assert result["status"] == "insufficient_data"


class TestInvariantChecker:
    """Tests for InvariantChecker class."""
    
    def test_check_pass_absolute(self):
        """Test invariant check that passes absolute tolerance."""
        spec = InvariantSpec(
            invariant_id="test",
            tolerance_abs=1e-8,
            tolerance_rel=0.0,
        )
        checker = InvariantChecker({"test": spec})
        
        passed, result = checker.check_with_value("test", 1e-10)
        
        assert passed
        assert result["status"] == "pass"
    
    def test_check_fail_absolute(self):
        """Test invariant check that fails absolute tolerance."""
        spec = InvariantSpec(
            invariant_id="test",
            tolerance_abs=1e-10,
            tolerance_rel=0.0,
        )
        checker = InvariantChecker({"test": spec})
        
        passed, result = checker.check_with_value("test", 1e-5)
        
        assert not passed
        assert result["status"] == "fail"
    
    def test_check_no_invariant(self):
        """Test check when invariant is not defined."""
        checker = InvariantChecker({})
        
        passed, result = checker.check_with_value("unknown", 0.5)
        
        assert passed
        assert result["status"] == "unknown_invariant"
    
    def test_check_with_residuals(self):
        """Test check with residuals dictionary."""
        spec = InvariantSpec(
            invariant_id="mass_cons",
            tolerance_abs=1e-10,
            source_metric="mass_residual",
        )
        checker = InvariantChecker({"mass_cons": spec})
        
        residuals = {"mass_residual": 1e-12}
        metrics = {}
        
        passed, result = checker.check("mass_cons", residuals, metrics)
        
        assert passed
        assert result["source_metric"] == "mass_residual"
    
    def test_check_missing_data(self):
        """Test check when source metric is missing."""
        spec = InvariantSpec(
            invariant_id="test",
            tolerance_abs=1e-10,
            source_metric="missing_metric",
        )
        checker = InvariantChecker({"test": spec})
        
        residuals = {"other_metric": 0.5}
        metrics = {}
        
        passed, result = checker.check("test", residuals, metrics)
        
        assert not passed
        assert result["status"] == "missing_data"
    
    def test_compute_residual_numeric(self):
        """Test residual computation for numeric values."""
        checker = InvariantChecker({})
        
        residual = checker.compute_residual("test", 1.5, 1.0)
        
        assert residual == 0.5
    
    def test_compute_residual_non_numeric(self):
        """Test residual computation for non-numeric values."""
        checker = InvariantChecker({})
        
        residual_eq = checker.compute_residual("test", "a", "a")
        residual_neq = checker.compute_residual("test", "a", "b")
        
        assert residual_eq == 0.0
        assert residual_neq == 1.0
    
    def test_check_all(self):
        """Test checking multiple invariants."""
        spec1 = InvariantSpec(invariant_id="inv1", tolerance_abs=1e-8, source_metric="inv1")
        spec2 = InvariantSpec(invariant_id="inv2", tolerance_abs=1e-8, source_metric="inv2")
        checker = InvariantChecker({"inv1": spec1, "inv2": spec2})
        
        residuals = {"inv1": 1e-10, "inv2": 1e-5}
        metrics = {}
        
        results, all_passed = checker.check_all(
            ["inv1", "inv2"],
            residuals,
            metrics,
        )
        
        assert results["inv1"]["passed"]
        assert not results["inv2"]["passed"]
        assert not all_passed


class TestReceiptGenerator:
    """Tests for ReceiptGenerator class."""
    
    def test_generator_initialization(self):
        """Test receipt generator initialization."""
        spec = LedgerSpec()
        gen = ReceiptGenerator(spec)
        
        assert gen.run_id is not None
        assert gen.run_id.startswith("R-")
        assert len(gen.receipts) == 0
    
    def test_generate_step_proposed(self):
        """Test step proposed receipt generation."""
        spec = LedgerSpec()
        gen = ReceiptGenerator(spec)
        
        rcpt = gen.generate_step_proposed(
            intent_id="intent_1",
            ops=["op1", "op2"],
            residuals={"res1": 1e-10},
        )
        
        assert rcpt.receipt_type == ReceiptType.STEP_PROPOSED
        assert rcpt.step_id == 1
        assert rcpt.intent_id == "intent_1"
        assert rcpt.hash is not None
        assert rcpt.hash_prev == gen.hash_chain.genesis_hash
    
    def test_generate_step_accepted(self):
        """Test step accepted receipt generation."""
        spec = LedgerSpec()
        gen = ReceiptGenerator(spec)
        
        proposed = gen.generate_step_proposed("intent_1", ["op1"], {"res1": 1e-10})
        accepted = gen.generate_step_accepted(
            proposed,
            {"gate1": {"status": "pass"}},
            {"metric1": 0.5},
        )
        
        assert accepted.receipt_type == ReceiptType.STEP_ACCEPTED
        assert accepted.step_id == proposed.step_id
        assert accepted.hash_prev == proposed.hash
    
    def test_generate_step_rejected(self):
        """Test step rejected receipt generation."""
        spec = LedgerSpec()
        gen = ReceiptGenerator(spec)
        
        proposed = gen.generate_step_proposed("intent_1", ["op1"], {"res1": 1e-10})
        rejected = gen.generate_step_rejected(
            proposed,
            {"gate1": {"status": "fail"}},
            "gate_failure",
        )
        
        assert rejected.receipt_type == ReceiptType.STEP_REJECTED
        assert rejected.status == "rejected"
    
    def test_generate_invariant_check_pass(self):
        """Test invariant check pass receipt."""
        spec = LedgerSpec()
        gen = ReceiptGenerator(spec)
        
        rcpt = gen.generate_invariant_check(
            invariant_id="mass_cons",
            passed=True,
            value=1e-12,
        )
        
        assert rcpt.receipt_type == ReceiptType.CHECK_INVARIANT
        assert rcpt.status == "pass"
        assert rcpt.residuals["value"] == 1e-12
    
    def test_generate_invariant_check_fail(self):
        """Test invariant check fail receipt."""
        spec = LedgerSpec()
        gen = ReceiptGenerator(spec)
        
        rcpt = gen.generate_invariant_check(
            invariant_id="mass_cons",
            passed=False,
            value=1e-5,
        )
        
        assert rcpt.status == "fail"
    
    def test_generate_run_summary(self):
        """Test run summary receipt generation."""
        spec = LedgerSpec()
        gen = ReceiptGenerator(spec)
        
        # Generate some receipts
        gen.generate_step_proposed("intent_1", ["op1"], {})
        gen.generate_step_proposed("intent_2", ["op2"], {})
        
        summary = gen.generate_run_summary()
        
        assert summary.receipt_type == ReceiptType.RUN_SUMMARY
        assert summary.metrics["total_steps"] == 2
        # Now total_receipts should be 3 (2 proposed + 1 summary)
        assert len(gen.receipts) == 3
    
    def test_hash_chain_continuity(self):
        """Test that receipts form a valid hash chain."""
        spec = LedgerSpec()
        gen = ReceiptGenerator(spec)
        
        # Generate sequence of receipts
        r1 = gen.generate_step_proposed("i1", ["op"], {"res": 0.1})
        r2 = gen.generate_step_accepted(r1, {"gate": {"status": "pass"}})
        r3 = gen.generate_invariant_check("inv1", True, 1e-12)
        
        # Validate chain
        valid, errors = gen.validate_chain()
        
        assert valid
        assert len(errors) == 0
    
    def test_receipt_counter(self):
        """Test step counter increments correctly."""
        spec = LedgerSpec()
        gen = ReceiptGenerator(spec)
        
        assert gen.step_counter == 0
        
        gen.generate_step_proposed("i1", ["op"], {})
        assert gen.step_counter == 1
        
        gen.generate_step_proposed("i2", ["op"], {})
        assert gen.step_counter == 2
    
    def test_get_receipts_by_type(self):
        """Test filtering receipts by type."""
        spec = LedgerSpec()
        gen = ReceiptGenerator(spec)
        
        gen.generate_step_proposed("i1", ["op"], {})
        gen.generate_step_proposed("i2", ["op"], {})
        gen.generate_invariant_check("inv1", True, 0.0)
        
        proposed = gen.get_receipts_by_type(ReceiptType.STEP_PROPOSED)
        invariant = gen.get_receipts_by_type(ReceiptType.CHECK_INVARIANT)
        
        assert len(proposed) == 2
        assert len(invariant) == 1


class TestLedgerValidator:
    """Tests for LedgerValidator class."""
    
    def test_validator_initialization(self):
        """Test validator initialization."""
        spec = LedgerSpec(
            gates=[GateSpec(gate_id="gate1", threshold=0.5)],
            invariants=[InvariantSpec(invariant_id="inv1", tolerance_abs=1e-8)],
        )
        validator = LedgerValidator(spec)
        
        assert validator.ledger_spec == spec
    
    def test_validate_full_ledger_valid(self):
        """Test validating a complete valid ledger."""
        spec = LedgerSpec(
            gates=[GateSpec(gate_id="residual_l2", threshold=1e-8, comparison="le")],
            invariants=[InvariantSpec(invariant_id="mass", tolerance_abs=1e-10)],
            required_receipts=[ReceiptType.STEP_PROPOSED, ReceiptType.RUN_SUMMARY],
        )
        
        # Create generator and produce receipts
        gen = ReceiptGenerator(spec)
        proposed = gen.generate_step_proposed("intent_1", ["op"], {"residual_l2": 1e-10})
        gen.generate_invariant_check("mass", True, 1e-12)
        gen.generate_run_summary()
        
        # Validate
        validator = LedgerValidator(spec)
        valid, results = validator.validate_full_ledger(gen.receipts)
        
        assert valid
        assert results["chain_valid"]
        assert results["all_gates_passed"]
        assert results["all_invariants_satisfied"]
        assert results["required_receipts_present"]
    
    def test_validate_full_ledger_missing_receipts(self):
        """Test validation fails with missing required receipts."""
        spec = LedgerSpec(
            required_receipts=[ReceiptType.STEP_PROPOSED, ReceiptType.RUN_SUMMARY],
        )
        
        gen = ReceiptGenerator(spec)
        gen.generate_step_proposed("intent_1", ["op"], {})
        
        validator = LedgerValidator(spec)
        valid, results = validator.validate_full_ledger(gen.receipts)
        
        assert not valid
        assert not results["required_receipts_present"]
        assert len(results["errors"]) > 0
    
    def test_validate_full_ledger_failed_invariant(self):
        """Test validation fails with failed invariant."""
        spec = LedgerSpec(
            invariants=[InvariantSpec(invariant_id="inv1", tolerance_abs=1e-10)],
        )
        
        gen = ReceiptGenerator(spec)
        gen.generate_invariant_check("inv1", False, 1e-5)
        gen.generate_run_summary()
        
        validator = LedgerValidator(spec)
        valid, results = validator.validate_full_ledger(gen.receipts)
        
        assert not valid
        assert not results["all_invariants_satisfied"]
    
    def test_validate_receipt_chain(self):
        """Test validating just the receipt chain."""
        spec = LedgerSpec()
        gen = ReceiptGenerator(spec)
        gen.generate_step_proposed("i1", ["op"], {})
        gen.generate_step_proposed("i2", ["op"], {})
        
        validator = LedgerValidator(spec)
        valid, errors = validator.validate_receipt_chain(gen.receipts)
        
        assert valid
    
    def test_validate_gates(self):
        """Test gate validation."""
        spec = LedgerSpec(
            gates=[GateSpec(gate_id="gate1", threshold=0.5, comparison="le")]
        )
        
        validator = Validator = LedgerValidator(spec)
        all_passed, results = validator.validate_gates({"gate1": 0.3})
        
        assert all_passed
        assert results["gate1"]["passed"]
    
    def test_validate_invariants(self):
        """Test invariant validation."""
        spec = LedgerSpec(
            invariants=[InvariantSpec(invariant_id="inv1", tolerance_abs=1e-8, source_metric="metric1")]
        )
        
        validator = LedgerValidator(spec)
        
        # Use the source_metric key that the spec expects
        all_passed, results = validator.validate_invariants({"metric1": 1e-10}, {})
        
        assert all_passed
    
    def test_get_missing_receipts(self):
        """Test getting list of missing required receipts."""
        spec = LedgerSpec(
            required_receipts=[
                ReceiptType.STEP_PROPOSED,
                ReceiptType.STEP_ACCEPTED,
                ReceiptType.RUN_SUMMARY,
            ]
        )
        
        gen = ReceiptGenerator(spec)
        gen.generate_step_proposed("i1", ["op"], {})
        gen.generate_run_summary()
        
        validator = LedgerValidator(spec)
        missing = validator.get_missing_receipts(gen.receipts)
        
        assert ReceiptType.STEP_ACCEPTED in missing
        assert len(missing) == 1
    
    def test_get_validation_summary(self):
        """Test getting validation summary."""
        spec = LedgerSpec()
        gen = ReceiptGenerator(spec)
        gen.generate_step_proposed("i1", ["op"], {})
        gen.generate_run_summary()
        
        validator = LedgerValidator(spec)
        summary = validator.get_validation_summary(gen.receipts)
        
        assert "valid" in summary
        assert "chain_valid" in summary
        assert "total_receipts" in summary


class TestLedgerLowerer:
    """Tests for LedgerLowerer class."""
    
    def test_lower_empty_program(self):
        """Test lowering an empty program."""
        lowerer = LedgerLowerer()
        
        class EmptyProgram:
            statements = []
        
        spec = lowerer.lower(EmptyProgram())
        
        assert isinstance(spec, LedgerSpec)
    
    def test_lower_with_directives(self):
        """Test lowering program with directives."""
        lowerer = LedgerLowerer()
        
        class MockDirective:
            def __init__(self, d_type, **kwargs):
                self.type = d_type
                self.__dict__.update(kwargs)
        
        class MockStatement:
            def __init__(self):
                self.directives = [
                    MockDirective("inv", invariant_id="mass_cons", tolerance_abs=1e-10),
                    MockDirective("gate", gate_id="residual_l2", threshold=1e-8),
                ]
        
        class Program:
            def __init__(self):
                self.statements = [MockStatement()]
        
        spec = lowerer.lower(Program())
        
        assert spec.get_invariant("mass_cons") is not None
        assert spec.get_gate("residual_l2") is not None
    
    def test_extract_gate_requirements(self):
        """Test extracting gate requirements."""
        lowerer = LedgerLowerer()
        
        class MockDirective:
            def __init__(self, d_type, **kwargs):
                self.type = d_type
                self.__dict__.update(kwargs)
        
        class MockStatement:
            def __init__(self):
                self.directives = [
                    MockDirective("gate", gate_id="gate1", threshold=0.5),
                    MockDirective("gate", gate_id="gate2", threshold=0.3),
                ]
        
        class Program:
            def __init__(self):
                self.statements = [MockStatement()]
        
        gates = lowerer.extract_gate_requirements(Program())
        
        assert "gate1" in gates
        assert "gate2" in gates
        assert gates["gate1"].threshold == 0.5
    
    def test_add_default_gates(self):
        """Test adding default gates."""
        lowerer = LedgerLowerer()
        lowerer.add_default_gates()
        
        spec = lowerer.get_ledger_spec()
        
        assert spec.get_gate("residual_l2") is not None
        assert spec.get_gate("cfl_number") is not None
    
    def test_add_default_invariants(self):
        """Test adding default invariants."""
        lowerer = LedgerLowerer()
        lowerer.add_default_invariants()
        
        spec = lowerer.get_ledger_spec()
        
        assert spec.get_invariant("mass_conservation") is not None
        assert spec.get_invariant("energy_conservation") is not None
    
    def test_build_from_directives(self):
        """Test building ledger spec from directive dictionaries."""
        lowerer = LedgerLowerer()
        
        inv_directives = [
            {"invariant_id": "inv1", "tolerance_abs": 1e-10},
        ]
        gate_directives = [
            {"gate_id": "gate1", "threshold": 0.5},
        ]
        
        spec = lowerer.build_from_directives(inv_directives, gate_directives)
        
        assert spec.get_invariant("inv1") is not None
        assert spec.get_gate("gate1") is not None


class TestReceiptSerialization:
    """Tests for receipt JSON serialization."""
    
    def test_receipt_json_roundtrip(self):
        """Test receipt serialization and deserialization."""
        original = Receipt(
            receipt_type=ReceiptType.STEP_ACCEPTED,
            timestamp=1234567890.0,
            run_id="R-test",
            step_id=1,
            ops=["op1", "op2"],
            residuals={"res1": 1e-10, "res2": 2e-10},
            metrics={"metric1": 0.5},
            status="accepted",
        )
        
        # Convert to dict
        data = original.to_dict()
        
        # Serialize to JSON
        json_str = json.dumps(data)
        
        # Deserialize from JSON
        parsed = json.loads(json_str)
        
        # Convert back to receipt
        restored = Receipt.from_dict(parsed)
        
        assert restored.receipt_type == original.receipt_type
        assert restored.run_id == original.run_id
        assert restored.step_id == original.step_id
        assert len(restored.ops) == len(original.ops)
        assert restored.residuals["res1"] == original.residuals["res1"]
    
    def test_receipt_list_serialization(self):
        """Test serializing list of receipts."""
        spec = LedgerSpec()
        gen = ReceiptGenerator(spec)
        
        gen.generate_step_proposed("i1", ["op"], {})
        gen.generate_step_proposed("i2", ["op"], {})
        
        receipts = gen.get_receipts()
        
        # Serialize
        json_data = json.dumps([r.to_dict() for r in receipts])
        
        # Deserialize
        parsed = json.loads(json_data)
        restored = [Receipt.from_dict(p) for p in parsed]
        
        assert len(restored) == 2
        assert restored[0].receipt_type == ReceiptType.STEP_PROPOSED


class TestHysteresisBehavior:
    """Tests for gate hysteresis behavior."""
    
    def test_hysteresis_prevents_flip_flop(self):
        """Test that hysteresis prevents rapid gate state changes."""
        spec = GateSpec(
            gate_id="test",
            threshold=1.0,
            hysteresis=0.1,
            comparison="le"
        )
        evaluator = GateEvaluator({"test": spec})
        
        # Just above threshold but within hysteresis
        passed1, result1 = evaluator.evaluate("test", 1.05)
        
        # Value within hysteresis should go to review
        assert result1["status"] == "review"
        assert result1["hysteresis"] == 0.1
    
    def test_hysteresis_ge_comparison(self):
        """Test hysteresis with greater-than-or-equal comparison."""
        spec = GateSpec(
            gate_id="test",
            threshold=0.5,
            hysteresis=0.1,
            comparison="ge"
        )
        evaluator = GateEvaluator({"test": spec})
        
        # Just below threshold but within hysteresis
        passed, result = evaluator.evaluate("test", 0.45)
        
        assert result["status"] == "review"
        assert result["hysteresis"] == 0.1


class TestFullLedgerWorkflow:
    """Integration tests for complete ledger workflow."""
    
    def test_complete_execution_workflow(self):
        """Test a complete execution with all ledger features."""
        # Create ledger spec
        spec = LedgerSpec(
            gates=[
                GateSpec(gate_id="residual_l2", threshold=1e-8, comparison="le"),
                GateSpec(gate_id="cfl", threshold=0.5, comparison="le"),
            ],
            invariants=[
                InvariantSpec(invariant_id="mass", tolerance_abs=1e-10),
                InvariantSpec(invariant_id="energy", tolerance_abs=1e-8),
            ],
            required_receipts=[
                ReceiptType.STEP_PROPOSED,
                ReceiptType.STEP_ACCEPTED,
                ReceiptType.CHECK_INVARIANT,
                ReceiptType.RUN_SUMMARY,
            ],
        )
        
        # Create generator and validator
        gen = ReceiptGenerator(spec)
        validator = LedgerValidator(spec)
        
        # Simulate execution steps
        steps = [
            {"intent": "init", "ops": ["setup"], "residuals": {"residual_l2": 1e-9, "cfl": 0.3}},
            {"intent": "step1", "ops": ["advance"], "residuals": {"residual_l2": 5e-9, "cfl": 0.4}},
            {"intent": "step2", "ops": ["advance"], "residuals": {"residual_l2": 2e-9, "cfl": 0.35}},
        ]
        
        for step in steps:
            # Step proposed
            proposed = gen.generate_step_proposed(
                step["intent"],
                step["ops"],
                step["residuals"],
            )
            
            # Evaluate gates
            all_passed, gate_results = validator.validate_gates(step["residuals"])
            
            # Step accepted or rejected
            if all_passed:
                gen.generate_step_accepted(proposed, gate_results)
            else:
                gen.generate_step_rejected(proposed, gate_results, "gate_failure")
        
        # Check invariants
        gen.generate_invariant_check("mass", True, 1e-12)
        gen.generate_invariant_check("energy", True, 1e-10)
        
        # Generate summary
        gen.generate_run_summary()
        
        # Validate full ledger
        valid, results = validator.validate_full_ledger(gen.receipts)
        
        assert valid
        assert results["chain_valid"]
        assert results["all_gates_passed"]
        assert results["all_invariants_satisfied"]
        assert results["required_receipts_present"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
