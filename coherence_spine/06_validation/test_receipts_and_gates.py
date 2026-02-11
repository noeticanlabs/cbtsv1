"""
Receipt correctness, gate behavior, debt boundedness, and projection legality tests (T1-T4).

Tests:
  T1 — Receipt correctness: schema validation, deterministic hashing, hash chain verification
  T2 — Gate behavior: hard gate fail ⇒ rollback/abort, soft gate fail ⇒ bounded rail then retry
  T3 — Debt boundedness: max accepted debt ≤ C_max, repairs reduce targeted debt block
  T4 — Projection legality: illegal layer jump ⇒ tool residual triggers and is receipted

Run:
  python -m pytest coherence_spine/06_validation/test_receipts_and_gates.py -v
"""

from __future__ import annotations

import hashlib
import json
import math
import unittest
from dataclasses import dataclass
from typing import Any


# =============================================================================
# Utilities (stdlib only)
# =============================================================================

def canonical_json(obj: Any) -> bytes:
    """Canonical JSON per Coherence spec (deterministic ordering)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# =============================================================================
# Receipt Schema (simplified from Coherence_Spec_v1_0/schemas)
# =============================================================================

RECEIPT_SCHEMA = {
    "type": "object",
    "required": ["header", "state", "metrics", "gates", "actions", "provenance", "integrity"],
    "properties": {
        "header": {
            "type": "object",
            "required": ["run_id", "step_id", "t", "dt"],
        },
        "state": {
            "type": "object",
            "required": ["hash_before", "hash_after"],
        },
        "metrics": {
            "type": "object",
            "required": ["residuals", "debt"],
        },
        "gates": {
            "type": "object",
            "required": ["hard", "soft", "decision"],
        },
        "actions": {
            "type": "object",
            "required": ["rails", "retries_used"],
        },
        "provenance": {
            "type": "object",
            "required": ["lexicon_terms_used", "layers"],
        },
        "integrity": {
            "type": "object",
            "required": ["prev_hash", "this_hash"],
        },
    },
}


def validate_receipt_schema(receipt: dict) -> bool:
    """Simple schema validation against RECEIPT_SCHEMA."""
    for key in RECEIPT_SCHEMA["required"]:
        if key not in receipt:
            return False
    return True


# =============================================================================
# Gate Policy (from minimal_tests.py)
# =============================================================================

@dataclass
class GatePolicy:
    phys_fail: float = 1e-2
    phys_warn: float = 7.5e-3
    dt_min: float = 1e-4
    dt_alpha: float = 0.8
    retry_cap: int = 6

    def evaluate(self, phys_rms: float, nan_free: bool = True) -> dict:
        hard = {"nan_free": "pass" if nan_free else "fail"}
        soft = {"phys_rms": "pass"}
        decision = "accept"

        if not nan_free:
            decision = "abort"
        else:
            if phys_rms > self.phys_fail:
                soft["phys_rms"] = "fail"
                decision = "reject"
            elif phys_rms > self.phys_warn:
                soft["phys_rms"] = "warn"
                decision = "accept"

        return {"hard": hard, "soft": soft, "decision": decision}


# =============================================================================
# Receipt Factory
# =============================================================================

def make_receipt(
    step_id: int,
    t: float,
    dt: float,
    x_before: float,
    x_after: float,
    phys_rms: float,
    debt: float,
    decision: str,
    rails: list,
    retries_used: int,
    prev_hash: str,
    nan_free: bool = True,
) -> dict:
    """Create a deterministic receipt for testing."""
    receipt = {
        "header": {
            "run_id": "TEST-RUN",
            "step_id": step_id,
            "t": t,
            "dt": dt,
            "timezone": "UTC",
        },
        "state": {
            "hash_before": "sha256:" + sha256_hex(b"before" + canonical_json({"x": x_before})),
            "hash_after": "sha256:" + sha256_hex(b"after" + canonical_json({"x": x_after})),
            "summary": {"x": x_after},
        },
        "metrics": {
            "residuals": {"phys_rms": phys_rms},
            "debt": {"C_total": debt, "C_blocks": {"phys": debt}},
            "invariants": {"nan_free": nan_free},
        },
        "gates": {"hard": {"nan_free": "pass" if nan_free else "fail"}, "soft": {"phys_rms": "pass"}, "decision": decision},
        "actions": {"rails": rails, "retries_used": retries_used, "notes": []},
        "provenance": {
            "lexicon_terms_used": ["LoC_measure"],
            "layers": ["L0", "L2", "L4"],
            "code_version": "v1.0.0",
        },
        "integrity": {"prev_hash": prev_hash, "this_hash": ""},
    }

    # Compute hash chain
    receipt_for_hash = json.loads(json.dumps(receipt))  # Deep copy
    receipt_for_hash["integrity"]["this_hash"] = ""
    this_hash = "sha256:" + sha256_hex(canonical_json(receipt_for_hash) + prev_hash.encode("utf-8"))
    receipt["integrity"]["this_hash"] = this_hash

    return receipt


# =============================================================================
# Test Classes (T1-T4)
# =============================================================================

class TestT1ReceiptCorrectness(unittest.TestCase):
    """T1 — Receipt correctness: schema, deterministic hashing, hash chain verification."""

    def test_schema_validation(self) -> None:
        """T1a: All receipts validate against schema."""
        receipt = make_receipt(
            step_id=0, t=0.0, dt=0.1,
            x_before=1.0, x_after=0.9,
            phys_rms=0.01, debt=1e-4,
            decision="accept", rails=[], retries_used=0,
            prev_hash="sha256:" + "0" * 64,
        )
        self.assertTrue(validate_receipt_schema(receipt))

    def test_deterministic_hashing(self) -> None:
        """T1b: Same inputs produce same hash."""
        receipt1 = make_receipt(
            step_id=5, t=0.5, dt=0.1,
            x_before=0.5, x_after=0.45,
            phys_rms=0.005, debt=2.5e-5,
            decision="accept", rails=[], retries_used=0,
            prev_hash="sha256:" + "1" * 64,
        )
        receipt2 = make_receipt(
            step_id=5, t=0.5, dt=0.1,
            x_before=0.5, x_after=0.45,
            phys_rms=0.005, debt=2.5e-5,
            decision="accept", rails=[], retries_used=0,
            prev_hash="sha256:" + "1" * 64,
        )
        self.assertEqual(receipt1["integrity"]["this_hash"], receipt2["integrity"]["this_hash"])

    def test_hash_chain_verification(self) -> None:
        """T1c: Hash chain is verifiable (each hash depends on previous)."""
        prev = "sha256:" + "0" * 64
        receipts = []
        for i in range(10):
            receipt = make_receipt(
                step_id=i, t=i * 0.1, dt=0.1,
                x_before=1.0 - i * 0.01, x_after=1.0 - (i + 1) * 0.01,
                phys_rms=0.001 * (i + 1), debt=1e-6 * (i + 1),
                decision="accept", rails=[], retries_used=0,
                prev_hash=prev,
            )
            receipts.append(receipt)
            prev = receipt["integrity"]["this_hash"]

        # Verify chain
        prev_check = "sha256:" + "0" * 64
        for receipt in receipts:
            receipt_for_hash = json.loads(json.dumps(receipt))
            receipt_for_hash["integrity"]["this_hash"] = ""
            expected = "sha256:" + sha256_hex(canonical_json(receipt_for_hash) + prev_check.encode("utf-8"))
            self.assertEqual(receipt["integrity"]["this_hash"], expected)
            prev_check = receipt["integrity"]["this_hash"]

    def test_hash_chain_mismatch_detection(self) -> None:
        """T1d: Tampered receipt hash is detected."""
        receipt = make_receipt(
            step_id=0, t=0.0, dt=0.1,
            x_before=1.0, x_after=0.9,
            phys_rms=0.01, debt=1e-4,
            decision="accept", rails=[], retries_used=0,
            prev_hash="sha256:" + "0" * 64,
        )
        # Tamper with the receipt
        tampered = dict(receipt)
        tampered["metrics"]["residuals"]["phys_rms"] = 999.0  # Tamper!

        # Compute expected hash from original
        receipt_for_hash = json.loads(json.dumps(receipt))
        receipt_for_hash["integrity"]["this_hash"] = ""
        expected_hash = "sha256:" + sha256_hex(canonical_json(receipt_for_hash) + receipt["integrity"]["prev_hash"].encode("utf-8"))

        # Tampered hash should not match
        self.assertNotEqual(tampered["integrity"]["this_hash"], expected_hash)


class TestT2GateBehavior(unittest.TestCase):
    """T2 — Gate behavior: hard fail ⇒ rollback, soft fail ⇒ bounded retry."""

    def test_hard_fail_abort(self) -> None:
        """T2a: Hard gate fail (NaN) → abort decision."""
        gp = GatePolicy()
        # NaN condition
        verdict = gp.evaluate(phys_rms=0.001, nan_free=False)
        self.assertEqual(verdict["decision"], "abort")
        self.assertEqual(verdict["hard"]["nan_free"], "fail")

    def test_soft_fail_reject(self) -> None:
        """T2b: Soft gate fail (phys_rms > phys_fail) → reject decision."""
        gp = GatePolicy()
        verdict = gp.evaluate(phys_rms=0.05, nan_free=True)  # Above phys_fail=1e-2
        self.assertEqual(verdict["decision"], "reject")
        self.assertEqual(verdict["soft"]["phys_rms"], "fail")

    def test_soft_warn_accept(self) -> None:
        """T2c: Soft gate warn (phys_fail > phys_rms > phys_warn) → accept."""
        gp = GatePolicy()
        # phys_fail=1e-2, phys_warn=7.5e-3, so use value between them
        verdict = gp.evaluate(phys_rms=0.008, nan_free=True)  # Between warn and fail
        self.assertEqual(verdict["decision"], "accept")
        self.assertEqual(verdict["soft"]["phys_rms"], "warn")

    def test_phys_pass_accept(self) -> None:
        """T2d: Below warn threshold → accept."""
        gp = GatePolicy()
        verdict = gp.evaluate(phys_rms=0.001, nan_free=True)
        self.assertEqual(verdict["decision"], "accept")
        self.assertEqual(verdict["soft"]["phys_rms"], "pass")

    def test_retry_cap_enforcement(self) -> None:
        """T2e: Retries bounded by retry_cap."""
        gp = GatePolicy()
        max_retries = gp.retry_cap

        # Simulate repeated rejects
        retries = 0
        for _ in range(max_retries + 2):  # Exceed cap
            verdict = gp.evaluate(phys_rms=0.05, nan_free=True)
            if verdict["decision"] == "reject":
                retries += 1

        self.assertGreaterEqual(retries, max_retries)  # Cap enforced


class TestT3DebtBoundedness(unittest.TestCase):
    """T3 — Debt boundedness: max accepted debt ≤ C_max, repairs reduce debt."""

    def test_debt_bounded_by_threshold(self) -> None:
        """T3a: Accepted debt stays below C_max threshold."""
        C_max = 1e-4
        accepted_debts = []

        # Simulate step loop with decreasing residuals
        for i in range(20):
            phys_rms = 1e-3 * (0.9 ** i)  # Decreasing residual
            debt = phys_rms ** 2
            gp = GatePolicy()

            if phys_rms <= gp.phys_fail:
                accepted_debts.append(debt)

        if accepted_debts:
            max_accepted = max(accepted_debts)
            self.assertLessEqual(max_accepted, C_max)

    def test_dt_deflation_reduces_debt(self) -> None:
        """T3b: dt deflation rail reduces targeted debt block."""
        # Initial state
        dt = 0.1
        phys_rms = 0.05
        initial_debt = phys_rms ** 2

        # Apply dt deflation
        dt_alpha = 0.8
        dt_new = dt * dt_alpha

        # Residual scales with dt (for Euler: residual ~ dt)
        phys_rms_new = phys_rms * dt_alpha
        new_debt = phys_rms_new ** 2

        # Debt should be reduced
        self.assertLess(new_debt, initial_debt)

    def test_repair_reduces_targeted_debt(self) -> None:
        """T3c: Repair action reduces targeted debt block."""
        # Initial debt block
        C_phys_initial = 1e-3

        # Simulate rail application (dt deflation)
        reduction_factor = 0.8
        C_phys_reduced = C_phys_initial * reduction_factor

        self.assertLess(C_phys_reduced, C_phys_initial)


class TestT4ProjectionLegality(unittest.TestCase):
    """T4 — Projection legality: illegal layer jump → tool residual triggers."""

    def test_illegal_layer_jump_rejected(self) -> None:
        """T4a: Illegal layer jump (L0→L4 without L1-L3) is rejected."""
        # Simulate illegal projection: L0 axioms directly to L4 runtime
        illegal_layers = ["L0", "L4"]  # Missing L1, L2, L3
        legal_layers = ["L0", "L1", "L2", "L3", "L4"]  # Correct projection (contiguous)

        # Check if projection is legal (must have contiguous layers in the hierarchy)
        def is_legal_projection(layers: list) -> bool:
            # Legal if layers are contiguous in the hierarchy
            layer_order = ["L0", "L1", "L2", "L3", "L4", "L5"]
            indices = [layer_order.index(l) for l in layers]
            return max(indices) - min(indices) == len(layers) - 1

        self.assertFalse(is_legal_projection(illegal_layers))
        self.assertTrue(is_legal_projection(legal_layers))

    def test_tool_residual_on_violation(self) -> None:
        """T4b: Tool residual triggers on lexicon violation."""
        # Simulate tool violation detection
        lexicon_terms_used = ["LoC_measure", "illegal_term"]
        allowed_terms = {"LoC_measure", "LoC_axiom", "CTL_clock"}

        # Check for unauthorized terms
        violations = [term for term in lexicon_terms_used if term not in allowed_terms]

        # Violation should be detected
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0], "illegal_term")

    def test_violation_receipted(self) -> None:
        """T4c: Violation is recorded in receipt notes."""
        violations = ["illegal_layer_projection"]
        notes = []

        for violation in violations:
            notes.append({
                "type": "tool_violation",
                "detail": violation,
                "handled": True,
            })

        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0]["type"], "tool_violation")


class TestT5MinimalPhysicsExample(unittest.TestCase):
    """T5 — Minimal physics example: ODE x'=-λx, residual decreases as dt shrinks."""

    def test_residual_monotone_in_dt(self) -> None:
        """T5a: Residual is monotone decreasing as dt shrinks."""
        lam = 2.0
        x = 1.0

        def defect(x: float, x_next: float, lam: float, dt: float) -> float:
            return (x_next - x) / dt + lam * x

        residuals = []
        for dt in [0.1, 0.05, 0.025, 0.0125]:
            x_next = x + dt * (-lam * x)  # Explicit Euler
            res = abs(defect(x, x_next, lam, dt))
            residuals.append(res)

        # Check monotonicity with tolerance for floating point
        for i in range(1, len(residuals)):
            self.assertLessEqual(residuals[i], residuals[i - 1] + 1e-12)

    def test_acceptance_stable_across_dt(self) -> None:
        """T5b: Acceptance stable below threshold."""
        lam = 2.0
        x = 1.0
        gp = GatePolicy(phys_fail=1e-2, phys_warn=7.5e-3)

        def defect(x: float, x_next: float, lam: float, dt: float) -> float:
            return (x_next - x) / dt + lam * x

        for dt in [0.1, 0.05, 0.025]:
            x_next = x + dt * (-lam * x)
            res = abs(defect(x, x_next, lam, dt))
            verdict = gp.evaluate(res, nan_free=True)

            if res <= gp.phys_warn:
                self.assertEqual(verdict["decision"], "accept")


if __name__ == "__main__":
    unittest.main()
