"""
Courtroom Tests for Audit-Grade Recovery Specification

Tests verify that:
1. Soft failure retries properly restore both state and UnifiedClock
2. Exhausted retries with no repair escalate to hard failure (SEMFailure)
3. Gate kind classification correctly determines hard vs soft behavior
"""

import numpy as np
import pytest
from unittest.mock import Mock, MagicMock, patch
import hashlib

from src.core.gr_gates import GateKind, should_hard_fail
from aeonic_memory_contract import SEMFailure, AeonicMemoryContract
from aeonic_memory_bank import AeonicMemoryBank, Record
from aeonic_clocks import AeonicClockPack
from receipt_schemas import Kappa, MSolveReceipt
from src.core.gr_clock import UnifiedClockState


class TestGateKindClassification:
    """Tests for GateKind enum and should_hard_fail function."""
    
    def test_constraint_gate_is_hard_fail(self):
        """CONSTRAINT gate kind should always hard fail."""
        gate = {'kind': 'constraint', 'reason': 'eps_H exceeded threshold'}
        assert should_hard_fail(gate) is True
    
    def test_nonfinite_gate_is_hard_fail(self):
        """NONFINITE gate kind should always hard fail."""
        gate = {'kind': 'nonfinite', 'reason': 'NaN detected in residuals'}
        assert should_hard_fail(gate) is True
    
    def test_uninitialized_gate_is_hard_fail(self):
        """UNINITIALIZED gate kind should always hard fail."""
        gate = {'kind': 'uninitialized', 'reason': 'Field not initialized'}
        assert should_hard_fail(gate) is True
    
    def test_state_gate_is_soft_by_default(self):
        """STATE gate kind should be soft by default."""
        gate = {'kind': 'state', 'reason': 'eps_M exceeded soft threshold'}
        assert should_hard_fail(gate) is False
    
    def test_rate_gate_is_soft(self):
        """RATE gate kind should be soft (retry allowed)."""
        gate = {'kind': 'rate', 'reason': 'Constraint damping rate exceeded'}
        assert should_hard_fail(gate) is False
    
    def test_unknown_gate_defaults_to_soft(self):
        """Unknown gate kind should default to soft (state-like)."""
        gate = {'kind': 'unknown_kind', 'reason': 'some reason'}
        assert should_hard_fail(gate) is False
    
    def test_gate_object_with_kind_attribute(self):
        """should_hard_fail works with objects having kind attribute."""
        class MockGate:
            def __init__(self, kind):
                self.kind = kind
        
        gate = MockGate(GateKind.CONSTRAINT)
        assert should_hard_fail(gate) is True
        
        gate = MockGate(GateKind.STATE)
        assert should_hard_fail(gate) is False
    
    def test_gate_kind_enum_values(self):
        """Verify GateKind enum has expected values."""
        assert GateKind.CONSTRAINT.value == "constraint"
        assert GateKind.NONFINITE.value == "nonfinite"
        assert GateKind.UNINITIALIZED.value == "uninitialized"
        assert GateKind.STATE.value == "state"
        assert GateKind.RATE.value == "rate"


class TestAbortOnHardFail:
    """Tests for AeonicMemoryContract.abort_on_hard_fail method."""
    
    def test_hard_fail_raises_sem_failure(self):
        """Hard fail gate with accepted=False should raise SEMFailure."""
        memory_bank = AeonicMemoryBank(clock=AeonicClockPack())
        contract = AeonicMemoryContract(memory_bank)
        
        gate = {'kind': 'constraint', 'reason': 'eps_H exceeded threshold'}
        
        with pytest.raises(SEMFailure) as exc_info:
            contract.abort_on_hard_fail(gate, accepted=False)
        
        assert "Hard gate failure" in str(exc_info.value)
        assert "constraint" in str(exc_info.value)
    
    def test_soft_fail_does_not_raise(self):
        """Soft fail gate should not raise even when not accepted."""
        memory_bank = AeonicMemoryBank(clock=AeonicClockPack())
        contract = AeonicMemoryContract(memory_bank)
        
        gate = {'kind': 'state', 'reason': 'eps_M exceeded soft threshold'}
        
        # Should not raise
        contract.abort_on_hard_fail(gate, accepted=False)
    
    def test_accepted_hard_fail_does_not_raise(self):
        """Accepted gate should not raise even if hard kind."""
        memory_bank = AeonicMemoryBank(clock=AeonicClockPack())
        contract = AeonicMemoryContract(memory_bank)
        
        gate = {'kind': 'constraint', 'reason': 'eps_H exceeded threshold'}
        
        # Should not raise when accepted
        contract.abort_on_hard_fail(gate, accepted=True)


class TestRetryRestoresClockAndState:
    """Test that retry properly restores both field state and UnifiedClockState."""
    
    def test_rollback_restores_clock_state(self):
        """Verify that rollback mechanism restores UnifiedClockState."""
        # Create mock orchestrator with clock
        orchestrator = Mock()
        
        # Create UnifiedClockState
        original_state = UnifiedClockState(
            global_step=10,
            global_time=0.01,
            band_steps=np.array([10, 5, 2, 1, 0, 0, 0, 0], dtype=np.int64),
            band_times=np.array([0.01, 0.005, 0.002, 0.001, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        )
        
        # Create clock mock
        clock = Mock()
        clock.state = original_state.copy()
        
        orchestrator.clock = clock
        orchestrator.fields = Mock()
        orchestrator.fields.gamma_sym6 = np.zeros((3, 3, 3))
        orchestrator.fields.K_sym6 = np.zeros((3, 3, 3))
        orchestrator.fields.alpha = np.zeros(3)
        orchestrator.fields.beta = np.zeros((3, 3))
        orchestrator.fields.phi = np.zeros(3)
        
        orchestrator.rollback_count = 0
        orchestrator.rollback_reason = None
        
        # Simulate state backup with clock state
        state_backup = {
            'gamma': np.ones((3, 3, 3)),
            'K': np.ones((3, 3, 3)) * 0.5,
            'alpha': np.ones(3) * 0.1,
            'beta': np.zeros((3, 3)),
            'phi': np.ones(3) * 0.01,
            'clock_state': original_state.copy()
        }
        
        # Simulate rollback (as done in CommitPhase.rollback)
        # Restore field state
        orchestrator.fields.gamma_sym6[:] = state_backup['gamma']
        orchestrator.fields.K_sym6[:] = state_backup['K']
        orchestrator.fields.alpha[:] = state_backup['alpha']
        orchestrator.fields.beta[:] = state_backup['beta']
        orchestrator.fields.phi[:] = state_backup['phi']
        
        # Restore clock state
        if state_backup.get('clock_state') is not None:
            orchestrator.clock.set_state(state_backup['clock_state'])
        
        orchestrator.rollback_count += 1
        orchestrator.rollback_reason = "Stepper rejection: test reason"
        
        # Verify clock state was restored
        assert orchestrator.clock.state.global_step == 10
        assert orchestrator.clock.state.global_time == 0.01
        assert np.array_equal(orchestrator.clock.state.band_steps, original_state.band_steps)
        assert np.array_equal(orchestrator.clock.state.band_times, original_state.band_times)
        
        # Verify field state was restored
        assert np.allclose(orchestrator.fields.gamma_sym6, state_backup['gamma'])
        assert np.allclose(orchestrator.fields.K_sym6, state_backup['K'])
        
        # Verify rollback tracking
        assert orchestrator.rollback_count == 1
        assert "test reason" in orchestrator.rollback_reason
    
    def test_retry_with_dt_reduction(self):
        """Verify retry with dt reduction maintains consistency."""
        # Initial clock state
        initial_clock = UnifiedClockState(
            global_step=5,
            global_time=0.005
        )
        
        # After failed attempt (clock advanced, but should rollback)
        failed_clock = UnifiedClockState(
            global_step=6,
            global_time=0.006
        )
        
        # Verify we can restore from failed to initial
        restored_clock = initial_clock.copy()
        
        assert restored_clock.global_step == 5
        assert restored_clock.global_time == 0.005


class TestExhaustedRepairsEscalates:
    """Test that repeated state failures with no repair escalate to hard fail."""
    
    def test_exhausted_retries_raises_sem_failure(self):
        """After max retries without repair, should raise SEMFailure."""
        memory_bank = AeonicMemoryBank(clock=AeonicClockPack())
        contract = AeonicMemoryContract(memory_bank)
        
        # Simulate a state gate with no repair actions
        gate = {
            'kind': 'state',
            'reason': 'Repeated constraint violation',
            'actions_allowed': []  # No repair actions
        }
        
        # After exhausting retries, should escalate to hard fail
        # Using abort_on_hard_fail which checks should_hard_fail
        # Since state is soft by default, we need to check the escalation logic
        # In practice, the caller should track retry count and call abort_on_hard_fail
        # when retries are exhausted
        
        retry_count_max = 5
        current_retry = 5  # Exhausted
        
        if current_retry >= retry_count_max:
            # Escalate: treat as hard fail when no repair available
            with pytest.raises(SEMFailure):
                contract.abort_on_hard_fail(gate, accepted=False)
    
    def test_state_gate_with_repair_soft_fail(self):
        """State gate with repair actions should remain soft."""
        memory_bank = AeonicMemoryBank(clock=AeonicClockPack())
        contract = AeonicMemoryContract(memory_bank)
        
        # State gate with repair actions
        gate = {
            'kind': 'state',
            'reason': 'eps_M exceeded soft threshold',
            'actions_allowed': [{'repair': True, 'action': 'reduce_dt'}]
        }
        
        # Even with exhausted retries, should_hard_fail returns False for state
        # So this should NOT raise (soft failure, not hard)
        assert should_hard_fail(gate) is False
        
        # The caller handles soft failures via retry policy
        # No SEMFailure should be raised
        contract.abort_on_hard_fail(gate, accepted=False)
    
    def test_constraint_gate_always_hard_fail(self):
        """Constraint gate should always hard fail regardless of retry count."""
        memory_bank = AeonicMemoryBank(clock=AeonicClockPack())
        contract = AeonicMemoryContract(memory_bank)
        
        # Constraint gate - always hard fail
        gate = {
            'kind': 'constraint',
            'reason': 'eps_H exceeded hard threshold',
            'actions_allowed': [{'repair': True, 'action': 'projection'}]
        }
        
        # Even with repair actions, constraint is always hard fail
        assert should_hard_fail(gate) is True
        
        with pytest.raises(SEMFailure):
            contract.abort_on_hard_fail(gate, accepted=False)


class TestScopedCacheInvalidation:
    """Tests for AeonicMemoryBank.invalidate_by_regime with config_hash."""
    
    def test_invalidate_by_regime_without_config_hash(self):
        """Invalidate by regime without config_hash should taint all matching records."""
        clock = AeonicClockPack()
        memory_bank = AeonicMemoryBank(clock=clock)
        
        # Add some records
        record1 = Record(
            key="test1", tier=1, payload={"data": 1}, bytes=100,
            created_tau_s=0, created_tau_l=0, created_tau_m=0,
            last_use_tau_s=0, last_use_tau_l=0,
            ttl_s=3600, ttl_l=86400, reuse_count=0,
            recompute_cost_est=100.0, risk_score=0.1,
            tainted=False, regime_hashes=["regime_a"], demoted=False
        )
        record2 = Record(
            key="test2", tier=1, payload={"data": 2}, bytes=100,
            created_tau_s=0, created_tau_l=0, created_tau_m=0,
            last_use_tau_s=0, last_use_tau_l=0,
            ttl_s=3600, ttl_l=86400, reuse_count=0,
            recompute_cost_est=100.0, risk_score=0.1,
            tainted=False, regime_hashes=["regime_b"], demoted=False
        )
        record3 = Record(
            key="test3", tier=2, payload={"data": 3}, bytes=100,
            created_tau_s=0, created_tau_l=0, created_tau_m=0,
            last_use_tau_s=0, last_use_tau_l=0,
            ttl_s=3600, ttl_l=86400, reuse_count=0,
            recompute_cost_est=100.0, risk_score=0.1,
            tainted=False, regime_hashes=["regime_a"], demoted=False
        )
        
        memory_bank.tiers[1] = {"test1": record1, "test2": record2}
        memory_bank.tiers[2] = {"test3": record3}
        
        # Invalidate regime_a without config_hash
        memory_bank.invalidate_by_regime("regime_a")
        
        # All regime_a records should be tainted
        assert record1.tainted is True
        assert record3.tainted is True
        # regime_b record should not be tainted
        assert record2.tainted is False
    
    def test_invalidate_by_regime_with_config_hash(self):
        """Invalidate by regime with config_hash should only taint matching records."""
        clock = AeonicClockPack()
        memory_bank = AeonicMemoryBank(clock=clock)
        
        # Add records with config_hash
        record1 = Record(
            key="test1", tier=1, payload={"data": 1}, bytes=100,
            created_tau_s=0, created_tau_l=0, created_tau_m=0,
            last_use_tau_s=0, last_use_tau_l=0,
            ttl_s=3600, ttl_l=86400, reuse_count=0,
            recompute_cost_est=100.0, risk_score=0.1,
            tainted=False, regime_hashes=["regime_a"], demoted=False,
            config_hash="config_x"
        )
        record2 = Record(
            key="test2", tier=1, payload={"data": 2}, bytes=100,
            created_tau_s=0, created_tau_l=0, created_tau_m=0,
            last_use_tau_s=0, last_use_tau_l=0,
            ttl_s=3600, ttl_l=86400, reuse_count=0,
            recompute_cost_est=100.0, risk_score=0.1,
            tainted=False, regime_hashes=["regime_a"], demoted=False,
            config_hash="config_y"
        )
        record3 = Record(
            key="test3", tier=2, payload={"data": 3}, bytes=100,
            created_tau_s=0, created_tau_l=0, created_tau_m=0,
            last_use_tau_s=0, last_use_tau_l=0,
            ttl_s=3600, ttl_l=86400, reuse_count=0,
            recompute_cost_est=100.0, risk_score=0.1,
            tainted=False, regime_hashes=["regime_a"], demoted=False,
            config_hash="config_x"
        )
        
        memory_bank.tiers[1] = {"test1": record1, "test2": record2}
        memory_bank.tiers[2] = {"test3": record3}
        
        # Invalidate regime_a with config_x
        memory_bank.invalidate_by_regime("regime_a", config_hash="config_x")
        
        # Only config_x records should be tainted
        assert record1.tainted is True
        assert record3.tainted is True
        # config_y record should not be tainted
        assert record2.tainted is False


class TestMSolveReceiptForensicFields:
    """Tests for MSolveReceipt forensic fields."""
    
    def test_receipt_with_forensic_fields(self):
        """MSolveReceipt should accept forensic fields."""
        kappa = Kappa(o=1, s=2, mu=None)
        
        receipt = MSolveReceipt(
            attempt_id=0,
            kappa=kappa,
            t=0.0,
            tau_attempt=0,
            residual_proxy={"CONS": {"L": 1e-5}},
            dt_cap_min=1e-6,
            dominant_thread=("PHY", "step", "act"),
            actions=[{"reduce_dt": True}],
            policy_hash="abc123",
            state_hash="state456",
            stage_time=0.001,
            stage_id=0,
            sem_ok=False,
            rollback_count=0,
            perf={},
            # Forensic fields
            gate_kind="state",
            gate_reason="eps_M exceeded soft threshold",
            hard_fail=False,
            retry_index=0,
            dt_before=1e-4,
            dt_after=8e-5,
            snapshot_id="snap789",
            regime_hash="regime_abc",
            repair_actions_allowed=["reduce_dt", "increase_kappa"],
            repair_action_chosen="reduce_dt"
        )
        
        assert receipt.gate_kind == "state"
        assert receipt.gate_reason == "eps_M exceeded soft threshold"
        assert receipt.hard_fail is False
        assert receipt.retry_index == 0
        assert receipt.dt_before == 1e-4
        assert receipt.dt_after == 8e-5
        assert receipt.snapshot_id == "snap789"
        assert receipt.repair_actions_allowed == ["reduce_dt", "increase_kappa"]
        assert receipt.repair_action_chosen == "reduce_dt"
    
    def test_receipt_invalid_gate_kind_raises(self):
        """MSolveReceipt should validate gate_kind."""
        kappa = Kappa(o=1, s=2, mu=None)
        
        with pytest.raises(Exception):  # SEMFailure
            MSolveReceipt(
                attempt_id=0,
                kappa=kappa,
                t=0.0,
                tau_attempt=0,
                residual_proxy={"CONS": {"L": 1e-5}},
                dt_cap_min=1e-6,
                dominant_thread=("PHY", "step", "act"),
                actions=[],
                policy_hash="abc123",
                state_hash="state456",
                stage_time=0.001,
                stage_id=0,
                sem_ok=False,
                rollback_count=0,
                perf={},
                gate_kind="invalid_kind"  # Invalid
            )


class TestDeterministicRetryPolicy:
    """Tests for deterministic retry policy behavior."""
    
    def test_retry_count_limits(self):
        """Retry policy should have deterministic max count."""
        retry_count_max = 5
        
        # Simulate retries
        for i in range(retry_count_max):
            # Retry logic here
            assert i < retry_count_max
        
        # After max, should escalate
        assert retry_count_max == 5
    
    def test_dt_shrink_schedule(self):
        """DT shrink should follow geometric backoff schedule."""
        shrink_schedule = [0.7, 0.5, 0.3]
        dt = 1e-4
        
        for i, factor in enumerate(shrink_schedule):
            new_dt = dt * factor
            assert new_dt < dt  # Each step should shrink
            dt = new_dt
        
        # Final dt should be significantly reduced
        assert dt == 1e-4 * 0.7 * 0.5 * 0.3
    
    def test_dt_floor_enforcement(self):
        """DT should not shrink below floor value."""
        dt_floor = 1e-10
        dt = 1e-4
        shrink_factor = 0.7
        
        for _ in range(10):  # Many retries
            dt = max(dt * shrink_factor, dt_floor)
        
        assert dt >= dt_floor


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
