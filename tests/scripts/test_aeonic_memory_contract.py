import numpy as np
from aeonic_memory_bank import AeonicMemoryBank
from aeonic_memory_contract import AeonicMemoryContract, SEMFailure
from phaseloom_27 import PhaseLoom27
from receipt_schemas import Kappa, MSolveReceipt, MStepReceipt
from aeonic_clocks import AeonicClockPack
from aeonic_receipts import AeonicReceipts

class MockState:
    def __init__(self):
        self.alpha = np.ones(10)
        self.beta = np.zeros((10, 3))

class MockGeometry:
    def __init__(self):
        self.det_gamma = np.ones(10)

def test_attempt_accepted_separation():
    """Test that attempt_id increments always, step_id only on acceptance."""
    clock = AeonicClockPack()
    receipts = AeonicReceipts()
    bank = AeonicMemoryBank(clock, receipts)
    memory = AeonicMemoryContract(bank, receipts)

    # Create attempt receipt
    kappa = Kappa(o=0, s=0, mu=0)
    attempt = MSolveReceipt(
        attempt_id=0, kappa=kappa, t=0.0, tau_attempt=0,
        residual_proxy={'PHY': {'L': 1e-7}}, dt_cap_min=0.1,
        dominant_thread=('PHY', 'L', 'FAST'), actions=[],
        policy_hash="abc123", state_hash="def456",
        stage_time=0.0, stage_id=0, sem_ok=True, rollback_count=0, perf={}
    )

    memory.put_attempt_receipt(kappa, attempt)
    assert memory.attempt_counter == 1

    # Create step receipt (only if accepted)
    step = MStepReceipt(
        step_id=0, kappa=Kappa(o=0, s=0, mu=None), t=0.1, tau_step=0,
        residual_full={'PHY': {'L': 1e-8}}, enforcement_magnitude=0.0,
        dt_used=0.1, rollback_count=0,
        gate_after={'pass': True}, actions_applied=[]
    )

    memory.put_step_receipt(kappa, step)
    assert memory.step_counter == 1

def test_gate_classification():
    """Test gate classification: dt vs state vs sem."""
    phaseloom = PhaseLoom27()

    # Test SEM gate
    residuals = {('SEM', 'L'): float('inf')}
    dt_cap, dominant = phaseloom.arbitrate_dt(residuals)
    gate = phaseloom.get_gate_classification(dominant, float('inf'))
    assert gate['kind'] == 'sem'

    # Test dt gate (PHY)
    residuals = {('PHY', 'L'): 1e-3}
    dt_cap, dominant = phaseloom.arbitrate_dt(residuals)
    gate = phaseloom.get_gate_classification(dominant, 1e-3)
    assert gate['kind'] == 'dt'

def test_sem_barriers():
    """Test SEM safety: no silent zeros, no nonfinite."""
    clock = AeonicClockPack()
    receipts = AeonicReceipts()
    bank = AeonicMemoryBank(clock, receipts)
    memory = AeonicMemoryContract(bank, receipts)

    # Test no silent zeros
    state = MockState()
    geometry = MockGeometry()
    assert memory.check_no_silent_zeros(state, geometry)

    # Break it
    state.alpha[0] = 0
    assert not memory.check_no_silent_zeros(state, geometry)

    # Test no nonfinite
    assert memory.check_no_nonfinite(np.array([1.0, 2.0, 3.0]))
    assert not memory.check_no_nonfinite(np.array([1.0, np.nan, 3.0]))

def test_state_gate_abort():
    """Test abort on state gate with no repair action."""
    clock = AeonicClockPack()
    receipts = AeonicReceipts()
    bank = AeonicMemoryBank(clock, receipts)
    memory = AeonicMemoryContract(bank, receipts)

    # Should abort
    gate = {'kind': 'state', 'actions_allowed': []}
    try:
        memory.abort_on_state_gate_no_repair(gate)
        assert False, "Should have raised SEMFailure"
    except SEMFailure:
        pass  # Expected

    # Should not abort if repair available
    gate = {'kind': 'state', 'actions_allowed': [{'repair': True}]}
    memory.abort_on_state_gate_no_repair(gate)  # No exception

if __name__ == "__main__":
    test_attempt_accepted_separation()
    test_gate_classification()
    test_sem_barriers()
    test_state_gate_abort()
    print("All tests passed!")