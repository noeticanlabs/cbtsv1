"""Test AML (Aeonic Memory Language) legality layer."""

from aml import AML, ThreadTag
from receipt_schemas import Kappa, SEMFailure
import json

def test_aml_initialization():
    """Test AML initializes with coupling policy."""
    aml = AML()

    assert aml.compartments == ["SOLVE", "STEP", "ORCH", "S_PHY"]
    assert "SOLVE" in aml.coupling_matrix
    assert aml.gate_thresholds["eps_H_max"] == 1e-4
    assert aml.ops_allowed_s_phy == ["projection", "enforcement", "clamp"]

def test_thread_tagging():
    """Test thread tagging."""
    aml = AML()
    kappa = Kappa(o=0, s=0, mu=0)

    tag = aml.tag_thread("SOLVE", "solve", kappa)
    assert tag.compartment == "SOLVE"
    assert tag.operation == "solve"
    assert tag.kappa == kappa

    current_tag = aml.get_current_tag()
    assert current_tag == tag

def test_compartment_enforcement():
    """Test compartment transition enforcement."""
    aml = AML()

    # Valid transitions
    aml.enforce_compartment_transition("SOLVE", "STEP")  # OK
    aml.enforce_compartment_transition("STEP", "ORCH")   # OK
    aml.enforce_compartment_transition("ORCH", "S_PHY")  # OK

    # Invalid transition
    try:
        aml.enforce_compartment_transition("S_PHY", "SOLVE")  # Not allowed
        assert False, "Should have raised SEMFailure"
    except SEMFailure:
        pass

def test_s_phy_protection():
    """Test S_PHY compartment protection."""
    aml = AML()

    # Tag as S_PHY
    aml.tag_thread("S_PHY", "projection")  # Allowed

    # Should not raise
    aml.protect_s_phy("projection")
    aml.protect_s_phy("enforcement")
    aml.protect_s_phy("clamp")

    # Should raise for disallowed
    try:
        aml.protect_s_phy("invalid_op")
        assert False, "Should have raised SEMFailure"
    except SEMFailure:
        pass

def test_gate_protocol():
    """Test gate threshold enforcement."""
    aml = AML()

    # Should pass
    assert aml.validate_gate(1e-5, 1e-5)

    # Should fail
    try:
        aml.validate_gate(1e-3, 1e-5)  # eps_H too high
        assert False, "Should have raised SEMFailure"
    except SEMFailure:
        pass

    try:
        aml.validate_gate(0.0, 0.0)  # SEM hard failure
        assert False, "Should have raised SEMFailure"
    except SEMFailure:
        pass

def test_retry_schedules():
    """Test retry logic."""
    aml = AML()

    # First retry
    can_retry, new_dt = aml.retry_logic(0)
    assert can_retry
    assert new_dt > 0

    # Max attempts
    try:
        aml.retry_logic(10)  # Max attempts exceeded
        assert False, "Should have raised SEMFailure"
    except SEMFailure:
        pass

def test_transaction_commit_rollback():
    """Test transaction management."""
    aml = AML()
    tag = ThreadTag("SOLVE", "solve")

    # Begin transaction
    aml.begin_transaction(tag)
    assert len(aml.transaction_stack) == 1

    # Commit
    result = aml.commit_transaction()
    assert result
    assert len(aml.transaction_stack) == 0

    # Test rollback
    aml.begin_transaction(tag)
    aml.rollback_transaction()
    assert len(aml.transaction_stack) == 0

def test_execute_operation():
    """Test executing operations within AML."""
    aml = AML()
    kappa = Kappa(o=0, s=0, mu=0)

    def dummy_solve():
        return {
            't': 0.0,
            'residuals': {'domain1': {'scale1': 1e-6}},
            'dt_min': 1e-6,
            'actions': [],
            'state': {},
            'stage_time': 0.1,
            'stage_id': 1,
            'sem_ok': True,
            'rollback_count': 0,
            'perf': {}
        }

    # Execute in SOLVE compartment
    result = aml.execute_operation(
        dummy_solve,
        compartment="SOLVE",
        operation="solve",
        kappa=kappa
    )

    assert result['t'] == 0.0
    assert result['sem_ok'] == True

    # Check that receipt was generated (via memory_contract counters)
    assert aml.memory_contract.attempt_counter == 1

def test_nsc_gr_coupling():
    """Test NSC↔GR/NR coupling enforcement via AML."""
    aml = AML()
    kappa = Kappa(o=0, s=0, mu=0)

    # Simulate NSC initiating solve in SOLVE compartment
    def nsc_solve():
        # NSC computes symbolic solution
        return {
            't': 0.1,
            'residuals': {'domain1': {'scale1': 1e-6}},
            'dt_min': 1e-6,
            'actions': [{'type': 'solve_attempt', 'status': 'ok'}],
            'state': {'phi': 'symbolic_expr'},
            'stage_time': 0.05,
            'stage_id': 1,
            'sem_ok': True,
            'rollback_count': 0,
            'perf': {'nsc_time': 0.05}
        }

    # Execute NSC solve
    nsc_result = aml.execute_operation(
        nsc_solve,
        compartment="SOLVE",
        operation="solve",
        kappa=kappa
    )
    print(f"NSC solve result: {nsc_result}")

    # Now simulate GR/NR step in STEP compartment
    # First enforce transition SOLVE -> STEP
    aml.enforce_compartment_transition("SOLVE", "STEP")

    def gr_step():
        # GR/NR numerical step
        eps_H = 5e-5  # Below threshold
        eps_M = 3e-5
        aml.validate_gate(eps_H, eps_M)  # Should pass

        return {
            't': 0.1,
            'residuals': {'domain1': {'scale1': eps_H}, 'domain2': {'scale1': eps_M}},
            'enforcement': 1e-4,
            'dt_used': 1e-6,
            'rollback_count': 0,
            'gate_result': {'pass': True},
            'actions': [{'type': 'enforce_constraints'}]
        }

    # Execute GR step
    gr_result = aml.execute_operation(
        gr_step,
        compartment="STEP",
        operation="step",
        kappa=kappa
    )
    print(f"GR step result: {gr_result}")

    # Simulate ORCH promotion
    aml.enforce_compartment_transition("STEP", "ORCH")

    def orch_promote():
        return {
            'o': 1,
            'window_steps': [1, 2, 3],
            'quantiles': {},
            'dominance_histogram': {'thread1': 10},
            'chatter_score': 0.1,
            'regime_label': 'stable',
            'promotions': [{'canon': 'v1.0'}],
            'min_history': 1
        }

    orch_result = aml.execute_operation(
        orch_promote,
        compartment="ORCH",
        operation="promote",
        kappa=kappa
    )
    print(f"ORCH result: {orch_result}")

    # Test S_PHY operation
    aml.enforce_compartment_transition("ORCH", "S_PHY")

    def s_phy_enforce():
        # Projection operation
        return {'enforced': True}

    s_phy_result = aml.execute_operation(
        s_phy_enforce,
        compartment="S_PHY",
        operation="enforcement",
        kappa=kappa
    )
    print(f"S_PHY result: {s_phy_result}")

    # Check counters
    assert aml.memory_contract.attempt_counter >= 1
    assert aml.memory_contract.step_counter >= 1

if __name__ == "__main__":
    test_aml_initialization()
    test_thread_tagging()
    test_compartment_enforcement()
    test_s_phy_protection()
    test_gate_protocol()
    test_retry_schedules()
    test_transaction_commit_rollback()
    test_execute_operation()
    test_nsc_gr_coupling()
    print("All AML tests passed!")