"""
Unit test for WP1 Global Smoothness Bootstrap Certificate structure.
Tests certificate schema without running full simulation.
"""
import json
import sys
import os
from datetime import datetime
from unittest.mock import Mock, patch

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from wp1_global_smoothness_bootstrap import (
    create_wph1_certificate,
    EnergyHistory,
    ConstraintHistory,
    RailHistory,
    OperatorReceipt,
    compute_state_hash,
    compute_decay_trend
)


def test_certificate_structure():
    """Verify certificate contains all required fields."""
    # Create mock histories
    energy_history = EnergyHistory(
        times=[0.0, 0.1, 0.2],
        energies=[100.0, 105.0, 102.0],
        gamma_norms=[10.0, 10.2, 10.1],
        K_norms=[1.0, 1.05, 1.02]
    )
    
    constraint_history = ConstraintHistory(
        times=[0.0, 0.1, 0.2],
        eps_H=[1e-8, 5e-9, 2e-9],
        eps_M=[1e-9, 5e-10, 2e-10],
        eps_H_linf=[1e-7, 5e-8, 2e-8],
        eps_M_linf=[1e-8, 5e-9, 2e-9]
    )
    
    rail_history = RailHistory(
        times=[0.0, 0.1, 0.2],
        rail_actions=[0, 2, 3],
        dt_ratios=[1.0, 0.95, 0.98],
        det_min=[1.0, 0.99, 0.995]
    )
    
    operator_receipts = [
        OperatorReceipt(
            timestamp=datetime.utcnow().isoformat() + 'Z',
            step_id=10,
            mms_defect=1e-10,
            isolation_hash='abc123',
            state_hash_before='hash1',
            state_hash_after='hash2'
        )
    ]
    
    test_params = {
        'N': 16,
        'grid_size': 8.0,
        'T_max': 0.1,
        'energy_bound_factor': 10.0,
        'eps_H_threshold': 1e-6,
        'eps_M_threshold': 1e-6,
        'det_threshold': 0.2,
        'max_rail_actions': 10000,
        'mms_defect_threshold': 1e-8
    }
    
    certificate = create_wph1_certificate(
        test_params=test_params,
        energy_history=energy_history,
        constraint_history=constraint_history,
        rail_history=rail_history,
        operator_receipts=operator_receipts,
        mms_defect=1e-10,
        isolation_hash='abc123'
    )
    
    # Verify all required fields
    assert 'certificate_type' in certificate
    assert certificate['certificate_type'] == 'WP1_Global_Smoothness_Bootstrap'
    
    assert 'version' in certificate
    assert 'timestamp' in certificate
    
    # Test energy_boundedness
    assert 'energy_boundedness' in certificate
    eb = certificate['energy_boundedness']
    assert 'max' in eb
    assert 'initial' in eb
    assert 'ratio_to_initial' in eb
    assert 'margin' in eb
    assert 'bound_witness' in eb
    assert 'times' in eb
    assert 'values' in eb
    
    # Test constraint_coherence
    assert 'constraint_coherence' in certificate
    cc = certificate['constraint_coherence']
    assert 'max_hamiltonian' in cc
    assert 'max_momentum' in cc
    assert 'hamiltonian_threshold' in cc
    assert 'momentum_threshold' in cc
    assert 'decay_trend_H' in cc
    assert 'decay_trend_M' in cc
    assert 'constraint_satisfied' in cc
    
    # Test rail_spending
    assert 'rail_spending' in certificate
    rs = certificate['rail_spending']
    assert 'total_actions' in rs
    assert 'max_rail_actions' in rs
    assert 'bounded' in rs
    assert 'det_min' in rs
    assert 'det_threshold' in rs
    
    # Test operator_receipts
    assert 'operator_receipts' in certificate
    ors = certificate['operator_receipts']
    assert 'mms_defect' in ors
    assert 'isolation_hash' in ors
    assert 'mms_defect_threshold' in ors
    assert 'defect_satisfied' in ors
    assert 'receipt_count' in ors
    
    # Test ledger_proofs
    assert 'ledger_proofs' in certificate
    lp = certificate['ledger_proofs']
    assert 'receipt_chain_valid' in lp
    assert 'hash_chain' in lp
    
    # Test summary
    assert 'summary' in certificate
    s = certificate['summary']
    assert 'energy_bounded' in s
    assert 'constraints_coherent' in s
    assert 'rails_bounded' in s
    assert 'mms_verified' in s
    assert 'bootstrap_passed' in s
    
    print("✅ Certificate structure validation passed!")
    print(f"Certificate type: {certificate['certificate_type']}")
    print(f"Version: {certificate['version']}")
    print(f"Bootstrap passed: {certificate['summary']['bootstrap_passed']}")
    
    return certificate


def test_decay_trend_computation():
    """Test decay trend computation."""
    times = [0.0, 0.1, 0.2, 0.3]
    values = [1e-6, 5e-7, 2.5e-7, 1.25e-7]  # Exponential decay
    
    trend = compute_decay_trend(times, values)
    
    assert trend is not None
    assert 'decay_rate' in trend
    assert 'intercept' in trend
    assert 'r_squared' in trend
    assert trend['decay_rate'] < 0  # Should be decaying
    
    print(f"✅ Decay trend computed: rate={trend['decay_rate']:.2f}, R²={trend['r_squared']:.4f}")
    return trend


def test_state_hash():
    """Test state hash computation."""
    mock_solver = Mock()
    # Return float values directly (simulating numpy scalar)
    mock_solver.fields.gamma_sym6.mean.return_value = 1.0
    mock_solver.fields.K_sym6.mean.return_value = 0.5
    mock_solver.fields.alpha.mean.return_value = 1.0
    mock_solver.t = 0.0
    
    hash_val = compute_state_hash(mock_solver)
    
    assert hash_val is not None
    assert len(hash_val) == 16  # SHA256 hexdigest truncated to 16 chars
    
    print(f"✅ State hash computed: {hash_val}")
    return hash_val


def test_certificate_json_roundtrip():
    """Test certificate can be serialized and deserialized."""
    cert = test_certificate_structure()
    
    # Serialize
    json_str = json.dumps(cert, indent=2)
    
    # Deserialize
    cert_loaded = json.loads(json_str)
    
    assert cert_loaded['certificate_type'] == cert['certificate_type']
    assert cert_loaded['summary']['bootstrap_passed'] == cert['summary']['bootstrap_passed']
    
    print("✅ JSON roundtrip test passed!")
    return True


if __name__ == "__main__":
    print("="*60)
    print("WP1 Bootstrap Certificate Unit Tests")
    print("="*60)
    print()
    
    test_state_hash()
    print()
    
    test_decay_trend_computation()
    print()
    
    test_certificate_structure()
    print()
    
    test_certificate_json_roundtrip()
    print()
    
    print("="*60)
    print("All unit tests passed!")
    print("="*60)
