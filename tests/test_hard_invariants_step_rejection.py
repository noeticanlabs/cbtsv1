"""
Test suite for hard invariant step rejection in GRStepper.

Verifies that steps are properly rejected when hard invariants are violated
before acceptance, as required by Theorem Lemma 1.
"""

import pytest
import numpy as np
import logging
from unittest.mock import Mock, MagicMock, patch, call
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.hard_invariants import HardInvariantChecker


class TestStepRejectionOnHardInvariantViolation:
    """Test suite for step rejection when hard invariants are violated."""
    
    def test_step_rejected_on_negative_alpha(self):
        """Test that step is rejected when alpha is negative."""
        from core.hard_invariants import HardInvariantChecker
        
        # Create fields with negative alpha
        fields = Mock()
        fields.alpha = np.array([[[-1.5]]])  # Negative lapse
        
        fields.gamma_sym6 = np.zeros((6, 1, 1, 1))
        fields.gamma_sym6[0] = 1.0
        fields.gamma_sym6[3] = 1.0
        fields.gamma_sym6[5] = 1.0
        
        fields.K_sym6 = np.zeros((6, 1, 1, 1))
        
        checker = HardInvariantChecker()
        is_valid, violations, margins = checker.check_hard_invariants(fields)
        
        # Verify step would be rejected
        assert is_valid is False
        assert len(violations) > 0
        assert margins['alpha_min'] < 0
        
        # Verify violation describes the issue
        assert any("alpha" in v.lower() for v in violations)
    
    def test_step_rejected_on_non_positive_definite_metric(self):
        """Test step rejection when metric is not positive definite."""
        from core.hard_invariants import HardInvariantChecker
        
        fields = Mock()
        fields.alpha = np.ones((1, 1, 1))
        
        # Non-positive definite metric (one eigenvalue is negative)
        fields.gamma_sym6 = np.zeros((6, 1, 1, 1))
        fields.gamma_sym6[0, 0, 0, 0] = -0.5   # gamma_11 (negative)
        fields.gamma_sym6[1, 0, 0, 0] = 0.0    # gamma_12
        fields.gamma_sym6[2, 0, 0, 0] = 0.0    # gamma_13
        fields.gamma_sym6[3, 0, 0, 0] = 1.0    # gamma_22
        fields.gamma_sym6[4, 0, 0, 0] = 0.0    # gamma_23
        fields.gamma_sym6[5, 0, 0, 0] = 1.0    # gamma_33
        
        fields.K_sym6 = np.zeros((6, 1, 1, 1))
        
        checker = HardInvariantChecker()
        is_valid, violations, margins = checker.check_hard_invariants(fields)
        
        assert is_valid is False
        assert any("positive definite" in v for v in violations)
    
    def test_step_rejected_on_inf_in_metric(self):
        """Test step rejection when metric contains infinity."""
        from core.hard_invariants import HardInvariantChecker
        
        fields = Mock()
        fields.alpha = np.ones((1, 1, 1))
        
        fields.gamma_sym6 = np.zeros((6, 1, 1, 1))
        fields.gamma_sym6[0] = 1.0
        fields.gamma_sym6[0, 0, 0, 0] = np.inf  # Infinity in metric
        fields.gamma_sym6[3] = 1.0
        fields.gamma_sym6[5] = 1.0
        
        fields.K_sym6 = np.zeros((6, 1, 1, 1))
        
        checker = HardInvariantChecker()
        is_valid, violations, margins = checker.check_hard_invariants(fields)
        
        assert is_valid is False
        assert any("NaN/inf" in v for v in violations)
    
    def test_step_rejected_on_nan_in_k(self):
        """Test step rejection when extrinsic curvature contains NaN."""
        from core.hard_invariants import HardInvariantChecker
        
        fields = Mock()
        fields.alpha = np.ones((1, 1, 1))
        
        fields.gamma_sym6 = np.zeros((6, 1, 1, 1))
        fields.gamma_sym6[0] = 1.0
        fields.gamma_sym6[3] = 1.0
        fields.gamma_sym6[5] = 1.0
        
        fields.K_sym6 = np.zeros((6, 1, 1, 1))
        fields.K_sym6[0, 0, 0, 0] = np.nan  # NaN in K
        
        checker = HardInvariantChecker()
        is_valid, violations, margins = checker.check_hard_invariants(fields)
        
        assert is_valid is False
        assert any("NaN/inf" in v for v in violations)
    
    def test_multiple_violations_detected(self):
        """Test detection of multiple simultaneous violations."""
        from core.hard_invariants import HardInvariantChecker
        
        fields = Mock()
        fields.alpha = np.array([[[-1.0]]])  # Negative AND inf
        fields.alpha[0, 0, 0] = np.inf
        
        fields.gamma_sym6 = np.zeros((6, 1, 1, 1))
        fields.gamma_sym6[0, 0, 0, 0] = np.nan  # NaN in metric
        fields.gamma_sym6[3] = 1.0
        fields.gamma_sym6[5] = 1.0
        
        fields.K_sym6 = np.zeros((6, 1, 1, 1))
        
        checker = HardInvariantChecker()
        is_valid, violations, margins = checker.check_hard_invariants(fields)
        
        assert is_valid is False
        # Should detect multiple violations
        assert len(violations) >= 2
    
    def test_acceptance_logic_with_violation(self):
        """Test that acceptance logic properly rejects steps with violations."""
        from core.hard_invariants import HardInvariantChecker
        
        # Simulate the acceptance decision logic
        fields = Mock()
        fields.alpha = np.array([[[-0.5]]])  # Negative lapse
        
        fields.gamma_sym6 = np.zeros((6, 1, 1, 1))
        fields.gamma_sym6[0] = 1.0
        fields.gamma_sym6[3] = 1.0
        fields.gamma_sym6[5] = 1.0
        
        fields.K_sym6 = np.zeros((6, 1, 1, 1))
        
        checker = HardInvariantChecker()
        
        # Simulate gate acceptance
        accepted_gates = True
        hard_fail_gates = False
        reasons = []
        
        # Check hard invariants
        is_valid, violations, margins_inv = checker.check_hard_invariants(fields)
        
        if not is_valid:
            accepted_gates = False
            hard_fail_gates = True
            reasons.append(f"Hard invariant violation: {', '.join(violations)}")
        
        # Verify step would be rejected
        assert accepted_gates is False
        assert hard_fail_gates is True
        assert len(reasons) > 0
        assert "Hard invariant violation" in reasons[0]
    
    def test_margin_tracking_for_near_violations(self):
        """Test that safety margins are properly tracked."""
        from core.hard_invariants import HardInvariantChecker
        
        fields = Mock()
        fields.alpha = np.full((2, 2, 2), 0.01)  # Very small but positive
        
        fields.gamma_sym6 = np.zeros((6, 2, 2, 2))
        fields.gamma_sym6[0] = 1.0
        fields.gamma_sym6[3] = 1.0
        fields.gamma_sym6[5] = 1.0
        
        fields.K_sym6 = np.zeros((6, 2, 2, 2))
        
        checker = HardInvariantChecker()
        is_valid, violations, margins = checker.check_hard_invariants(fields)
        
        # Should still be valid but with low margin
        assert is_valid is True
        assert margins['alpha_min'] == 0.01
    
    def test_determinant_margin_tracking(self):
        """Test determinant margin tracking for near-degenerate metrics."""
        from core.hard_invariants import HardInvariantChecker
        
        fields = Mock()
        fields.alpha = np.ones((1, 1, 1))
        
        # Nearly degenerate metric (det = 0.001)
        fields.gamma_sym6 = np.zeros((6, 1, 1, 1))
        fields.gamma_sym6[0, 0, 0, 0] = 1.0
        fields.gamma_sym6[3, 0, 0, 0] = 1.0
        fields.gamma_sym6[5, 0, 0, 0] = 0.001  # Small determinant
        
        fields.K_sym6 = np.zeros((6, 1, 1, 1))
        
        checker = HardInvariantChecker()
        is_valid, violations, margins = checker.check_hard_invariants(fields)
        
        assert is_valid is True
        assert 'metric_det_min' in margins
        assert margins['metric_det_min'] > 0
        # Margin is small
        assert margins['metric_det_min'] < 0.01
    
    def test_violation_report_details(self):
        """Test that violation reports contain detailed information."""
        from core.hard_invariants import HardInvariantChecker
        
        fields = Mock()
        fields.alpha = np.array([[[-1.0]]])
        
        fields.gamma_sym6 = np.zeros((6, 1, 1, 1))
        fields.gamma_sym6[0, 0, 0, 0] = -1.0  # Also non-positive definite
        fields.gamma_sym6[3] = 1.0
        fields.gamma_sym6[5] = 1.0
        
        fields.K_sym6 = np.zeros((6, 1, 1, 1))
        
        checker = HardInvariantChecker()
        is_valid, violations, margins = checker.check_hard_invariants(fields)
        
        report = checker.get_report()
        
        assert report['status'] == 'failed'
        assert report['num_violations'] == 1  # One check event
        assert 'violations' in report
        
        # Check that violations list has detail
        assert len(report['violations']) > 0
        assert 'violations' in report['violations'][0]
        assert 'margins' in report['violations'][0]
    
    def test_consistent_alpha_min_across_checks(self):
        """Test that alpha_min is consistently the minimum value."""
        from core.hard_invariants import HardInvariantChecker
        
        alpha_vals = np.array([[[0.5, 1.0, 0.1]]])
        
        fields = Mock()
        fields.alpha = alpha_vals
        
        fields.gamma_sym6 = np.zeros((6, 1, 1, 3))
        fields.gamma_sym6[0] = 1.0
        fields.gamma_sym6[3] = 1.0
        fields.gamma_sym6[5] = 1.0
        
        fields.K_sym6 = np.zeros((6, 1, 1, 3))
        
        checker = HardInvariantChecker()
        is_valid, violations, margins = checker.check_hard_invariants(fields)
        
        assert is_valid is True
        assert margins['alpha_min'] == 0.1  # Minimum value in alpha


class TestConfigurationDrivenChecking:
    """Test configuration-driven hard invariant checking."""
    
    def test_checking_disabled_by_config(self):
        """Test that checking can be disabled via config."""
        from core.gate_policy import GatePolicy
        
        policy = GatePolicy()
        policy.hard_invariants['check_before_acceptance'] = False
        
        assert policy.hard_invariants['check_before_acceptance'] is False
        
        # Validate should still pass
        policy.validate()
    
    def test_tolerance_configuration(self):
        """Test tolerance configuration."""
        from core.gate_policy import GatePolicy
        
        policy = GatePolicy()
        
        # Change tolerance
        policy.hard_invariants['tolerance'] = 1e-8
        policy.validate()
        assert policy.hard_invariants['tolerance'] == 1e-8
        
        # Very tight tolerance
        policy.hard_invariants['tolerance'] = 1e-16
        policy.validate()
        assert policy.hard_invariants['tolerance'] == 1e-16
    
    def test_halt_on_violation_config(self):
        """Test halt_on_violation configuration."""
        from core.gate_policy import GatePolicy
        
        policy = GatePolicy()
        
        # Default is False (don't halt, just log)
        assert policy.hard_invariants['halt_on_violation'] is False
        
        # Can be set to True
        policy.hard_invariants['halt_on_violation'] = True
        policy.validate()
        assert policy.hard_invariants['halt_on_violation'] is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
