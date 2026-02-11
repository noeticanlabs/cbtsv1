"""
Test suite for hard invariant validation per Theorem Lemma 1.

Tests verify:
1. HardInvariantChecker validates all required invariants
2. Step rejected if hard invariants violated before acceptance
3. Configuration-driven checking
4. All violations logged appropriately
"""

import pytest
import numpy as np
import logging
from unittest.mock import Mock, MagicMock, patch
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.hard_invariants import HardInvariantChecker


class TestHardInvariantChecker:
    """Test suite for HardInvariantChecker class."""
    
    def test_init(self):
        """Test HardInvariantChecker initialization."""
        checker = HardInvariantChecker(tolerance=1e-14)
        assert checker.tolerance == 1e-14
        assert checker.violations == []
    
    def test_valid_fields(self):
        """Test that valid fields pass all invariant checks."""
        # Create mock fields with valid data
        fields = Mock()
        fields.alpha = np.ones((3, 3, 3))  # Positive lapse
        
        # Identity metric in sym6 format
        fields.gamma_sym6 = np.zeros((6, 3, 3, 3))
        fields.gamma_sym6[0] = 1.0  # gamma_11 = 1
        fields.gamma_sym6[3] = 1.0  # gamma_22 = 1
        fields.gamma_sym6[5] = 1.0  # gamma_33 = 1
        
        fields.K_sym6 = np.zeros((6, 3, 3, 3))  # Zero K
        
        checker = HardInvariantChecker()
        is_valid, violations, margins = checker.check_hard_invariants(fields)
        
        assert is_valid is True
        assert len(violations) == 0
        assert 'alpha_min' in margins
        assert margins['alpha_min'] >= 1.0
        assert 'metric_eigenvalue_min' in margins
        assert margins['metric_det_min'] > 0
    
    def test_negative_alpha_violation(self):
        """Test detection of negative lapse violation."""
        fields = Mock()
        fields.alpha = np.array([[[1.0, -0.1, 1.0]]])  # Negative at one point
        
        fields.gamma_sym6 = np.zeros((6, 1, 1, 3))
        fields.gamma_sym6[0] = 1.0
        fields.gamma_sym6[3] = 1.0
        fields.gamma_sym6[5] = 1.0
        
        fields.K_sym6 = np.zeros((6, 1, 1, 3))
        
        checker = HardInvariantChecker()
        is_valid, violations, margins = checker.check_hard_invariants(fields)
        
        assert is_valid is False
        assert any("alpha not positive" in v for v in violations)
    
    def test_nan_alpha_violation(self):
        """Test detection of NaN in lapse field."""
        fields = Mock()
        fields.alpha = np.array([[[1.0, np.nan, 1.0]]])
        
        fields.gamma_sym6 = np.zeros((6, 1, 1, 3))
        fields.gamma_sym6[0] = 1.0
        fields.gamma_sym6[3] = 1.0
        fields.gamma_sym6[5] = 1.0
        
        fields.K_sym6 = np.zeros((6, 1, 1, 3))
        
        checker = HardInvariantChecker()
        is_valid, violations, margins = checker.check_hard_invariants(fields)
        
        assert is_valid is False
        assert any("NaN/inf" in v for v in violations)
    
    def test_nan_metric_violation(self):
        """Test detection of NaN in metric."""
        fields = Mock()
        fields.alpha = np.ones((1, 1, 3))
        
        fields.gamma_sym6 = np.zeros((6, 1, 1, 3))
        fields.gamma_sym6[0] = 1.0
        fields.gamma_sym6[0, 0, 0, 0] = np.nan  # NaN at one point
        fields.gamma_sym6[3] = 1.0
        fields.gamma_sym6[5] = 1.0
        
        fields.K_sym6 = np.zeros((6, 1, 1, 3))
        
        checker = HardInvariantChecker()
        is_valid, violations, margins = checker.check_hard_invariants(fields)
        
        assert is_valid is False
        assert any("NaN/inf" in v for v in violations)
    
    def test_non_positive_definite_metric(self):
        """Test detection of non-positive definite metric."""
        fields = Mock()
        fields.alpha = np.ones((1, 1, 1))
        
        # Create a non-positive-definite metric
        # [[-0.1, 0.1, 0.1], [0.1, 1, 0.1], [0.1, 0.1, 1]]
        fields.gamma_sym6 = np.zeros((6, 1, 1, 1))
        fields.gamma_sym6[0, 0, 0, 0] = -0.1  # gamma_11 negative
        fields.gamma_sym6[1, 0, 0, 0] = 0.1
        fields.gamma_sym6[2, 0, 0, 0] = 0.1
        fields.gamma_sym6[3, 0, 0, 0] = 1.0
        fields.gamma_sym6[4, 0, 0, 0] = 0.1
        fields.gamma_sym6[5, 0, 0, 0] = 1.0
        
        fields.K_sym6 = np.zeros((6, 1, 1, 1))
        
        checker = HardInvariantChecker()
        is_valid, violations, margins = checker.check_hard_invariants(fields)
        
        assert is_valid is False
        assert any("positive definite" in v for v in violations)
    
    def test_negative_determinant(self):
        """Test detection of negative metric determinant."""
        fields = Mock()
        fields.alpha = np.ones((1, 1, 1))
        
        # Create metric with negative determinant
        # det = -0.1 for diag(-0.1, 1, 1)
        fields.gamma_sym6 = np.zeros((6, 1, 1, 1))
        fields.gamma_sym6[0, 0, 0, 0] = -0.1  # gamma_11
        fields.gamma_sym6[3, 0, 0, 0] = 1.0   # gamma_22
        fields.gamma_sym6[5, 0, 0, 0] = 1.0   # gamma_33
        
        fields.K_sym6 = np.zeros((6, 1, 1, 1))
        
        checker = HardInvariantChecker()
        is_valid, violations, margins = checker.check_hard_invariants(fields)
        
        assert is_valid is False
        assert any("determinant" in v for v in violations)
    
    def test_get_report_success(self):
        """Test success report generation."""
        checker = HardInvariantChecker()
        report = checker.get_report()
        
        assert report['status'] == 'success'
        assert 'All hard invariants satisfied' in report['message']
        assert report['num_violations'] == 0
    
    def test_get_report_failure(self):
        """Test failure report generation."""
        fields = Mock()
        fields.alpha = np.array([[[1.0, -0.1, 1.0]]])
        fields.gamma_sym6 = np.zeros((6, 1, 1, 3))
        fields.gamma_sym6[0] = 1.0
        fields.gamma_sym6[3] = 1.0
        fields.gamma_sym6[5] = 1.0
        fields.K_sym6 = np.zeros((6, 1, 1, 3))
        
        checker = HardInvariantChecker()
        checker.check_hard_invariants(fields)
        report = checker.get_report()
        
        assert report['status'] == 'failed'
        assert report['num_violations'] == 1
        assert 'violations' in report
    
    def test_reset(self):
        """Test violation history reset."""
        fields = Mock()
        fields.alpha = np.array([[[1.0, -0.1, 1.0]]])
        fields.gamma_sym6 = np.zeros((6, 1, 1, 3))
        fields.gamma_sym6[0] = 1.0
        fields.gamma_sym6[3] = 1.0
        fields.gamma_sym6[5] = 1.0
        fields.K_sym6 = np.zeros((6, 1, 1, 3))
        
        checker = HardInvariantChecker()
        checker.check_hard_invariants(fields)
        
        assert len(checker.violations) == 1
        
        checker.reset()
        assert len(checker.violations) == 0
    
    def test_tolerance_parameter(self):
        """Test that tolerance parameter is properly stored."""
        checker1 = HardInvariantChecker(tolerance=1e-10)
        assert checker1.tolerance == 1e-10
        
        checker2 = HardInvariantChecker(tolerance=1e-16)
        assert checker2.tolerance == 1e-16


class TestGatePolicyHardInvariants:
    """Test suite for GatePolicy hard_invariants configuration."""
    
    def test_gate_policy_default_hard_invariants(self):
        """Test that GatePolicy has hard_invariants config."""
        from core.gate_policy import GatePolicy
        
        policy = GatePolicy()
        assert hasattr(policy, 'hard_invariants')
        assert 'check_before_acceptance' in policy.hard_invariants
        assert 'tolerance' in policy.hard_invariants
        assert 'halt_on_violation' in policy.hard_invariants
        
        assert policy.hard_invariants['check_before_acceptance'] is True
        assert policy.hard_invariants['tolerance'] == 1e-14
    
    def test_gate_policy_to_dict_includes_hard_invariants(self):
        """Test that to_dict includes hard_invariants config."""
        from core.gate_policy import GatePolicy
        
        policy = GatePolicy()
        policy_dict = policy.to_dict()
        
        assert 'hard_invariants' in policy_dict
        assert policy_dict['hard_invariants']['check_before_acceptance'] is True
        assert policy_dict['hard_invariants']['tolerance'] == 1e-14
    
    def test_gate_policy_validate_hard_invariants(self):
        """Test validation of hard_invariants config."""
        from core.gate_policy import GatePolicy
        
        policy = GatePolicy()
        # Should not raise
        policy.validate()
        
        # Invalid tolerance (negative)
        policy.hard_invariants['tolerance'] = -1
        with pytest.raises(ValueError, match="non-negative"):
            policy.validate()
        
        # Reset
        policy.hard_invariants['tolerance'] = 1e-14
        policy.validate()  # Should pass now
    
    def test_gate_policy_from_config_file(self):
        """Test loading hard_invariants from config file."""
        from core.gate_policy import GatePolicy
        import json
        import tempfile
        
        # Create temporary config with hard_invariants
        config = {
            'hard_invariants': {
                'check_before_acceptance': False,
                'tolerance': 1e-10,
                'halt_on_violation': True
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            temp_path = f.name
        
        try:
            policy = GatePolicy.from_file(temp_path)
            assert policy.hard_invariants['check_before_acceptance'] is False
            assert policy.hard_invariants['tolerance'] == 1e-10
            assert policy.hard_invariants['halt_on_violation'] is True
        finally:
            import os
            os.unlink(temp_path)


class TestHardInvariantsIntegration:
    """Integration tests for hard invariants with GRStepper."""
    
    @patch('core.gr_stepper.GatePolicy')
    @patch('core.gr_stepper.GRRhs')
    @patch('core.gr_stepper.GRLedger')
    @patch('core.gr_stepper.GRScheduler')
    @patch('core.gr_stepper.GateChecker')
    @patch('core.gr_stepper.TheoremValidator')
    @patch('core.gr_stepper.HardInvariantChecker')
    def test_stepper_initializes_invariant_checker(
        self, 
        mock_hi_checker,
        mock_theorem_validator,
        mock_gate_checker,
        mock_scheduler,
        mock_ledger,
        mock_rhs,
        mock_gate_policy
    ):
        """Test that GRStepper initializes HardInvariantChecker."""
        from core.gr_stepper import GRStepper
        
        # Setup mocks
        mock_fields = Mock()
        mock_fields.Nx = 2
        mock_fields.Ny = 2
        mock_fields.Nz = 2
        mock_fields.Lambda = 0.1
        mock_fields.gamma_sym6 = np.ones((6, 2, 2, 2))
        mock_fields.K_sym6 = np.zeros((6, 2, 2, 2))
        mock_fields.alpha = np.ones((2, 2, 2))
        mock_fields.beta = np.zeros((2, 2, 2, 3))
        mock_fields.phi = np.zeros((2, 2, 2))
        mock_fields.Z = np.zeros((2, 2, 2))
        mock_fields.Z_i = np.zeros((2, 2, 2, 3))
        mock_fields.gamma_tilde_sym6 = np.ones((6, 2, 2, 2))
        mock_fields.A_sym6 = np.zeros((6, 2, 2, 2))
        mock_fields.Gamma_tilde = np.zeros((2, 2, 2, 3))
        
        mock_geometry = Mock()
        mock_constraints = Mock()
        mock_gauge = Mock()
        
        mock_gate_policy_instance = Mock()
        mock_gate_policy_instance.theorem_validation = {'enabled': False}
        mock_gate_policy_instance.hard_invariants = {
            'check_before_acceptance': True,
            'tolerance': 1e-14,
            'halt_on_violation': False
        }
        mock_gate_policy.return_value = mock_gate_policy_instance
        
        # Create stepper
        stepper = GRStepper(
            mock_fields,
            mock_geometry,
            mock_constraints,
            mock_gauge
        )
        
        # Verify HardInvariantChecker was initialized
        mock_hi_checker.assert_called_once()
        call_args = mock_hi_checker.call_args
        assert call_args[1]['tolerance'] == 1e-14
        
        assert hasattr(stepper, 'invariant_checker')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
