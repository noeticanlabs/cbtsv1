"""Tests for theorem validators (Lemma 3 - Debt Boundedness)"""

import pytest
import logging
from src.core.theorem_validators import TheoremValidator

logger = logging.getLogger('test_theorem_validator')


class TestTheoremValidator:
    """Tests for TheoremValidator class"""
    
    def test_initialization_valid_gamma(self):
        """Test validator initialization with valid gamma"""
        validator = TheoremValidator(gamma=0.8, b=1e-4)
        assert validator.gamma == 0.8
        assert validator.b == 1e-4
        assert validator.enable_halt_on_violation is False
        assert len(validator.violations) == 0
    
    def test_initialization_invalid_gamma_zero(self):
        """Test validator raises on gamma=0"""
        with pytest.raises(AssertionError):
            TheoremValidator(gamma=0.0, b=1e-4)
    
    def test_initialization_invalid_gamma_one(self):
        """Test validator raises on gamma=1"""
        with pytest.raises(AssertionError):
            TheoremValidator(gamma=1.0, b=1e-4)
    
    def test_initialization_invalid_gamma_negative(self):
        """Test validator raises on negative gamma"""
        with pytest.raises(AssertionError):
            TheoremValidator(gamma=-0.5, b=1e-4)
    
    def test_initialization_invalid_gamma_greater_than_one(self):
        """Test validator raises on gamma > 1"""
        with pytest.raises(AssertionError):
            TheoremValidator(gamma=1.5, b=1e-4)
    
    def test_validate_contraction_valid(self):
        """Test contraction validation with valid debt"""
        validator = TheoremValidator(gamma=0.8, b=1e-4)
        
        # debt_after = 0.5, threshold = 0.8 * 1.0 + 1e-4 = 0.80010
        is_valid, margin, msg = validator.validate_contraction(
            debt_before=1.0,
            debt_after=0.5,
            step_num=1
        )
        
        assert is_valid is True
        assert margin > 0  # positive margin means threshold - debt_after > 0
        assert 'Contraction ok' in msg
        assert len(validator.violations) == 0
    
    def test_validate_contraction_invalid(self):
        """Test contraction validation with violated debt"""
        validator = TheoremValidator(gamma=0.8, b=1e-4, enable_halt_on_violation=False)
        
        # debt_after = 1.0, threshold = 0.8 * 1.0 + 1e-4 = 0.80010
        # 1.0 > 0.80010, so this violates the contraction
        is_valid, margin, msg = validator.validate_contraction(
            debt_before=1.0,
            debt_after=1.0,
            step_num=1
        )
        
        assert is_valid is False
        assert margin < 0  # negative margin means violation
        assert 'Contraction violated' in msg
        assert len(validator.violations) == 1
        assert validator.violations[0] == (1, 1.0, 1.0, 0.8 + 1e-4)
    
    def test_validate_contraction_halt_on_violation(self):
        """Test that halt_on_violation=True raises exception"""
        validator = TheoremValidator(gamma=0.8, b=1e-4, enable_halt_on_violation=True)
        
        with pytest.raises(RuntimeError):
            validator.validate_contraction(
                debt_before=1.0,
                debt_after=1.0,
                step_num=1
            )
    
    def test_multiple_violations_tracking(self):
        """Test that multiple violations are tracked"""
        validator = TheoremValidator(gamma=0.8, b=1e-4)
        
        # Step 1: violation
        validator.validate_contraction(debt_before=1.0, debt_after=1.0, step_num=1)
        
        # Step 2: valid
        validator.validate_contraction(debt_before=0.9, debt_after=0.5, step_num=2)
        
        # Step 3: violation
        validator.validate_contraction(debt_before=0.8, debt_after=0.9, step_num=3)
        
        assert len(validator.violations) == 2
        assert validator.violations[0][0] == 1  # first violation at step 1
        assert validator.violations[1][0] == 3  # second violation at step 3
    
    def test_get_violation_report_no_violations(self):
        """Test violation report with no violations"""
        validator = TheoremValidator(gamma=0.8, b=1e-4)
        
        validator.validate_contraction(debt_before=1.0, debt_after=0.5, step_num=1)
        
        report = validator.get_violation_report()
        
        assert report['num_violations'] == 0
        assert report['violations'] == []
        assert 'No contractions violated' in report['status']
    
    def test_get_violation_report_with_violations(self):
        """Test violation report with violations"""
        validator = TheoremValidator(gamma=0.8, b=1e-4)
        
        # Create some violations
        validator.validate_contraction(debt_before=1.0, debt_after=1.0, step_num=1)
        validator.validate_contraction(debt_before=0.9, debt_after=0.95, step_num=2)
        
        report = validator.get_violation_report()
        
        assert report['num_violations'] == 2
        assert report['first_violation_step'] == 1
        assert len(report['violations']) == 2
        assert 'violations detected' in report['status']
    
    def test_reset_violations(self):
        """Test that reset clears violations"""
        validator = TheoremValidator(gamma=0.8, b=1e-4)
        
        # Create violations
        validator.validate_contraction(debt_before=1.0, debt_after=1.0, step_num=1)
        assert len(validator.violations) == 1
        
        # Reset
        validator.reset()
        assert len(validator.violations) == 0
        
        # Verify can validate again
        is_valid, _, _ = validator.validate_contraction(debt_before=0.5, debt_after=0.3, step_num=2)
        assert is_valid is True
        assert len(validator.violations) == 0
    
    def test_boundary_case_exactly_at_threshold(self):
        """Test contraction validation at exact threshold"""
        validator = TheoremValidator(gamma=0.8, b=1e-4)
        
        # debt_after exactly equals threshold
        threshold = 0.8 * 1.0 + 1e-4
        is_valid, margin, msg = validator.validate_contraction(
            debt_before=1.0,
            debt_after=threshold,
            step_num=1
        )
        
        assert is_valid is True
        assert margin == pytest.approx(0.0, abs=1e-15)
        assert len(validator.violations) == 0


class TestTheoremValidatorIntegration:
    """Integration tests for theorem validator with realistic scenarios"""
    
    def test_contracting_debt_sequence(self):
        """Test with a contracting debt sequence"""
        validator = TheoremValidator(gamma=0.8, b=1e-4)
        
        # Simulate a solver run with contracting debt
        debts = [1.0, 0.8, 0.64, 0.512, 0.4096]
        
        prev_debt = 0.0
        for i, debt in enumerate(debts, 1):
            is_valid, margin, _ = validator.validate_contraction(
                debt_before=prev_debt,
                debt_after=debt,
                step_num=i
            )
            assert is_valid is True
            prev_debt = debt
        
        report = validator.get_violation_report()
        assert report['num_violations'] == 0
    
    def test_non_contracting_debt_sequence(self):
        """Test with a non-contracting debt sequence (should have violations)"""
        validator = TheoremValidator(gamma=0.8, b=1e-4)
        
        # Simulate debts that violate contraction
        debts = [1.0, 1.1, 1.2]  # increasing debts will violate contraction
        
        prev_debt = 0.0
        for i, debt in enumerate(debts, 1):
            validator.validate_contraction(
                debt_before=prev_debt,
                debt_after=debt,
                step_num=i
            )
            prev_debt = debt
        
        report = validator.get_violation_report()
        assert report['num_violations'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
