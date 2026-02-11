"""Simple tests for theorem validators (Lemma 3 - Debt Boundedness) - no pytest required"""

import sys
sys.path.insert(0, '/workspaces/cbtsv1')

from src.core.theorem_validators import TheoremValidator


def test_initialization_valid_gamma():
    """Test validator initialization with valid gamma"""
    validator = TheoremValidator(gamma=0.8, b=1e-4)
    assert validator.gamma == 0.8
    assert validator.b == 1e-4
    assert validator.enable_halt_on_violation is False
    assert len(validator.violations) == 0
    print("✓ test_initialization_valid_gamma passed")


def test_initialization_invalid_gamma_zero():
    """Test validator raises on gamma=0"""
    try:
        TheoremValidator(gamma=0.0, b=1e-4)
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        assert "gamma must be in (0,1)" in str(e)
        print("✓ test_initialization_invalid_gamma_zero passed")


def test_initialization_invalid_gamma_one():
    """Test validator raises on gamma=1"""
    try:
        TheoremValidator(gamma=1.0, b=1e-4)
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        assert "gamma must be in (0,1)" in str(e)
        print("✓ test_initialization_invalid_gamma_one passed")


def test_validate_contraction_valid():
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
    print("✓ test_validate_contraction_valid passed")


def test_validate_contraction_invalid():
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
    assert validator.violations[0][0] == 1  # step number
    print("✓ test_validate_contraction_invalid passed")


def test_validate_contraction_halt_on_violation():
    """Test that halt_on_violation=True raises exception"""
    validator = TheoremValidator(gamma=0.8, b=1e-4, enable_halt_on_violation=True)
    
    try:
        validator.validate_contraction(
            debt_before=1.0,
            debt_after=1.0,
            step_num=1
        )
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert 'Contraction violated' in str(e)
        print("✓ test_validate_contraction_halt_on_violation passed")


def test_multiple_violations_tracking():
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
    print("✓ test_multiple_violations_tracking passed")


def test_get_violation_report_no_violations():
    """Test violation report with no violations"""
    validator = TheoremValidator(gamma=0.8, b=1e-4)
    
    validator.validate_contraction(debt_before=1.0, debt_after=0.5, step_num=1)
    
    report = validator.get_violation_report()
    
    assert report['num_violations'] == 0
    assert report['violations'] == []
    assert 'No contractions violated' in report['status']
    print("✓ test_get_violation_report_no_violations passed")


def test_get_violation_report_with_violations():
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
    print("✓ test_get_violation_report_with_violations passed")


def test_reset_violations():
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
    print("✓ test_reset_violations passed")


def test_boundary_case_exactly_at_threshold():
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
    assert abs(margin) < 1e-10  # margin should be ~0
    assert len(validator.violations) == 0
    print("✓ test_boundary_case_exactly_at_threshold passed")


def test_contracting_debt_sequence():
    """Test with a contracting debt sequence"""
    validator = TheoremValidator(gamma=0.8, b=1e-4)
    
    # Simulate a solver run with contracting debt starting from a reasonable initial debt
    # Each step: debt_after <= 0.8 * debt_before + 1e-4
    debts = [0.8, 0.64, 0.512, 0.4096, 0.32768]  # Each value is ~0.8x previous
    
    prev_debt = 1.0  # Start with initial debt of 1.0
    for i, debt in enumerate(debts, 1):
        is_valid, margin, _ = validator.validate_contraction(
            debt_before=prev_debt,
            debt_after=debt,
            step_num=i
        )
        assert is_valid is True, f"Step {i}: debt {debt} should satisfy 0.8*{prev_debt} + 1e-4 = {0.8*prev_debt + 1e-4}"
        prev_debt = debt
    
    report = validator.get_violation_report()
    assert report['num_violations'] == 0
    print("✓ test_contracting_debt_sequence passed")


def test_non_contracting_debt_sequence():
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
    print("✓ test_non_contracting_debt_sequence passed")


def run_all_tests():
    """Run all tests"""
    tests = [
        test_initialization_valid_gamma,
        test_initialization_invalid_gamma_zero,
        test_initialization_invalid_gamma_one,
        test_validate_contraction_valid,
        test_validate_contraction_invalid,
        test_validate_contraction_halt_on_violation,
        test_multiple_violations_tracking,
        test_get_violation_report_no_violations,
        test_get_violation_report_with_violations,
        test_reset_violations,
        test_boundary_case_exactly_at_threshold,
        test_contracting_debt_sequence,
        test_non_contracting_debt_sequence,
    ]
    
    print("\n" + "="*70)
    print("Running TheoremValidator Tests")
    print("="*70 + "\n")
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} error: {e}")
            failed += 1
    
    print("\n" + "="*70)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*70 + "\n")
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
