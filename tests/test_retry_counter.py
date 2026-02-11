"""Tests for RetryCounter and bounded retry enforcement per Theorem Lemma 2."""

import unittest
import logging
from src.core.gr_stepper import RetryCounter
from src.core.gate_policy import GatePolicy


class TestRetryCounter(unittest.TestCase):
    """Test RetryCounter class enforces Theorem Lemma 2 bounds."""
    
    def test_retry_counter_initialization(self):
        """Test RetryCounter initializes with correct bounds."""
        counter = RetryCounter(max_retries=3)
        self.assertEqual(counter.max_retries, 3)
        self.assertEqual(counter.max_attempts, 4)  # 1 + max_retries
        self.assertEqual(counter.attempt, 0)
    
    def test_retry_counter_increment(self):
        """Test RetryCounter increments correctly."""
        counter = RetryCounter(max_retries=3)
        for i in range(1, 5):
            counter.increment()
            self.assertEqual(counter.attempt, i)
    
    def test_retry_counter_can_retry(self):
        """Test can_retry() returns correct values."""
        counter = RetryCounter(max_retries=3)
        
        # Initial state (attempt 0)
        self.assertTrue(counter.can_retry())
        
        # Attempt 1-3 should allow retry
        counter.increment()
        self.assertTrue(counter.can_retry())
        
        counter.increment()
        self.assertTrue(counter.can_retry())
        
        counter.increment()
        self.assertTrue(counter.can_retry())
        
        # Attempt 4 (max_attempts) should not allow retry
        counter.increment()
        self.assertFalse(counter.can_retry())
    
    def test_retry_counter_hard_fail(self):
        """Test RetryCounter raises RuntimeError when limit exceeded."""
        counter = RetryCounter(max_retries=3)
        
        # Increment up to max_attempts (4)
        for _ in range(4):
            counter.increment()
        
        # Next increment should raise RuntimeError
        with self.assertRaises(RuntimeError) as context:
            counter.increment()
        
        self.assertIn("Retry limit exceeded", str(context.exception))
        self.assertIn("5 > 4", str(context.exception))
    
    def test_retry_counter_reset(self):
        """Test RetryCounter resets correctly."""
        counter = RetryCounter(max_retries=3)
        
        # Increment a few times
        counter.increment()
        counter.increment()
        self.assertEqual(counter.attempt, 2)
        
        # Reset
        counter.reset()
        self.assertEqual(counter.attempt, 0)
        self.assertTrue(counter.can_retry())
    
    def test_retry_counter_default_max_retries(self):
        """Test RetryCounter uses default max_retries if not specified."""
        counter = RetryCounter()
        self.assertEqual(counter.max_retries, 3)
        self.assertEqual(counter.max_attempts, 4)


class TestGatePolicyRetryPolicy(unittest.TestCase):
    """Test GatePolicy retry_policy configuration."""
    
    def test_gate_policy_has_retry_policy(self):
        """Test GatePolicy initializes with retry_policy."""
        policy = GatePolicy()
        self.assertIn('max_retries', policy.retry_policy)
        self.assertIn('initial_dt_reduction', policy.retry_policy)
        self.assertIn('max_attempts_hard_fail', policy.retry_policy)
    
    def test_gate_policy_retry_policy_defaults(self):
        """Test GatePolicy retry_policy has correct default values."""
        policy = GatePolicy()
        self.assertEqual(policy.retry_policy['max_retries'], 3)
        self.assertEqual(policy.retry_policy['initial_dt_reduction'], 0.8)
        self.assertEqual(policy.retry_policy['max_attempts_hard_fail'], 4)
    
    def test_gate_policy_to_dict_includes_retry_policy(self):
        """Test to_dict exports retry_policy."""
        policy = GatePolicy()
        policy_dict = policy.to_dict()
        self.assertIn('retry_policy', policy_dict)
        self.assertEqual(policy_dict['retry_policy']['max_retries'], 3)
    
    def test_gate_policy_validate_retry_policy(self):
        """Test validate() checks retry_policy bounds."""
        policy = GatePolicy()
        # Should not raise
        policy.validate()
    
    def test_gate_policy_validate_retry_policy_invalid_max_retries(self):
        """Test validate() rejects invalid max_retries."""
        policy = GatePolicy()
        policy.retry_policy['max_retries'] = 0
        
        with self.assertRaises(ValueError) as context:
            policy.validate()
        self.assertIn("max_retries must be > 0", str(context.exception))
    
    def test_gate_policy_validate_retry_policy_invalid_dt_reduction(self):
        """Test validate() rejects invalid initial_dt_reduction."""
        policy = GatePolicy()
        policy.retry_policy['initial_dt_reduction'] = 1.5  # Must be in (0,1)
        
        with self.assertRaises(ValueError) as context:
            policy.validate()
        self.assertIn("initial_dt_reduction must be in (0,1)", str(context.exception))
    
    def test_gate_policy_validate_retry_policy_max_attempts_constraint(self):
        """Test validate() enforces max_attempts = 1 + max_retries (Lemma 2)."""
        policy = GatePolicy()
        # Violate Lemma 2 constraint
        policy.retry_policy['max_attempts_hard_fail'] = 5  # Should be 4 for max_retries=3
        
        with self.assertRaises(ValueError) as context:
            policy.validate()
        self.assertIn("must equal 1 + max_retries", str(context.exception))
    
    def test_gate_policy_from_file_loads_retry_policy(self):
        """Test from_file loads retry_policy from config."""
        # Use default config file
        config_path = 'config/gate_policy_default.json'
        try:
            policy = GatePolicy.from_file(config_path)
            self.assertEqual(policy.retry_policy['max_retries'], 3)
            self.assertEqual(policy.retry_policy['initial_dt_reduction'], 0.8)
            self.assertEqual(policy.retry_policy['max_attempts_hard_fail'], 4)
        except FileNotFoundError:
            self.skipTest("Config file not found")


class TestRetryCounterBounds(unittest.TestCase):
    """Test RetryCounter enforces Theorem Lemma 2 bounds."""
    
    def test_attempts_bounded_by_lemma_2(self):
        """Verify attempts ≤ (1 + N_retry) per Lemma 2."""
        N_retry = 3
        counter = RetryCounter(max_retries=N_retry)
        
        # Maximum number of attempts should be exactly 1 + N_retry
        self.assertEqual(counter.max_attempts, 1 + N_retry)
        
        # Should be able to do exactly max_attempts increments
        for _ in range(counter.max_attempts):
            counter.increment()
        
        # Should fail on the next increment
        with self.assertRaises(RuntimeError):
            counter.increment()
    
    def test_retry_counter_with_various_max_retries(self):
        """Test RetryCounter works correctly with various max_retries values."""
        for N_retry in [1, 2, 3, 5, 10]:
            counter = RetryCounter(max_retries=N_retry)
            
            # Verify max_attempts = 1 + N_retry
            self.assertEqual(counter.max_attempts, 1 + N_retry)
            
            # Verify we can do exactly max_attempts increments
            for _ in range(counter.max_attempts):
                counter.increment()
            
            # Verify hard fail on next increment
            with self.assertRaises(RuntimeError):
                counter.increment()


if __name__ == '__main__':
    unittest.main()
