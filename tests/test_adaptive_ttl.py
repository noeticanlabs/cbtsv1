"""
Tests for adaptive TTL behavior in the AEONIC memory system.
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.gr_ttl_calculator import TTLCalculator, AdaptiveTTLs, compute_adaptive_ttls
from aeonic_clocks import AeonicClockPack
from aeonic_memory_bank import AeonicMemoryBank
from aeonic_memory_contract import AeonicMemoryContract
from aeonic_receipts import AeonicReceipts


class TestTTLCalculator:
    """Test the TTLCalculator class."""
    
    def test_basic_calculation(self):
        """Test basic TTL calculation with standard parameters."""
        calculator = TTLCalculator(
            t_end=1000.0,
            dt_avg=0.01,
            N=64,
            problem_type='standard'
        )
        
        ttls = calculator.compute_ttls()
        
        # Check that all TTLs are positive integers
        assert isinstance(ttls.msolve_ttl_s, int)
        assert isinstance(ttls.msolve_ttl_l, int)
        assert isinstance(ttls.mstep_ttl_s, int)
        assert isinstance(ttls.mstep_ttl_l, int)
        assert isinstance(ttls.morch_ttl_s, int)
        assert isinstance(ttls.morch_ttl_l, int)
        
        # Check that all TTLs are within reasonable bounds
        assert ttls.msolve_ttl_s > 0
        assert ttls.msolve_ttl_l > 0
        assert ttls.mstep_ttl_s > 0
        assert ttls.mstep_ttl_l > 0
        assert ttls.morch_ttl_s > 0
        assert ttls.morch_ttl_l > 0
        
        # Long TTL should be >= short TTL for each tier
        assert ttls.msolve_ttl_l >= ttls.msolve_ttl_s
        assert ttls.mstep_ttl_l >= ttls.mstep_ttl_s
        assert ttls.morch_ttl_l >= ttls.morch_ttl_s
    
    def test_short_simulation(self):
        """Test TTL calculation for short simulation (should use minimum TTLs)."""
        calculator = TTLCalculator(
            t_end=1.0,  # Very short simulation
            dt_avg=0.001,
            N=64
        )
        
        ttls = calculator.compute_ttls()
        
        # Should use minimum TTLs for short simulation
        assert ttls.msolve_ttl_s >= TTLCalculator.MIN_MSOLVE_S
        assert ttls.mstep_ttl_s >= TTLCalculator.MIN_MSTEP_S
        assert ttls.morch_ttl_s >= TTLCalculator.MIN_MORCH_S
    
    def test_long_simulation(self):
        """Test TTL calculation for long simulation."""
        calculator = TTLCalculator(
            t_end=10000.0,  # Long simulation
            dt_avg=0.01,
            N=64
        )
        
        ttls = calculator.compute_ttls()
        
        # Should have larger TTLs for longer simulation
        # But should not exceed maximums
        assert ttls.msolve_ttl_s <= TTLCalculator.MAX_MSOLVE_S
        assert ttls.mstep_ttl_s <= TTLCalculator.MAX_MSTEP_S
        assert ttls.morch_ttl_s <= TTLCalculator.MAX_MORCH_S
    
    def test_grid_size_scaling(self):
        """Test that TTL scales with grid size."""
        calc_small = TTLCalculator(t_end=1000.0, dt_avg=0.01, N=32)
        calc_large = TTLCalculator(t_end=1000.0, dt_avg=0.01, N=128)
        
        ttls_small = calc_small.compute_ttls()
        ttls_large = calc_large.compute_ttls()
        
        # Larger grid should have larger TTLs (higher recompute cost)
        assert ttls_large.msolve_ttl_s >= ttls_small.msolve_ttl_s
        assert ttls_large.mstep_ttl_s >= ttls_small.mstep_ttl_s
        assert ttls_large.morch_ttl_s >= ttls_small.morch_ttl_s
    
    def test_problem_type_multipliers(self):
        """Test that problem type affects TTL multipliers."""
        calc_standard = TTLCalculator(t_end=100.0, dt_avg=0.01, N=64, problem_type='standard')
        calc_long_run = TTLCalculator(t_end=100.0, dt_avg=0.01, N=64, problem_type='long_run')
        calc_transient = TTLCalculator(t_end=100.0, dt_avg=0.01, N=64, problem_type='transient')
        
        ttls_standard = calc_standard.compute_ttls()
        ttls_long_run = calc_long_run.compute_ttls()
        ttls_transient = calc_transient.compute_ttls()
        
        # long_run should have longer TTLs
        assert ttls_long_run.morch_ttl_l >= ttls_standard.morch_ttl_l
        
        # transient should have shorter or equal TTLs
        assert ttls_transient.morch_ttl_l <= ttls_standard.morch_ttl_l
    
    def test_static_fallback(self):
        """Test static fallback TTL values."""
        calculator = TTLCalculator(t_end=1000.0, dt_avg=0.01, N=64)
        
        fallback = calculator.get_static_fallback()
        
        # Should match static values from contract
        assert fallback.msolve_ttl_s == TTLCalculator.MIN_MSOLVE_S  # 1 hour
        assert fallback.msolve_ttl_l == TTLCalculator.MIN_MSOLVE_L  # 1 day
        assert fallback.mstep_ttl_s == TTLCalculator.MIN_MSTEP_S    # 10 hours
        assert fallback.mstep_ttl_l == TTLCalculator.MIN_MSTEP_L    # 1 week
        assert fallback.morch_ttl_s == TTLCalculator.MIN_MORCH_S    # 30 days
        assert fallback.morch_ttl_l == TTLCalculator.MIN_MORCH_L    # 30 days
    
    def test_factory_method(self):
        """Test the create_from_simulation factory method."""
        calc = TTLCalculator.create_from_simulation(
            t_end=500.0,
            dt_avg=0.005,
            N=128,
            problem_type='critical'
        )
        
        assert calc.t_end == 500.0
        assert calc.dt_avg == 0.005
        assert calc.N == 128
        assert calc.problem_type == 'critical'
    
    def test_convenience_function(self):
        """Test the compute_adaptive_ttls convenience function."""
        ttls = compute_adaptive_ttls(
            t_end=1000.0,
            dt_avg=0.01,
            N=64,
            problem_type='standard'
        )
        
        assert isinstance(ttls, AdaptiveTTLs)
        assert ttls.msolve_ttl_s > 0


class TestAeonicMemoryContractWithTTL:
    """Test AeonicMemoryContract with adaptive TTL."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.clocks = AeonicClockPack()
        self.receipts = AeonicReceipts()
        self.memory_bank = AeonicMemoryBank(self.clocks, self.receipts)
    
    def test_no_calculator_uses_static_ttl(self):
        """Test that static TTL is used when no calculator is provided."""
        contract = AeonicMemoryContract(
            self.memory_bank,
            receipts_log=self.receipts,
            ttl_calculator=None
        )
        
        # Get TTLs should return static values
        ttls = contract._get_ttls()
        
        assert ttls.msolve_ttl_s == AeonicMemoryContract.STATIC_TTL_M_SOLVE_S
        assert ttls.msolve_ttl_l == AeonicMemoryContract.STATIC_TTL_M_SOLVE_L
        assert ttls.mstep_ttl_s == AeonicMemoryContract.STATIC_TTL_M_STEP_S
        assert ttls.mstep_ttl_l == AeonicMemoryContract.STATIC_TTL_M_STEP_L
        assert ttls.morch_ttl_s == AeonicMemoryContract.STATIC_TTL_M_ORCH_S
        assert ttls.morch_ttl_l == AeonicMemoryContract.STATIC_TTL_M_ORCH_L
    
    def test_with_calculator_uses_adaptive_ttl(self):
        """Test that adaptive TTL is used when calculator is provided."""
        ttl_calculator = TTLCalculator(
            t_end=5000.0,  # Longer simulation to get larger adaptive TTL
            dt_avg=0.01,
            N=64,
            problem_type='standard'
        )
        
        contract = AeonicMemoryContract(
            self.memory_bank,
            receipts_log=self.receipts,
            ttl_calculator=ttl_calculator
        )
        
        # Get TTLs should return adaptive values (different from static)
        adaptive_ttls = contract._get_ttls()
        
        # Static fallback values from TTL calculator
        static_ttls = ttl_calculator.get_static_fallback()
        
        # For a 5000 simulation, msolve TTL should be larger than static
        # adaptive = 5000 * 0.05 / 0.01 = 25000, clamped to MIN = 36000
        # For a longer simulation, the adaptive values will differ
        # Check that msolve_ttl_l is different (adaptive = 5000 * 0.05 * 4 / 0.01 = 100000)
        assert adaptive_ttls.msolve_ttl_l != static_ttls.msolve_ttl_l
    
    def test_set_ttl_calculator(self):
        """Test setting a new TTL calculator."""
        contract = AeonicMemoryContract(
            self.memory_bank,
            receipts_log=self.receipts,
            ttl_calculator=None
        )
        
        # Initially should use static
        initial_ttls = contract._get_ttls()
        assert initial_ttls.msolve_ttl_s == AeonicMemoryContract.STATIC_TTL_M_SOLVE_S
        
        # Set new calculator
        new_calculator = TTLCalculator(t_end=5000.0, dt_avg=0.01, N=64)
        contract.set_ttl_calculator(new_calculator)
        
        # Should now use adaptive TTLs
        new_ttls = contract._get_ttls()
        assert new_ttls.mstep_ttl_s > AeonicMemoryContract.STATIC_TTL_M_STEP_S


class TestBackwardCompatibility:
    """Test backward compatibility of the changes."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.clocks = AeonicClockPack()
        self.receipts = AeonicReceipts()
        self.memory_bank = AeonicMemoryBank(self.clocks, self.receipts)
    
    def test_old_constructor_signature(self):
        """Test that old constructor signature still works."""
        # Old code that doesn't pass ttl_calculator
        contract = AeonicMemoryContract(
            self.memory_bank,
            receipts_log=self.receipts
        )
        
        # Should work and use static TTLs
        assert contract.ttl_calculator is None
        ttls = contract._get_ttls()
        assert ttls.msolve_ttl_s == AeonicMemoryContract.STATIC_TTL_M_SOLVE_S
    
    def test_memory_bank_put_signature_unchanged(self):
        """Test that MemoryBank.put signature is unchanged."""
        # The put method should still accept ttl_s and ttl_l as before
        self.memory_bank.put(
            key="test_key",
            tier=1,
            payload={"data": "test"},
            bytes=100,
            ttl_s=3600,
            ttl_l=86400,
            recompute_cost_est=100.0,
            risk_score=0.1,
            tainted=False,
            regime_hashes=[],
            demoted=False
        )
        
        # Should have stored the record - access via tiers dict
        record = self.memory_bank.tiers[1]["test_key"]
        assert record is not None
        assert record.ttl_s == 3600
        assert record.ttl_l == 86400
    
    def test_static_ttl_constants_defined(self):
        """Test that static TTL constants are accessible."""
        assert hasattr(AeonicMemoryContract, 'STATIC_TTL_M_SOLVE_S')
        assert hasattr(AeonicMemoryContract, 'STATIC_TTL_M_SOLVE_L')
        assert hasattr(AeonicMemoryContract, 'STATIC_TTL_M_STEP_S')
        assert hasattr(AeonicMemoryContract, 'STATIC_TTL_M_STEP_L')
        assert hasattr(AeonicMemoryContract, 'STATIC_TTL_M_ORCH_S')
        assert hasattr(AeonicMemoryContract, 'STATIC_TTL_M_ORCH_L')
        
        # Verify they match the original hardcoded values
        assert AeonicMemoryContract.STATIC_TTL_M_SOLVE_S == 3600   # 1 hour
        assert AeonicMemoryContract.STATIC_TTL_M_SOLVE_L == 86400  # 1 day
        assert AeonicMemoryContract.STATIC_TTL_M_STEP_S == 36000  # 10 hours
        assert AeonicMemoryContract.STATIC_TTL_M_STEP_L == 604800 # 1 week
        assert AeonicMemoryContract.STATIC_TTL_M_ORCH_S == 2592000  # 30 days
        assert AeonicMemoryContract.STATIC_TTL_M_ORCH_L == 31536000 # 1 year


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
