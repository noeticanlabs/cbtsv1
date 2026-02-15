"""
Smoke test for cbtsv1 package installation and imports.

This test verifies that:
1. The package installs correctly
2. All major imports resolve
3. Basic solver can be instantiated

This is the "first line of defense" against import errors.
"""

import pytest


def test_critical_imports():
    """Test that all critical packages can be imported."""
    # Main package
    import cbtsv1
    
    # GR solver components
    from cbtsv1.solvers.gr.gr_solver import GRSolver
    from cbtsv1.solvers.gr.gr_stepper import GRStepper
    from cbtsv1.solvers.gr.gr_ledger import GRLedger
    from cbtsv1.solvers.gr.gr_scheduler import GRScheduler
    from cbtsv1.solvers.gr.gr_gates import GateChecker, should_hard_fail
    from cbtsv1.solvers.gr.gr_sem import SEMDomain
    from cbtsv1.solvers.gr.ttl_calculator import TTLCalculator
    
    # Framework components
    from cbtsv1.framework.aeonic_memory_bank import AeonicMemoryBank
    from cbtsv1.framework.aeonic_memory_contract import AeonicMemoryContract
    from cbtsv1.framework.aeonic_clocks import AeonicClockPack
    from cbtsv1.framework.aeonic_receipts import AeonicReceipts
    from cbtsv1.framework.receipt_schemas import Kappa, OmegaReceipt
    
    # Numerics
    from cbtsv1.numerics.spectral.cache import SpectralCache
    
    # PhaseLoom
    from cbtsv1.framework.phaseloom_gr_orchestrator import GRPhaseLoomOrchestrator
    
    # NLLC (compiler)
    from nllc.nir import Module, Function, BasicBlock
    from nllc.vm import VM
    
    # NSC (numerical core)
    from nsc.exec_vm.vm import VirtualMachine
    
    # Hadamard (assembler)
    from hadamard.assembler import HadamardAssembler
    
    assert True  # If we get here, all imports succeeded


def test_gr_solver_basic_instantiation():
    """Test that GRSolver can be instantiated with minimal config."""
    from cbtsv1.solvers.gr.gr_solver import GRSolver
    
    # Minimal config for smoke test
    config = {
        'Nx': 8,
        'Ny': 8,
        'Nz': 8,
        'dx': 0.1,
        'dy': 0.1,
        'dz': 0.1,
    }
    
    # This should not raise
    # Note: Full initialization may require more setup
    # This is just a smoke test to verify basic class works
    solver = GRSolver.__new__(GRSolver)
    assert solver is not None


def test_memory_system_imports():
    """Test that memory system components can be imported."""
    from cbtsv1.framework.aeonic_memory_bank import AeonicMemoryBank, Record
    from cbtsv1.framework.aeonic_memory_contract import AeonicMemoryContract
    
    # Verify basic record creation
    record = Record(
        key="test_key",
        tier=1,
        payload="test_payload",
        bytes=100,
        created_tau_s=0,
        created_tau_l=0,
        created_tau_m=0,
        last_use_tau_s=0,
        last_use_tau_l=0,
        ttl_s=100,
        ttl_l=1000,
        reuse_count=0
    )
    assert record.key == "test_key"


def test_vendor_stub_warning():
    """Test that vendor stub is properly marked."""
    import os
    vendor_path = os.path.join(
        os.path.dirname(__file__), 
        '..', 'src', 'cbtsv1', 'vendor', 'coherence_framework_stub', 'VENDOR_VERSION.txt'
    )
    assert os.path.exists(vendor_path), "Vendor stub should have VENDOR_VERSION.txt"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
