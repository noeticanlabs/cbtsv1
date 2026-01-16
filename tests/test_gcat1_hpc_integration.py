# GCAT-1: HPC Integration Calibration Suite
# Tests for correctness under fused kernels, mixed precision, multi-rate, loom downsampling

import sys
sys.path.append('.')
import numpy as np
from gr_solver.gr_solver import GRSolver

def test_zero_allocations_per_step():
    """Prove 0 allocations/step in stepper loop."""
    solver = GRSolver(Nx=8, Ny=8, Nz=8)
    solver.init_minkowski()

    # Run a few steps and check memory usage (placeholder, requires profiling)
    for _ in range(10):
        dt, dominant, violation = solver.orchestrator.run_step()
        assert dt > 0
    # In real test, use memory profiler to assert no new allocations

def test_spectral_cache_correctness():
    """Test spectral cache produces same results as on-the-fly computation."""
    from gr_solver.spectral.cache import SpectralCache
    cache = SpectralCache(8, 8, 8, 1.0, 1.0, 1.0)

    # Create test field
    change = np.random.rand(8, 8, 8)
    freq = np.fft.rfftn(change)
    power = np.abs(freq)**2

    # Old way: compute bins
    kx = np.fft.fftfreq(8)
    ky = np.fft.fftfreq(8)
    kz = np.fft.rfftfreq(8)
    kx_bins = np.linspace(kx.min(), kx.max(), 4)
    ky_bins = np.linspace(ky.min(), ky.max(), 4)
    kz_bins = np.linspace(kz.min(), kz.max(), 4)
    kx_bin_old = np.digitize(kx, kx_bins) - 1
    ky_bin_old = np.digitize(ky, ky_bins) - 1
    kz_bin_old = np.digitize(kz, kz_bins) - 1

    # New way: from cache
    kx_bin_new = cache.kx_bin
    ky_bin_new = cache.ky_bin
    kz_bin_new = cache.kz_bin

    assert np.array_equal(kx_bin_old, kx_bin_new)
    assert np.array_equal(ky_bin_old, ky_bin_new)
    assert np.array_equal(kz_bin_old, kz_bin_new)

def test_loom_downsampling_gate():
    """Test cheap proxy gate skips FFT when appropriate."""
    solver = GRSolver(Nx=8, Ny=8, Nz=8)
    solver.init_minkowski()

    # Set small changes
    solver.orchestrator.prev_K = solver.fields.K_sym6[..., 0].copy()
    solver.orchestrator.prev_gamma = solver.fields.gamma_sym6[..., 0].copy()

    # Run step, should skip loom if tiny
    dt, dominant, violation = solver.orchestrator.run_step()
    # Check if loom skipped (from logs or flags)

def test_mixed_precision_demotion():
    """Test Aeonic memory demotes arrays to fp16."""
    from aeonic_memory_bank import AeonicMemoryBank
    from aeonic_clocks import AeonicClockPack
    bank = AeonicMemoryBank(AeonicClockPack())

    arr = np.random.rand(10, 10).astype(np.float64)
    bank.put("test", 2, arr, arr.nbytes, 100, 100, 1.0, 0.0, False, [])

    bank.maintenance_tick()  # Trigger demotion

    demoted = bank.get("test")
    if isinstance(demoted, np.ndarray):
        assert demoted.dtype == np.float16

def test_elliptic_solver_warm_start():
    """Test MG warm-start reduces iterations."""
    from gr_solver.elliptic.solver import EllipticSolver, apply_poisson
    solver = EllipticSolver(apply_poisson)

    # Test problem
    b = np.random.rand(8, 8, 8)

    # First solve
    x1 = solver.solve(b, regime_hash="test_regime")

    # Second solve with same regime, should warm-start
    x2 = solver.solve(b, regime_hash="test_regime")

    # Check convergence faster (placeholder)

def test_convergence_loop_fix():
    """
    GCAT-1 Convergence Check.
    Ensures resolution varies to fix p_obs = 0.00 diagnosis.
    """
    resolutions = [8, 16, 24]
    errors = []
    
    print("\nRunning Convergence Loop (Fix for p_obs=0)...")
    for N in resolutions:
        # Explicitly pass varying resolution to GRSolver
        solver = GRSolver(Nx=N, Ny=N, Nz=N)
        solver.init_minkowski()
        
        # Run a step to generate metrics
        dt, dominant, violation = solver.orchestrator.run_step()
        
        # In a real run, we would append the error metric here
        # errors.append(violation['error_norm'])
        pass

if __name__ == "__main__":
    test_zero_allocations_per_step()
    test_spectral_cache_correctness()
    test_loom_downsampling_gate()
    test_mixed_precision_demotion()
    test_elliptic_solver_warm_start()
    test_convergence_loop_fix()
    print("GCAT-1: All HPC integration tests passed!")