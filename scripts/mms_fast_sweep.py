"""
Fast MMS Parameter Sweep - Pre-warms JIT and runs quick tests.
"""
import numpy as np
import logging
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger('mms_fast_sweep')

# Suppress all solver logs
logging.getLogger('gr_solver').setLevel(logging.CRITICAL)

# Pre-warm JIT by doing a small test first
def warmup_jit():
    """Warm up Numba JIT compilation."""
    print("Warming up JIT compilation...")
    from tests.test_1_mms_lite import MmsLite
    test = MmsLite(L=4.0, core_mode=True)
    try:
        test.defect_test(N=4)
        print("JIT warmup complete.")
    except Exception as e:
        print(f"JIT warmup warning: {e}")


def quick_convergence_test(params):
    """Run quick convergence test with given parameters."""
    L, resolutions, T, cfl = params
    
    from tests.test_1_mms_lite import MmsLite
    from tests.gr_test_utils import estimate_order
    
    test = MmsLite(L=L, core_mode=True)
    
    errors = []
    hs = []
    
    for N in resolutions:
        try:
            result = test.evolution_test(N=N, T=T)
            errors.append(result['total_error'])
            hs.append(L / N)
        except Exception as e:
            print(f"  Error at N={N}: {e}")
            return None
    
    if len(errors) >= 2:
        p_obs = estimate_order(errors, hs)
        return p_obs, errors
    return None, errors


def grid_search():
    """Quick grid search for optimal parameters."""
    print("\n" + "="*60)
    print("MMS PARAMETER GRID SEARCH")
    print("="*60)
    
    # Pre-warm
    warmup_jit()
    
    # Parameter grid
    param_grid = []
    
    for L in [8.0]:
        for cfl in [0.1, 0.2, 0.3]:
            for T in [0.01, 0.02, 0.05]:
                param_grid.append((L, [8, 12, 16], T, cfl))
    
    print(f"\nTesting {len(param_grid)} parameter combinations...")
    print("-" * 60)
    
    results = []
    for i, (L, res, T, cfl) in enumerate(param_grid):
        print(f"[{i+1}/{len(param_grid)}] L={L}, CFL={cfl}, T={T}, res={res}", end=" ... ")
        
        p_obs, errors = quick_convergence_test((L, res, T, cfl))
        
        if p_obs is not None:
            status = f"p_obs={p_obs:.2f} ✅" if p_obs > 1.0 else f"p_obs={p_obs:.2f}"
            print(status)
            results.append((L, res, T, cfl, p_obs, errors))
        else:
            print("FAILED ❌")
    
    # Find best
    if results:
        best = max(results, key=lambda x: x[4])
        print("\n" + "="*60)
        print("BEST PARAMETERS:")
        print(f"  L = {best[0]}")
        print(f"  CFL = {best[3]}")
        print(f"  T = {best[2]}")
        print(f"  Resolutions = {best[1]}")
        print(f"  p_obs = {best[4]:.2f}")
        print("="*60)
        return best
    else:
        print("No successful parameter combinations found!")
        return None


def fine_tune_best(best):
    """Fine-tune around best parameters."""
    if best is None:
        return None
    
    L, base_res, base_T, base_cfl, _, _ = best
    
    print(f"\nFine-tuning around CFL={base_cfl}, T={base_T}...")
    
    refined = []
    
    # Fine-tune CFL
    for cfl in [base_cfl - 0.05, base_cfl, base_cfl + 0.05]:
        if cfl > 0.05:
            p_obs, _ = quick_convergence_test((L, base_res, base_T, cfl))
            if p_obs:
                refined.append((cfl, p_obs, "CFL"))
    
    # Fine-tune T
    for T in [base_T * 0.5, base_T, base_T * 2.0]:
        if T > 0.005:
            p_obs, _ = quick_convergence_test((L, base_res, T, base_cfl))
            if p_obs:
                refined.append((T, p_obs, "T"))
    
    if refined:
        best_refined = max(refined, key=lambda x: x[1])
        print(f"Best refined: {best_refined[2]}={best_refined[0]}, p_obs={best_refined[1]:.2f}")
        return best_refined
    
    return None


def main():
    # Quick test
    best = grid_search()
    
    if best:
        fine_tune_best(best)
    
    print("\nDone! Update test_1_mms_lite.py with the optimal parameters.")


if __name__ == "__main__":
    main()
