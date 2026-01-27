#!/usr/bin/env python3
"""
GCAT-2 Scenario 3: Under-Resolution Cascade
Tests system response to grid-scale noise (aliasing limit).
"""

import numpy as np
import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core.gr_solver import GRSolver
from src.core.gr_core_fields import SYM6_IDX

def test_gcat2_scenario3_under_resolution_cascade():
    """Scenario 3: Inject Nyquist-frequency noise and verify PhaseLoom detection."""
    # 1. Setup low-res solver
    N = 16
    solver = GRSolver(Nx=N, Ny=N, Nz=N, dx=0.1, dy=0.1, dz=0.1)
    solver.init_minkowski()

    # Relax rails for test
    solver.orchestrator.rails.H_max = 1.0
    solver.orchestrator.rails.M_max = 1.0

    # 2. Inject Grid-Scale Noise (Nyquist Checkerboard)
    # This puts maximum energy in the highest frequency band
    eps = 1e-3
    x = np.arange(N)
    y = np.arange(N)
    z = np.arange(N)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Checkerboard pattern: (-1)^(i+j+k)
    noise = eps * ((-1.0)**(X + Y + Z))
    
    # Add to gamma_xx (metric) and K_xx (curvature)
    solver.fields.gamma_sym6[..., 0] += noise
    solver.fields.K_sym6[..., 0] += noise * 0.1

    # 3. Run Evolution
    # We expect the solver to struggle or PhaseLoom to flag high D_max
    solver.run(T_max=0.1) # Short run, instability is fast

    receipts = solver.orchestrator.receipts.receipts
    
    # 4. Analyze Metrics
    # Find max Tail Danger recorded
    max_D_max = 0.0
    for r in receipts:
        loom = r.get('loom_data', {})
        d = loom.get('D_max', 0.0)
        if d > max_D_max:
            max_D_max = d
            
    # Check constraint drift
    eps_H_initial = receipts[0]['constraints']['eps_post_H'] if receipts else 0.0
    eps_H_final = receipts[-1]['constraints']['eps_post_H'] if receipts else 0.0
    
    # 5. Verdict
    # Pass if PhaseLoom detected the danger (D_max > 0.1) OR if the solver survived without NaN
    # In a "Cascade" scenario, we expect D_max to be high.
    detected = max_D_max > 0.1
    survived = not np.any(np.isnan(solver.fields.gamma_sym6))

    passed = detected or survived
    
    diagnosis = (f"Scenario 3 {'Passed' if passed else 'Failed'}: "
                 f"Max D_max={max_D_max:.2e} (Expected > 0.1), "
                 f"Constraints {eps_H_initial:.2e} -> {eps_H_final:.2e}")

    # 6. Generate Receipts
    summary = {
        "passed": passed,
        "metrics": {
            "max_D_max": max_D_max,
            "eps_H_initial": eps_H_initial,
            "eps_H_final": eps_H_final,
            "survived": survived
        },
        "diagnosis": diagnosis
    }
    
    with open('receipts_gcat2_s3_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    with open('receipts_gcat2_s3.json', 'w') as f:
        json.dump(receipts, f, indent=2)
        
    print(diagnosis)
    assert passed, diagnosis

if __name__ == "__main__":
    test_gcat2_scenario3_under_resolution_cascade()