#!/usr/bin/env python3
"""
Diagnostic script for E1 convergence test in Analysis Mode.
Runs for N=8,12,16 with dt~dx, same final T, collects eps_H(T) and checks convergence (monotonic decrease).
"""

import numpy as np
import sys
import os
import json
import logging
sys.path.append(os.path.abspath('.'))

from src.core.gr_solver import GRSolver

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_e1_analysis(N, T_final=1.0):
    """Run E1 test in analysis mode for given N, to final T."""
    L = 4.0
    dx = L / N
    dt = dx  # dt ~ dx

    logger.info(f"Running E1 analysis for N={N}, dx={dx:.3f}, dt={dt:.3f}, T_final={T_final}")

    # Create solver with analysis_mode=True
    solver = GRSolver(N, N, N, dx, dx, dx, analysis_mode=True, log_level=logging.WARNING)

    # Initialize Minkowski
    solver.init_minkowski()

    # Compute number of steps
    num_steps = int(T_final / dt) + 1 if T_final / dt % 1 > 0 else int(T_final / dt)
    actual_T = num_steps * dt

    logger.info(f"Will run {num_steps} steps to T={actual_T:.3f}")

    eps_H_history = []
    t_history = []

    for step in range(num_steps):
        eps_H_before = float(solver.constraints.eps_H)
        eps_H_history.append(eps_H_before)
        t_history.append(float(solver.orchestrator.t))

        # Run one step
        dt_actual, _, _ = solver.orchestrator.run_step(dt_max=dt)

        # Check if reached T
        if solver.orchestrator.t >= actual_T - 1e-10:
            break

    # Final eps_H
    eps_H_final = float(solver.constraints.eps_H)
    eps_H_history.append(eps_H_final)
    t_history.append(float(solver.orchestrator.t))

    result = {
        'N': N,
        'dx': dx,
        'dt': dt,
        'T_final_target': T_final,
        'T_actual': float(solver.orchestrator.t),
        'num_steps': len(eps_H_history) - 1,
        'eps_H_history': eps_H_history,
        't_history': t_history,
        'eps_H_final': eps_H_final
    }

    logger.info(f"N={N}: eps_H_final = {eps_H_final:.6e}")

    return result

def main():
    T_final = 1.0  # Same final T for all
    results = []

    for N in [8, 12, 16]:
        result = run_e1_analysis(N, T_final)
        results.append(result)

    # Check convergence: eps_H should decrease in magnitude as N increases
    eps_H_finals = [r['eps_H_final'] for r in results]
    magnitudes = [abs(eps) for eps in eps_H_finals]

    monotone = all(magnitudes[i] >= magnitudes[i+1] for i in range(len(magnitudes)-1))

    convergence_result = {
        'monotone': monotone,
        'eps_H_finals': eps_H_finals,
        'magnitudes': magnitudes
    }

    output = {
        'convergence_check': convergence_result,
        'individual_results': results
    }

    with open('diagnostic_e1_analysis.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("E1 Analysis Mode Results:")
    print(f"eps_H finals: {eps_H_finals}")
    print(f"Magnitudes: {magnitudes}")
    print(f"Monotone convergence: {monotone}")

if __name__ == "__main__":
    main()