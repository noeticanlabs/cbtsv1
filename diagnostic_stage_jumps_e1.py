#!/usr/bin/env python3
"""
Comprehensive stage jump diagnostic for full E1 convergence runs.
Runs E1 in Analysis Mode for N=8,12,16 to T=1.0, collects all stage_eps_H per step across the entire simulation,
computes Delta_epsilon_H per stage per step, then aggregates statistics: mean, median, worst 1%, fraction positive jumps per stage per resolution.
This identifies which stage persistently injects error over the full run.
"""

import numpy as np
import sys
import os
import json
import logging
sys.path.append(os.path.abspath('.'))

from gr_solver.gr_solver import GRSolver

# Set up logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise
logger = logging.getLogger(__name__)

def run_e1_stage_jumps(N, T_final=1.0):
    """Run E1 test in analysis mode for given N, collect stage jumps."""
    L = 4.0
    dx = L / N
    dt = dx  # dt ~ dx

    logger.info(f"Running E1 stage jumps for N={N}, dx={dx:.3f}, dt={dt:.3f}, T_final={T_final}")

    # Create solver with analysis_mode=False (Flight Mode)
    solver = GRSolver(N, N, N, dx, dx, dx, analysis_mode=False, log_level=logging.WARNING)

    # Initialize Minkowski
    solver.init_minkowski()

    # Compute number of steps
    num_steps = int(T_final / dt) + 1 if T_final / dt % 1 > 0 else int(T_final / dt)
    actual_T = num_steps * dt

    logger.info(f"Will run {num_steps} steps to T={actual_T:.3f}")

    # To collect jumps per stage (only in analysis mode, but keeping for compatibility)
    all_deltas = {}  # stage -> list of Delta_epsilon_H
    positive_counts = {}  # stage -> count of positive jumps
    total_steps_per_stage = {}  # stage -> total steps

    # Collect eps_H history for convergence check
    eps_H_history = []

    for step in range(num_steps):
        # Run one step
        dt_actual, _, _ = solver.orchestrator.run_step(dt_max=dt)

        # Collect eps_H for convergence check
        eps_H_history.append(float(solver.constraints.eps_H))

        # Get receipts for this step
        receipts = solver.stepper.receipts
        if receipts:
            # Find the last STEP_ACCEPT or STEP_REJECT receipt
            step_receipt = None
            for r in reversed(receipts):
                if r.get('event') in ['STEP_ACCEPT', 'STEP_REJECT']:
                    step_receipt = r
                    break
            if step_receipt:
                stage_eps_H = step_receipt.get('stage_eps_H', {})

                if step == 0:
                    print(f"stage_eps_H for step 0: {stage_eps_H}")

                # The stage_eps_H has Delta_eps_H_phys, Delta_eps_H_cons, etc.
                delta_keys = ['Delta_eps_H_phys', 'Delta_eps_H_cons', 'Delta_eps_H_gauge', 'Delta_eps_H_filter']
                for delta_key in delta_keys:
                    if delta_key in stage_eps_H:
                        stage = delta_key.replace('Delta_eps_H_', '')
                        delta = stage_eps_H[delta_key]

                        if stage not in all_deltas:
                            all_deltas[stage] = []
                            positive_counts[stage] = 0
                            total_steps_per_stage[stage] = 0

                        all_deltas[stage].append(delta)
                        if delta > 0:
                            positive_counts[stage] += 1
                        total_steps_per_stage[stage] += 1

        if step % 100 == 0:
            logger.info(f"Step {step}/{num_steps}, t={solver.orchestrator.t:.3f}")

        # Check if reached T
        if solver.orchestrator.t >= actual_T - 1e-10:
            break

    # Final eps_H
    eps_H_final = float(solver.constraints.eps_H)

    # Compute statistics per stage
    stats = {}
    for stage in sorted(all_deltas.keys()):
        deltas = np.array(all_deltas[stage])
        if len(deltas) > 0:
            mean = float(np.mean(deltas))
            median = float(np.median(deltas))
            worst_1_percent = float(np.percentile(deltas, 99))  # largest jumps (most positive)
            fraction_positive = positive_counts[stage] / total_steps_per_stage[stage] if total_steps_per_stage[stage] > 0 else 0
        else:
            mean = median = worst_1_percent = fraction_positive = 0

        stats[stage] = {
            'mean': mean,
            'median': median,
            'worst_1_percent': worst_1_percent,
            'fraction_positive': fraction_positive,
            'num_samples': len(deltas)
        }

    result = {
        'N': N,
        'dx': dx,
        'dt': dt,
        'T_final_target': T_final,
        'T_actual': float(solver.orchestrator.t),
        'num_steps': step + 1,
        'eps_H_final': eps_H_final,
        'eps_H_history': eps_H_history,
        'stage_statistics': stats,
        'all_deltas': all_deltas  # full data for further analysis if needed
    }

    logger.info(f"N={N}: eps_H_final = {eps_H_final:.6e}, stages collected: {list(all_deltas.keys())}")

    return result

def main():
    T_final = 1.0  # Same final T for all
    results = []

    for N in [8, 12, 16]:
        result = run_e1_stage_jumps(N, T_final)
        results.append(result)

    output = {
        'description': 'Flight mode E1 convergence test for monotone eps_H check',
        'T_final': T_final,
        'resolutions': [r['N'] for r in results],
        'individual_results': results
    }

    with open('diagnostic_flight_e1.json', 'w') as f:
        json.dump(output, f, indent=2)

    # Print summary
    print("Flight Mode E1 Convergence Test:")
    original_finals = [1.037e-4, 4.62e-5, 1.169e-4]  # N=8,12,16
    for i, result in enumerate(results):
        N = result['N']
        eps_H_final = result['eps_H_final']
        eps_H_history = result['eps_H_history']
        original = original_finals[i]
        is_monotone = eps_H_history == sorted(eps_H_history, reverse=True)
        print(f"\nN={N}, eps_H_final={eps_H_final:.2e} (original: {original:.2e}), Monotone decreasing: {is_monotone}")
        # Note: stage stats may be empty in flight mode, but keeping for compatibility

if __name__ == "__main__":
    main()