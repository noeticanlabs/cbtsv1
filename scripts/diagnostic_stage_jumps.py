#!/usr/bin/env python3
"""
Diagnostic script for stage jump histogram.
Runs N=16 simulation for 100 steps, collects per-step stage jump data (Delta_epsilon_H per stage),
computes histogram statistics: mean, median, worst 1%, fraction of steps with positive jumps per stage.
"""

import numpy as np
import sys
import os
import json
import logging
sys.path.append(os.path.abspath('.'))

from src.core.gr_solver import GRSolver
import inspect_receipts

# Set up logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise
logger = logging.getLogger(__name__)

def run_stage_jump_histogram(N=16, num_steps=50):
    """Run N=16 for 100 steps, collect stage eps_H jumps."""
    # Clean up previous receipts
    if os.path.exists("aeonic_receipts.jsonl"):
        os.remove("aeonic_receipts.jsonl")

    L = 4.0
    dx = L / N
    dt = dx / 4 # dt ~ dx / 4 for stability

    logger.info(f"Running stage jump histogram for N={N}, dx={dx:.3f}, dt={dt:.3f}, steps={num_steps}")

    # Create solver
    solver = GRSolver(N, N, N, dx, dx, dx, analysis_mode=True, log_level=logging.WARNING)

    # Initialize Minkowski
    solver.init_minkowski()

    # Relax thresholds for this diagnostic run to accommodate initial perturbations (k=10 => R ~ 1e-4)
    if hasattr(solver.orchestrator, 'rails'):
        solver.orchestrator.rails.H_max = 1.0
        solver.orchestrator.rails.M_max = 1.0
        solver.orchestrator.rails.H_warn = 0.8
        solver.orchestrator.rails.H_max = 10.0
        solver.orchestrator.rails.M_max = 10.0
        solver.orchestrator.rails.H_warn = 8.0

    # Relax thread thresholds to prevent dt crash (threads drive dt proposal)
    if hasattr(solver.orchestrator, 'threads'):
        solver.orchestrator.threads.eps_H_max = 10.0
        solver.orchestrator.threads.eps_M_max = 10.0

    # Also relax SEM hard limits which are checked in VerifyPhase
    if hasattr(solver.orchestrator, 'sem_domain'):
        solver.orchestrator.sem_domain.kappa_validator.eps_H_hard = 1.0
        solver.orchestrator.sem_domain.kappa_validator.eps_M_hard = 1.0
        solver.orchestrator.sem_domain.kappa_validator.eps_H_hard = 10.0
        solver.orchestrator.sem_domain.kappa_validator.eps_M_hard = 10.0

    # Also relax the SEM validation thresholds
    if hasattr(solver.orchestrator.sem_domain, 'kappa_validator'):
        solver.orchestrator.sem_domain.kappa_validator.eps_H_hard = 1.0
        solver.orchestrator.sem_domain.kappa_validator.eps_M_hard = 1.0
        solver.orchestrator.sem_domain.kappa_validator.eps_H_soft = 0.01
        solver.orchestrator.sem_domain.kappa_validator.eps_M_soft = 0.01

    # Relax orchestrator step thresholds
    if hasattr(solver.orchestrator, 'eps_H_target'):
        solver.orchestrator.eps_H_target = 0.01

    # To collect jumps
    all_deltas = {}  # stage -> list of Delta_epsilon_H
    all_values = {}  # stage -> list of raw eps_H values
    positive_counts = {}  # stage -> count of positive jumps
    total_steps_per_stage = {}  # stage -> total steps (should be num_steps for each)

    for step in range(num_steps):
        eps_H_before_step = float(solver.constraints.eps_H)

        # Run one step
        dt_actual, _, _ = solver.orchestrator.run_step(dt_max=dt)

        # Get receipts for this step
        receipts = solver.stepper.receipts
        if receipts:
            last_receipt = receipts[-1]
            stage_eps_H = last_receipt.get('stage_eps_H', {})

            # Debug: print first few receipts
            if step < 3:
                print(f"Step {step}: receipt event = {last_receipt.get('event', 'N/A')}")
                print(f"  stage_eps_H = {stage_eps_H}")

            # Also track the raw values at each stage
            for key in ['eps_H_pre', 'eps_H_post_phys', 'eps_H_post_gauge', 'eps_H_post_cons', 'eps_H_post_filter']:
                if key in stage_eps_H:
                    stage_name = key.replace('eps_H_', '')
                    if stage_name not in all_values:
                        all_values[stage_name] = []
                    all_values[stage_name].append(stage_eps_H[key])

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
        else:
            if step < 3:
                print(f"Step {step}: NO RECEIPTS FOUND (step rejected?)")
                print(f"  eps_H_before = {eps_H_before_step}")
                print(f"  eps_H_after = {float(solver.constraints.eps_H)}")
                print(f"  orchestrator step accepted = {solver.orchestrator.step}")

        if step % 20 == 0:
            logger.info(f"Step {step}/{num_steps}, t={solver.orchestrator.t:.3f}")

    # Compute statistics
    stats = {}
    for stage in sorted(all_deltas.keys()):
        deltas = np.array(all_deltas[stage])
        mean = float(np.mean(deltas))
        median = float(np.median(deltas))
        worst_1_percent = float(np.percentile(deltas, 99))  # largest jumps
        fraction_positive = positive_counts[stage] / total_steps_per_stage[stage] if total_steps_per_stage[stage] > 0 else 0

        stats[stage] = {
            'mean': mean,
            'median': median,
            'worst_1_percent': worst_1_percent,
            'fraction_positive': fraction_positive,
            'num_samples': len(deltas)
        }

    result = {
        'N': N,
        'num_steps': num_steps,
        'dx': dx,
        'dt': dt,
        'final_t': float(solver.orchestrator.t),
        'stage_statistics': stats,
        'all_deltas': all_deltas,  # full data for histogram if needed
        'all_values': all_values  # raw eps_H values at each stage
    }

    with open('diagnostic_stage_jumps.json', 'w') as f:
        json.dump(result, f, indent=2)

    # Print summary
    print("Stage Jump Statistics:")
    for stage, stat in stats.items():
        print(f"Stage {stage}: mean={stat['mean']:.2e}, median={stat['median']:.2e}, worst_1%={stat['worst_1_percent']:.2e}, frac_pos={stat['fraction_positive']:.3f}")

    return result

if __name__ == "__main__":
    run_stage_jump_histogram()
    print("\nRunning Receipt Inspection:")
    inspect_receipts.main()