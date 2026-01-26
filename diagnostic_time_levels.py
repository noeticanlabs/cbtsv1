#!/usr/bin/env python3
"""
Diagnostic script for time-level sanity check.
Runs a single step, evaluates constraints at different time levels (t_n, t_{n+1}, intermediate if applicable),
and logs which time level is used for receipts.
"""

import numpy as np
import sys
import os
import json
import logging
sys.path.append(os.path.abspath('.'))

from gr_solver.gr_solver import GRSolver

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_time_level_check(N=16):
    """Run one step and check time levels for constraint evaluation."""
    L = 4.0
    dx = L / N
    dt = dx  # dt ~ dx

    logger.info(f"Running time level check for N={N}, dx={dx:.3f}, dt={dt:.3f}")

    solver = GRSolver(N, N, N, dx, dx, dx, log_level=logging.WARNING)

    solver.init_minkowski()

    t_initial = float(solver.orchestrator.t)
    eps_H_initial = float(solver.constraints.eps_H)

    logger.info(f"Initial t={t_initial}, eps_H={eps_H_initial}")

    # Run one step
    dt_actual, _, _ = solver.orchestrator.run_step(dt_max=dt)

    t_final = float(solver.orchestrator.t)
    eps_H_final = float(solver.constraints.eps_H)

    logger.info(f"After step t={t_final}, eps_H={eps_H_final}")

    # Get receipts from file
    receipts_file = "aeonic_receipts.jsonl"
    receipts = []
    if os.path.exists(receipts_file):
        with open(receipts_file, 'r') as f:
            for line in f:
                receipts.append(json.loads(line.strip()))
    if receipts:
        last_receipt = receipts[-1]
        stage_eps_H = last_receipt.get('stage_eps_H', {})

        # The receipt has eps_H at different stages, but the time level is not directly logged.
        # But we can infer: eps_H_pre is at t_n, eps_H_post_* at t_{n+1} (since computed after evolution)
        time_levels = {
            'eps_H_pre': t_initial,
            'eps_H_post_phys': t_final,  # After physical evolution
            'eps_H_post_cons': t_final,  # After constraint damping
            'eps_H_post_gauge': t_final,  # After gauge evolution
            'eps_H_post_filter': t_final   # After filtering
        }

        logged_time_levels = {}
        for key in stage_eps_H:
            if key in time_levels:
                logged_time_levels[key] = {
                    'eps_H': stage_eps_H[key],
                    't_used': time_levels[key]
                }

        result = {
            'N': N,
            'dx': dx,
            'dt': dt,
            't_n': t_initial,
            't_n_plus_1': t_final,
            'eps_H_t_n': eps_H_initial,
            'eps_H_t_n_plus_1': eps_H_final,
            'receipt_stage_eps_H': stage_eps_H,
            'inferred_time_levels': logged_time_levels
        }

        with open('diagnostic_time_levels.json', 'w') as f:
            json.dump(result, f, indent=2)

        print("Time Level Check Results:")
        print(f"t_n: {t_initial}, eps_H: {eps_H_initial}")
        print(f"t_n+1: {t_final}, eps_H: {eps_H_final}")
        print("Stage eps_H with inferred time levels:")
        for key, val in logged_time_levels.items():
            print(f"  {key}: eps_H={val['eps_H']:.2e} at t={val['t_used']:.3f}")

    else:
        print("No receipts found")

if __name__ == "__main__":
    run_time_level_check()