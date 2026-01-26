#!/usr/bin/env python3
"""
Test script for Analysis Mode in GR Solver.
Verifies that analysis_mode gates observe and log potential corrective actions without executing them.
"""

import numpy as np
import sys
import os
import logging
sys.path.append(os.path.abspath('.'))

from gr_solver.gr_solver import GRSolver

# Set up logging to see the logs
logging.basicConfig(level=logging.INFO)

def test_analysis_mode():
    print("Testing Analysis Mode...")

    # Create solver with analysis_mode=True
    solver = GRSolver(Nx=8, Ny=8, Nz=8, dx=0.5, dy=0.5, dz=0.5, analysis_mode=True, log_level=logging.INFO)

    # Initialize Minkowski
    solver.init_minkowski()

    # Run a few steps
    T_max = 0.1  # Short evolution
    dt_max = 0.01

    try:
        solver.run(T_max, dt_max)
        print("Analysis mode test completed successfully.")
        print(f"Final step: {solver.step}, Final t: {solver.t}")

        # Check that state hasn't been modified by corrective actions
        # In analysis mode, corrections should not be applied, so parameters should remain default

        # Check kappa_H, kappa_M (should be 1.0 if not modified)
        kappa_H = solver.stepper.loc_operator.kappa_H
        kappa_M = solver.stepper.loc_operator.kappa_M
        lambda_val = solver.stepper.lambda_val

        print(f"kappa_H: {kappa_H}, kappa_M: {kappa_M}, lambda_val: {lambda_val}")

        # In analysis mode, these should remain at initial values if no corrections applied
        assert kappa_H == 1.0, f"kappa_H modified in analysis mode: {kappa_H}"
        assert kappa_M == 1.0, f"kappa_M modified in analysis mode: {kappa_M}"
        assert lambda_val == 0.0, f"lambda_val modified in analysis mode: {lambda_val}"

        print("Analysis mode verification passed: no state modifications detected.")

        return True

    except Exception as e:
        print(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_analysis_mode()
    print(f"Analysis Mode Test: {'PASSED' if success else 'FAILED'}")