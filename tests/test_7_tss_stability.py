"""
Test 7 TSS Stability
"""

import numpy as np
import logging
import copy

class Test7Tss:
    def __init__(self, gr_solver):
        self.gr_solver = gr_solver
        self.logger = logging.getLogger(__name__)

    def run(self):
        # Initialize solver with Minkowski + small perturbations
        self.gr_solver.init_minkowski()
        pert = 1e-12
        self.gr_solver.fields.gamma_sym6[0, 0, 0, 0] += pert
        self.gr_solver.fields.K_sym6[0, 0, 0, 0] += pert

        # Compute dt0 from scheduler for T=0.1
        self.gr_solver.constraints.compute_hamiltonian()
        self.gr_solver.constraints.compute_momentum()
        eps_H = self.gr_solver.constraints.eps_H
        eps_M = self.gr_solver.constraints.eps_M
        dt0 = self.gr_solver.scheduler.compute_dt(eps_H, eps_M)

        # Run evolutions with dt_multipliers = [0.5, 1.0, 2.0, 4.0], each to T=0.1, track max eps_H during run
        dt_multipliers = [0.5, 1.0, 2.0, 4.0]
        eps_H_max_list = []
        for mult in dt_multipliers:
            solver = copy.deepcopy(self.gr_solver)
            # Override compute_dt to force dt = dt0 * mult
            solver.scheduler.compute_dt = lambda eps_H, eps_M: dt0 * mult
            # Reset time and step
            solver.orchestrator.t = 0.0
            solver.orchestrator.step = 0
            T = 0.1
            max_eps_H = 0.0
            while solver.orchestrator.t < T:
                dt, _, _ = solver.orchestrator.run_step()
                # eps_H is computed in run_step, but ensure
                solver.constraints.compute_hamiltonian()
                max_eps_H = max(max_eps_H, solver.constraints.eps_H)
            eps_H_max_list.append(max_eps_H)

        # Check that eps_H increases with dt_multiplier, but for multiplier=0.5 and 1.0 it's stable, for 2.0 and 4.0 it degrades but doesn't crash immediately
        # Pass if eps_H_curve is monotonic increasing and for small dt it's low
        eps_H_curve = eps_H_max_list
        monotonic_increasing = all(eps_H_curve[i] <= eps_H_curve[i+1] for i in range(len(eps_H_curve)-1))
        small_dt_low = eps_H_curve[0] < 1e-6 and eps_H_curve[1] < 1e-6
        passed = monotonic_increasing and small_dt_low

        diagnosis = f"Monotonic increasing: {monotonic_increasing}, small dt low: {small_dt_low}, eps_H_max: {eps_H_max_list}"

        return {'passed': passed, 'metrics': {'dt_multipliers': dt_multipliers, 'eps_H_max': eps_H_max_list}, 'diagnosis': diagnosis}