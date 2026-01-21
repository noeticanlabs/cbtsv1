import numpy as np
import logging

logger = logging.getLogger('gr_solver.gates')

class GateChecker:
    def __init__(self, constraints):
        self.constraints = constraints
        self.eps_H_state = 'normal'  # Track state for eps_H hysteresis: 'normal', 'warn', 'fail'

    def check_gates(self):
        eps_H = float(self.constraints.eps_H)
        eps_M = float(self.constraints.eps_M)
        eps_proj = float(self.constraints.eps_proj)
        eps_clk = float(self.constraints.eps_clk) if self.constraints.eps_clk is not None else 0.0

        eps_H_warn = 7.5e-5
        eps_H_fail = 1e-4
        eps_M_soft_max = 1e-2
        eps_M_hard_max = 1e-1
        eps_proj_soft_max = 1e-2
        eps_proj_hard_max = 1e-1
        eps_clk_soft_max = 1e-2
        eps_clk_hard_max = 1e-1

        if np.isnan(eps_H) or np.isinf(eps_H) or np.isnan(eps_M) or np.isinf(eps_M) or np.isnan(eps_proj) or np.isinf(eps_proj) or np.isnan(eps_clk) or np.isinf(eps_clk):
            logger.error("Hard fail: NaN or infinite residuals in gates")
            return False, True, float('inf'), ["NaN/inf residuals"], {}, {}, {}

        accepted = True
        hard_fail = False
        penalty = 0.0
        reasons = []
        margins = {}
        corrections = {}

        # Hysteresis for eps_H
        enter_warn = 7.5e-5
        exit_warn = 6.0e-5
        enter_fail = 1e-4

        if self.eps_H_state == 'normal':
            if eps_H > enter_warn:
                self.eps_H_state = 'warn'
                penalty += (eps_H - enter_warn) / enter_warn
                reasons.append(f"Warn: eps_H = {eps_H:.2e} > {enter_warn:.2e}")
                margins['eps_H'] = enter_warn - eps_H
                corrections = {'reduce_dt': True, 'increase_kappa_budget': True, 'increase_projection_freq': True}
        elif self.eps_H_state == 'warn':
            if eps_H > enter_fail:
                self.eps_H_state = 'fail'
                hard_fail = True
                accepted = False
                reasons.append(f"eps_H = {eps_H:.2e} > {enter_fail:.2e}")
                margins['eps_H'] = enter_fail - eps_H
            elif eps_H < exit_warn:
                self.eps_H_state = 'normal'
            else:
                # Stay in warn
                penalty += (eps_H - enter_warn) / enter_warn
                reasons.append(f"Warn: eps_H = {eps_H:.2e} > {enter_warn:.2e}")
                margins['eps_H'] = enter_warn - eps_H
                corrections = {'reduce_dt': True, 'increase_kappa_budget': True, 'increase_projection_freq': True}
        elif self.eps_H_state == 'fail':
            # Always hard fail once in fail state
            hard_fail = True
            accepted = False
            reasons.append(f"eps_H = {eps_H:.2e} > {enter_fail:.2e}")
            margins['eps_H'] = enter_fail - eps_H

        for eps, name, soft, hard in [
            (eps_M, 'eps_M', eps_M_soft_max, eps_M_hard_max),
            (eps_proj, 'eps_proj', eps_proj_soft_max, eps_proj_hard_max),
            (eps_clk, 'eps_clk', eps_clk_soft_max, eps_clk_hard_max)
        ]:
            if eps > hard:
                hard_fail = True
                accepted = False
                reasons.append(f"{name} = {eps:.2e} > {hard:.2e}")
                margins[name] = hard - eps
            elif eps > soft:
                penalty += (eps - soft) / soft
                reasons.append(f"Soft: {name} = {eps:.2e} > {soft:.2e}")
                margins[name] = soft - eps

        spike_norms = {
            'alpha_spike': np.max(np.abs(np.gradient(self.constraints.fields.alpha))),
            'K_spike': np.max(np.abs(np.gradient(self.constraints.fields.K_sym6)))
        }
        spike_soft_max = 1e2
        spike_hard_max = 1e3
        for field, spike in spike_norms.items():
            if np.isnan(spike) or np.isinf(spike):
                hard_fail = True
                accepted = False
                reasons.append(f"Hard: {field} NaN/inf")
                margins[field] = float('-inf')
            elif spike > spike_hard_max:
                hard_fail = True
                accepted = False
                reasons.append(f"{field} spike = {spike:.2e} > {spike_hard_max:.2e}")
                margins[field] = spike_hard_max - spike
            elif spike > spike_soft_max:
                penalty += (spike - spike_soft_max) / spike_soft_max
                reasons.append(f"Soft: {field} spike = {spike:.2e} > {spike_soft_max:.2e}")
                margins[field] = spike_soft_max - spike

        return accepted, hard_fail, penalty, reasons, margins, corrections