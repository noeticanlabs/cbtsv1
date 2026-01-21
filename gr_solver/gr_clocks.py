import numpy as np
from .gr_receipts import ReceiptEmitter

class ClockManager:
    def __init__(self, receipt_emitter: ReceiptEmitter):
        self.receipt_emitter = receipt_emitter

    def compute_clocks(self, dt_candidate, fields):
        c_max = np.sqrt(np.max(fields.alpha)**2 + np.max(np.linalg.norm(fields.beta, axis=-1))**2)
        h_min = min(fields.dx, fields.dy, fields.dz)
        dt_CFL = h_min / c_max if c_max > 0 else float('inf')

        beta_norm = np.max(np.linalg.norm(fields.beta, axis=-1))
        dt_gauge = fields.alpha.max() * h_min / (1 + beta_norm)

        dt_coh = dt_CFL * 0.5  # Coherence clock based on CFL

        K_norm = np.max(np.linalg.norm(fields.K_sym6, axis=-1))
        dt_res = h_min / max(np.sqrt(K_norm), 1e-6)

        dt_sigma = float('inf')

        dt_used = min(dt_candidate, dt_CFL, dt_gauge, dt_coh, dt_res, dt_sigma)

        clocks = {
            'dt_CFL': dt_CFL,
            'dt_gauge': dt_gauge,
            'dt_coh': dt_coh,
            'dt_res': dt_res,
            'dt_sigma': dt_sigma,
            'dt_used': dt_used
        }

        return clocks, dt_used

    def emit_clock_decision_receipt(self, step, t, dt, clocks):
        self.receipt_emitter.emit_clock_decision_receipt(step, t, dt, clocks)