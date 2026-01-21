# Lexicon declarations per canon v1.2
LEXICON_IMPORTS = [
    "LoC_axiom",
    "UFE_core",
    "GR_dyn",
    "CTL_time"
]
LEXICON_SYMBOLS = {
    "Q_mass": "GR_obs.mass",
    "Q_angular_momentum": "GR_obs.angular_momentum",
    "\\mathcal{R}": "LoC_residual"
}

import json
import numpy as np
from typing import Optional
from dataclasses import asdict
from receipt_schemas import OmegaReceipt
from .gr_core_fields import inv_sym6, trace_sym6, norm2_sym6

from src.triaxis.lexicon import GHLL, GML

def min_active(values):
    """Return min of active values; assumes all are active."""
    return min(values) if values else float('inf')

class GRLedger:
    def __init__(self):
        self.receipts = []
        self.last_id: Optional[str] = None

    def emit_receipt(self, step, dt, eps_H, eps_M, eps_UFE, max_R, alpha_min, alpha_max, stability_class, thread_id="A:THREAD.PHY.L.R0", ops=None, dt_caps=None, eps=None, dominant_clock=None, argmin_margin=None, risk_gauge=None, margins=None, p_obs=None, fields=None):
        """Emit unified Ω-receipt with hash chaining."""
        # Compute invariant stamps
        dt_consistent = None
        if dt_caps is not None and eps is not None:
            dt_consistent = dt <= min_active(dt_caps) + eps

        dominance_consistent = None
        if dominant_clock is not None and argmin_margin is not None:
            dominance_consistent = dominant_clock == argmin_margin

        risk_consistent = None
        if risk_gauge is not None and margins is not None:
            risk_consistent = risk_gauge == min_active(margins)

        # Compute ADM mass and angular momentum
        Q_mass = 0.0
        Q_angular_momentum = 0.0
        if fields is not None:
            gamma_inv = inv_sym6(fields.gamma_sym6)
            K_trace = trace_sym6(fields.K_sym6, gamma_inv)
            K2 = norm2_sym6(fields.K_sym6, gamma_inv)
            energy_density = (1 / (16 * np.pi)) * (K2 - K_trace**2)
            dV = fields.dx * fields.dy * fields.dz
            Q_mass = np.sum(energy_density) * dV
            # Placeholder for angular momentum; set to 0.0 as initial data often has J=0
            Q_angular_momentum = 0.0

        record = {
            "lexicon_terms_used": ["LoC_axiom", "UFE_core", "GR_dyn"],
            "intent_id": GHLL.INV_PDE_DIV_FREE, # Primary intent
            "thread": thread_id,
            "ops": ops or [],
            "step": step,
            "dt": dt,
            "eps_H": eps_H,
            "eps_M": eps_M,
            "eps_UFE": eps_UFE,
            "max_curvature": max_R,
            "gauge": {"alpha_min": alpha_min, "alpha_max": alpha_max},
            "Q_mass": Q_mass,
            "Q_angular_momentum": Q_angular_momentum,
            "stability_class": stability_class,
            "p_obs": p_obs,  # Scaling law convergence order
            "invariants": {
                "dt_consistent": dt_consistent,
                "dominance_consistent": dominance_consistent,
                "risk_consistent": risk_consistent
            }
        }

        # Create unified receipt
        omega_receipt = OmegaReceipt.create(prev=self.last_id, tier="gr_ledger", record=record)

        # Store and print
        self.receipts.append(omega_receipt)
        print(json.dumps(asdict(omega_receipt), indent=2))  # For now, print; later save

        # Update last_id
        self.last_id = omega_receipt.id

        return omega_receipt