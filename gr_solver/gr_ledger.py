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

def min_active(values):
    """Return min of active values; assumes all are active."""
    return min(values) if values else float('inf')

class GRLedger:
    def __init__(self):
        self.receipts = []

    def emit_receipt(self, step, dt, eps_H, eps_M, eps_UFE, max_R, alpha_min, alpha_max, stability_class, dt_caps=None, eps=None, dominant_clock=None, argmin_margin=None, risk_gauge=None, margins=None):
        """Emit Ω-receipt."""
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

        receipt = {
            "lexicon_terms_used": ["LoC_axiom", "UFE_core", "GR_dyn"],
            "step": step,
            "dt": dt,
            "eps_H": eps_H,
            "eps_M": eps_M,
            "eps_UFE": eps_UFE,
            "max_curvature": max_R,
            "gauge": {"alpha_min": alpha_min, "alpha_max": alpha_max},
            "Q_mass": 0.0,  # Placeholder
            "Q_angular_momentum": 0.0,  # Placeholder
            "stability_class": stability_class,
            "invariants": {
                "dt_consistent": dt_consistent,
                "dominance_consistent": dominance_consistent,
                "risk_consistent": risk_consistent
            }
        }
        self.receipts.append(receipt)
        print(json.dumps(receipt, indent=2))  # For now, print; later save