# Lexicon declarations per canon v1.2
LEXICON_IMPORTS = [
    "LoC_axiom",
    "UFE_core",
    "GR_dyn",
    "CTL_time",
    "PhaseLoom",
    "\\Omega_ledger"
]
LEXICON_SYMBOLS = {
    "\\Omega": "\\Omega_ledger.receipt",
    "CLOCK_DOMINANCE": "PhaseLoom.event.clock_dominance",
    "RAIL_WARNING": "PhaseLoom.event.rail_warning",
    "ROLLBACK": "PhaseLoom.event.rollback",
    "GAUGE_UPDATE": "PhaseLoom.event.gauge_update",
    "MILESTONE": "PhaseLoom.event.milestone",
    "COHERENCE_DROP": "Aeonic_Phaseloom.event.coherence_drop",
    "TAIL_DANGER_SPIKE": "Aeonic_Phaseloom.event.tail_danger_spike",
    "TIER_ALERT": "Aeonic_Phaseloom.event.tier_alert"
}

import json
from datetime import datetime
import numpy as np

class GRPhaseLoomReceipts:
    def __init__(self):
        self.receipts = []
        self.events = []

    def emit_receipt(self, step, t, dt, dominant_thread, threads, eps_pre_H, eps_pre_M, eps_post_H, eps_post_M, d_eps_H, d_eps_M, max_R, det_gamma_min, mu_H, mu_M, rollback_count, rollback_reason=None, loom_data=None, t_expected=0.0, t_err=0.0, dt_loom=None, risk_gauge=None, tight_threads=None, consistency_ok=None, rail_margins=None, lambda_min=None, lambda_max=None, cond_gamma=None, repair_applied=False, repair_type=None, lambda_min_pre=None, lambda_min_post=None, t_prev=None, t_next=None, dt_selected=None, dt_applied=None, substeps=None, commit_ok=None):
        """Emit Ω-receipt for audit"""
        loom_dict = {}
        if loom_data:
            loom_dict = {
                "C_global": loom_data.get('C_global', 0.0),
                "C_band": loom_data.get('C_band', []).tolist(),
                "D_band": loom_data.get('D_band', []).tolist(),
                "dt_loom": dt_loom
            }
            D_band_arr = loom_data.get('D_band', [])
            if len(D_band_arr) > 0 and any(d > 0 for d in D_band_arr):
                D_band = np.array(D_band_arr)
                loom_dict["D_max"] = float(np.max(D_band))
                loom_dict["D_argmax"] = int(np.argmax(D_band))
                sorted_indices = np.argsort(D_band)[::-1][:3]
                loom_dict["top3_bands"] = sorted_indices.tolist()
        receipt = {
            "step": step,
            "t": t,
            "dt": dt,
            "dominant_clock": dominant_thread,
            "threads": threads,
            "constraints": {
                "eps_pre_H": eps_pre_H,
                "eps_pre_M": eps_pre_M,
                "eps_post_H": eps_post_H,
                "eps_post_M": eps_post_M,
                "d_eps_H": d_eps_H,
                "d_eps_M": d_eps_M
            },
            "geometry": {
                "det_gamma_min": det_gamma_min,
                "R_max": max_R,
                "lambda_min": lambda_min,
                "lambda_max": lambda_max,
                "cond_gamma": cond_gamma
            },
            "damping": {
                "mu_H": mu_H,
                "mu_M": mu_M
            },
            "rails": {
                "rollback": rollback_count > 0,
                "reason": rollback_reason,
                "margins": rail_margins or {},
                "repair_applied": repair_applied,
                "repair_type": repair_type,
                "lambda_min_pre": lambda_min_pre,
                "lambda_min_post": lambda_min_post
            },
            "time_audit": {
                "t_expected": t_expected,
                "t_err": t_err
            },
            "loom": loom_dict,
            "risk_gauge": risk_gauge,
            "tight_threads": tight_threads or [],
            "commitment": {
                "t_prev": t_prev,
                "t_next": t_next,
                "dt_selected": dt_selected,
                "dt_applied": dt_applied,
                "substeps": substeps,
                "commit_ok": commit_ok
            },
            "consistency_ok": consistency_ok,
            "timestamp": datetime.utcnow().isoformat(),
            "lexicon": "canon_v1_2",
            "modules": ["gr_solver"]
        }

        # Clean threads for JSON
        cleaned_threads = {}
        for name, data in threads.items():
            cleaned_data = data.copy()
            if 'dt' in cleaned_data and (isinstance(cleaned_data['dt'], float) and np.isinf(cleaned_data['dt'])):
                cleaned_data['dt'] = None
            cleaned_threads[name] = cleaned_data
        receipt["threads"] = cleaned_threads

        self.receipts.append(receipt)
        # In real impl, write to file or DB
        # print(f"Ω-Receipt: {json.dumps(receipt, indent=2)}")  # Commented out to reduce output

    def emit_event(self, event_type, data):
        """Emit PhaseLoom timeline event"""
        event = {
            "type": event_type,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.events.append(event)
        print(f"PhaseLoom Event: {event_type} - {data}")

    def emit_clock_dominance(self, dominant_thread):
        self.emit_event("CLOCK_DOMINANCE", {"dominant": dominant_thread})

    def emit_rail_warning(self, rail, margin):
        self.emit_event("RAIL_WARNING", {"rail": rail, "margin": margin})

    def emit_rollback(self, cause, dt_new):
        self.emit_event("ROLLBACK", {"cause": cause, "dt_new": dt_new})

    def emit_gauge_update(self, parameter, value):
        self.emit_event("GAUGE_UPDATE", {"parameter": parameter, "value": value})

    def emit_milestone(self, milestone):
        self.emit_event("MILESTONE", {"milestone": milestone})

    def emit_coherence_drop(self, C_old, C_new, band):
        self.emit_event("COHERENCE_DROP", {"C_old": C_old, "C_new": C_new, "band": band})

    def emit_tail_danger_spike(self, D_old, D_new, band):
        self.emit_event("TAIL_DANGER_SPIKE", {"D_old": D_old, "D_new": D_new, "band": band})

    def emit_tier_alert(self, tier, issue):
        self.emit_event("TIER_ALERT", {"tier": tier, "issue": issue})