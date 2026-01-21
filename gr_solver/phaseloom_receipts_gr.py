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
import hashlib
from .logging_config import ReceiptLevels
from receipt_schemas import OmegaReceipt

class GRPhaseLoomReceipts:
    def __init__(self):
        self.receipts = []  # Keep for compatibility, but use omega_receipts
        self.omega_receipts = []
        self.events = []
        self.config = ReceiptLevels()
        self.last_id = None
        self.macro_aggregates = {
            'eps_post_H': [],
            'eps_post_M': [],
            'd_eps_H': [],
            'd_eps_M': [],
            'det_gamma_min': [],
            'R_max': [],
            'lambda_min': [],
            'lambda_max': [],
            'cond_gamma': [],
            'mu_H': [],
            'mu_M': [],
            'rollback_count': []
        }

    def emit_m_solve(self, step, t, dt, dominant_thread, threads, eps_pre_H, eps_pre_M, eps_post_H, eps_post_M, d_eps_H, d_eps_M, max_R, det_gamma_min, mu_H, mu_M, rollback_count, rollback_reason=None, loom_data=None, t_expected=0.0, t_err=0.0, dt_loom=None, risk_gauge=None, tight_threads=None, consistency_ok=None, rail_margins=None, lambda_min=None, lambda_max=None, cond_gamma=None, repair_applied=False, repair_type=None, lambda_min_pre=None, lambda_min_post=None, t_prev=None, t_next=None, dt_selected=None, dt_applied=None, substeps=None, commit_ok=None, policy_hash=None):
        """Emit M_solve receipt for high-frequency diagnostics"""
        loom_dict = {}
        if loom_data:
            loom_dict = {
                "C_global": loom_data.get('C_global', 0.0),
                "C_band": loom_data.get('C_band', []).tolist(),
                "D_band": loom_data.get('D_band', []).tolist(),
                "C_o": loom_data.get('C_o', 0.0),
                "coherence_drop": loom_data.get('coherence_drop', 0.0),
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
            "level": "M_solve",
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
            "policy_hash": policy_hash,
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

        # Create OmegaReceipt for unified LoC ledger
        omega_record = receipt.copy()
        omega_receipt = OmegaReceipt.create(prev=self.last_id, tier="gr_phaseloom", record=omega_record)
        self.omega_receipts.append(omega_receipt)
        self.last_id = omega_receipt.id

        # Legacy receipts for compatibility
        self.receipts.append(receipt)
        # Aggregate for macro receipts
        self.macro_aggregates['eps_post_H'].append(eps_post_H)
        self.macro_aggregates['eps_post_M'].append(eps_post_M)
        self.macro_aggregates['d_eps_H'].append(d_eps_H)
        self.macro_aggregates['d_eps_M'].append(d_eps_M)
        self.macro_aggregates['det_gamma_min'].append(det_gamma_min)
        self.macro_aggregates['R_max'].append(max_R)
        self.macro_aggregates['lambda_min'].append(lambda_min)
        self.macro_aggregates['lambda_max'].append(lambda_max)
        self.macro_aggregates['cond_gamma'].append(cond_gamma)
        self.macro_aggregates['mu_H'].append(mu_H)
        self.macro_aggregates['mu_M'].append(mu_M)
        self.macro_aggregates['rollback_count'].append(rollback_count)
        # In real impl, write to file or DB
        # print(f"Ω-Receipt: {json.dumps(asdict(omega_receipt), indent=2)}")  # Commented out to reduce output

    def emit_event(self, event_type, data):
        """Emit PhaseLoom timeline event"""
        event = {
            "type": event_type,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.events.append(event)
        print(f"PhaseLoom Event: {event_type} - {data}")

    def emit_m_step(self, step, t, dt, dominant_thread, threads, eps_pre_H, eps_pre_M, eps_post_H, eps_post_M, d_eps_H, d_eps_M, max_R, det_gamma_min, mu_H, mu_M, rollback_count, rollback_reason=None, loom_data=None, t_expected=0.0, t_err=0.0, dt_loom=None, risk_gauge=None, tight_threads=None, consistency_ok=None, rail_margins=None, lambda_min=None, lambda_max=None, cond_gamma=None, repair_applied=False, repair_type=None, lambda_min_pre=None, lambda_min_post=None, t_prev=None, t_next=None, dt_selected=None, dt_applied=None, substeps=None, commit_ok=None, policy_hash=None):
        """Emit M_step receipt for sparse canonical data"""
        if not self.config.enable_M_step:
            return
        # Minimal threads summary
        minimal_threads = {k: {'dt': v.get('dt'), 'active': v.get('active', False)} for k, v in threads.items()}
        receipt = {
            "level": "M_step",
            "step": step,
            "t": t,
            "dt": dt,
            "dominant_clock": dominant_thread,
            "threads": minimal_threads,
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
                "reason": rollback_reason
            },
            "time_audit": {
                "t_expected": t_expected,
                "t_err": t_err
            },
            "commitment": {
                "t_prev": t_prev,
                "t_next": t_next,
                "dt_selected": dt_selected,
                "dt_applied": dt_applied,
                "substeps": substeps,
                "commit_ok": commit_ok
            },
            "consistency_ok": consistency_ok,
            "policy_hash": policy_hash,
            "timestamp": datetime.utcnow().isoformat(),
            "lexicon": "canon_v1_2",
            "modules": ["gr_solver"]
        }
        # Create OmegaReceipt
        omega_record = receipt.copy()
        omega_receipt = OmegaReceipt.create(prev=self.last_id, tier="gr_phaseloom", record=omega_record)
        self.omega_receipts.append(omega_receipt)
        self.last_id = omega_receipt.id

        self.receipts.append(receipt)

    def emit_macro(self, step):
        """Emit macro receipt with aggregates every K steps"""
        if not self.config.enable_macro:
            return
        aggregates = {}
        for k, v in self.macro_aggregates.items():
            if v:
                aggregates[k] = {
                    'min': min(v),
                    'max': max(v),
                    'mean': sum(v) / len(v),
                    'count': len(v)
                }
        # Hash of recent receipts
        recent = self.receipts[-self.config.K:] if len(self.receipts) >= self.config.K else self.receipts
        receipt_str = json.dumps(recent, sort_keys=True)
        hash_val = hashlib.sha256(receipt_str.encode()).hexdigest()
        aggregates['receipt_chain_hash'] = hash_val
        macro_receipt = {
            "level": "macro",
            "step": step,
            "aggregates": aggregates,
            "timestamp": datetime.utcnow().isoformat(),
            "lexicon": "canon_v1_2",
            "modules": ["gr_solver"]
        }
        # Create OmegaReceipt
        omega_record = macro_receipt.copy()
        omega_receipt = OmegaReceipt.create(prev=self.last_id, tier="gr_phaseloom", record=omega_record)
        self.omega_receipts.append(omega_receipt)
        self.last_id = omega_receipt.id

        self.receipts.append(macro_receipt)
        # Reset aggregates
        for k in self.macro_aggregates:
            self.macro_aggregates[k] = []

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