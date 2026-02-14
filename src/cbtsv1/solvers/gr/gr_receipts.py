import hashlib
import json
import logging
from datetime import datetime
import numpy as np

logger = logging.getLogger('gr_solver.receipts')


def compute_debt_from_residuals(eps_H, eps_M, eps_proj, spike_norms):
    """
    Compute canonical debt decomposition per Spec 04_Coherence_Functionals.
    
    Per Axiom A2 (Attribution), all debt is decomposed into named causes:
    - conservation_defect: Hamiltonian constraint violation (ε_H²)
    - reconstruction_error: Momentum constraint violation (ε_M²)
    - tool_mismatch: Projection error (ε_proj²)
    - thrash_penalty: Aggregate spike violations (max of spike norms)
    
    Args:
        eps_H (float): Hamiltonian constraint residual
        eps_M (float): Momentum constraint residual
        eps_proj (float): Projection constraint residual
        spike_norms (dict): Dictionary of spike norms (field -> norm value)
    
    Returns:
        dict: Debt decomposition with four named components and total_debt
    """
    spike_penalty = float(max(spike_norms.values()) if spike_norms else 0.0)
    return {
        'conservation_defect': float(eps_H ** 2),
        'reconstruction_error': float(eps_M ** 2),
        'tool_mismatch': float(eps_proj ** 2),
        'thrash_penalty': spike_penalty,
        'total_debt': float(eps_H ** 2 + eps_M ** 2 + eps_proj ** 2 + spike_penalty)
    }

class ReceiptEmitter:
    def __init__(self, receipts_file="aeonic_receipts.jsonl"):
        self.receipts_file = receipts_file
        self.prev_receipt_hash = "0" * 64

    def _emit_receipt(self, receipt):
        receipt_str = json.dumps(receipt, sort_keys=True)
        receipt_hash = hashlib.sha256(receipt_str.encode()).hexdigest()
        receipt['receipt_hash'] = receipt_hash
        receipt['prev_receipt_hash'] = self.prev_receipt_hash
        self.prev_receipt_hash = receipt_hash

        with open(self.receipts_file, 'a') as f:
            f.write(json.dumps(receipt) + '\n')

    def emit_stage_rhs_receipt(self, step, t, dt, stage, stage_time, rhs_norms, fields, debt_decomposition=None):
        operator_code = hashlib.sha256(b"compute_rhs").hexdigest()
        operator_hash = hashlib.sha256(operator_code.encode()).hexdigest()
        state_str = f"{fields.alpha.sum():.6e}_{fields.gamma_sym6.sum():.6e}"
        state_hash = hashlib.sha256(state_str.encode()).hexdigest()

        receipt = {
            'run_id': 'gr_solver_run_001',
            'step': step,
            'event': 'STAGE_RHS',
            't': t,
            'dt': dt,
            'stage': stage,
            'stage_time': stage_time,
            'grid': {
                'Nx': fields.Nx,
                'Ny': fields.Ny,
                'Nz': fields.Nz,
                'h': [fields.dx, fields.dy, fields.dz],
                'domain': 'cartesian',
                'periodic': False
            },
            'hash': {
                'state_before': state_hash,
                'rhs': operator_hash,
                'operators': operator_hash
            },
            'ledgers': {},
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
        receipt['rhs_norms'] = rhs_norms
        if debt_decomposition is not None:
            receipt['debt_decomposition'] = debt_decomposition
        self._emit_receipt(receipt)

    def emit_clock_decision_receipt(self, step, t, dt, clocks):
        receipt = {
            'run_id': 'gr_solver_run_001',
            'step': step,
            'event': 'CLOCK_DECISION',
            't': t,
            'dt': dt,
            'stage': None,
            'grid': {
                'Nx': 0, 'Ny': 0, 'Nz': 0, 'h': [0,0,0], 'domain': 'cartesian', 'periodic': False  # Placeholder
            },
            'clocks': clocks,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
        self._emit_receipt(receipt)

    def emit_ledger_eval_receipt(self, step, t, dt, ledgers, fields, debt_decomposition=None):
        receipt = {
            'run_id': 'gr_solver_run_001',
            'step': step,
            'event': 'LEDGER_EVAL',
            't': t + dt,
            'dt': dt,
            'stage': None,
            'grid': {
                'Nx': fields.Nx,
                'Ny': fields.Ny,
                'Nz': fields.Nz,
                'h': [fields.dx, fields.dy, fields.dz],
                'domain': 'cartesian',
                'periodic': False
            },
            'ledgers': ledgers,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
        if debt_decomposition is not None:
            receipt['debt_decomposition'] = debt_decomposition
        self._emit_receipt(receipt)

    def emit_step_receipt(self, step, t, dt, accepted, ledgers, gates, rejection_reason, fields, debt_decomposition=None):
        event = 'STEP_ACCEPT' if accepted else 'STEP_REJECT'
        receipt = {
            'run_id': 'gr_solver_run_001',
            'step': step,
            'event': event,
            't': t + dt if accepted else t,
            'dt': dt,
            'stage': None,
            'grid': {
                'Nx': fields.Nx,
                'Ny': fields.Ny,
                'Nz': fields.Nz,
                'h': [fields.dx, fields.dy, fields.dz],
                'domain': 'cartesian',
                'periodic': False
            },
            'ledgers': ledgers,
            'gates': gates,
            'rejection_reason': rejection_reason,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
        if debt_decomposition is not None:
            receipt['debt_decomposition'] = debt_decomposition
        self._emit_receipt(receipt)