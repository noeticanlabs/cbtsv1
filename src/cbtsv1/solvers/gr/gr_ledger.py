import hashlib
import json
import inspect
from datetime import datetime
import numpy as np
from cbtsv1.solvers.gr.receipts import compute_debt_from_residuals

class GRLedger:
    def __init__(self, receipts_file="aeonic_receipts.jsonl"):
        self.receipts_file = receipts_file
        self.prev_receipt_hash = "0" * 64
        self.receipts = []

    def _emit(self, receipt):
        # Hash and chain
        receipt_str = json.dumps(receipt, sort_keys=True)
        receipt_hash = hashlib.sha256(receipt_str.encode()).hexdigest()
        receipt['receipt_hash'] = receipt_hash
        receipt['prev_receipt_hash'] = self.prev_receipt_hash
        self.prev_receipt_hash = receipt_hash

        # Store in memory
        self.receipts.append(receipt)

        # Write receipt
        with open(self.receipts_file, 'a') as f:
            f.write(json.dumps(receipt) + '\n')

    def emit_layout_violation_receipt(self, step, t, dt, fields, violations):
        receipt = {
            'run_id': 'gr_solver_run_001',
            'step': step,
            'event': 'LAYOUT_VIOLATION',
            't': t,
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
            'layout': {
                'ok': False,
                'violations': violations,
                'first_bad_tensor': violations[0]['field'] if violations else None,
                'got_shape': violations[0]['actual_shape'] if violations else None,
                'expected_shape': violations[0]['expected_shape'] if violations else None
            },
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
        self._emit(receipt)

    def emit_stage_rhs_receipt(self, step, t, dt, stage, stage_time, rhs_norms, fields, compute_rhs_method, debt_decomposition=None):
        # Compute operator hash (simplified - hash of compute_rhs method source)
        try:
            operator_code = inspect.getsource(compute_rhs_method)
        except (OSError, TypeError):
            operator_code = "source_not_available"
        operator_hash = hashlib.sha256(operator_code.encode()).hexdigest()

        # State hash (simplified fingerprint)
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

        # Add rhs norms
        receipt['rhs_norms'] = rhs_norms
        if debt_decomposition is not None:
            receipt['debt_decomposition'] = debt_decomposition
        self._emit(receipt)

    def emit_clock_decision_receipt(self, step, t, dt, fields, clocks):
        receipt = {
            'run_id': 'gr_solver_run_001',
            'step': step,
            'event': 'CLOCK_DECISION',
            't': t,
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
            'clocks': clocks,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
        self._emit(receipt)

    def emit_ledger_eval_receipt(self, step, t, dt, fields, ledgers, debt_decomposition=None):
        receipt = {
            'run_id': 'gr_solver_run_001',
            'step': step,
            'event': 'LEDGER_EVAL',
            't': t,
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
        self._emit(receipt)

    def emit_step_receipt(self, step, t, dt, fields, accepted, ledgers, gates, stage_eps_H=None, rejection_reason=None, corrections_applied=None, debt_decomposition=None):
        event = 'STEP_ACCEPT' if accepted else 'STEP_REJECT'
        receipt = {
            'run_id': 'gr_solver_run_001',
            'step': step,
            'event': event,
            't': t,
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
            'stage_eps_H': stage_eps_H or {},
            'rejection_reason': rejection_reason,
            'corrections_applied': corrections_applied or {},
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
        if debt_decomposition is not None:
            receipt['debt_decomposition'] = debt_decomposition
        self._emit(receipt)
