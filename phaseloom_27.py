import numpy as np
from typing import Dict, Any, Tuple, Optional
from aeonic_memory_contract import SEMFailure

class PhaseLoom27:
    """Full 27-thread PhaseLoom lattice: 3 domains × 3 scales × 3 responses."""

    DOMAINS = ['PHY', 'CONS', 'SEM']
    SCALES = ['L', 'M', 'H']
    RESPONSES = ['FAST', 'MID', 'SLOW']

    def __init__(self):
        # Thread storage: (domain, scale, response) -> thread state
        self.threads = {}
        for d in self.DOMAINS:
            for s in self.SCALES:
                for r in self.RESPONSES:
                    self.threads[(d, s, r)] = {
                        'residual_threshold': self._default_threshold(d, s, r),
                        'dt_cap': float('inf'),
                        'actions_allowed': self._default_actions(d, s, r)
                    }

    def _default_threshold(self, domain: str, scale: str, response: str) -> float:
        """Default thresholds per thread."""
        base = {'L': 1e-6, 'M': 1e-8, 'H': 1e-10}
        if domain == 'SEM':
            return 0.0  # Binary
        return base.get(scale, 1e-8)

    def _default_actions(self, domain: str, scale: str, response: str) -> list:
        """Default actions per thread."""
        if response == 'FAST':
            return [{'type': 'dt_shrink', 'factor': 0.5}]
        elif response == 'MID':
            if domain == 'CONS':
                return [{'type': 'enforce_constraint', 'magnitude': 0.1}]
            return [{'type': 'gauge_adjust'}]
        else:  # SLOW
            return [{'type': 'regime_change'}]

    def compute_residuals(self, state: Any, geometry: Any, gauge: Any) -> Dict[Tuple[str, str], float]:
        """Compute proxy residuals at μ (solve clock). SEM checks prereqs."""
        residuals = {}

        # PHY residuals (proxy)
        for scale in self.SCALES:
            if scale == 'L':
                # Large-scale: energy in low modes
                residuals[('PHY', 'L')] = np.abs(state.alpha).mean()  # Placeholder using alpha
            elif scale == 'M':
                # Transfer band: gradients
                residuals[('PHY', 'M')] = np.abs(np.gradient(state.alpha)).mean()
            else:  # H
                # High freq: dissipation proxy
                residuals[('PHY', 'H')] = np.abs(state.alpha).max()

        # CONS residuals (proxy)
        for scale in self.SCALES:
            if hasattr(geometry, 'constraints'):
                cons_val = geometry.compute_constraint_proxy(scale)
                residuals[('CONS', scale)] = abs(cons_val)
            else:
                residuals[('CONS', scale)] = 0.0

        # SEM residuals (hard barriers)
        for scale in self.SCALES:
            sem_ok = True
            if hasattr(state, 'alpha') and np.any(state.alpha <= 0):
                sem_ok = False
            if hasattr(geometry, 'det_gamma') and np.any(geometry.det_gamma <= 0):
                sem_ok = False
            residuals[('SEM', scale)] = 0.0 if sem_ok else float('inf')

        return residuals

    def arbitrate_dt(self, residuals: Dict[Tuple[str, str], float]) -> Tuple[float, Tuple[str, str, str]]:
        """Find min dt_cap and dominant thread."""
        dt_min = float('inf')
        dominant = ('PHY', 'L', 'FAST')

        for (d, s), r in residuals.items():
            for resp in self.RESPONSES:
                thread_key = (d, s, resp)
                if thread_key in self.threads:
                    threshold = self.threads[thread_key]['residual_threshold']
                    if r > threshold:
                        # Compute dt_cap based on residual excess
                        dt_cap = self._compute_dt_cap(r, threshold, d, s, resp)
                        self.threads[thread_key]['dt_cap'] = dt_cap
                        if dt_cap < dt_min:
                            dt_min = dt_cap
                            dominant = thread_key

        return dt_min, dominant

    def _compute_dt_cap(self, residual: float, threshold: float, domain: str, scale: str, response: str) -> float:
        """Compute dt cap for thread."""
        excess = max(0, residual - threshold)
        if response == 'FAST':
            return 0.1 / (1 + excess * 1000)  # Aggressive shrink
        elif response == 'MID':
            return 0.5 / (1 + excess * 100)
        else:  # SLOW
            return 1.0 / (1 + excess * 10)

    def get_gate_classification(self, dominant_thread: Tuple[str, str, str], residual: float) -> Dict[str, Any]:
        """Classify gate: dt, state, or sem."""
        d, s, resp = dominant_thread

        if d == 'SEM':
            return {
                'kind': 'sem',
                'code': 'sem_violation',
                'actions_allowed': []
            }
        elif d == 'CONS' and resp in ['MID', 'SLOW']:
            return {
                'kind': 'state',
                'code': f'{d}_{s}_constraint',
                'actions_allowed': [{'repair': True, 'type': 'enforce_constraint'}]
            }
        else:
            return {
                'kind': 'dt',
                'code': f'{d}_{s}_cfl',
                'actions_allowed': [{'repair': False, 'type': 'dt_shrink'}]
            }

    def get_rails(self, dominant_thread: Tuple[str, str, str]) -> list:
        """Return corrective actions for dominant thread."""
        return self.threads[dominant_thread]['actions_allowed']

    def validate_sem(self, residuals: Dict[Tuple[str, str], float]) -> bool:
        """SEM validation: no silent zeros, no NaN/Inf."""
        for (d, s), r in residuals.items():
            if d == 'SEM' and not np.isfinite(r):
                return False
            if not np.isfinite(r):
                return False
        return True