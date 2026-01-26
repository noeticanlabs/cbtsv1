from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any, List

@dataclass
class ThreadState:
    domain: str
    scale: str
    response: str
    residual: float = 0.0
    dt_cap: float = float('inf')
    active: bool = True
    action_suggestion: Optional[Dict[str, Any]] = None

class PhaseLoom27:
    """
    The 27-thread invariant lattice for PhaseLoom control.
    
    Structure:
    - Domains: PHY (Physics), CONS (Constraints), SEM (Semantics)
    - Scales: L (Large), M (Medium), H (High/Tail)
    - Responses: FAST (R0 - Immediate), MID (R1 - Stabilizing), SLOW (R2 - Governance)
    
    Implements the 'Sensor-Governor' lattice defined in Aeonic PhaseLoom Canon v1.0.
    """
    DOMAINS = ['PHY', 'CONS', 'SEM']
    SCALES = ['L', 'M', 'H']
    RESPONSES = ['R0', 'R1', 'R2']

    # Finalized Gating Thresholds (LoC-GR Standard)
    # Violating these triggers immediate rollback (Gate_step failure).
    # SEM: Hard semantic barriers (e.g., causality, positivity). Tolerance: 0.0.
    # CONS: Constraint manifold deviation (H, M). Tolerance: 1e-6 (High Fidelity).
    # PHY: Evolution discretization error. Tolerance: 1e-4.
    DEFAULT_THRESHOLDS = {
        'SEM': 0.0,
        'CONS': 1.0e-6,
        'PHY': 1.0e-4
    }

    def __init__(self):
        # Map (domain, scale, response) -> ThreadState
        self.threads: Dict[Tuple[str, str, str], ThreadState] = {}
        self._initialize_lattice()

    def _initialize_lattice(self):
        """Initialize the 27 threads."""
        for d in self.DOMAINS:
            for s in self.SCALES:
                for r in self.RESPONSES:
                    self.threads[(d, s, r)] = ThreadState(domain=d, scale=s, response=r)

    def update_residual(self, domain: str, scale: str, value: float):
        """
        Update residual for a domain/scale pair. 
        Propagates the residual value to all response tiers (R0, R1, R2) for that domain/scale.
        """
        if domain not in self.DOMAINS or scale not in self.SCALES:
            raise ValueError(f"Invalid domain/scale: {domain}/{scale}")
        
        # Update all response threads for this D,S
        for r in self.RESPONSES:
            self.threads[(domain, scale, r)].residual = value

    def update_thread_state(self, domain: str, scale: str, response: str, dt_cap: float = None, residual: float = None, active: bool = None):
        """Update specific thread state."""
        key = (domain, scale, response)
        if key in self.threads:
            thread = self.threads[key]
            if dt_cap is not None:
                thread.dt_cap = dt_cap
            if residual is not None:
                thread.residual = residual
            if active is not None:
                thread.active = active

    def get_thread(self, domain: str, scale: str, response: str) -> ThreadState:
        """Retrieve a specific thread state."""
        return self.threads.get((domain, scale, response))

    def arbitrate_dt(self) -> Tuple[float, Tuple[str, str, str]]:
        """
        Find the minimum dt cap across all threads and identify the dominant thread.
        
        Returns:
            (min_dt, dominant_thread_key)
        """
        min_dt = float('inf')
        dominant_key = None

        for key, thread in self.threads.items():
            if not thread.active:
                continue
                
            if thread.dt_cap < min_dt:
                min_dt = thread.dt_cap
                dominant_key = key
        
        if dominant_key is None:
            # Default fallback if no caps are set (unconstrained evolution)
            dominant_key = ('PHY', 'L', 'R1') 
        
        return min_dt, dominant_key

    def check_gate_step(self, thresholds: Optional[Dict[str, float]] = None) -> Tuple[bool, List[str]]:
        """
        Enforce LoC inequalities for Gate_step.

        Checks:
        1. SEM Hard Barrier: r[SEM] <= theta[SEM] (usually 0)
        2. CONS Barrier: r[CONS] <= theta[CONS]
        3. PHY Barrier: r[PHY] <= theta[PHY]

        Args:
            thresholds: Dictionary of thresholds keyed by domain ('SEM', 'CONS', 'PHY').

        Returns:
            (passed, reasons)
        """
        reasons = []
        passed = True

        if thresholds is None:
            thresholds = self.DEFAULT_THRESHOLDS

        # Aggregate max residuals per domain from active threads
        max_residuals = {d: 0.0 for d in self.DOMAINS}
        for (d, s, r), thread in self.threads.items():
            if thread.active:
                max_residuals[d] = max(max_residuals[d], thread.residual)

        for domain in self.DOMAINS:
            limit = thresholds.get(domain, float('inf'))
            val = max_residuals[domain]
            if val > limit:
                passed = False
                reasons.append(f"{domain} Barrier violation: max_r={val:.2e} > {limit:.2e}")

        return passed, reasons

    def check_gate_orch(self, window_stats: Dict[str, Any], thresholds: Dict[str, float]) -> Tuple[bool, List[str]]:
        """
        Enforce LoC inequalities for Gate_orch (Regime Stability).

        Checks:
        1. Chatter score <= theta_chatter
        2. Max residual over window <= theta_stable

        Args:
            window_stats: Dictionary containing 'chatter_score' and 'max_residual'.
            thresholds: Dictionary of thresholds keyed by 'chatter' and 'residual'.

        Returns:
            (passed, reasons)
        """
        passed = True
        reasons = []

        chatter = window_stats.get('chatter_score', 1.0)
        max_res = window_stats.get('max_residual', 1.0)

        if chatter > thresholds.get('chatter', 0.5):
            passed = False
            reasons.append(f"Chatter too high: {chatter:.2f} > {thresholds.get('chatter', 0.5)}")

        if max_res > thresholds.get('residual', 1e-4):
            passed = False
            reasons.append(f"Window residual too high: {max_res:.2e} > {thresholds.get('residual', 1e-4)}")

        return passed, reasons

    def get_rails(self, dominant_thread: Any) -> List[Dict[str, Any]]:
        """
        Get active rails (corrective actions) based on the dominant thread.

        Section 6.2: Rails
        Rails are pre-authorized interventions triggered when specific threads dominate the clock.

        Args:
            dominant_thread: Tuple (domain, scale, response) or string "DOMAIN_SCALE_RESPONSE".

        Returns:
            List of rail actions (dicts).
        """
        if isinstance(dominant_thread, str):
            parts = dominant_thread.split('_')
            if len(parts) == 3:
                domain, scale, response = parts
            else:
                return []
        elif isinstance(dominant_thread, tuple) and len(dominant_thread) == 3:
            domain, scale, response = dominant_thread
        else:
            return []

        rails = []

        # 1. PHY Rails (Physics/CFL/Stiffness)
        if domain == 'PHY':
            if scale == 'H':  # High-frequency instability (e.g., grid noise)
                rails.append({'action': 'increase_dissipation', 'strength': 1.5, 'reason': 'PHY.H dominance'})
            elif response == 'R2':  # Audit/Rollback territory
                rails.append({'action': 'trigger_audit', 'scope': 'full', 'reason': 'PHY.R2 dominance'})

        # 2. CONS Rails (Constraint Violations)
        elif domain == 'CONS':
            if scale == 'L':  # Macro violation (e.g., Hamiltonian drift)
                rails.append({'action': 'enforce_projection', 'method': 'kreiss-oliger', 'reason': 'CONS.L dominance'})
            elif scale == 'M':  # Mid-scale (e.g., boundary reflection)
                rails.append({'action': 'adjust_boundary', 'mode': 'absorb', 'reason': 'CONS.M dominance'})
            elif scale == 'H':  # Micro violation (e.g., spikes)
                rails.append({'action': 'increase_dissipation', 'strength': 2.0, 'reason': 'CONS.H dominance'})

        # 3. SEM Rails (Semantic/Policy)
        elif domain == 'SEM':
            if response == 'R2':
                rails.append({'action': 'halt_and_dump', 'reason': 'SEM.R2 dominance'})
            else:
                rails.append({'action': 'log_warning', 'reason': 'SEM dominance'})

        return rails