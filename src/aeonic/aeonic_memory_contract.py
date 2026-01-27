from typing import Dict, List, Any, Tuple, Optional
from src.aeonic.aeonic_memory_bank import AeonicMemoryBank
from src.receipts.receipt_schemas import Kappa, MSolveReceipt, MStepReceipt, MOrchReceipt
from src.core.gr_ttl_calculator import TTLCalculator, AdaptiveTTLs, TimeUnit, TTLValue
from src.core.gr_gates import should_hard_fail, GateKind
import hashlib
import json
import numpy as np

class SEMFailure(Exception):
    """SEM hard failure - immediate abort."""
    pass

class AeonicMemoryContract:
    """
    Contract-compliant interface mapping tiered memory to M_solve/M_step/M_orch.
    
    Supports multiple time units: microseconds (us), seconds (s), and minutes (m).
    All TTLs are stored internally in seconds for consistency.
    """

    # Static TTL fallback constants (in seconds)
    # These are the minimum fallback values when no adaptive TTL is configured
    STATIC_TTL_M_SOLVE_S = 1e-6   # 1 microsecond minimum for attempts
    STATIC_TTL_M_SOLVE_L = 1.0    # 1 second minimum
    STATIC_TTL_M_STEP_S = 10.0    # 10 seconds minimum for steps
    STATIC_TTL_M_STEP_L = 600.0   # 10 minutes minimum
    STATIC_TTL_M_ORCH_S = 3600.0  # 1 hour minimum for canon
    STATIC_TTL_M_ORCH_L = 86400.0 # 1 day minimum

    # Alternative static TTLs in different units (for convenience)
    @classmethod
    def get_static_ttl_msolve_s(cls, unit: TimeUnit = TimeUnit.SECONDS) -> float:
        """Get M_solve short TTL in specified unit."""
        return cls._convert_time(cls.STATIC_TTL_M_SOLVE_S, unit)
    
    @classmethod
    def get_static_ttl_msolve_l(cls, unit: TimeUnit = TimeUnit.SECONDS) -> float:
        """Get M_solve long TTL in specified unit."""
        return cls._convert_time(cls.STATIC_TTL_M_SOLVE_L, unit)
    
    @classmethod
    def get_static_ttl_mstep_s(cls, unit: TimeUnit = TimeUnit.SECONDS) -> float:
        """Get M_step short TTL in specified unit."""
        return cls._convert_time(cls.STATIC_TTL_M_STEP_S, unit)
    
    @classmethod
    def get_static_ttl_mstep_l(cls, unit: TimeUnit = TimeUnit.SECONDS) -> float:
        """Get M_step long TTL in specified unit."""
        return cls._convert_time(cls.STATIC_TTL_M_STEP_L, unit)
    
    @classmethod
    def get_static_ttl_morch_s(cls, unit: TimeUnit = TimeUnit.SECONDS) -> float:
        """Get M_orch short TTL in specified unit."""
        return cls._convert_time(cls.STATIC_TTL_M_ORCH_S, unit)
    
    @classmethod
    def get_static_ttl_morch_l(cls, unit: TimeUnit = TimeUnit.SECONDS) -> float:
        """Get M_orch long TTL in specified unit."""
        return cls._convert_time(cls.STATIC_TTL_M_ORCH_L, unit)
    
    @staticmethod
    def _convert_time(seconds: float, unit: TimeUnit) -> float:
        """Convert seconds to the specified time unit."""
        if unit == TimeUnit.MICROSECONDS:
            return seconds * 1e6
        elif unit == TimeUnit.MINUTES:
            return seconds / 60.0
        return seconds

    def __init__(self, memory_bank: AeonicMemoryBank, receipts_log=None, ttl_calculator: Optional[TTLCalculator] = None):
        """
        Initialize the AeonicMemoryContract.
        
        Args:
            memory_bank: The AeonicMemoryBank instance
            receipts_log: Optional receipts logger
            ttl_calculator: Optional TTLCalculator for adaptive TTL computation
        """
        self.memory_bank = memory_bank
        self.receipts_log = receipts_log
        self.ttl_calculator = ttl_calculator
        self.attempt_counter = 0
        self.step_counter = 0
        
        # Cache for computed TTLs (lazily computed)
        self._cached_ttls: Optional[AdaptiveTTLs] = None

        # Initialize and map tiers to contract memories
        for tier in [1, 2, 3]:
            if tier not in memory_bank.tiers:
                memory_bank.tiers[tier] = {}
                memory_bank.tier_bytes[tier] = 0

        self.M_solve = memory_bank.tiers[1]  # Ring buffer for attempts
        self.M_step = memory_bank.tiers[2]   # Only accepted steps
        self.M_orch = memory_bank.tiers[3]   # Canon promotions

    def _get_ttls(self) -> AdaptiveTTLs:
        """Get TTLs from calculator or compute cached value."""
        if self.ttl_calculator is None:
            # Return static TTLs as AdaptiveTTLs for backward compatibility
            return AdaptiveTTLs(
                msolve_ttl_s=self.STATIC_TTL_M_SOLVE_S,
                msolve_ttl_l=self.STATIC_TTL_M_SOLVE_L,
                mstep_ttl_s=self.STATIC_TTL_M_STEP_S,
                mstep_ttl_l=self.STATIC_TTL_M_STEP_L,
                morch_ttl_s=self.STATIC_TTL_M_ORCH_S,
                morch_ttl_l=self.STATIC_TTL_M_ORCH_L
            )
        
        if self._cached_ttls is None:
            self._cached_ttls = self.ttl_calculator.compute_ttls()
        return self._cached_ttls

    def set_ttl_calculator(self, ttl_calculator: TTLCalculator):
        """Set or replace the TTL calculator and invalidate cache."""
        self.ttl_calculator = ttl_calculator
        self._cached_ttls = None  # Invalidate cache

    def put_attempt_receipt(self, kappa: Kappa, receipt: MSolveReceipt):
        """Store attempt receipt in M_solve (ring buffer)."""
        self.attempt_counter += 1
        key = f"attempt_{kappa.o}_{kappa.s}_{kappa.mu}_{receipt.attempt_id}"
        
        # Get TTLs (adaptive or static)
        ttls = self._get_ttls()
        
        self.memory_bank.put(
            key=key,
            tier=1,
            payload=receipt,
            bytes=1024,  # Estimate
            ttl_s=ttls.msolve_ttl_s,
            ttl_l=ttls.msolve_ttl_l,
            recompute_cost_est=100.0,
            risk_score=0.1,
            tainted=False,
            regime_hashes=[receipt.policy_hash]
        )
        if self.receipts_log:
            self.receipts_log.emit_event("ATTEMPT_RECEIPT", {
                "kappa": (kappa.o, kappa.s, kappa.mu),
                "attempt_id": receipt.attempt_id,
                "sem_ok": receipt.sem_ok,
                "policy_hash": receipt.policy_hash
            })

    def put_step_receipt(self, kappa: Kappa, receipt: MStepReceipt):
        """Store accepted step in M_step."""
        self.step_counter += 1
        key = f"step_{kappa.o}_{kappa.s}"
        
        # Get TTLs (adaptive or static)
        ttls = self._get_ttls()
        
        self.memory_bank.put(
            key=key,
            tier=2,
            payload=receipt,
            bytes=2048,  # Estimate
            ttl_s=ttls.mstep_ttl_s,
            ttl_l=ttls.mstep_ttl_l,
            recompute_cost_est=1000.0,
            risk_score=0.05,
            tainted=False,
            regime_hashes=[receipt.kappa.o]  # Use o as regime hash for now
        )
        if self.receipts_log:
            self.receipts_log.emit_event("STEP_RECEIPT", {
                "kappa": (kappa.o, kappa.s, kappa.mu),
                "step_id": receipt.step_id,
                "t": receipt.t,
                "dt_used": receipt.dt_used
            })

    def put_orch_receipt(self, receipt: MOrchReceipt):
        """Store canon promotion in M_orch (append-only)."""
        key = f"orch_{receipt.o}"
        
        # Get TTLs (adaptive or static)
        ttls = self._get_ttls()
        
        self.memory_bank.put(
            key=key,
            tier=3,
            payload=receipt,
            bytes=4096,  # Estimate
            ttl_s=ttls.morch_ttl_s,
            ttl_l=ttls.morch_ttl_l,
            recompute_cost_est=10000.0,
            risk_score=0.01,
            tainted=False,
            regime_hashes=[receipt.policy_hash],
            demoted=False  # Canon never demoted
        )
        if self.receipts_log:
            self.receipts_log.emit_event("ORCH_RECEIPT", {
                "o": receipt.o,
                "regime_label": receipt.regime_label,
                "policy_hash": receipt.policy_hash
            })

    def get_by_kappa(self, kappa_range: Tuple[Kappa, Kappa]) -> List[Any]:
        """Retrieve receipts within kappa range."""
        results = []
        for key, record in self.M_step.items():
            if record.tainted:
                continue
            # Parse key: "step_{o}_{s}"
            parts = key.split('_')
            if len(parts) >= 3 and parts[0] == 'step':
                try:
                    o = int(parts[1])
                    s = int(parts[2])
                    kappa_lower = kappa_range[0]
                    kappa_upper = kappa_range[1]
                    if (kappa_lower.o <= o <= kappa_upper.o and
                        kappa_lower.s <= s <= kappa_upper.s):
                        results.append(record.payload)
                except ValueError:
                    continue  # Skip malformed keys
        return results

    def promote_to_canon(self, step_receipts: List[MStepReceipt], gate_result: Dict[str, Any]) -> bool:
        """Evaluate promotion to canon."""
        # Simple gate: check residuals below threshold
        if gate_result.get('pass', False):
            # Create orch receipt
            orch_receipt = MOrchReceipt(
                o=max(r.kappa.o for r in step_receipts) + 1,
                window_steps=[r.step_id for r in step_receipts],
                quantiles={},  # TODO: compute
                dominance_histogram={},
                chatter_score=0.0,
                regime_label='stable',
                promotions=[],
                verification_threshold=1e-6,
                verification_norm='L2',
                min_accepted_history=len(step_receipts),
                policy_hash=step_receipts[0].kappa.o  # Placeholder
            )
            self.put_orch_receipt(orch_receipt)
            return True
        return False

    def validate_min_accepted_history(self, min_steps: int) -> bool:
        """Check minimum accepted history before verification."""
        return len([r for r in self.M_step.values() if not r.tainted]) >= min_steps

    def abort_on_hard_fail(self, gate: Dict[str, Any], accepted: bool):
        """
        Fast-exit for hard gate failures.
        
        Args:
            gate: Gate dictionary with 'kind' and 'reason' keys
            accepted: Whether the gate was accepted (True) or rejected (False)
            
        Raises:
            SEMFailure: If gate is hard-fail kind and not accepted
        """
        if not accepted and should_hard_fail(gate):
            gate_kind = gate.get('kind', 'unknown')
            gate_reason = gate.get('reason', 'no reason specified')
            raise SEMFailure(f"Hard gate failure: kind={gate_kind}, reason={gate_reason}")
        # Soft failure handling is delegated to retry policy in the caller

    def abort_on_state_gate_no_repair(self, gate: Dict[str, Any]):
        """Fast-exit for state gate with no repair action. LEGACY: use abort_on_hard_fail."""
        if gate.get('kind') == 'state' and not any(action.get('repair', False) for action in gate.get('actions_allowed', [])):
            raise SEMFailure("State gate violation with no repair action - abort immediately")

    def compute_policy_hash(self, policy_dict: Dict[str, Any]) -> str:
        """Compute policy fingerprint."""
        policy_str = json.dumps(policy_dict, sort_keys=True)
        return hashlib.md5(policy_str.encode()).hexdigest()[:16]

    def validate_policy_consistency(self, current_policy_hash: str, window_policy_hash: str):
        """SEM barrier: policy hash must match window policy."""
        if current_policy_hash != window_policy_hash:
            raise SEMFailure(f"Policy hash mismatch: {current_policy_hash} != {window_policy_hash}")

    def check_no_silent_zeros(self, state: Any, geometry: Any) -> bool:
        """SEM invariant: prerequisites initialized, no zero defaults."""
        if hasattr(state, 'alpha') and np.any(state.alpha == 0):
            return False  # Lapse not initialized
        if hasattr(geometry, 'det_gamma') and np.any(geometry.det_gamma == 0):
            return False  # Metric not initialized
        return True

    def check_no_nonfinite(self, values: Any) -> bool:
        """SEM invariant: no NaN/Inf in residuals or state."""
        if isinstance(values, dict):
            for v in values.values():
                if not self.check_no_nonfinite(v):
                    return False
        elif hasattr(values, '__iter__') and not isinstance(values, str):
            return np.all(np.isfinite(values))
        return True
    
    def get_ttl_as_value(self, ttl_type: str, unit: TimeUnit = TimeUnit.SECONDS) -> TTLValue:
        """
        Get a TTL as a TTLValue object.
        
        Args:
            ttl_type: Type of TTL ('msolve_s', 'msolve_l', 'mstep_s', 'mstep_l', 'morch_s', 'morch_l')
            unit: Time unit for the value
        
        Returns:
            TTLValue object
        """
        ttls = self._get_ttls()
        ttl_mapping = {
            'msolve_s': ttls.msolve_ttl_s,
            'msolve_l': ttls.msolve_ttl_l,
            'mstep_s': ttls.mstep_ttl_s,
            'mstep_l': ttls.mstep_ttl_l,
            'morch_s': ttls.morch_ttl_s,
            'morch_l': ttls.morch_ttl_l,
        }
        seconds = ttl_mapping.get(ttl_type, self.STATIC_TTL_M_SOLVE_S)
        return TTLValue(value=seconds, unit=unit)
    
    def get_ttls_in_unit(self, unit: TimeUnit) -> Dict[str, float]:
        """
        Get all TTLs in the specified unit.
        
        Args:
            unit: Time unit for the values
        
        Returns:
            Dictionary with TTL values in the specified unit
        """
        ttls = self._get_ttls()
        return {
            'msolve_s': self._convert_time(ttls.msolve_ttl_s, unit),
            'msolve_l': self._convert_time(ttls.msolve_ttl_l, unit),
            'mstep_s': self._convert_time(ttls.mstep_ttl_s, unit),
            'mstep_l': self._convert_time(ttls.mstep_ttl_l, unit),
            'morch_s': self._convert_time(ttls.morch_ttl_s, unit),
            'morch_l': self._convert_time(ttls.morch_ttl_l, unit),
        }
    
    def __repr__(self):
        ttls = self._get_ttls()
        return (f"AeonicMemoryContract(M_solve={ttls.msolve_ttl_s_us:.1f}us/{ttls.msolve_ttl_l_s:.2f}s, "
                f"M_step={ttls.mstep_ttl_s_s:.2f}s/{ttls.mstep_ttl_l_m:.2f}m, "
                f"M_orch={ttls.morch_ttl_s_m:.2f}m/{ttls.morch_ttl_l_s:.2f}s)")