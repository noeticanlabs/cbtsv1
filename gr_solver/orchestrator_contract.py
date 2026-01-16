from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple


class OrchestratorContract(ABC):
    """
    Abstract base class for the Orchestrator Contract.

    This contract defines the mandatory interface for orchestrators, enforcing aggregation of only accepted step receipts into canonical windows, regime transitions based on coherence and performance metrics, and issuing promotions or quarantines. The orchestrator maintains canon integrity by never aggregating rejected attempts and never declaring verified without sufficient accepted history.

    Violations of this contract constitute SEM-hard failures, halting the system and requiring manual intervention.

    See Technical Data/orchestrator_contract.md for full specification.

    Enforces hard rules:
    - Never aggregate rejected attempts (unless labeled)
    - Never declare verified without accepted history
    - Include thresholds/margins in claims
    - Regime labels: stable, constraint-risk, semantic-risk, perf-risk
    """

    SEM_FAILURE = "SEM_FAILURE"

    @abstractmethod
    def aggregate_window(self, accepted_step_receipts: List[Any], window_def: Dict[str, Any], perf_counters: Dict[str, Any]) -> Tuple[Any, List[Dict[str, Any]], str]:
        """
        Aggregate accepted step receipts into a canonical window, evaluate regime, and decide promotions/quarantines.

        Subclasses must implement window aggregation, ensuring:
        - Only accepted receipts are aggregated; rejected receipts are ignored unless labeled
        - Sufficient accepted history is required for verification
        - Claims include thresholds and margins (e.g., "PASS because residual < residual_threshold - margin")
        - Regime labels: 'stable', 'constraint-risk', 'semantic-risk', 'perf-risk'
        - Issue promotions if stable, quarantines if risk detected

        Args:
            accepted_step_receipts: List of accepted M_step receipt objects (only accepted, no rejected attempts)
            window_def: W dictionary defining aggregation window (e.g., {'num_steps': int} or {'delta_tau': float})
            perf_counters: Dictionary of performance metrics (e.g., {'cpu_time_per_step': float, 'memory_peak': float})

        Returns:
            On success: (window_receipt, promotions_quarantines, regime_label)
                - window_receipt: M_orch object summarizing window statistics and regime
                - promotions_quarantines: List of action dictionaries (e.g., [{'action': 'promote', 'target': 'state_X'}, {'action': 'quarantine', 'reason': 'constraint-risk'}])
                - regime_label: String label ('stable', 'constraint-risk', 'semantic-risk', 'perf-risk')
            On failure: (SEM_FAILURE, reason_str) but since abstract, subclasses handle
        """
        pass