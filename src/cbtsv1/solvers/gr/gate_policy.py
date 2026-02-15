"""Gate policy module for externalized threshold configuration per Axiom A3.

Axiom A3 (Bounded Correction): All bounds must be declared and externalized.
This module provides the GatePolicy class to satisfy this requirement by
centralizing all gate thresholds that were previously hardcoded in gr_gates.py.
"""

import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger('gr_solver.gate_policy')


class GatePolicy:
    """Externalized gate thresholds per Coherence Axiom A3.
    
    Manages all threshold values for the GateChecker system. Supports:
    - Default initialization
    - Loading from JSON/YAML config files
    - Validation of threshold values
    - Export to dict for reproducibility (Axiom A5)
    """
    
    def __init__(self):
        """Initialize GatePolicy with default threshold values."""
        self.eps_H = {
            'warn': 5e-4,        # Enter warn state
            'fail': 1e-3,        # Enter fail state (hard fail)
            'warn_exit': 4e-4    # Exit warn state (hysteresis)
        }
        self.eps_M = {
            'soft': 1e-2,        # Soft threshold (penalty)
            'hard': 1e-1         # Hard threshold (hard fail)
        }
        self.eps_proj = {
            'soft': 1e-2,        # Soft threshold
            'hard': 1e-1         # Hard threshold
        }
        self.eps_clk = {
            'soft': 1e-2,        # Soft threshold
            'hard': 1e-1         # Hard threshold
        }
        self.spike = {
            'soft': 1e2,         # Soft threshold for spikes
            'hard': 1e3          # Hard threshold for spikes
        }
        self.theorem_validation = {
            'enabled': True,                # Enable theorem validation by default
            'gamma': 0.8,                   # Contraction coefficient (0 < γ < 1)
            'b': 1e-4,                      # Affine offset
            'halt_on_violation': False      # Log warnings but don't halt on violation
        }
        self.retry_policy = {
            'max_retries': 3,               # N_retry per Theorem Lemma 2
            'initial_dt_reduction': 0.8,    # Factor for dt reduction on retry
            'max_attempts_hard_fail': 4     # Hard fail after this many attempts (1 + N_retry)
        }
        self.hard_invariants = {
            'check_before_acceptance': True,  # Enforce hard invariants per Theorem Lemma 1
            'tolerance': 1e-14,               # Numerical tolerance for invariant checks
            'halt_on_violation': False        # Log warnings but don't halt (step is rejected)
        }

    @classmethod
    def from_file(cls, config_path: str) -> 'GatePolicy':
        """Load GatePolicy from JSON config file.
        
        Args:
            config_path: Path to JSON config file
            
        Returns:
            GatePolicy instance with loaded values
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config format is invalid
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}")
        
        policy = cls()
        
        # Update policy with loaded values
        if 'eps_H' in config_dict:
            policy.eps_H.update(config_dict['eps_H'])
        if 'eps_M' in config_dict:
            policy.eps_M.update(config_dict['eps_M'])
        if 'eps_proj' in config_dict:
            policy.eps_proj.update(config_dict['eps_proj'])
        if 'eps_clk' in config_dict:
            policy.eps_clk.update(config_dict['eps_clk'])
        if 'spike' in config_dict:
            policy.spike.update(config_dict['spike'])
        if 'theorem_validation' in config_dict:
            policy.theorem_validation.update(config_dict['theorem_validation'])
        if 'retry_policy' in config_dict:
            policy.retry_policy.update(config_dict['retry_policy'])
        if 'hard_invariants' in config_dict:
            policy.hard_invariants.update(config_dict['hard_invariants'])
        
        policy.validate()
        logger.info(f"Loaded gate policy from {config_path}")
        
        return policy

    def to_dict(self) -> Dict[str, Any]:
        """Export policy to dict for reproducibility (Axiom A5).
        
        Returns:
            Dictionary containing all threshold values
        """
        return {
            'eps_H': dict(self.eps_H),
            'eps_M': dict(self.eps_M),
            'eps_proj': dict(self.eps_proj),
            'eps_clk': dict(self.eps_clk),
            'spike': dict(self.spike),
            'theorem_validation': dict(self.theorem_validation),
            'retry_policy': dict(self.retry_policy),
            'hard_invariants': dict(self.hard_invariants)
        }

    def validate(self) -> None:
        """Validate that all thresholds are positive and ordered correctly.
        
        Raises:
            ValueError: If any threshold is invalid
        """
        # Check eps_H ordering
        if self.eps_H['warn_exit'] >= self.eps_H['warn']:
            raise ValueError(
                f"eps_H warn_exit ({self.eps_H['warn_exit']}) must be less than "
                f"warn ({self.eps_H['warn']})"
            )
        if self.eps_H['warn'] >= self.eps_H['fail']:
            raise ValueError(
                f"eps_H warn ({self.eps_H['warn']}) must be less than "
                f"fail ({self.eps_H['fail']})"
            )
        
        # Check all thresholds are positive
        all_thresholds = {
            'eps_H.warn': self.eps_H['warn'],
            'eps_H.fail': self.eps_H['fail'],
            'eps_H.warn_exit': self.eps_H['warn_exit'],
            'eps_M.soft': self.eps_M['soft'],
            'eps_M.hard': self.eps_M['hard'],
            'eps_proj.soft': self.eps_proj['soft'],
            'eps_proj.hard': self.eps_proj['hard'],
            'eps_clk.soft': self.eps_clk['soft'],
            'eps_clk.hard': self.eps_clk['hard'],
            'spike.soft': self.spike['soft'],
            'spike.hard': self.spike['hard']
        }
        
        for name, value in all_thresholds.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric, got {type(value)}")
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        
        # Check soft < hard ordering
        for prefix in ['eps_M', 'eps_proj', 'eps_clk', 'spike']:
            soft_val = getattr(self, prefix)['soft']
            hard_val = getattr(self, prefix)['hard']
            if soft_val >= hard_val:
                raise ValueError(
                    f"{prefix}: soft threshold ({soft_val}) must be less than "
                    f"hard threshold ({hard_val})"
                )
        
        # Validate theorem_validation config
        if self.theorem_validation.get('enabled', True):
            gamma = self.theorem_validation.get('gamma')
            b = self.theorem_validation.get('b')
            
            if gamma is not None:
                if not isinstance(gamma, (int, float)):
                    raise ValueError(f"theorem_validation.gamma must be numeric, got {type(gamma)}")
                if not (0 < gamma < 1):
                    raise ValueError(
                        f"theorem_validation.gamma must be in (0,1), got {gamma}"
                    )
            
            if b is not None:
                if not isinstance(b, (int, float)):
                    raise ValueError(f"theorem_validation.b must be numeric, got {type(b)}")
                if b < 0:
                    raise ValueError(
                        f"theorem_validation.b must be non-negative, got {b}"
                    )
        
        # Validate retry_policy config (Theorem Lemma 2)
        if self.retry_policy:
            max_retries = self.retry_policy.get('max_retries')
            dt_reduction = self.retry_policy.get('initial_dt_reduction')
            max_attempts = self.retry_policy.get('max_attempts_hard_fail')
            
            if max_retries is not None:
                if not isinstance(max_retries, int):
                    raise ValueError(f"retry_policy.max_retries must be int, got {type(max_retries)}")
                if max_retries <= 0:
                    raise ValueError(
                        f"retry_policy.max_retries must be > 0, got {max_retries}"
                    )
            
            if dt_reduction is not None:
                if not isinstance(dt_reduction, (int, float)):
                    raise ValueError(f"retry_policy.initial_dt_reduction must be numeric, got {type(dt_reduction)}")
                if not (0 < dt_reduction < 1):
                    raise ValueError(
                        f"retry_policy.initial_dt_reduction must be in (0,1), got {dt_reduction}"
                    )
            
            if max_attempts is not None:
                if not isinstance(max_attempts, int):
                    raise ValueError(f"retry_policy.max_attempts_hard_fail must be int, got {type(max_attempts)}")
                if max_attempts <= 1:
                    raise ValueError(
                        f"retry_policy.max_attempts_hard_fail must be > 1, got {max_attempts}"
                    )
                # Verify max_attempts = 1 + max_retries (Lemma 2 constraint)
                if max_retries is not None and max_attempts != (1 + max_retries):
                    raise ValueError(
                        f"retry_policy.max_attempts_hard_fail ({max_attempts}) must equal 1 + max_retries ({1 + max_retries})"
                    )
        
        # Validate hard_invariants config (Theorem Lemma 1)
        if self.hard_invariants:
            tolerance = self.hard_invariants.get('tolerance')
            check_enabled = self.hard_invariants.get('check_before_acceptance')
            
            if tolerance is not None:
                if not isinstance(tolerance, (int, float)):
                    raise ValueError(f"hard_invariants.tolerance must be numeric, got {type(tolerance)}")
                if tolerance < 0:
                    raise ValueError(
                        f"hard_invariants.tolerance must be non-negative, got {tolerance}"
                    )
            
            if check_enabled is not None:
                if not isinstance(check_enabled, bool):
                    raise ValueError(
                        f"hard_invariants.check_before_acceptance must be bool, got {type(check_enabled)}"
                    )
        
        logger.info("Gate policy validation passed")
