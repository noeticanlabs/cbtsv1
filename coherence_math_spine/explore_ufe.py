#!/usr/bin/env python3
"""
Explore Toy Unified Flow Equation (UFE)

This script defines a toy UFE, computes its discrete residual,
and emits a minimal receipt JSON to stdout demonstrating the gate predicates.

Usage:
    python coherence_math_spine/explore_ufe.py [--verbose]

The toy UFE demonstrates the core coherence principle:
    coherence(X, Y) = residual(X, Y) + debt(X, Y)

where:
    - X is the current state
    - Y is the target state
    - residual measures deviation from coherence
    - debt accumulates over time
"""

import json
import hashlib
import argparse
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any


class ToyUFE:
    """
    A toy Unified Flow Equation for demonstration.

    The UFE defines the evolution of a system state X under a coherence functional.
    """

    def __init__(self, lambda_param: float = 0.5, dt: float = 0.1):
        """
        Initialize the toy UFE.

        Args:
            lambda_param: Coherence rate parameter (0 < lambda < 1)
            dt: Discrete time step
        """
        self.lambda_param = lambda_param
        self.dt = dt
        self.step_count = 0

    def coherence_functional(self, x: float, y: float) -> float:
        """
        Compute coherence functional: C(x, y) = λ·x + (1-λ)·y

        This is a simple weighted average as a toy coherence functional.
        """
        return self.lambda_param * x + (1 - self.lambda_param) * y

    def residual(self, x: float, y: float) -> float:
        """
        Compute discrete residual: R = |x - C(x, y)|

        Measures how far the current state is from coherence.
        """
        c = self.coherence_functional(x, y)
        return abs(x - c)

    def debt_functional(self, residual: float) -> float:
        """
        Compute debt accumulation: D += R·dt

        Debt accumulates proportionally to residual over time.
        """
        return residual * self.dt

    def next_state(self, x: float, y: float) -> float:
        """
        Compute next state via UFE: x_{k+1} = x_k - dt·C(x_k, y_k)

        This represents flow toward the coherence manifold.
        """
        c = self.coherence_functional(x, y)
        return x - self.dt * c

    def step(self, x: float, y: float) -> Tuple[float, Dict[str, float]]:
        """
        Perform one UFE step.

        Returns:
            Tuple of (new_state, metrics_dict)
        """
        self.step_count += 1

        residual = self.residual(x, y)
        debt = self.debt_functional(residual)
        next_x = self.next_state(x, y)

        metrics = {
            'residual': residual,
            'debt': debt,
            'coherence': self.coherence_functional(x, y),
            'state': next_x,
        }

        return next_x, metrics


class GatePredicate:
    """
    Gate predicates that control flow through the UFE.

    These predicates determine when the system can proceed
    and when it must wait or halt.
    """

    def __init__(self, residual_threshold: float = 0.01, debt_limit: float = 1.0):
        """
        Initialize gate predicates.

        Args:
            residual_threshold: Max residual for gate to pass
            debt_limit: Max accumulated debt before halt
        """
        self.residual_threshold = residual_threshold
        self.debt_limit = debt_limit
        self.accumulated_debt = 0.0

    def affordability_gate(self, residual: float, coherence: float) -> Dict[str, Any]:
        """
        Affordability gate: checks if residual is affordable.

        Predicate: residual < threshold AND coherence > 0
        """
        affordable = residual < self.residual_threshold and coherence > 0
        return {
            'gate': 'affordability',
            'condition': f'residual ({residual:.4f}) < threshold ({self.residual_threshold}) AND coherence > 0',
            'passed': affordable,
        }

    def solvency_gate(self, total_debt: float) -> Dict[str, Any]:
        """
        Solvency gate: checks if debt is within limits.

        Predicate: accumulated_debt < debt_limit
        """
        solvent = total_debt < self.debt_limit
        return {
            'gate': 'solvency',
            'condition': f'total_debt ({total_debt:.4f}) < limit ({self.debt_limit})',
            'passed': solvent,
        }

    def coherence_gate(self, residual: float) -> Dict[str, Any]:
        """
        Coherence gate: checks if system is cohering.

        Predicate: residual is decreasing (not checked in single step)
        """
        return {
            'gate': 'coherence',
            'condition': f'residual ({residual:.4f}) > 0 (system evolving)',
            'passed': residual > 0,
        }

    def all_gates(self, residual: float, coherence: float, total_debt: float) -> List[Dict[str, Any]]:
        """
        Evaluate all gate predicates.

        Returns:
            List of gate results
        """
        return [
            self.affordability_gate(residual, coherence),
            self.solvency_gate(total_debt),
            self.coherence_gate(residual),
        ]


def compute_hash(data: Dict[str, Any]) -> str:
    """
    Compute SHA-256 hash of receipt data for tamper-evidence.
    """
    json_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode()).hexdigest()


def generate_receipt(
    ufe: ToyUFE,
    gate: GatePredicate,
    initial_x: float,
    target_y: float,
    steps: int,
    metrics: Dict[str, float],
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Generate a minimal receipt JSON for the UFE computation.
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    # Evaluate all gate predicates
    gate_results = gate.all_gates(
        metrics['residual'],
        metrics['coherence'],
        metrics['debt']
    )

    all_passed = all(g['passed'] for g in gate_results)

    receipt = {
        'receipt_version': '1.0',
        'timestamp': timestamp,
        'ufe_parameters': {
            'lambda': ufe.lambda_param,
            'dt': ufe.dt,
            'step_count': ufe.step_count,
        },
        'initial_state': initial_x,
        'target_state': target_y,
        'final_metrics': {
            'residual': round(metrics['residual'], 6),
            'debt': round(metrics['debt'], 6),
            'coherence': round(metrics['coherence'], 6),
            'state': round(metrics['state'], 6),
        },
        'gate_evaluation': {
            'all_passed': all_passed,
            'gates': gate_results,
        },
        'certificate': {
            'type': 'toy_ufe_coherence',
            'hash': '',  # Will be filled below
        }
    }

    # Add hash certificate
    receipt['certificate']['hash'] = compute_hash(receipt)

    if verbose:
        print("=" * 60)
        print("Toy UFE Computation Receipt")
        print("=" * 60)
        print(f"Parameters: λ={ufe.lambda_param}, dt={ufe.dt}, steps={steps}")
        print(f"Initial State: x₀ = {initial_x}")
        print(f"Target State: y = {target_y}")
        print(f"Final State: x_{steps} = {metrics['state']:.6f}")
        print(f"Residual: {metrics['residual']:.6f}")
        print(f"Debt: {metrics['debt']:.6f}")
        print(f"Coherence: {metrics['coherence']:.6f}")
        print("-" * 60)
        print("Gate Predicates:")
        for g in gate_results:
            status = "✓ PASS" if g['passed'] else "✗ FAIL"
            print(f"  [{status}] {g['gate']}: {g['condition']}")
        print("-" * 60)
        print(f"Certificate Hash: {receipt['certificate']['hash'][:16]}...")
        print("=" * 60)

    return receipt


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Explore Toy Unified Flow Equation (UFE)'
    )
    parser.add_argument(
        '--steps', '-n',
        type=int,
        default=5,
        help='Number of UFE steps to run (default: 5)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed output'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file for receipt JSON (default: stdout)'
    )

    args = parser.parse_args()

    # Initialize toy UFE and gate predicates
    ufe = ToyUFE(lambda_param=0.5, dt=0.1)
    gate = GatePredicate(residual_threshold=0.1, debt_limit=10.0)

    # Initial conditions
    x = 1.0  # Initial state
    y = 0.0  # Target state (equilibrium)

    if args.verbose:
        print(f"Initializing Toy UFE with λ={ufe.lambda_param}, dt={ufe.dt}")
        print(f"Running {args.steps} steps from x={x} to y={y}")

    # Run UFE for specified steps
    for step in range(args.steps):
        x, metrics = ufe.step(x, y)
        if args.verbose:
            print(f"Step {step + 1}: x={x:.6f}, residual={metrics['residual']:.6f}")

    # Generate receipt
    receipt = generate_receipt(ufe, gate, 1.0, y, args.steps, metrics, args.verbose)

    # Output receipt
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(receipt, f, indent=2)
        print(f"Receipt written to {args.output}")
    else:
        print(json.dumps(receipt, indent=2))

    return 0


if __name__ == '__main__':
    exit(main())
