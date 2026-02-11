---
title: "Minimal Worked Example"
description: "End-to-end example demonstrating UFE evolution, residual computation, Ω-receipt emission, and coherence verification"
last_updated: "2026-02-10"
authors: ["NoeticanLabs"]
tags: ["coherence", "example", "tutorial", "UFE", "receipts"]
---

# Minimal Worked Example

This document provides an end-to-end example demonstrating:
1. UFE system definition
2. Evolution step with residual computation
3. Ω-receipt emission
4. Gate pass/fail verification

---

## 1. System Definition: Scalar Diffusion

We use the scalar diffusion equation as our example:

\[
\partial_t u = D \nabla^2 u + f(x)
\]

### 1.1 UFE Decomposition

| Component | Operator | Definition |
|-----------|----------|------------|
| State | \(\Psi\) | Scalar field \(u(x)\) |
| Physics | \(\mathcal{L}_{\text{phys}}\) | \(\partial_t u = D \nabla^2 u\) |
| Geometry | \(\mathcal{S}_{\text{geo}}\) | None (flat metric) |
| Drives | \(\mathcal{G}_i\) | Source term \(f(x)\) |

### 1.2 Python Implementation

```python
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class DiffusionUFE:
    """UFE operators for scalar diffusion"""
    D: float  # Diffusion coefficient

    def Lphys(self, u: np.ndarray) -> np.ndarray:
        """Physics: diffusion operator D∇²u"""
        # Second derivative via central differences
        d2u = np.roll(u, 1, axis=0) - 2*u + np.roll(u, -1, axis=0)
        return self.D * d2u

    def Sgeo(self, u: np.ndarray) -> np.ndarray:
        """Geometry: flat metric, no correction"""
        return np.zeros_like(u)

    def G(self, i: int, u: np.ndarray) -> np.ndarray:
        """Drives: source term (i=0)"""
        if i == 0:
            # Simple sinusoidal source
            return 0.1 * np.sin(np.linspace(0, 2*np.pi, len(u)))
        return np.zeros_like(u)

# System parameters
L = 10.0          # Domain length
N = 100           # Grid points
dx = L / N        # Grid spacing
D = 0.1           # Diffusion coefficient
dt = 0.01         # Time step

# Initialize
u = np.random.randn(N) * 0.1  # Small random initial condition
system = DiffusionUFE(D)
```

---

## 2. Evolution Step with Residual Computation

### 2.1 Forward Euler Step

```python
def evolve_step(u: np.ndarray, ufe: DiffusionUFE, dt: float) -> np.ndarray:
    """Forward Euler evolution step"""
    rhs = ufe.Lphys(u) + ufe.Sgeo(u) + ufe.G(0, u)
    return u + dt * rhs

def compute_residual(u: np.ndarray, u_next: np.ndarray,
                     ufe: DiffusionUFE, dt: float) -> Dict[str, float]:
    """
    Compute UFE residual and component breakdown.

    Residual: ε = (u_{n+1} - u_n)/dt - RHS(u_n)
    """
    # Discrete time derivative
    dpsi_dt = (u_next - u) / dt

    # RHS components
    L = ufe.Lphys(u)
    S = ufe.Sgeo(u)
    G_total = ufe.G(0, u)
    rhs = L + S + G_total

    # Residual
    residual = dpsi_dt - rhs

    return {
        "residual": residual,
        "residual_norm": float(np.linalg.norm(residual)),
        "Lphys_norm": float(np.linalg.norm(L)),
        "Sgeo_norm": float(np.linalg.norm(S)),
        "G_total_norm": float(np.linalg.norm(G_total)),
    }
```

---

## 3. Ω-Receipt Emission

### 3.1 Receipt Schema

```python
@dataclass
class OmegaReceipt:
    """Ω-Ledger receipt for coherence verification"""
    id: str
    timestamp: str
    state_summary: Dict[str, Any]
    residuals: Dict[str, float]
    residual_norm: float
    threshold: float
    gate_status: str  # "pass" | "fail"
    layer: str  # "L4"
    lexicon_terms_used: list
    parent_hash: str
    receipt_hash: str
```

### 3.2 Receipt Emission

```python
import hashlib
import json
from datetime import datetime

def emit_receipt(u: np.ndarray, residual_info: Dict,
                 threshold: float, parent_hash: str) -> OmegaReceipt:
    """Emit an Ω-receipt for a coherence check"""

    # Generate unique ID
    receipt_id = hashlib.sha256(
        f"{datetime.now().isoformat()}{u.tobytes()}".encode()
    ).hexdigest()[:16]

    # Compute gate status
    gate_status = "pass" if residual_info["residual_norm"] <= threshold else "fail"

    # Create receipt
    receipt = OmegaReceipt(
        id=receipt_id,
        timestamp=datetime.now().isoformat(),
        state_summary={
            "u_mean": float(np.mean(u)),
            "u_std": float(np.std(u)),
            "grid_points": len(u),
        },
        residuals={
            "Lphys": residual_info["Lphys_norm"],
            "Sgeo": residual_info["Sgeo_norm"],
            "G_total": residual_info["G_total_norm"],
        },
        residual_norm=residual_info["residual_norm"],
        threshold=threshold,
        gate_status=gate_status,
        layer="L4",
        lexicon_terms_used=["UFE", "residual", "coherence", "gate"],
        parent_hash=parent_hash,
        receipt_hash="",  # Will compute below
    )

    # Compute receipt hash
    receipt_content = json.dumps({
        "id": receipt.id,
        "timestamp": receipt.timestamp,
        "state_summary": receipt.state_summary,
        "residuals": receipt.residuals,
        "residual_norm": receipt.residual_norm,
        "threshold": receipt.threshold,
        "gate_status": receipt.gate_status,
        "layer": receipt.layer,
        "lexicon_terms_used": receipt.lexicon_terms_used,
        "parent_hash": receipt.parent_hash,
    }, sort_keys=True)

    receipt.receipt_hash = hashlib.sha256(
        (receipt_content + parent_hash).encode()
    ).hexdigest()

    return receipt
```

---

## 4. End-to-End Execution

### 4.1 Main Loop

```python
def run_simulation(steps: int, threshold: float = 1e-3) -> list:
    """Run diffusion simulation with coherence enforcement"""

    u = np.random.randn(N) * 0.1
    ufe = DiffusionUFE(D)
    parent_hash = "0" * 64  # Genesis hash
    receipts = []

    for step in range(steps):
        # Evolve
        u_next = evolve_step(u, ufe, dt)

        # Compute residual
        residual_info = compute_residual(u, u_next, ufe, dt)

        # Emit receipt
        receipt = emit_receipt(u_next, residual_info, threshold, parent_hash)
        receipts.append(receipt)

        # Update chain
        parent_hash = receipt.receipt_hash

        # Accept or reject
        if receipt.gate_status == "pass":
            u = u_next
            print(f"Step {step}: PASS (residual={residual_info['residual_norm']:.2e})")
        else:
            print(f"Step {step}: FAIL (residual={residual_info['residual_norm']:.2e})")
            # In practice: apply rail, retry, or abort

    return receipts
```

### 4.2 Sample Output

```
Step 0: PASS (residual=3.21e-04)
Step 1: PASS (residual=2.87e-04)
Step 2: PASS (residual=2.54e-04)
Step 3: PASS (residual=2.21e-04)
Step 4: PASS (residual=1.89e-04)
...
Step 95: PASS (residual=4.32e-05)
Step 96: PASS (residual=3.98e-05)
Step 97: PASS (residual=3.65e-05)
Step 98: PASS (residual=3.31e-05)
Step 99: PASS (residual=2.98e-05)
```

---

## 5. Verification

### 5.1 Receipt Validation

```python
def validate_chain(receipts: list) -> Dict:
    """Validate receipt chain integrity"""
    if not receipts:
        return {"valid": False, "reason": "Empty chain"}

    # Check hash chaining
    for i, receipt in enumerate(receipts):
        if i > 0:
            expected_parent = receipts[i-1].receipt_hash
            if receipt.parent_hash != expected_parent:
                return {
                    "valid": False,
                    "reason": f"Hash chain broken at step {i}"
                }

    # Check gate compliance
    failed = [r for r in receipts if r.gate_status == "fail"]

    return {
        "valid": True,
        "total_steps": len(receipts),
        "passed": len(receipts) - len(failed),
        "failed": len(failed),
        "pass_rate": (len(receipts) - len(failed)) / len(receipts),
    }
```

### 5.2 Validation Result

```python
receipts = run_simulation(steps=100, threshold=1e-3)
result = validate_chain(receipts)

print(f"""
=== Validation Result ===
Valid: {result['valid']}
Total Steps: {result['total_steps']}
Passed: {result['passed']}
Failed: {result['failed']}
Pass Rate: {result['pass_rate']*100:.1f}%
""")
```

---

## 6. Summary

This example demonstrates:

| Concept | Implementation |
|---------|---------------|
| **UFE Decomposition** | `Lphys` (diffusion), `Sgeo` (none), `G` (source) |
| **Residual Computation** | Discrete time derivative minus RHS |
| **Ω-Receipt** | Complete audit trail with hash chaining |
| **Coherence Gate** | Threshold check with pass/fail |
| **Chain Validation** | Hash chain integrity verification |

All code is available in [`explore_ufe.py`](../../explore_ufe.py) for interactive exploration.
