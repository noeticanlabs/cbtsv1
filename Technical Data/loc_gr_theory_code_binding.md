# LoC/GR Theory-Code Binding

**Status:** Active
**Date:** 2026-01-24
**Scope:** Mapping PhaseLoom thresholds to LoC Principles.

## 1. Threshold Definitions (PhaseLoom)

The `PhaseLoom27` governor enforces three primary gating thresholds defined in `DEFAULT_THRESHOLDS` (see `phaseloom_27.py`):

| Domain | Threshold ($\theta$) | Code Symbol | Meaning |
| :--- | :--- | :--- | :--- |
| **SEM** | `0.0` | `DEFAULT_THRESHOLDS['SEM']` | **Semantic Barrier.** Hard logical or physical invalidity (e.g., causality violation, NaN, negative lapse). |
| **CONS** | `1.0e-6` | `DEFAULT_THRESHOLDS['CONS']` | **Constraint Manifold.** Maximum allowable deviation from $\mathcal{H}=0, \mathcal{M}=0$. |
| **PHY** | `1.0e-4` | `DEFAULT_THRESHOLDS['PHY']` | **Physical Fidelity.** Discretization error tolerance for evolution equations. |

## 2. Theoretical Binding (LoC Principles)

This section maps the runtime thresholds to the theoretical lemmas established in `loc_principle_v1_0.md`.

### 2.1 SEM Binding: LoC-1 & LoC-6
*   **Theory:**
    *   **LoC-1 (Tangency/Invariance):** The state must remain within the domain of definition of the evolution operator.
    *   **LoC-6 (Representation Coherence):** The mapping between mathematical state and machine representation must be valid (no NaNs, valid types).
*   **Implementation:**
    *   `Gate_step` checks `r[SEM] > 0.0`.
    *   Any violation implies the state $\Psi \notin \text{Dom}(F)$, requiring immediate rollback.

### 2.2 CONS Binding: LoC-3 & LoC-4
*   **Theory:**
    *   **LoC-3 (Damped Coherence):** Dynamics must include damping $\lambda K(\Psi)$ to suppress constraint drift.
    *   **LoC-4 (Witness Inequality):** The discrete residual $\varepsilon^n$ must be bounded to guarantee global stability.
*   **Implementation:**
    *   `Gate_step` checks `r[CONS] <= 1e-6`.
    *   This defines the "thickness" of the numerical constraint manifold $\mathcal{M}_\epsilon$.
    *   Violations trigger **Rails** (damping increase) or **Gates** (rollback) to force the trajectory back towards $\mathcal{M}$.

### 2.3 PHY Binding: LoC-5
*   **Theory:**
    *   **LoC-5 (Clock Coherence):** The time step $\Delta t$ must respect the fastest timescale $\tau(\Psi)$ to ensure the discrete operator $\mathcal{E}_{\Delta t}$ approximates the continuous flow $F$.
*   **Implementation:**
    *   `Gate_step` checks `r[PHY] <= 1e-4`.
    *   This monitors the local truncation error or high-frequency noise (chatter).
    *   Violations drive $\Delta t$ reduction via the **Arbitrator**.

## 3. Operational Verification

The binding is verified if **GCAT-GR-1A** (Discrete Defect) and **GCAT-0.5** (Lie Detector) pass using these thresholds.

*   **GCAT-GR-1A:** Proves `r[CONS]` decreases when damping is active (LoC-3).
*   **GCAT-0.5:** Proves `r[SEM]` triggers on causality violations (LoC-1).

---
**Reference:**
- `phaseloom_27.py`: Implementation of thresholds.
- `loc_principle_v1_0.md`: Definition of LoC lemmas.