# Aeonic Memory + PhaseLoom

## Canon Spec v1.0 — Parts Defined So Far (0–4 + partial 5/6)

---

## Section 0 — Executive Abstract

### 0.1 What problem this subsystem solves

Modern multi-physics solvers (UFE/RFE/GR/NR) fail in two recurring ways:

1. **Representation incoherence**
   The code claims it is evolving
   $$
   \dot X = \mathcal F(X,t) + S(X,t),
   $$
   but evaluates mismatched components (wrong RK stage time, inconsistent gauge/enforcement timing, filtering applied at the wrong layer, etc.). This produces false symptoms: no convergence, precision floors, mysterious blowups, and “speedups” that are actually cheating.

2. **Unverifiable speed claims**
   Optimization reduces wall time but changes effective dynamics (more constraint enforcement, more filtering, different dominance behavior). Without an audit spine, “faster” can mean “more coercive,” not “more efficient.”

**Aeonic Memory + PhaseLoom fixes both** by forcing every mutating computation to be:

* **clock-stamped** (who did what, when),
* **scale-resolved** (octave tiers L/M/H),
* **domain-resolved** (PHY/CONS/SEM),
* **audit-receipted** (M_solve / M_step / M_orch),
* and tracked as a **trajectory** through a fixed lattice (PhaseLoom).

---

### 0.2 Core objects (minimal formalism)

State ($X(t)$) is a vector of fields (e.g., GR: $(\tilde\gamma_{ij},\tilde A_{ij},\phi,K,\tilde\Gamma^i,\alpha,\beta^i,\dots)$).

Intended method-of-lines evolution:
$$
\frac{dX}{dt} = \mathcal F(X,t) + S(X,t),
$$
where ($S$) may be 0 (physical run) or nonzero (MMS, control, glyph drive).

**PhaseLoom does not change ($\mathcal F$)**. It wraps runtime with:

* residual sensing ($r$),
* dt caps ($\Delta t_{\text{cap}}$),
* rail actions ($a$),
* promotion policies ($P$) for memory.

---

### 0.3 “3 clocks + 3 memory sets + 1 trajectory”

**Clocks (responsibility boundaries):**

* solve clock ($\mu$): RK stages, IMEX iterations, subcycling
* stepper clock ($s$): step attempt/accept/reject + dt negotiation + rollback
* orchestrator clock ($o$): windows/slabs + promotion + regime labels + budgets

Clock stamp:
$$
\kappa=(o,s,\mu), \quad \text{plus } (t,\tau),
$$
where:

* ($t$) = physical PDE time
* ($\tau$) = coherence/audit time (monotone even under rollback in ($t$))

**Memory sets:**

* ($M_{\text{solve}}$): ring buffer of stage receipts (fast overwriteable)
* ($M_{\text{step}}$): accepted step receipts (persistent run history)
* ($M_{\text{orch}}$): window aggregates + canon promotions (earned truth)

**Trajectory:**
$$
\mathcal T = \{\Pi(\kappa)\}_{\kappa\in \text{sampled indices}}
$$
where ($\Pi(\kappa)$) is a PhaseLoom “point” (dominant thread, residual tiers, dt caps, actions, perf counters).

---

### 0.4 What PhaseLoom is

PhaseLoom is a fixed **27-thread** sensor-and-governor lattice; structure is invariant across solvers.

Thread indices:

* domain ($D\in\{\text{PHY, CONS, SEM}\}$)
* scale tier ($S\in\{\text{L, M, H}\}$)
* response tier ($C\in\{\text{FAST, MID, SLOW}\}$)

So threads:
$$
T[D,S,C],\quad 3\times 3\times 3=27.
$$

Each thread emits:

* residual scalar ($r[D,S]$) (proxy at solve clock allowed),
* dt cap ($\Delta t_{\text{cap}}[D,S,C]$),
* optional action suggestion ($a[D,S,C]$).

---

### 0.5 Falsifiable outputs

Subsystem works only if it can emit receipts certifying:

1. **Coherence correctness**
   Stage/step/orch time representations agree.
   Example invariant: time-dependent sources are evaluated at RK stage times ($t^{(\mu)}$); `policy_hash` constant within a window unless versioned.

2. **Performance truth**
   Speedups correspond to stable or improved PhaseLoom trajectory, not hidden coercion.
   Example: if runtime drops but enforcement magnitude rises, label the speedup as coercion-driven.

---

---

## Section 1 — Design Axioms (Non-negotiables)

### A1 — Clock-stamped causality

Every mutating computation must be attributable to ($\kappa=(o,s,\mu)$) and ($(t,\tau)$).

### A2 — No canon truth from solve memory

Nothing in ($M_{\text{solve}}$) can be promoted directly:
$$
P(M_{\text{solve}})=\varnothing.
$$

### A3 — Accepted history is the only statistical substrate

Orchestrator aggregates use **accepted steps only**. Rejected attempts may be stored, but excluded by default.

### A4 — SEM violations are hard barriers

SEM failures are invalid states → immediate FAST response (rollback/reject) + window classification.

### A5 — Scale awareness is mandatory

PHY and CONS residuals must be tiered L/M/H (even approximately at solve-clock).

### A6 — Performance is part of coherence

Perf-risk is a real regime label; promotion can be blocked by performance regression.

### A7 — Conservation vs enforcement are not conflated

Track diagnostic constraint residual ($r_{\text{diag}}$) separately from enforcement magnitude ($e_{\text{enf}}$).

### A8 — Deterministic policy fingerprinting

Each window records `policy_hash` from gauge/enforcement/filter/dt arbitration/receipt cadence. Mid-window changes require version bump or trigger SEM.

---

---

## Section 2 — Clock Architecture (Solve / Stepper / Orchestrator)

### 2.1 Formal clock model

Indices:

* ($o\in\mathbb N$): orchestrator windows
* ($s\in\mathbb N$): accepted steps within ($o$)
* ($\mu\in\mathbb N$): micro-ops within step attempt

Stamp: $\kappa=(o,s,\mu)$

Times:

* physical ($t$)
* coherence ($\tau$) (must be monotone even if rollback decreases ($t$))

Rollback rule: rollback may decrease ($t$) by reverting state, but may not decrease ($\tau$). Rollback is recorded as an event in ($\tau$).

---

### 2.2 Solve clock (µ): responsibilities & allowed operations

Example RK4 stage times:
$$
t^{(\mu)}=t_n + c_\mu \Delta t,\quad c=(0,\tfrac12,\tfrac12,1).
$$

Allowed at µ:

* evaluate staged RHS:
  $$
  k_\mu=\mathcal F(X^{(\mu)},t^{(\mu)}) + S(X^{(\mu)},t^{(\mu)})
  $$
* update temporary stage states
* compute proxy residuals ($r^{\text{proxy}}[D,S]$)
* compute dt caps for FAST/MID tiers
* apply emergency rails: shrink dt, abort stage, clamp SEM-illegal values (policy-authorized)

Forbidden at µ:

* canon promotion
* expensive global transforms unless cached
* slow governance actions (labels/promotion)

---

### 2.3 Stepper clock (s): acceptance authority

Stepper attempts a step with candidate dt:

* run solve-clock to produce ($X_{n+1}^{\text{cand}}$)
* compute full residuals + constraints + enforcement magnitudes
* evaluate `Gate_step`

If reject:

* rollback to ($X_n$)
* shrink dt by factor ($\eta\in(0,1)$)
* increment rollback counter and retry

If accept:

* commit ($X_{n+1}$), advance ($t\leftarrow t+\Delta t$)
* increment accepted step count ($s$)
* write step receipt into ($M_{\text{step}}$)

---

### 2.4 Orchestrator clock (o): governance & promotion

Orchestrator defines windows by:

* fixed accepted-step count ($W$), or
* fixed coherence-time length ($\Delta\tau_{\text{win}}$)

It aggregates accepted steps only:

* residual quantiles per ($D,S$)
* dominance histogram over 27 threads
* chatter score (dominant switching rate)
* rail activity rates
* performance envelope stats

It produces:

* window regime label
* promotions into ($M_{\text{orch}}$) iff `Gate_orch` passes

---

---

## Section 3 — Memory Architecture (M_solve / M_step / M_orch)

### 3.1 Three memory sets (epistemic separation)

1. ($M_{\text{solve}}$): ephemeral evidence
2. ($M_{\text{step}}$): accepted history
3. ($M_{\text{orch}}$): earned canon

This separation prevents speed optimizations from corrupting truth.

---

### 3.2 M_solve (ring buffer) — minimal and fast

Key: $((o,s,\mu))$

Purpose: within-step diagnostics, rollback reasoning, stage coherence checks, microprofiling.

Required fields (PhaseLoom point):

* stamps: $((o,s,\mu))$, ($t_{\text{stage}}$), ($\tau$)
* dt: dt_candidate, top-k dt caps
* proxies: ($r^{\text{proxy}}[\text{PHY/CONS/SEM}][L/M/H]$)
* dominant thread + margin
* actions applied at solve-clock
* invariants: `policy_hash`, `stage_source_hash`
* perf: stage CPU/wall time, rhs call counts, optional alloc deltas

Hard constraints:

* bounded storage (ring)
* no large arrays; only summaries + hashes

---

### 3.3 M_step (persistent receipts) — authoritative run history

Key: $((o,s))$ for accepted steps; rejected attempts optional (`accepted=false`).

Required fields:

* stamps: $((o,s))$, physical ($t$), coherence ($\tau$)
* dt chosen, accept/reject, rollback count
* full residuals ($r[D,S]$) for all domains and tiers
* constraint diagnostics (system-specific)

  * GR/NR: ($|\mathcal H|_{L2}$), ($|\mathcal M|_{L2}$), algebraic constraints
* enforcement magnitudes ($e_{\text{enf}}$) if applied
* performance counters (step wall/cpu time, call counts, allocations if measured)

This is the dataset used to prove: “same dynamics, less compute.”

---

### 3.4 M_orch (canon) — window aggregates + promotions

Key: ($o$)

Required fields:

* window definition (which steps)
* quantiles p50/p90/p99 of residuals per ($D,S$)
* dominance histogram over 27 threads
* chatter score
* rail activity rates
* performance envelope stats
* regime label
* promotion list (canonized patterns/params)

Canon rule: append-only; corrections are new windows, never rewrites.

---

---

## Section 4 — PhaseLoom Core (27-thread invariant lattice)

### 4.1 Thread indexing (invariant)

Thread:
$$
T_{D,S,C},\quad D\in\{\text{PHY,CONS,SEM}\},\ S\in\{\text{L,M,H}\},\ C\in\{\text{FAST,MID,SLOW}\}.
$$

### 4.2 Thread inputs and outputs

Inputs:

* residual scalar ($r[D,S]$) (proxy at µ, full at s/o)
* stiffness indicator ($\sigma[D,S]$) (optional)
* policy parameters ($\pi$) (thresholds, caps, allowed rails)
* optional performance indicators ($p$)

Outputs:

1. dt cap:
   $$
   \Delta t_{\text{cap}}[D,S,C] = f_{D,S,C}(r[D,S],\sigma[D,S],\pi,p)
   $$
2. action suggestion ($a[D,S,C]$) from a policy-defined action set
3. normalized margin:
   $$
   m[D,S,C] = \frac{r[D,S]}{\theta[D,S,C]}
   $$

### 4.3 Domain meanings (fixed)

* PHY: solver follows declared evolution
* CONS: constraints satisfied + enforcement magnitude tracked separately
* SEM: legality/audit/typing policy compliance

SEM dominates: any SEM hard failure implies immediate reject/abort behavior.

### 4.4 Scale tier meanings (fixed)

Octave tiers:

* L: large-scale structure
* M: transfer/cascade band
* H: high-frequency tail / instability risk

### 4.5 Response tier meanings (fixed)

FAST/MID/SLOW are response profiles:

* FAST: immediate (µ allowed)
* MID: stabilizing (s)
* SLOW: governance (o)

### 4.6 Dominance arbitration

Choose dt by:
$$
\Delta t = \min_{D,S,C}\Delta t_{\text{cap}}[D,S,C]
$$
Dominant thread:
$$
(D^*,S^*,C^*) = \arg\min_{D,S,C}\Delta t_{\text{cap}}[D,S,C]
$$
Chatter score (window):
$$
\text{chatter} = \frac{#\text{dominance switches}}{#\text{accepted steps}}
$$

---

---

## Section 5 — Residual Semantics (Completed)

### 5.1 General Residual Definition & Contracts

**Evolution Defect:**
$$
\varepsilon = \partial_t X - (\mathcal F + S)
$$
**PhaseLoom Measure:**
$$
r[D,S] = |P_S(\varepsilon_D)|_{\mathcal N(D)}
$$

**Contract: Proxy Residuals ($\tilde r$)**
*   **Timing:** Computed at solve-clock ($\mu$).
*   **Cost:** Must be $\ll$ full residual (e.g., sparse sampling, local stencil check, or algebraic invariant).
*   **Validity:** Must strictly upper-bound or strongly correlate ($R^2 > 0.9$) with the full residual.
*   **Usage:** Triggers FAST rails (stage abort) or dt cap shrinking.

**Contract: Full Residuals ($r$)**
*   **Timing:** Computed at stepper-clock ($s$) and orchestrator-clock ($o$).
*   **Cost:** Full grid sweep allowed ($O(N)$).
*   **Validity:** Exact L2 or L-infinity norms of the discretized equations.
*   **Usage:** Acceptance/Rejection (`Gate_step`), Promotion (`Gate_orch`).

### 5.2 PHY Residual Registry (Physical Evolution)

**Definition:** Measures how well the discrete step satisfies the discretized PDE evolution operator.

**GR/NR (3+1 BSSN/Z4):**
*   **Full ($r_{\text{PHY}}$):** Norm of the finite difference residual $\varepsilon_{\text{evol}} = ||\frac{\Psi^{n+1}-\Psi^n}{\Delta t} - \text{RHS}(\Psi^{n+1/2})||$.
*   **Proxy ($\tilde r_{\text{PHY}}$):**
    *   *KO-Dissipation Proxy:* Energy in the highest spectral octave (from `gr_geometry` stencil fusion).
    *   *Stage Defect:* $||\Psi^{(\mu)} - \Psi^{(\text{pred})}||$ in predictor-corrector schemes.

**RFE (Resonance Field):**
*   **Full ($r_{\text{PHY}}$):** $||\partial_t \rho + \nabla \cdot (\rho^2 \nabla \theta) - S_\rho||$.
*   **Proxy ($\tilde r_{\text{PHY}}$):** Local flux divergence at detector points.

### 5.3 CONS Residual Registry (Constraints & Conservation)

**Definition:** Measures violation of algebraic or differential constraints that should be identically zero.

**GR/NR:**
*   **Full ($r_{\text{CONS}}$):**
    *   Hamiltonian: $||\mathcal H||_{L2}$.
    *   Momentum: $||\mathcal M^i||_{L2}$.
    *   Z-damping: $||Z||$ (if Z4).
*   **Proxy ($\tilde r_{\text{CONS}}$):**
    *   *Algebraic:* $||\det(\tilde\gamma) - 1||_{\infty}$ (computed pointwise during evolution).
    *   *Trace:* $||\text{tr}(\tilde A)||_{\infty}$.
    *   *Sampled H:* $\max_{x \in \text{samples}} |\mathcal H(x)|$.

**RFE:**
*   **Full ($r_{\text{CONS}}$):** Coherence ledger imbalance $\mathcal R_C$.
*   **Proxy ($\tilde r_{\text{CONS}}$):** Phase winding number $\oint \nabla \theta \cdot dl$ around plaquettes (singularity detection).

### 5.4 SEM Residual Registry (Semantic/Legality)

**Definition:** Measures violation of invariant properties (positivity, causality, typing).

**GR/NR:**
*   **Hard Barriers:**
    *   Lapse positivity: $\alpha \le 0 \implies r_{\text{SEM}} = \infty$.
    *   Metric determinant: $\det(\gamma) \le 0 \implies r_{\text{SEM}} = \infty$.
    *   Causality: Characteristic speed $> c_{\text{max}} \implies r_{\text{SEM}} = \infty$.

**RFE:**
*   **Hard Barriers:**
    *   Density positivity: $\rho < 0 \implies r_{\text{SEM}} = \infty$.

### 5.5 Normalization & Margins

**Risk Ratio:**
$$
m[D,S,C]=\frac{r[D,S]}{\theta[D,S,C]}
$$
*   $m \le 1$: Safe / Admissible.
*   $m > 1$: Violation.

**Normalization Rules:**
*   **Scale-Relative:** $\theta[D,S]$ scales with $h^p$ (grid spacing) for convergent quantities.
*   **Background-Relative:** $\theta_{\text{CONS}} \propto ||\text{Sources}|| + \epsilon$.
*   **Policy-Fixed:** $\theta_{\text{SEM}}$ is often binary (0 or $\infty$) or fixed by precision ($\epsilon_{\text{mach}}$).

Thresholds are part of `policy_hash`; changing without version bump is SEM violation.

---

## Section 6 — Rails & Gates (started; structure defined)

* rails: corrective actions (µ/s)
* gates: accept/reject (s) and promote/quarantine (o)
* FAST rails: cheap, reversible, logged
* MID rails: policy-authorized stabilization, enforcement magnitude logged
* SLOW rails: regime classification + promotion controls
* Gate_step: SEM hard barrier first, then CONS, then PHY, plus rollback/min dt limits
* Gate_orch: window-level promotion conditions include low residuals, low enforcement magnitude, low SEM chatter, stable dominance, acceptable performance envelope

---

## Summary: What “all parts so far” are

You now have **the full skeleton** of the subsystem:

* **Purpose + falsifiable outputs** (Section 0)
* **Axioms that prevent silent drift** (Section 1)
* **Clock architecture with rollback-safe audit time** (Section 2)
* **Three-tier memory with epistemic separation** (Section 3)
* **PhaseLoom 27-thread invariant lattice + dominance arbitration** (Section 4)
* **Residual semantics framework + domain definitions** (Section 5 partial but structurally complete)
* **Rails & gates framework + acceptance/promotion logic** (Section 6 partial but structurally complete)

---

Next in-order is exactly where the system becomes “provably useful”:

**Section 5 completion** (proxy/full contracts, required norms, GR/NR + RFE-specific residual registries)
**Section 6 completion** (exact legal rail sets per domain, exact Gate_step and Gate_orch inequalities)
Then:

* **Section 7 — PhaseLoom Trajectory (formal metrics, distances, dominance entropy, chatter regimes)**
* **Section 8 — Performance as an Invariant (coercion vs compute speed certificates)**