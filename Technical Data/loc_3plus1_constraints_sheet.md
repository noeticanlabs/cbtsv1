# Solver Spec: LoC-GR as constraints + clocks + gates + Ω-receipts

**Status:** Canon Extension
**Theme:** Turning "LoC ⇒ GR" into a CPU-safe machine
**Context:** Concrete implementation details for the 3+1 GR solver.

---

## 0) Canonical goal

Design a 3+1 GR evolution such that:

1.  **Physical evolution** advances stably.
2.  **Constraint energy** stays bounded / decays:
    $$
    \mathcal C(t):=|\mathcal H(t)|_2^2 + |\mathcal M(t)|_2^2 + |\mathcal G(t)|_2^2
    $$
    ($\mathcal G$ = any formulation-specific auxiliary constraint, e.g. BSSN $\tilde\Gamma^i$ constraint, Z4 fields, etc.)
3.  Every step emits an auditable **Ω-receipt**.

---

## 1) State vector (BSSN-ish, explicit)

Pick a concrete representation (BSSN is a good CPU baseline):

$$
\Psi :=
\big(\phi,\ \tilde\gamma_{ij},\ K,\ \tilde A_{ij},\ \tilde\Gamma^i,\ \alpha,\ \beta^i,\ B^i,\ \Phi_{\text{matter}}\big)
$$

Constraints (minimum set you must compute every step):

*   $\mathcal H$ (Hamiltonian)
*   $\mathcal M^i$ (Momentum)
*   $\det(\tilde\gamma_{ij})-1=0$ (conformal metric constraint)
*   $\tilde A^i{}_i=0$ (tracefree constraint)
*   $\mathcal G^i := \tilde\Gamma^i - \tilde\gamma^{jk}\tilde\Gamma^i{}_{jk}=0$ (Gamma constraint, formulation dependent)

---

## 2) UFE/LoC evolution wrapper (the one line that matters)

Write evolution as:
$$
\dot\Psi = B(\Psi) + \lambda K(\Psi).
$$

*   $B(\Psi)$: baseline BSSN + gauge driver equations.
*   $K(\Psi)$: **coherence operator** that targets constraint growth and operator mismatch.
*   $\lambda$: coherence gain (clock-aware; bounded).

### 2.1 Coherence operator ($K$) (no hand-waving: define what it does)

Use a **three-stage** enforcement that's CPU cheap:

**(K1) Constraint damping injection (continuous-like)**
Add terms to the evolution RHS proportional to constraints, e.g.
$$
\partial_t K \leftarrow \partial_t K - \lambda_H \alpha \mathcal H
$$
$$
\partial_t \tilde\Gamma^i \leftarrow \partial_t \tilde\Gamma^i - \lambda_M \alpha \tilde\gamma^{ij}\mathcal M_j
$$
and similarly for $\mathcal G^i$ if used.
(Exact placement depends on your chosen formulation; the LoC requirement is: the added terms must be **constraint-consistent** and **dimensionally compatible**.)

**(K2) Projection / renormalization (discrete exactness)**
After updating:

*   enforce $\det(\tilde\gamma)=1$ by rescaling $\tilde\gamma_{ij}$,
*   enforce $\tilde A^i{}_i=0$ by trace subtraction,
*   optionally clamp gauge variables if they exceed stability envelopes.

**(K3) Spectral/LP filter on incoherent high-k tails**
Apply a conservative filter to the highest bands (your Route-style tail barrier logic), but only on variables known to tolerate it (often $\tilde A_{ij}$, $\tilde\Gamma^i$, sometimes gauge drivers).
Filter strength is tied to the Aeonic clock margin (below).

That's the concrete LoC lever set: **damp + project + de-spike**.

---

## 3) Residuals and norms (what gets measured)

Compute these each step (grid spacing $h$, volume $V$):

*   RMS Hamiltonian:
    $$
    \varepsilon_H := \left(\frac{1}{V}\int \mathcal H^2 dV\right)^{1/2}
    $$
*   RMS Momentum:
    $$
    \varepsilon_M := \left(\frac{1}{V}\int \gamma_{ij}\mathcal M^i\mathcal M^j dV\right)^{1/2}
    $$
*   Optional $L_\infty$ versions for "spike detection":
    $$
    \varepsilon_{H,\infty} := |\mathcal H|_\infty,\quad
    \varepsilon_{M,\infty} := |\mathcal M|_\infty
    $$

**Constraint energy:**
$$
\mathcal C := \varepsilon_H^2 + \varepsilon_M^2 + \varepsilon_G^2.
$$

Default **target** (canon-grade CPU test threshold):

*   pass if late-window median satisfies
    $\operatorname{median}(\varepsilon_H) \le 10^{-6}$ and $\operatorname{median}(\varepsilon_M)\le 10^{-6}$
    (scale this with resolution if you prefer a strict convergence gate; see GCAT below).

---

## 4) Aeonic clock rule (GR-appropriate, explicit)

Use the stiffest of these clocks:

### 4.1 CFL / wave clock

GR characteristic speeds (in units $c=1$) are roughly $\alpha \pm \beta$ relative to the grid. A safe CFL step is:
$$
\Delta t_{\text{CFL}} = C_{\text{CFL}}\min_{\text{grid}}\frac{h}{\alpha + |\beta|}.
$$

### 4.2 Gauge driver clock (if using Gamma-driver shift)

If $B^i$ has damping $\eta$, use
$$
\Delta t_{\text{gauge}} = C_g \eta^{-1}.
$$

### 4.3 Coherence / enforcement clock

Tie the enforcement intensity to how fast constraints are changing:
$$
\Delta t_{\text{coh}} = C_c \frac{\mathcal C}{|\dot{\mathcal C}|+\epsilon}
$$
(with a tiny $\epsilon$ to avoid division by zero; choose $\epsilon=10^{-30}$ in float64).

### 4.4 Final Aeonic step

$$
\Delta t = \min(\Delta t_{\text{CFL}},\Delta t_{\text{gauge}},\Delta t_{\text{coh}},\Delta t_{\max}).
$$

Canonical safe constants (good CPU defaults):

*   $C_{\text{CFL}}=0.25$
*   $C_g=0.5$
*   $C_c=0.25$
*   $\Delta t_{\max}$ chosen by scenario (e.g. $10^{-2}$ in code units)

---

## 5) Gates (LoC pass/fail logic)

### 5.1 Late-window gate

Define a late window fraction ($w=0.2$): use the final 20% of steps to judge coherence (avoids startup transients).

Pass if:

*   $\text{median}_{\text{late}}(\varepsilon_H) \le \tau_H$
*   $\text{median}_{\text{late}}(\varepsilon_M) \le \tau_M$
*   no spike: $\varepsilon_{H,\infty}$ and $\varepsilon_{M,\infty}$ below spike thresholds (e.g. $10^3\tau$).

### 5.2 Convergence gate (GCAT style)

For spatial refinement ($h, h/2, h/4$):
$$
p_{\text{obs}} = \log_2\left(\frac{E(h)}{E(h/2)}\right),
$$
where $E(h)$ is the late-window median of $\varepsilon_H+\varepsilon_M$.

Pass if $p_{\text{obs}}\ge p_{\min}$. Canon default: $p_{\min}=1.5$ for "is it really converging?" sanity; stricter if you want.

---

## 6) GCAT tests (the ones you keep rerunning forever)

### GCAT-GR-1A: Discrete defect injection (constraint damping reality check)

1.  Initialize with constraint-satisfying data (or as close as your initializer allows).
2.  Inject a localized constraint defect:
    $$
    K \leftarrow K + \delta \exp(-|x-x_0|^2/\sigma^2)
    $$
    (or perturb $\tilde A_{ij}$ similarly), chosen so $\varepsilon_H$ jumps by a known amount.
3.  Run with $K(\Psi)$ on.
4.  Require $\mathcal C(t)$ decays (or at least does not grow) and that the decay rate increases monotonically with $\lambda_H,\lambda_M$ in a stable range.

**Pass condition (canon):**
$$
\mathcal C(t_{\text{end}}) \le 0.1 \mathcal C(t_{\text{inj}})
$$
for a fixed duration, without destabilizing the main fields.

### GCAT-GR-1B: Operator-only convergence (no physics excuses)

Hold physics scenario fixed, sweep resolution and timestep under the Aeonic clock, and require the measured order is consistent with your discretization.

This test detects your classic failure mode: "errors flat vs dt" (operator mismatch or sources not staged), and it's pure LoC: **if your operators don't compose coherently, refinement won't help.**

---

## 7) Ω-receipt schema (minimum viable audit record)

Emit one JSON object per step (JSONL). Required fields:

*   `run_id`, `step`, `t`, `dt`
*   `grid`: `{Nx,Ny,Nz,h,domain,periodic}`
*   `gauge`: `{alpha_min, alpha_max, beta_max, eta}`
*   `constraints`: `{eps_H, eps_M, eps_G, eps_H_inf, eps_M_inf, C}`
*   `clocks`: `{dt_CFL, dt_gauge, dt_coh, dt_used}`
*   `gates`: `{late_window:false/true, pass_H:true/false, pass_M:true/false, spike:false/true}`
*   `lambda`: `{lambda_H, lambda_M, lambda_G}`
*   `projection`: `{det_tilde_gamma_err, tr_tildeA_err}`
*   `hashes`: `{state_hash, rhs_hash}` (even a simple deterministic hash is fine at first)

That's enough to make failures diagnosable instead of mystical.

---

## 8) Connection

*   **Part I** says: GR is the **unique minimal divergence-free geometric closure** compatible with matter ledger closure.
*   **Part II** says: your solver is coherent only if it preserves that closure **numerically** via constraints, clocks, and receipts.

In other words: **LoC → GR in theory, and constraints → LoC in practice.**