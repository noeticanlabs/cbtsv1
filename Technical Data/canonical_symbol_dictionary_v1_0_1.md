# Canonical Symbol Dictionary v1.0.1 (Full Updated)

**Scope:** LoC × Noetica × GM-OS × Aeonica/AML × PhaseLoom (Symbolic ↔ Resonant ↔ Field)
**Design rule:** *One symbol → one semantic role.* Variants use **subscripts/superscripts**—never silent redefinition.
**Version note (v1.0 → v1.0.1):**

1. LoC residual uses **δ** (difference), never **Δ**.
2. PhaseLoom response axis canonized to **OBS / CTRL / AUD**.
3. Slab interval uses **ϑ** (vartheta) for slab size; **θ** remains phase.
4. Added **λ** (coherence coupling) + required bounds/receipt hooks.
5. Clarified **Ω** (instance chain) vs **𝔹Ω / 𝕆mega** (ledger system).

---

## 0) Notation conventions (non-negotiable)

### 0.1 Typography = meaning

* **Scalars:** (a,\ \phi,\ \theta,\ \kappa,\ C,\ \lambda)
* **Vectors:** (\mathbf v,\ \mathbf x,\ \mathbf J)
* **Tensors:** (g_{\mu\nu},\ T_{\mu\nu})
* **Operators:** (\nabla,\ \partial_t,\ \Delta) (Laplacian), (\mathcal L,\ \mathcal F)
* **Sets / spaces:** (\mathbb R,\ \mathbb Z,\ \mathbb T^d)
* **Calligraphic letters:** always operators/functionals (\mathcal E,\ \mathcal D,\ \mathcal L,\ \mathcal F)
* **Blackboard / Fraktur:** glyph primitives / semantic atoms (\mathbb G,\ \mathfrak p,\ \mathbb\Omega)

### 0.2 “Same letter” disambiguation policy (locked)

* (\Delta) **means Laplacian** only. “Difference” is **(\delta)** or explicit subtraction.
* (\rho) **means density** (carrier), never correlation coefficient.
* (\phi) **means amplitude/order parameter**; (\theta) **means phase**. Never swap.
* (\Lambda) is **never bare**: must be (\Lambda_{\text{type}}) or (\Lambda_{\text{cosmo}}), etc.
* (\lambda) is **coherence coupling** (this canon). If you need wavelength, write (\lambda_{\text{wave}}).

### 0.3 Canonical code identifiers

Every symbol has a stable identifier used in IR/receipts:

* (\phi \rightarrow) `phi_amp`
* (\theta \rightarrow) `theta_phase`
* (\kappa \rightarrow) `kappa_K`
* (\lambda \rightarrow) `lambda_cpl`
* (C \rightarrow) `C_tot`, etc.

---

## 1) Coherence core (LoC)

### (C) — Coherence (generic)

* **Type:** scalar in ([0,1]) unless stated otherwise
* **Meaning:** compatibility measure of a system’s internal descriptions under evolution
* **Canonical factorization:**
  [
  C_{\text{tot}} = C_{\text{syn}}\cdot C_{\text{sem}}\cdot C_{\text{ph}}
  ]
* **Code:** `C_tot`, `C_syn`, `C_sem`, `C_ph`
* **Constraint:** (C_{\text{tot}}\in[0,1]). If any factor undefined → **fail closed** (no silent defaults).

### (C_{\text{syn}}) — Syntactic coherence

* **Meaning:** grammar/typing/IR well-formedness (parse + typing + SSA invariants)
* **Typical computation:** parser validity, type constraints satisfied, SSA dominance, no missing receipts
* **Constraint:** computable from artifacts alone (no “human intuition” inputs)

### (C_{\text{sem}}) — Semantic coherence

* **Meaning:** meaning preservation under transforms (rewrites, lowering, scheduling)
* **Typical computation:** equivalence checks, interpreter replay agreement, contract assertions

### (C_{\text{ph}}) — Physical coherence

* **Meaning:** PDE/physics invariants consistent (constraints, conservation, stability margins)
* **Typical computation:** residual norms, constraint violations, energy drift bounds

---

### (\varepsilon_{\text{LoC}}) — LoC residual / mismatch witness

* **Type:** scalar (\ge 0)
* **Meaning:** evidence of incoherence (predicted vs observed mismatch)
* **Canonical (ledger form, Δ-free):**
  [
  \boxed{\varepsilon_{\text{LoC}}(t)=\left|\delta X_{\text{obs}}(t)-\delta X_{\text{pred}}(t)\right|*{\mathcal N}}
  ]
  with
  [
  \delta X*{\text{obs}} := X_{\text{obs}}(t+\Delta t)-X_{\text{obs}}(t),\qquad
  \delta X_{\text{pred}} := \int_t^{t+\Delta t}\mathcal F(X(s),s),ds
  ]
  and (|\cdot|_{\mathcal N}) an explicitly declared norm (RMS/L2/energy norm).
* **Code:** `eps_loc`
* **Receipt required fields:** `state_id`, `norm_spec`, `X_obs_ref`, `X_pred_ref`, `dt`

---

## 2) Ledger spine (Ω family)

### (\mathbb\Omega) — Ledger system / receipts spine (the *mechanism*)

* **Meaning:** the system that enforces tamper-evident step logging (schemas + hashing + canonical serialization)
* **Code:** `omega_ledger_system`
* **Invariant:** every state mutation has a receipt with hash chaining

### (\Omega) — Receipt chain instance (the *run*)

* **Meaning:** a specific run’s realized receipt chain (instances of the ledger system)
* **Code:** `omega_chain`
* **Core fields:** `receipt_hash`, `prev_hash`, `artifact_hash`, `step_id`, `t`, `dt`, `tau`, `dtau`

**Hash standard:** **BLAKE3 or SHA-256** only (CRC32 allowed only as a fast checksum; never audit truth).

---

## 3) Amplitude–phase fields (the φ vs θ anchor)

### (\phi) — Amplitude / order parameter (never “phase”)

* **Type:** scalar field (\phi(\mathbf x,t))
* **Meaning:** magnitude / intensity / coherence-carrier amplitude
* **Units:** model-dependent; must be declared in module header if physical
* **Code:** `phi_amp`
* **Hard rule:** if you need a different scalar potential, name it (u) or (\psi), not (\phi).

### (\theta) — Phase field (never “amplitude”)

* **Type:** scalar field (\theta(\mathbf x,t)) (often angle-like)
* **Meaning:** phase / potential whose gradient drives flux/currents
* **Units:** radians (dimensionless) if treated as angle; still dimensionless even if “phase units”
* **Code:** `theta_phase`
* **Canonical coupling template:**
  [
  \mathbf J \propto \phi^2 \nabla \theta
  ]
  (coefficients belong in model tags, not in the symbol role)

### (\Phi(s)) — Spectral/Mellin lift (zeta side)

* **Meaning:** analytic transform of a test function into the spectral domain
* **Rule:** capital (\Phi) is **never** interchangeable with (\phi).

---

## 4) Density / carrier fields

### (\rho) — Density (mass/energy/coherence carrier)

* **Type:** scalar field (\rho(\mathbf x,t)\ge 0)
* **Meaning:** “stuff per volume” (fluid density / probability density / coherence carrier)
* **Units:** declared per module (e.g. kg·m(^{-3}) for fluids)
* **Code:** `rho`
* **Constraint:** if used as carrier for a current, must remain nonnegative (barrier/positivity enforcement)

### (\mathbf v) — Velocity field

* **Type:** vector field (\mathbf v(\mathbf x,t))
* **Units:** m·s(^{-1}) under physical fluid interpretation
* **Code:** `v`
* **Constraint:** incompressible mode requires (\nabla\cdot\mathbf v=0).

---

## 5) Currents, fluxes, continuity

### (\mathbf J) — Generic spatial flux/current

* **Type:** vector field
* **Meaning:** transport of a tracked quantity (mass/probability/coherence, etc.)
* **Code:** `J`

### (J_C^\mu) — Coherence 4-current (relativistic form)

* **Type:** 4-vector
* **Meaning:** coherence flow in spacetime
* **Canonical pattern:**
  [
  \nabla_\mu J_C^\mu = S_C
  ]
* **Code:** `Jc_mu`, `Sc`

### Continuity with source

[
\partial_t \rho + \nabla\cdot \mathbf J = S
]

* **Rule:** must define (i) (\mathbf J), (ii) (S), (iii) boundary conditions.

---

## 6) K-resource and enforcement (κ family)

### (\kappa) — K-resource / enforcement budget (LoC controller fuel)

* **Type:** scalar or scalar field (\kappa(t)) or (\kappa(\mathbf x,t))
* **Meaning:** “how much coherence-enforcement power is available”
* **Code:** `kappa_K`
* **Canonical barrier parameterization:**
  [
  \kappa = \kappa_{\min} + e^{\chi}
  ]
* **Constraint:** (\kappa \ge \kappa_{\min}&gt;0) always (hard floor prevents depletion catastrophe)

### (\kappa_{\min}) — Hard floor

* **Meaning:** minimum enforcement budget guaranteed by design
* **Code:** `kappa_min`
* **Rule:** explicit in receipts; never implicit.

### (\chi) — Log-budget coordinate

* **Meaning:** unconstrained variable that keeps (\kappa) positive
* **Code:** `chi_kappa`
* **Constraint:** (\chi) evolves; (\kappa) derived.

---

## 7) Coherence coupling (λ family)

### (\lambda) — Coherence coupling (rails-to-dynamics coupling)

* **Type:** scalar (dimensionless unless module declares units)
* **Meaning:** coupling strength that controls how strongly coherence rails (or glyph actuation) influence dynamics, or how dynamics feed back into coherence measures—**must be directionally specified in the module**
* **Code:** `lambda_cpl`
* **Required declarations:** (\lambda_{\min},\lambda_{\max}) and saturation rule (or clamp)
* **Constraints:**

  * bounds must be declared: (\lambda\in[\lambda_{\min},\lambda_{\max}])
  * any change to (\lambda) must be receipt-logged (`lambda_before`, `lambda_after`, `policy_ref`)
* **Forbidden overload:** (\lambda) is not wavelength; use (\lambda_{\text{wave}}) if needed.

---

## 8) Source terms and forcing

### (S) — Source term (generic)

* **Type:** scalar or vector (depends on PDE)
* **Meaning:** external injection/removal, forcing, control actuation
* **Code:** `S`
* **Rule:** all sources must be receipt-logged with provenance tags.

### (S_C) — Coherence source/sink

* **Meaning:** adds/removes coherence (or enforcement budgets)
* **Rule:** must be justified; otherwise it’s a hidden cheat channel.

---

## 9) Time, clocks, and Aeonic contract

### (t) — Physical/solver time

* **Meaning:** independent variable used by the PDE/ODE integrator
* **Code:** `t`
* **Rule:** never mix with coherence time without an explicit map.

### (\Delta t) / `dt` — Step size in (t)

* **Code:** `dt`
* **Constraint:** must appear in receipts for every step.

### (\tau) — Coherence time (Aeonic clock coordinate)

* **Meaning:** reparameterized time tracking progress by coherence rather than wall time
* **Code:** `tau`
* **Rule:** mapping must be explicit:
  [
  \frac{d\tau}{dt} = h(C_{\text{tot}},\kappa,\lambda,\ldots),\quad h&gt;0
  ]
* **Receipt fields:** `tau_n`, `dtau`, `time_policy`, `h_eval`

### (\tau_n) — Coherence time at step (n)

* **Code:** `tau_n`

### (\Delta\tau) / `dtau`

* **Meaning:** coherence-time increment per step
* **Constraint:** positive. If it hits 0 (stall), PhaseLoom must trigger dominance/bottleneck diagnostics.

---

### (\mathsf{TimePolicy}) — Aeonic time policy object (typed, auditable)

* **Meaning:** declared rule set that defines (h), clamps, and monotonicity guarantees
* **Required fields (minimum):**

  * `h_form` (identifier of formula family)
  * `h_params` (constants)
  * `clamps` (min/max for `dtau`, optional for `dt`)
  * `monotone_proof_tag` (how we guarantee (h&gt;0))
* **Code:** `time_policy`

---

## 10) PhaseLoom / multi-thread decomposition

### Threads (27) — domain × scale × response

* **Domains (3):** PHY (physical evolution), CONS (constraints/compatibility), SEM (semantic/meaning)
* **Scales (3):** L/M/H (low/mid/high octave bands or dyadic shells)
* **Responses (3):**

  * **OBS** (observation/metrics)
  * **CTRL** (actuation/control)
  * **AUD** (audit/receipt/proof layer)

**Canonical thread label:**
[
T[\text{domain},\text{scale},\text{response}]
]
**Code:** `T_PHY_L_OBS`, `T_CONS_H_CTRL`, `T_SEM_M_AUD`, etc.

### Dominant thread

* **Meaning:** thread with highest risk/impact at a step (largest normalized residual / strongest urgency)
* **Rule:** dominance selection must be reproducible and receipt-logged:

  * `dominant_thread`
  * `dominance_metric`
  * `dominance_scores` (optional but recommended)

---

## 11) Control and feedback symbols

### (u_{\text{glyph}}) — Glyph control input (actuation)

* **Type:** scalar/vector depending on target
* **Meaning:** control applied by glyph logic (rails-only control)
* **Code:** `u_glyph`
* **Constraint:** saturate:
  [
  |u_{\text{glyph}}|\le u_{\max}
  ]
* **Receipt fields:** `u_glyph`, `u_max`, `actuation_targets`

### (K_p) — Proportional gain

* **Meaning:** maps residual to actuation magnitude
* **Code:** `Kp`
* **Constraint:** stability bounds must be declared in controller spec and referenced in receipts.

### (\eta) — Noise / stochastic term (optional)

* **Meaning:** modeled uncertainty
* **Rule:** if (\eta\neq 0), seed and distribution must be receipt-logged.

---

## 12) PDE/analysis operators

### (\nabla) — Gradient

* **Code:** `grad`

### (\nabla\cdot) — Divergence

* **Code:** `div`

### (\Delta) — Laplacian

* **Meaning:** (\Delta = \nabla\cdot\nabla)
* **Code:** `laplacian`
* **Hard rule:** never use (\Delta) to mean “difference” in proofs; use (\delta) or explicit subtraction.

### (\delta) — Difference operator (ledger / discrete-time difference)

* **Meaning:** discrete difference or bookkeeping delta; never a spatial operator
* **Code:** `delta_op` (or explicit `x_next - x_now` in code)
* **Rule:** must specify what time step or slab it refers to

### (\mathcal L) — Linear operator (system-defined)

* **Meaning:** linear part of evolution (diffusion, wave operator, etc.)
* **Rule:** declare domain + boundary conditions per module.

### (\mathcal F) — Nonlinear operator (system-defined)

* **Meaning:** nonlinear part (advection, couplings, constraint terms, etc.)

---

## 13) Spectral / octave / dyadic shell symbols

### (j) — Dyadic scale index

* **Meaning:** octave/shell number (higher (j) = higher frequency)
* **Constraint:** integer, typically (j\ge 0)

### (E_{\ge j}(t)) — High-frequency tail energy

* **Meaning:** energy in frequencies (\ge 2^j)
* **Rule:** must be defined by chosen decomposition (Littlewood–Paley / FFT bands / wavelets)

### (D_{\ge j}(t)) — High-frequency dissipation rate

* **Meaning:** dissipation contributed by shells (\ge j)

### Slab interval (disambiguated)

[
\boxed{I_{j,n}=\big[n,\vartheta,2^{-j},\ (n+1),\vartheta,2^{-j}\big]}
]

* **Meaning:** time slab at scale (j), slab index (n)
* **Rule:** slab size parameter is **(\vartheta)** (or (\theta_{\text{slab}})); never reuse (\theta).

---

## 14) Stability / synchronization metrics

### (R) — Kuramoto order parameter (synchrony)

* **Type:** scalar in ([0,1])
* **Meaning:** phase synchrony measure
  [
  R e^{i\psi}=\frac{1}{N}\sum_{k=1}^N e^{i\theta_k}
  ]
* **Code:** `kuramoto_R`
* **Rule:** only use (R) for this; if you need radius, use (r).

### (H) — Energy / Hamiltonian (system-defined)

* **Meaning:** conserved/monitored quantity
* **Code:** `H_energy`
* **Constraint:** drift bounds declared (e.g. (|\delta H|&lt;\epsilon) per interval).

---

## 15) GR/NR (if/when invoked)

### (g_{\mu\nu}) — Metric tensor

### (T_{\mu\nu}) — Stress-energy tensor

### (\nabla_\mu) — Covariant derivative

### Constraints (canonical symbols)

* (\mathcal H) — Hamiltonian constraint
* (\mathcal M_i) — Momentum constraint
* **Code:** `H_constraint`, `M_constraint`

---

## 16) Noetica / AML language symbols (glyph semantics)

### (\mathbb G) — Glyph alphabet / glyph set

* **Meaning:** set of atomic glyph primitives available to the language
* **Rule:** glyphs are tokens; semantics assigned via compiler dictionary, not vibes.

### (\Lambda) — Typing/regime tag (never bare)

* **Meaning (Noetica):** type/regime annotation controlling interpretation, units, constraints
* **Code:** `Lambda_tag`
* **Rule:** must be explicit: (\Lambda_{\text{type}}, \Lambda_{\text{obs}}, \Lambda_{\text{adpt}}, \Lambda_{\text{cosmo}}), etc.

### (\Psi) — Semantic program/meaning state (canonical)

* **Meaning:** “what the program is intending/representing” at semantic level
* **Code:** `Psi_sem`
* **Rule:** wavefunction must be (\Psi_{\text{QM}}) explicitly.

### (\zeta) — Zeta-gated constructs / analytic gate

* **Meaning:** zeta/prime-ledger gating, spectral constraints, analytic receipts
* **Code:** `zeta_gate`
* **Rule:** (\zeta) is always “analytic gate” in Noetica contexts.

### (\mathfrak p) — Prime primitive (arithmetical atom)

* **Meaning:** prime-indexed semantic atom for ledger coupling
* **Code:** `p_prime`

---

## 17) Forbidden overload list (enforced)

* Never use (\phi) to mean phase. Phase is (\theta).
* Never use (\Delta) to mean “difference.” Use (\delta) or explicit subtraction.
* Never reuse (\theta) as slab-size parameter. Use (\vartheta) or (\theta_{\text{slab}}).
* Never use bare (\Lambda). Must be subscripted.
* Never let “checksum” pretend to be “receipt hash.” CRC32 ≠ audit.
* Never use (\lambda) as wavelength in this canon; use (\lambda_{\text{wave}}).

---

## 18) Canonical symbol → code mapping (minimum set)

* (\phi(\mathbf x,t)) → `phi_amp[x,t]`
* (\theta(\mathbf x,t)) → `theta_phase[x,t]`
* (\rho(\mathbf x,t)) → `rho[x,t]`
* (\kappa(t)) or (\kappa(\mathbf x,t)) → `kappa_K`
* (\kappa_{\min}) → `kappa_min`
* (\chi) → `chi_kappa`
* (\lambda) → `lambda_cpl`
* (C_{\text{tot}}) → `C_tot`
* (\varepsilon_{\text{LoC}}) → `eps_loc`
* (t, dt) → `t, dt`
* (\tau, dtau) → `tau, dtau`
* (\mathsf{TimePolicy}) → `time_policy`
* (u_{\text{glyph}}, K_p) → `u_glyph, Kp`
* (R) → `kuramoto_R`
* (H) → `H_energy`
* Ledger system (\mathbb\Omega) → `omega_ledger_system`
* Run chain (\Omega) → `omega_chain`
* Receipts spine → `prev_hash`, `receipt_hash`, `artifact_hash`

---

### Coherence hardening note (practical, canon-friendly)

Adopt one enforcement mechanism in the compiler/validator:

**Reject any IR transform that changes a symbol’s canonical role without an explicit remap receipt** (`symbol_remap` event, with old/new role, justification tag, and hash).

That turns symbol drift into a compile error instead of a future catastrophe.