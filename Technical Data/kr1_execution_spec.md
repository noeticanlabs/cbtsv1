# KR-1 Execution Spec: LoC-YM Mass Gap Validation

**Document ID:** KR-1-Exec-Spec-v1.0  
**Based on:** LoC-YM-Beta-v1.1.1 Canvas  
**Status:** Draft  
**Stamp:** 2026-01-18 (America/Chicago)  

---

## 0) Mission statement

This spec defines the executable procedures, thresholds, failure diagnostics, required plots/metrics, and canonical receipt bundle layout for KR-1 validation. KR-1 validates the coherence of Lemma 5.1 (UV gate) and Lemma 5.2 (budget stability) in SU(2) Yang-Mills theory, enabling evidence-grade mass gap claims upon pass.

---

## 1) Scope and objectives

KR-1 validates:
- Lemma 5.1: UV coherence gate (renormalization convergence)
- Lemma 5.2: YM coherence budget (anti-overstabilization)

For SU(2) gauge group, lattice spacings a ∈ {0.05, 0.03, 0.02}, Landau gauge, Wilson flow times τ ∈ {0.1, 0.2, 0.4}.

Outputs: deterministic receipts for all gates, hash-chained.

Disposition: PASS if all criteria met; FAIL otherwise (gap claims rejected).

---

## 2) Ensemble specifications

- **Gauge group:** SU(2)
- **Lattice sizes:** To maintain approximate physical volume V ≈ (1.6)^4 fm^4 (assuming fm units), set:
  - a = 0.05: L = 32 (physical L_phys = L*a ≈ 1.6)
  - a = 0.03: L = 53 (physical ≈ 1.59)
  - a = 0.02: L = 80 (physical ≈ 1.6)
- **Number of configurations:** N = 1000 per ensemble
- **Gauge:** Landau (∂_μ A_μ = 0)
- **Flow times:** τ ∈ {0.1, 0.2, 0.4} (in lattice units, physical τ_phys = a^2 τ_latt)

Note: Finite size effects checked by ensuring L_phys >> ξ_expected ≈ 1/m_gap, where m_gap ≈ 0.4 from prior estimates.

---

## 3) Validation procedures

### 3.1 Ensemble generation
- Use Hybrid Monte Carlo (HMC) algorithm.
- Tune step size for acceptance rate > 80%.
- Thermalization: 1000 HMC sweeps.
- Measurement: Collect 1000 configs, separated by 10 sweeps to decorrelate.
- Log: Plaquette, action history to ensure equilibrium.

### 3.2 Gauge fixing
- Minimize functional ∫ d^4x tr(A_μ(x)^2) using overrelaxation with mixing parameter 1.7, target precision 10^{-8}.
- Compute gauge residual ε_GF = √( < ||∂_μ A_μ||^2 >_{volume} / V )
- Spend: D_GF = α_GF ε_GF^2 with α_GF = 1.0

### 3.3 Wilson flow evolution
- Evolve each config with Wilson flow: ∂_τ A_μ = - D_ν F_νμ
- Integration: RK4 with step size 0.01 in τ.
- Measure at τ = 0.1, 0.2, 0.4:
  - Flowed action S(τ) = ∫ tr(F^2)/4
  - Flow energy E(τ) = (1/4) ∫ tr(F_{\mu\nu} F_{\mu\nu})
- Log: τ, S(τ), E(τ), monotonicity check.

### 3.4 UV gate (Lemma 5.1)
- **Renormalization scheme:** MSbar at scale μ = 1/a (lattice units).
- **Counterterms CT(a, μ):** Perturbative expansion up to 2 loops, fitted from high-precision perturbative data (e.g., from literature).
  - CT = c1 g^2 + c2 g^4 + ... where g = 4π/β, β = 4/(g_0^2)
- **Residual computation:** For each a, ε_ren(a) = | < S_E - (S_latt - CT) > |_{configs}
  - S_E: continuum action (approximated as S_latt at a=0)
  - Fit ε_ren(a) = c0 + c1 a + c2 a^2 using linear regression on {0.02,0.03,0.05}
  - Extrapolate c0 (a→0 value)
- Log: scheme, μ, a, CT coeffs, ε_ren per a, fit coeffs.

### 3.5 Budget gate (Lemma 5.2)
- **Initialization:** λ_GF = 1.0, λ_RP = 1.0, λ_min = 0.1
- **Spend rules:**
  - D_GF = α_GF ε_GF^2, α_GF = 1.0
  - D_RP = β_RP |min_eigenvalue(RP_matrix)| if negative, else 0, β_RP = 1.0
- **Recharge:** S_GF = s0 σ(ε_GF - ε0), s0 = 0.1, ε0 = 0.001, σ(x) = log(1 + exp(x)) (softplus)
  - Similarly for S_RP if applicable.
- **Evolution:** At each "step" (e.g., post-gauge-fix, post-flow), λ_{new} = λ_{old} - D + S
- **Stability check:** Compute linearized loop gain L = s0 σ'(ε0) / (∂D/∂λ)
  - σ'(x) = exp(x)/(1+exp(x)) at ε0 ≈ 0.001
  - ∂D/∂λ ≈ 1 for GF (linear), approximate for RP.
- Log: λ_GF(t), λ_RP(t), D_GF, D_RP, S_GF, S_RP, L, floor checks.

### 3.6 Positivity gate (GCAT-YM-1)
- Construct RP_matrix from time-reflected correlator blocks (e.g., using smeared operators).
- Compute eigenvalues, find min_eigen.
- Pass if min_eigen ≥ -δ_RP, δ_RP = 0.01
- Log: construction method, min_eigen, δ_RP, matrix dims.

### 3.7 Ward/Slavnov-Taylor gate (GCAT-YM-2)
- Evaluate identity residuals for a set of gauge-invariant correlators (e.g., < D_μ J_μ O > = < δO >).
- Compute ε_Ward = max residual over identities.
- Pass if ε_Ward ≤ δ_Ward, δ_Ward = 0.001
- Log: identity set, ε_Ward, δ_Ward.

### 3.8 Gap extraction (Section 13.1 canvas)
- Observable: tr(F_{\mu\nu} F_{\mu\nu})
- Correlator: C(t) = < O(t) O(0) >_c
- Effective mass: m_eff(t) = log(C(t)/C(t+1))
- Fit plateau in t ∈ [5,15] with constant + exp(-m t), extract m, chi2/dof.
- Refinement: Repeat for all a, check m(a) convergence.
- Gauge invariance check: Re-run in Coulomb gauge, compare m within 5%.
- Log: full receipt as per canvas Section 13.1.

---

## 4) Thresholds and acceptance criteria (hard)

- **UV gate:** Extrapolated ε_ren(a→0) < 0.01
- **Budget gate:** L ≤ 0.8 and λ_GF > λ_min and λ_RP > λ_min throughout run
- **Positivity gate:** min_eigen ≥ -0.01
- **Ward gate:** ε_Ward ≤ 0.001
- **Gap extraction:** chi2/dof ≤ 1.5, m stable across a (slope of m vs 1/a < 0.05), gauge invariance within 5%

**Overall:** All gates pass. If any fail, FAIL with diagnostic code.

---

## 5) Failure modes (forensic diagnosis codes)

Based on canvas FM-1 to FM-5:

- **FM-1: Gauge-breaking fake gap**
  - Symptom: ε_Ward > 0.01
  - Evidence: Ward residuals spike
  - Disposition: Reject, log gauge residual plot

- **FM-2: Positivity failure**
  - Symptom: min_eigen < -0.1
  - Evidence: RP matrix not positive
  - Disposition: Reject, log eigen spectrum

- **FM-3: Discretization hallucination**
  - Symptom: m_eff depends strongly on a
  - Evidence: Refinement slope > 0.1
  - Disposition: Reject, log m vs 1/a plot

- **FM-4: Overstabilization**
  - Symptom: λ hits floor or L > 0.9
  - Evidence: Budget traces show depletion
  - Disposition: Reject, log budget evolution plots

- **FM-5: Operator contamination**
  - Symptom: Inconsistent m across observables
  - Evidence: chi2 > 5 or gauge variation > 10%
  - Disposition: Reject, log comparison plots

---

## 6) Required plots and metrics

All plots saved as PNG/PDF with data points, fits, error bars.

- **UV gate:** ε_ren vs a (linear scale), with fit curve extrapolated to a=0; metric: extrapolated value
- **Budget gate:** Time series of λ_GF, λ_RP, D_GF, S_GF; metric: L value, min λ
- **Positivity gate:** Histogram of min_eigen per config; metric: min over all
- **Ward gate:** Scatter plot ε_Ward vs config index; metric: max ε_Ward
- **Gap extraction:** m_eff(t) with plateau fit; m vs 1/a; metric: m, chi2, slope
- **General:** Action vs sweep (thermalization), plaquette history

Metrics logged: all residuals, averages, std devs, fit parameters.

---

## 7) Canonical receipt bundle layout and serialization

Receipt bundle is a JSON object with top-level "kr1_bundle".

**Structure:**
```json
{
  "kr1_bundle": {
    "uv_receipt": {
      "scheme": "MSbar",
      "mu": "1/a",
      "a_values": [0.02, 0.03, 0.05],
      "eps_ren": [0.005, 0.007, 0.012],
      "fit_coeffs": [0.001, 0.1, 2.0],
      "extrapolated_eps": 0.001
    },
    "budget_receipt": {
      "L": 0.5,
      "min_lambda_gf": 0.15,
      "min_lambda_rp": 0.12,
      "traces": {
        "lambda_gf": [1.0, 0.9, ...],
        "d_gf": [0.01, 0.02, ...]
      }
    },
    "positivity_receipt": {
      "min_eigen": -0.005,
      "delta_rp": 0.01,
      "pass": true
    },
    "ward_receipt": {
      "eps_ward": 0.0005,
      "delta_ward": 0.001,
      "pass": true
    },
    "gap_receipt": {
      "operator": "tr_FF",
      "method": "effective_mass",
      "t_min": 5.0,
      "t_max": 15.0,
      "m_eff": 0.42,
      "m_eff_err": 0.03,
      "chi2_dof": 1.2,
      "converged": true,
      "refinement_series": [
        {"a": 0.05, "m_eff": 0.38, "err": 0.05},
        {"a": 0.03, "m_eff": 0.41, "err": 0.04},
        {"a": 0.02, "m_eff": 0.42, "err": 0.03}
      ],
      "gauge_invariant": true,
      "gauge_choice": "Landau",
      "gauge_residual": 1.2e-5,
      "pass": true
    }
  }
}
```

**Serialization order (canonical for hashing):**
1. uv_receipt
2. budget_receipt
3. positivity_receipt
4. ward_receipt
5. gap_receipt

Fields within each sub-receipt sorted alphabetically for determinism.

Hash: SHA256 of canonical JSON string.

---

## 8) Implementation notes

- Use open-source lattice QCD libraries (e.g., Grid, MILC) for computations.
- Parallelize over configs using HPC.
- Automate receipt generation to ensure determinism.
- Version control all code/scripts used.

---

## 9) Change log

- v1.0: Initial draft based on LoC-YM Canvas v1.1.1, with detailed procedures and layouts.