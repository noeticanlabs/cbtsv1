# Validation Test Plan (L5)

Goal: not “unit tests”, but **tests of coherence claims** (gates + receipts + bounded drift).

## T1 — Receipt correctness
- schema validation
- deterministic hashing
- hash chain verification

Pass: 100% receipts validate; chain verifies.

## T2 — Gate behavior
- hard gate fail ⇒ rollback/abort
- soft gate fail ⇒ bounded rail then retry
- hysteresis prevents chatter-as-default

Pass: observed decisions match policy; retries bounded.

## T3 — Debt boundedness (Lemma 4)
Inject disturbance → observe repair → confirm accepted debt stays ≤ C_max.

Pass: max accepted debt ≤ C_max; repairs reduce targeted debt block.

## T4 — Projection legality (lexicon enforcement)
Inject illegal layer jump → ensure tool residual triggers and is receipted.

Pass: reject or explicit violation receipt.

## T5 — Minimal physics example
Stable ODE x'=-λx: residual decreases as dt shrinks; gates pass below threshold.

Pass: residual monotone in dt; acceptance stable.

## T6 — Rails System (Sprint 3)
Test R1-R4 rail actions bounded correctness.
- R1 deflation reduces residual magnitude
- R2 projection removes specified direction
- R3 damping smooths oscillations
- R4 prioritized mitigation addresses largest blocks

Pass: all rails reduce debt; rails maintain hard invariants.

## T7 — Debt Penalties (Sprint 3)
Test penalty term computation and integration.
- Consistency penalties for hard gate violations
- Affordability penalties for budget overruns
- Domain penalties for forbidden regions
- Complete debt with all penalty types

Pass: penalty computation matches specification; debt increases with violations.

## T8 — Coercivity Analysis (Sprint 3)
Test coercivity property verification.
- Check C(x) → ∞ as ||x - x*|| → ∞
- Compute Hessian eigenvalue margin
- Verify margin positive for well-posed debt

Pass: coercivity detectable; margin computation stable.

## T9 — Lyapunov Augmentation (Sprint 3)
Test Lyapunov-based step validation.
- V(t) = C(x(t)) debt at current time
- Steps valid if V(t+1) < V(t)
- Margin computation ΔV = V(t) - V(t+1)

Pass: step validation consistent with debt decrease.

## T10 — UFE Decomposition (Sprint 3)
Test universal field equation operator decomposition.
- dΨ/dt = L_phys + S_geo + Σ G_i residual
- Component extraction and norm computation
- Decomposition validation within tolerance

Pass: residual norm matches specification; validation triggers appropriately.

## T11 — Kuramoto and Phase Coherence (Sprint 3)
Test advanced gate types for oscillator systems.
- Kuramoto order parameter R ∈ [0,1]
- Mean phase Φ computation
- Phase coherence temporal detection

Pass: R = 1 for perfect coherence, R ≈ 0 for random phases; detection stable.
