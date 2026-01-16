# HPC Integration Plan for Aeonic/PhaseLoom/solver Architecture

## Overview

This plan integrates high-performance computing (HPC) concepts into the Aeonic/PhaseLoom/solver architecture, turning them into named modules with invariants, cache/memory contracts, clocks, receipts, and GCAT tests. The goal is to create a "canon-grade" GR solver that is runnable, auditable, and optimized for low-end hardware while maintaining numerical correctness.

Based on the provided review-for-integration document, this plan maps HPC ideas to the existing codebase and adds the necessary structure.

## 1) Fit Check: Where Each HPC Concept Lands in the Stack

### A) Cache-tiled Stencil Engine

**Where it belongs:** `gr_solver/physics_kernels/` and `gr_solver/constraint_kernels/`

**Existing Integration:** Already present in `gr_geometry.py` for spatial derivatives and `gr_constraints.py` for constraint computations.

**Solver Integration Point:**
- Any grid-space operator: constraint damping, Kreiss–Oliger smoothing, diffusion-like terms, local fluxes, gauge-driver relaxations.
- Fused with diagnostics to compute max norms, residual deltas, and cheap danger proxies during stencil passes.

**Contract for Integration:**
- SoA (Structure of Arrays) layout for fields (e.g., `gamma_sym6`, `K_sym6` as separate arrays).
- Preallocated scratch buffers for halo loads and temporary registers.
- Fused passes: evolve fields while accumulating incremental stats (omega proxies, max norms).

**PhaseLoom Tie-in:**
- Loom consumes per-tile partial stats without rescanning arrays.

**Action:** Extend `gr_geometry.py` and `gr_constraints.py` to support tiled stencils and fused diagnostics.

### B) Fast Poisson/Projection/Elliptic Solves (MG or FFT)

**Where it belongs:** New module `gr_solver/elliptic/`

**Solver Integration Point:**
- ADM/constraint-cleanup steps, gauge correction, projection.
- Warm-starting for NR (Newton-Raphson) and MG cycles.

**Contract:**
- Matrix-free operator: `apply_A(x) -> y` as tiled stencil.
- MG hierarchy buffers allocated once per resolution.
- Residual thresholds in rails units.

**Aeonic Memory Tie-in:**
- Store last solution as initial guess, keyed by regime hash.
- Store MG convergence signature (cycles needed).
- Store diagonal/Jacobi weights if stable.

**Action:** Create `gr_solver/elliptic/solver.py` with MG and FFT backends, integrate into `gr_constraints.py`.

### C) Krylov with Matrix-free Ops

**Where it belongs:** `gr_solver/elliptic/krylov.py`

**Solver Integration Point:**
- Fallback when MG unstable or domain not periodic.
- GMRES(m) with small restart, CPU-friendly via SoA dot-products.

**Memory Contract:**
- Preallocate Krylov vectors (e.g., 20 vectors for GMRES(20)).
- Fused SoA operations to reduce passes.

**Action:** Add Krylov backend to elliptic module.

### D) Spectral Speed Tricks + Caching

**Where it belongs:** `gr_solver/spectral/plan_cache.py`

**Solver Integration Point:**
- FFT usage in omega/loom, pseudospectral NSE, periodic GR.

**Aeonic Memory Tie-in:**
- Tier-2 LoomCache: Store `k2`, projection factors, dealias masks, viscosity multipliers.
- These are calibration-bank style, rarely change.

**Action:** Create `gr_solver/spectral/cache.py` for k-vectors, bins, etc.

### E) Multi-rate Clocks (Octave Bands)

**Where it belongs:** `gr_solver/scheduler/multirate.py`

**Solver Integration Point:**
- Stepper chooses dt (physical, gauge), and which bands to update.
- Loom and maintenance on independent cadences.

**Critical Constraint:**
- Multi-rate must not break conservation/constraints: use coupling rules, rails for drift detection.

**Action:** Extend `gr_scheduler.py` with multi-rate logic.

### F) Aeonic Memory as HPC Weapon

**Already Implemented:** `aeonic_memory_bank.py`, `aeonic_clocks.py`, `aeonic_receipts.py`.

**Enhancements:**
- Incremental stats, lossy snapshots (fp16), predict-then-correct for expensive solves.

## 2) The Missing Piece: "Canon-grade" Structure

To achieve "canon-grade," add:

1. **Strict Data Layout Spec:** SoA arrays, 64-byte alignment (Python/virtual, but design for C/Numba).
2. **Kernel Fusion Plan:** Passes that minimize memory sweeps.
3. **Multi-Clock Schedule:** Domains for stepper, loom, maintenance.
4. **Memory Bank Policy:** Prune/demote/taint for tiers 0-3.
5. **Calibration Suite:** GCAT-1 tests proving speed + correctness.

## 3) Integration Blueprint (Implementation-ready)

### 3.1 Data Layout v1 (Low-end CPU Optimized)

**Field Storage Rules:**
- Primary: SoA arrays, C-order contiguous.
- For sym6 tensors: 6 separate arrays or SoA view wrapper mapping `[...,6]` to six buffers.
- Alignment: Design for 64-byte alignment in future C extensions.

**No Allocations per Step:**
- All buffers created once: stencil scratch, MG buffers, FFT work, loom summaries, receipts ring.

**Invariants:**
- Buffers sized at initialization, no `np.zeros` in hot loops.
- SoA access: `gamma_xx[iz, iy, ix]` etc.

### 3.2 Kernel Fusion v1

**Fused Passes:**
- **Pass A: Evolve Core Fields** - Update K, gamma, alpha/beta; accumulate norms, deltas, proxies.
- **Pass B: Constraints + Damping** - Apply damping, accumulate deltas, residual estimates.
- **Pass C: Optional Expensive Ops** - MG/FFT only if rails trigger.

**Benefits:** Fewer sweeps, cheaper diagnostics.

**Contracts:** Each pass has pre/post invariants (e.g., norms computed).

### 3.3 Multi-Clock Scheduler v1

**Clock Domains:**
- `τ_step`: Increments every attempted step.
- `τ_loom`: Every loom update.
- `τ_maint`: On pruning/compaction.

**Trigger Rules:**
- Loom: Every 4 accepted steps OR if proxy trips.
- Maintenance: Every 8 steps OR on pressure/regime shift.

**Regime Shifts:** Invalidate caches on dt change (>2x), rollback, residual slope sign change, D_max bands, dominant band change.

### 3.4 Aeonic Memory Policies v1

**Tiers:**
- Tier0: Step-local scratch (cleared on rollback).
- Tier1: StepperCache (NR warm-starts, dt suggestions).
- Tier2: LoomCache (omega_band, D_band, summaries).
- Tier3: CalibrationBank (k-vectors, masks, ETD factors).

**Demotion:** Tier2 summaries after ttl_l loom ticks.

**Taint:** Mark tainted if suggestion causes fail/rollback/violation; disable for N steps.

**Receipts:** Log all ops with costs/risks.

### 3.5 Receipts and GCAT Tests

**Receipts:** Emit events for put/get/taint/etc., tied to clocks.

**GCAT-1:** Test numerics preserved under fusion, precision, multi-rate; measure error/op counts.

## 4) What to Add Immediately (High Impact)

Based on logs, current kit has loom summaries, dt logic, damping.

**Immediate Upgrades:**
1. Cache k-bin maps: Replace 3x3x3 digitize with precomputed `kx_bin/ky_bin/kz_bin` in Tier3.
2. Cheap proxy gate before FFT: Skip if ΔK_inf and slope tiny.
3. Warm-start elliptic: Use last solution for MG/NR.
4. Downsample loom: Periodic + trigger-based.

**Expected Outcome:** Like "laptop grew a GPU."

## 5) GCAT-1 Calibration Suite Mapping

**Tests:**
- Correctness: Fused kernels, mixed precision, multi-rate, loom downsampling preserve error norms.
- Performance: 0 allocations/step, reduced FFT/MG calls, improved runtime.

**Metrics:** Error norms, op counts, runtime.

## 6) Integration Build Order

### Phase 1 — Mechanical Wins
1. Preallocate all buffers (prove 0 allocations/step).
2. Cache k-vectors, masks, bin maps (Tier3).
3. Loom downsampling + cheap proxy trigger.

### Phase 2 — Solver Speed Wins
4. StepperMemory warm starts for NR.
5. MG warm-start + "1–2 cycle then check" policy.

### Phase 3 — Physics-aligned Wins
6. Multi-rate band updates (octaves).
7. Mixed precision for cold layers.

### Phase 4 — Proof-grade
8. Receipts + taint logic.
9. GCAT-1 calibration suite.

## 7) Critical Caution

Multi-rate can break constraints if coupling sloppy. Always reconcile at macro boundaries with projection and rails check.

This plan turns HPC advice into a runnable system.