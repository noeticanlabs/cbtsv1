# GML_GLYPH_LIBRARY_DICTIONARY.md
# Triaxis (NPA) — GML Glyph Library / Dictionary (v1.2)

**Alphabet:** GML (Aeonica / Memory-Time / Witness)  
**Purpose:** Concrete dictionary of GML glyph IDs: PhaseLoom threads, receipt events, clock policies, checkpoint/rollback events, alerts, and summary records — with required fields and evidence obligations.

This document is **normative** for v1.2.

---

## 0) Conventions

### 0.1 GML ID format
- `A:<KIND>.<PATH>`

### 0.2 Kinds (v1.x closed set)
`THREAD | RCPT | CLOCK | CKPT | ALERT | SUM | LINK`

### 0.3 Evidence-grade rule (always on)
Any record of kind `RCPT` MUST:
- be immutable,
- be hash-chained via `hash_prev` → `hash`,
- use SHA-256 over canonical JSON v1.2 bytes.

---

## 1) THREAD glyphs (PhaseLoom 27-thread lattice)

### 1.1 Thread axes (canonical)
- Domain: `PHY | CONS | SEM`
- Scale: `L | M | H`
- Response: `R0 | R1 | R2`

### 1.2 Full 27-thread set (hard-defined)
**PHY**
- `A:THREAD.PHY.L.R0` `A:THREAD.PHY.L.R1` `A:THREAD.PHY.L.R2`
- `A:THREAD.PHY.M.R0` `A:THREAD.PHY.M.R1` `A:THREAD.PHY.M.R2`
- `A:THREAD.PHY.H.R0` `A:THREAD.PHY.H.R1` `A:THREAD.PHY.H.R2`

**CONS**
- `A:THREAD.CONS.L.R0` `A:THREAD.CONS.L.R1` `A:THREAD.CONS.L.R2`
- `A:THREAD.CONS.M.R0` `A:THREAD.CONS.M.R1` `A:THREAD.CONS.M.R2`
- `A:THREAD.CONS.H.R0` `A:THREAD.CONS.H.R1` `A:THREAD.CONS.H.R2`

**SEM**
- `A:THREAD.SEM.L.R0` `A:THREAD.SEM.L.R1` `A:THREAD.SEM.L.R2`
- `A:THREAD.SEM.M.R0` `A:THREAD.SEM.M.R1` `A:THREAD.SEM.M.R2`
- `A:THREAD.SEM.H.R0` `A:THREAD.SEM.H.R2` `A:THREAD.SEM.H.R1`

> Note: ordering above is not semantic; IDs are.

### 1.3 Thread assignment rule (mandatory)
Every `A:RCPT.step.*` record must include exactly one `thread` value that is one of the 27 IDs.

---

## 2) CLOCK glyphs (clock policy and modes)

### 2.1 `A:CLOCK.policy.triaxis_v1`
- **kind:** CLOCK  
- **meaning:** Canonical clock policy binding `(t,dt)` and `(tau,dtau)` for Triaxis evidence-grade runs.  
- **required fields in manifest:**  
  - `clock_policy="A:CLOCK.policy.triaxis_v1"`
  - `tau_definition` (string; e.g., `"coherence_time"` or `"real_time"`)
  - `dt_arbitration` (string; deterministic method)
- **required fields in step receipts:** `t,dt,tau,dtau`

### 2.2 `A:CLOCK.mode.real_time`
- **meaning:** `tau := t` and `dtau := dt` (identity coherence clock).

### 2.3 `A:CLOCK.mode.coherence_time`
- **meaning:** `tau` evolves by coherence-time arbitration (must be deterministic and witnessed).
- **receipt requirement:** must record `dtau` decision metric if adaptive.

---

## 3) RCPT glyphs (receipt events)

All `A:RCPT.*` events are **receipt records** and MUST satisfy hash-chain rules.

### 3.1 Step receipts
#### `A:RCPT.step.proposed`
- **meaning:** A step was constructed but not yet accepted/rejected.
- **required fields:** `run_id, step_id, event, thread, t, dt, tau, dtau, intent_id, intent_hash, ops[], hash_prev, hash`
- **optional fields:** `gates{}` (partial), `residuals{}` (partial), `metrics{}`

#### `A:RCPT.step.accepted`
- **meaning:** Step committed as accepted evidence.
- **required fields:**  
  - identity: `run_id, step_id, event, thread`
  - clocks: `t, dt, tau, dtau`
  - seam joins: `intent_id, intent_hash, ops[]`
  - evidence: `gates{}, residuals{}, metrics{}`
  - decision: `status="accepted"`
  - ledger: `hash_prev, hash`

#### `A:RCPT.step.rejected`
- **meaning:** Step rejected (must be replay-derivable why).
- **required fields:** same as accepted +  
  - `status="rejected"`
  - `action_on_fail` (e.g., `"rollback"` or `"halt"`)
  - `rollback_to` (if rollback is taken)

---

### 3.2 Gate receipts (optional if embedded in step receipts)
#### `A:RCPT.gate.pass`
#### `A:RCPT.gate.fail`
- **required fields:** `run_id, step_id, event, thread, gate_name, status, hash_prev, hash`
- **recommended fields:** `margin`, `snapshot` (small metrics dict)

---

### 3.3 Checkpoint receipt
#### `A:RCPT.ckpt.created`
- **required fields:**  
  - `run_id, step_id, event, ckpt_id, thread`
  - `t, tau`
  - `reason`
  - `state_digest` (string)
  - `hash_prev, hash`

---

### 3.4 Rollback receipt
#### `A:RCPT.rollback.executed`
- **required fields:**  
  - `run_id, step_id, event, thread`
  - `t, tau`
  - `rollback_to`
  - `restored_state_digest`
  - `policy` (GHLL policy ID that demanded rollback)
  - `hash_prev, hash`

---

### 3.5 Invariant check receipt (optional; usually embedded)
#### `A:RCPT.check.invariant`
- **required fields:** `run_id, step_id, event, thread, intent_id, intent_hash, checks{}, hash_prev, hash`
- **checks{} format:** map of `check_name -> {value, tol_abs, tol_rel, pass}` (all decimal strings except booleans)

---

### 3.6 Run summary receipt (mandatory for evidence-grade runs)
#### `A:RCPT.run.summary`
- **required fields:**  
  - `run_id, event`
  - `steps_total, steps_accepted, steps_rejected`
  - `first_step_id, last_step_id`
  - `clock_policy`
  - `dominant_threads[]`
  - `invariants_enforced[]`
  - `final_ckpt`
  - `final_head_hash`
  - `hash_prev, hash`

---

## 4) CKPT glyphs (checkpoint IDs and helpers)

### 4.1 `A:CKPT.id`
- **meaning:** Naming rule for checkpoint IDs.
- **format:** `CKPT-<step6>-<tag>`
  - `<step6>` = step_id zero-padded to 6 digits
  - `<tag>` = short ASCII label (e.g., `A`, `B`, `SAFE`)

Examples:
- `CKPT-000128-A`
- `CKPT-000450-SAFE`

Checkpoint IDs must be unique within a run.

---

## 5) ALERT glyphs (witnessed anomalies)

Alerts are receipts if they are hash-chained records; they MUST be `A:RCPT.*` if you want them counted in evidence-grade stream.
We also allow `A:ALERT.*` IDs as **event categories** that appear inside a receipt payload.

### 5.1 `A:ALERT.bottleneck`
- **meaning:** Performance stall or dominance saturation detected.
- **required fields (inside a receipt metrics snapshot):**
  - `thread`, `step_id`, `reason`, `wall_ms`, `residual_trend`

### 5.2 `A:ALERT.decode_margin_low`
- **meaning:** Hadamard decode margin fell below policy threshold.
- **required fields:** `decode_margin_min`, `threshold`, `op_id`

### 5.3 `A:ALERT.stage_mismatch`
- **meaning:** Stage coherence violated.
- **required fields:** `delta_stage_t_max`

### 5.4 `A:ALERT.hash_chain_fail`
- **meaning:** Ledger verification failed.
- **required fields:** `hash_fail_count`, `first_fail_step_id`

---

## 6) SUM glyphs (rollups)

### 6.1 `A:SUM.thread_dominance`
- **meaning:** Aggregate dominance per thread across a run.
- **required fields:** `dominance_table` (map thread -> score as decimal string)

### 6.2 `A:SUM.gate_failures`
- **meaning:** Aggregate failures per gate.
- **required fields:** `gate_fail_table` (map gate_name -> count)

---

## 7) LINK glyphs (optional explicit joins)

### 7.1 `A:LINK.intent_to_ops`
- **meaning:** Explicit mapping record: which ops were used to enforce an intent (per run).
- **required fields:** `intent_id, intent_hash, ops[], map_id`
- **note:** This is optional because step receipts already provide joins; it’s useful for quick indexing.

---

## 8) Minimal “evidence bundle” for one accepted step (must be possible)
To claim a step is accepted, the GML stream must enable an auditor to answer:

1. Which meaning was enforced? (`intent_id`, `intent_hash`)
2. Which actions ran? (`ops[]`)
3. Under which clocks? (`t,dt,tau,dtau`)
4. Under which gates and outcomes? (`gates{}`)
5. What residual evidence exists? (`residuals{}`)
6. Why accepted? (`status`, derivable from witness rules)
7. Untampered? (`hash_prev`, `hash` chain verifies)

If any answer is impossible, the run is **not evidence-grade**.

---

## 9) Summary
This dictionary is the “truth alphabet” of Triaxis:
- fixed thread IDs (27-thread PhaseLoom),
- fixed receipt event IDs,
- strict schemas + deterministic hashing,
- replay-grade evidence requirements.

GHLL can be poetic. GLLL can be fast.
GML must be **honest**.