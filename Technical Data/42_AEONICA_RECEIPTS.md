# 42_AEONICA_RECEIPTS.md
# Aeonica (A) — Receipts, Hash Chain, and Replay Spec (v1.2)

**Role:** Aeonica is Triaxis’s witness layer: time + memory + audit truth.  
**Alphabet:** GML (`A:*`).  
**Ledger rule:** Receipts are append-only, hash-chained, replayable.

---

## 1) Receipt encoding rules (normative)

### 1.1 JSON number policy (determinism)
To avoid float formatting drift across runtimes:
- All real-valued fields (`t`, `dt`, `tau`, `dtau`, residuals, margins, budgets) are encoded as **decimal strings**.
- Integers remain JSON integers.

Example:
- `"dt": "0.010000000"` is valid
- `"dt": 0.01` is not valid in receipts v1.2

### 1.2 Canonical key order (required)
Receipts **MUST** be serialized with keys in this order when hashing:

`run_id, step_id, event, thread, t, dt, tau, dtau, intent_id, intent_hash, ops, gates, residuals, metrics, status, action_on_fail, rollback_to, ckpt_id, reason, state_digest, restored_state_digest, policy, steps_total, steps_accepted, steps_rejected, first_step_id, last_step_id, clock_policy, dominant_threads, invariants_enforced, final_ckpt, final_head_hash, hash_prev`

Keys not applicable to a given receipt are omitted (not null).

### 1.3 Hash algorithm (required)
- Hash function: **SHA-256**
- Encoding: UTF-8
- Hex lowercase output (64 chars)

**Computation rule:**
1) Construct receipt object with all fields **except** `hash`.
2) Serialize using canonical rules (1.1–1.2) with separators `,` and `:` and no extra whitespace.
3) Compute `hash = sha256(serialized_receipt_without_hash)`.

### 1.4 Hash chaining rule
Each receipt contains:
- `hash_prev`: hash of the previous receipt in the run
- `hash`: its own computed hash

The first receipt in a run uses:
- `hash_prev = "0000…0000"` (64 zeros)

---

## 2) Required join keys (binding truth to meaning + mechanism)

Every `A:RCPT.step.*` receipt **MUST** include:
- `intent_id` (GHLL ID)
- `intent_hash` (SHA-256 of canonical Noetica glyph object)
- `ops[]` (GLLL opcode IDs used in the step)

This is how audits join:
- Meaning (`N:*`) → Action (`H64:*`) → Truth (`A:*`)

---

## 3) Receipt types and schemas

### 3.1 Step receipts
Events:
- `A:RCPT.step.proposed`
- `A:RCPT.step.accepted`
- `A:RCPT.step.rejected`

**Minimum fields (accepted/rejected):**
- `run_id`, `step_id`, `event`, `thread`
- `t`, `dt`, `tau`, `dtau`
- `intent_id`, `intent_hash`
- `ops[]`
- `gates` (at least those active)
- `residuals` (at least those checked)
- `metrics` (at least `decode_margin_min`)
- `status`
- `hash_prev`, `hash`

### 3.2 Gate receipts (optional if embedded in step)
Events:
- `A:RCPT.gate.pass`
- `A:RCPT.gate.fail`

Gate receipts may be emitted in addition to embedding `gates{}` inside step receipts.

### 3.3 Checkpoint receipts
Event:
- `A:RCPT.ckpt.created`

Minimum fields:
- `run_id`, `step_id`, `event`
- `ckpt_id`, `thread`, `t`, `tau`
- `reason`, `state_digest`
- `hash_prev`, `hash`

### 3.4 Rollback receipts
Event:
- `A:RCPT.rollback.executed`

Minimum fields:
- `run_id`, `step_id`, `event`
- `rollback_to`, `thread`, `t`, `tau`
- `restored_state_digest`
- `policy`
- `hash_prev`, `hash`

### 3.5 Run summary receipt
Event:
- `A:RCPT.run.summary`

Minimum fields:
- `run_id`, `event`
- `steps_total`, `steps_accepted`, `steps_rejected`
- `first_step_id`, `last_step_id`
- `clock_policy`
- `dominant_threads[]`
- `invariants_enforced[]`
- `final_ckpt`
- `final_head_hash`
- `hash_prev`, `hash`

---

## 4) Normative examples (hashes are real and consistent)

### 4.1 Step accepted receipt (example)
```json
{"run_id":"R-2026-01-19-0001","step_id":128,"event":"A:RCPT.step.accepted","thread":"A:THREAD.PHY.M.R1","t":"12.800000000","dt":"0.010000000","tau":"12.734500000","dtau":"0.009820000","intent_id":"N:INV.pde.div_free","intent_hash":"2f3d4c5b22e4b6b70b1eacbbcd1a56f3f3f9e91a0b8d3abf25a4a3f9246e5f1b","ops":["H64:r48","H64:r53","H64:r50","H64:r49","H64:r56","H64:r61"],"gates":{"energy_identity":{"status":"pass","margin":"0.000002100"},"tail_barrier":{"status":"pass","S_j_max":"0.420000000","c_star":"0.500000000"},"clock_stage_coherence":{"status":"pass","delta_stage_t":"0.000000000"}},"residuals":{"eps_div":"0.000000000032","eps_loc":"0.000000008400","eps_gate":"0.000000000118"},"metrics":{"decode_margin_min":22,"kappa_used":"0.031000000","wall_ms":"1.73"},"status":"accepted","hash_prev":"0000000000000000000000000000000000000000000000000000000000000000","hash":"dbcd3b52e6619ef5afea0679bc4bdba3d0b68a34985c8049e71d8d8719e3c34c"}
```

### 4.2 Checkpoint created receipt (example)

```json
{"run_id":"R-2026-01-19-0001","step_id":128,"event":"A:RCPT.ckpt.created","ckpt_id":"CKPT-000128-A","thread":"A:THREAD.CONS.M.R0","t":"12.800000000","tau":"12.734500000","reason":"periodic","state_digest":"sha256:5f3a9f5c7fb7c8b4b9a6b915e90d71e4cdbf1b0b4c8b2f2f3a1d1a7c9e4b6c2d","hash_prev":"dbcd3b52e6619ef5afea0679bc4bdba3d0b68a34985c8049e71d8d8719e3c34c","hash":"ee0c3a6fc93b3600e2cdcc8fe724873fb94fe8cf55e8b7045fb849bd130da91d"}
```

### 4.3 Step rejected receipt (example)

```json
{"run_id":"R-2026-01-19-0001","step_id":129,"event":"A:RCPT.step.rejected","thread":"A:THREAD.PHY.H.R2","t":"12.810000000","dt":"0.010000000","tau":"12.744320000","dtau":"0.009800000","intent_id":"N:INV.pde.energy_nonincreasing","intent_hash":"c4d6c5a3e5c9d4f1a6b7c8d9e0f11223344556677889900aabbccddeeff00112","ops":["H64:r48","H64:r52","H64:r50","H64:r49","H64:r56"],"gates":{"energy_identity":{"status":"pass","margin":"0.000000400"},"tail_barrier":{"status":"fail","S_j_max":"0.612000000","c_star":"0.500000000"}},"residuals":{"eps_energy":"0.000000000900","eps_tail":"0.000112000000"},"metrics":{"decode_margin_min":21,"kappa_used":"0.041000000","wall_ms":"1.96"},"status":"rejected","action_on_fail":"rollback","rollback_to":"CKPT-000128-A","hash_prev":"ee0c3a6fc93b3600e2cdcc8fe724873fb94fe8cf55e8b7045fb849bd130da91d","hash":"fe39bbdadf00411685c73e302c03a9e304e13208d53f92a11ab0f17ca694c8bb"}
```

### 4.4 Rollback executed receipt (example)

```json
{"run_id":"R-2026-01-19-0001","step_id":129,"event":"A:RCPT.rollback.executed","thread":"A:THREAD.CONS.M.R2","t":"12.810000000","tau":"12.744320000","rollback_to":"CKPT-000128-A","restored_state_digest":"sha256:5f3a9f5c7fb7c8b4b9a6b915e90d71e4cdbf1b0b4c8b2f2f3a1d1a7c9e4b6c2d","policy":"N:POLICY.rollback_on_gate_fail","hash_prev":"fe39bbdadf00411685c73e302c03a9e304e13208d53f92a11ab0f17ca694c8bb","hash":"ca20fa1ea4308496e195b6c3590ea306e7652b2a61056caa7207c5061f9d74ab"}
```

### 4.5 Run summary receipt (example)

```json
{"run_id":"R-2026-01-19-0001","event":"A:RCPT.run.summary","steps_total":130,"steps_accepted":129,"steps_rejected":1,"first_step_id":0,"last_step_id":129,"clock_policy":"A:CLOCK.policy.triaxis_v1","dominant_threads":["A:THREAD.PHY.M.R1","A:THREAD.CONS.M.R0","A:THREAD.PHY.H.R2"],"invariants_enforced":["N:INV.pde.div_free","N:INV.pde.energy_nonincreasing","N:INV.clock.stage_coherence","N:INV.ledger.hash_chain_intact"],"final_ckpt":"CKPT-000128-A","final_head_hash":"ca20fa1ea4308496e195b6c3590ea306e7652b2a61056caa7207c5061f9d74ab","hash_prev":"ca20fa1ea4308496e195b6c3590ea306e7652b2a61056caa7207c5061f9d74ab","hash":"9e48590a4b74677342e1a3d931e74798b697ca9cedd572e3236f867e2d931fde"}
```

---

## 5) Replay protocol (required)

A conformant replay engine must:

1. Load receipts in order.
2. Verify each receipt’s `hash` matches recomputation of the receipt without `hash`.
3. Verify `hash_prev` equals the prior receipt’s `hash`.
4. Recompute step decisions (accepted/rejected) from stored gate outcomes and ensure consistency.

If any check fails, the run is **not** evidence-grade.

---

## 6) Cross-document dependencies

* Opcode IDs and semantics: `14_TRIASIS_GLYPH_CODEBOOK_NPA.md`
* Execution step lifecycle and gate semantics: `30_PRAXICA_SPEC.md`
* Namespace + seam + truth axiom: `12_GLYPH_TAXONOMY.md`