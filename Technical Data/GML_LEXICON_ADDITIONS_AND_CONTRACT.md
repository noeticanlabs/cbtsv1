# GML_LEXICON_ADDITIONS_AND_CONTRACT.md
# Triaxis (NPA) — GML Lexicon Additions & Contract (v1.2)

**Alphabet:** GML (Aeonica / Memory-Time / Witness)  
**Role:** Append-only truth records (receipts), PhaseLoom thread identity, clock policy stamps, checkpoint/rollback evidence, and replay-grade auditing.  
**Ledger axiom:** Truth ≡ hash-chained receipts (replayable, deterministic).

This document is **normative** for:
1) how new GML glyphs are introduced (lexicon additions),
2) what every GML glyph **must** specify (contract),
3) strict immutability + hash-chain requirements,
4) PhaseLoom thread taxonomy (27 threads) and dominance tagging,
5) replay and evidence validity criteria.

---

## 1) What “GML Lexicon” means (and what it does not)

### 1.1 GML lexicon = witness event registry
The **GML lexicon** is the authoritative registry of event types and witness tags:

\[
A:\text{KIND.PATH} \;\;\longleftrightarrow\;\; \text{(required fields, hash rules, replay obligations)}.
\]

GML glyphs record **what happened**, not what it meant (GHLL) and not how it was executed (GLLL).

### 1.2 What GML is NOT
- Not an execution layer (no ops; only evidence).
- Not editable state (no in-place mutation).
- Not “optional logging.” If a run claims evidence-grade truth, required receipts must exist.

---

## 2) Namespaces + immutability rules

### 2.1 Canonical ID format
- `A:<KIND>.<PATH>`

Examples:
- `A:THREAD.PHY.M.R1`
- `A:RCPT.step.accepted`
- `A:CKPT.created`
- `A:ALERT.gate_fail`
- `A:SUM.run`

### 2.2 Kinds (closed set for v1.x)
GML kinds are fixed in v1.x:
- `THREAD` — PhaseLoom thread IDs (27)
- `RCPT`   — receipts (step/gate/check/ckpt/rollback/run summary)
- `CLOCK`  — clock policy and mode stamps
- `CKPT`   — checkpoint identifiers and events
- `ALERT`  — warnings/errors (witnessed anomalies)
- `SUM`    — rollups and run summaries
- `LINK`   — explicit join records (optional but allowed)

### 2.3 Immutability rule (hard)
A GML record, once emitted, is immutable:
- it cannot be modified,
- it cannot be deleted (in an evidence-grade run),
- it can only be superseded by later records (e.g., a correction note), but the original remains.

---

## 3) Hash chain contract (ledger truth)

### 3.1 Required hash algorithm and encoding
- Hash: **SHA-256**
- Receipt canonical encoding: **canonical JSON v1.2**
- Output: lowercase hex (64 chars)

### 3.2 Canonical JSON requirements (normative)
To be evidence-grade, receipts must satisfy:
- JSON numbers encoded as **decimal strings** for all real values
- keys serialized in **canonical key order** (defined in `42_AEONICA_RECEIPTS.md`)
- no extraneous whitespace in the serialized bytes used for hashing

### 3.3 Hash computation rule
For each GML receipt object:
1) Serialize the object with all fields **except** `hash`.
2) Compute `hash = sha256(serialized_without_hash)`.
3) Record `hash_prev` = prior receipt hash (or 64 zeros for first receipt).

### 3.4 Evidence validity rule (hard)
A run is evidence-grade only if:
- every receipt’s `hash` recomputes correctly, and
- every receipt’s `hash_prev` matches the previous receipt’s `hash`, and
- required receipts exist for enforced intents/policies.

---

## 4) The GML contract: required fields by GML kind

### 4.1 Universal fields (all receipt-like records)
All GML records that are receipts **MUST** include:
- `run_id` (string)
- `event` (string, a valid `A:RCPT.*` ID)
- `hash_prev` (64-hex string)
- `hash` (64-hex string)

All step-like and time-like receipts **MUST** include:
- `step_id` (int) when applicable
- `t`, `dt`, `tau`, `dtau` as decimal strings (when applicable)

### 4.2 Step receipts (mandatory)
Events:
- `A:RCPT.step.proposed`
- `A:RCPT.step.accepted`
- `A:RCPT.step.rejected`

**Required fields (accepted/rejected):**
- `run_id`, `step_id`, `event`, `thread`
- `t`, `dt`, `tau`, `dtau`
- `intent_id`, `intent_hash` (GHLL join keys)
- `ops[]` (GLLL join keys, in execution order)
- `gates{}` (all active gates + outcomes)
- `residuals{}` (all checked residuals)
- `metrics{}` (must include `decode_margin_min` if decoding occurred)
- `status` (`accepted|rejected`)
- `hash_prev`, `hash`

### 4.3 Checkpoint receipts (mandatory when checkpointing)
Events:
- `A:RCPT.ckpt.created`

Required fields:
- `run_id`, `step_id`, `event`
- `ckpt_id`
- `thread`
- `t`, `tau`
- `reason`
- `state_digest`
- `hash_prev`, `hash`

### 4.4 Rollback receipts (mandatory when rollback occurs)
Events:
- `A:RCPT.rollback.executed`

Required fields:
- `run_id`, `step_id`, `event`
- `rollback_to`
- `thread`
- `t`, `tau`
- `restored_state_digest`
- `policy` (GHLL policy ID that triggered rollback)
- `hash_prev`, `hash`

### 4.5 Gate receipts (optional if embedded in step receipts)
Events:
- `A:RCPT.gate.pass`
- `A:RCPT.gate.fail`

If emitted separately, required fields:
- `run_id`, `step_id`, `event`, `thread`
- `gate_name`
- `status`
- `margin` (decimal string, if applicable)
- `hash_prev`, `hash`

### 4.6 Run summary receipt (mandatory for evidence-grade runs)
Event:
- `A:RCPT.run.summary`

Required fields:
- `run_id`, `event`
- `steps_total`, `steps_accepted`, `steps_rejected`
- `first_step_id`, `last_step_id`
- `clock_policy` (GML clock policy ID)
- `dominant_threads[]`
- `invariants_enforced[]` (GHLL IDs)
- `final_ckpt`
- `final_head_hash`
- `hash_prev`, `hash`

---

## 5) PhaseLoom threads (GML THREAD lexicon)

### 5.1 Thread axes (canonical)
- Domain: `PHY | CONS | SEM`
- Scale: `L | M | H`
- Response: `R0 | R1 | R2`

Thread ID format:
- `A:THREAD.<DOMAIN>.<SCALE>.<RESP>`

### 5.2 Full 27-thread set (hard-defined)
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
- `A:THREAD.SEM.H.R0` `A:THREAD.SEM.H.R1` `A:THREAD.SEM.H.R2`

### 5.3 Thread selection rule (mandatory)
Every step receipt **MUST** include exactly one `thread`.

Selection is determined by:
- the dominant engine domain (PHY/CONS/SEM),
- the active scale band (L/M/H),
- the response mode (R0/R1/R2) according to the PhaseLoom scheduler policy.

---

## 6) Dominance and bottleneck tagging (GML metrics contract)

### 6.1 Dominant thread metric
Step receipts **MAY** include:
- `metrics.dominant_thread_score` (decimal string)
- `metrics.dominant_thread_prev` (thread ID)
- `metrics.dominant_thread_now` (thread ID)

Run summary **MUST** include:
- `dominant_threads[]` (top threads by cumulative score)

### 6.2 Bottleneck alert rules (canonical)
Emit `A:ALERT.bottleneck` when any of these conditions hold for a configured window:
- `wall_ms` increases by > X% while residuals do not improve
- repeated gate failures on same thread
- decode margin falls below threshold repeatedly
- κ-budget usage spikes beyond policy limits

Alert receipt must include:
- `thread`, `step_id`, `reason`, `metrics_snapshot`

---

## 7) Lexicon additions (how new GML glyphs are accepted)

### 7.1 Additions are schema-first
A new GML event type can be added only if:
- it is assigned a stable ID (`A:KIND.path`),
- its schema is explicitly defined (required fields + optional fields),
- it defines its **hash participation** (all fields included except `hash`),
- it defines replay obligations (what must be verifiable from it).

### 7.2 Hard prohibitions
- No event types that require mutating old receipts.
- No “freeform blobs” that cannot be replay-verified.
- No new thread IDs outside the 27-thread lattice in v1.x.

---

## 8) GML lexicon versioning (required)

### 8.1 Version fields
Evidence-grade runs **MUST** include:
- `gml_lexicon_version` (string) in run manifest
- and in every step receipt’s `metrics`:
  - `metrics.gml_lexicon_version`

### 8.2 Version stability rule
- Existing event IDs keep their schema stable in v1.x.
- Additive-only changes are allowed:
  - adding optional fields is permitted if they do not affect replay verification,
  - required fields must not be removed.

---

## 9) Conformance tests required for any GML addition

Any implementation claiming GML v1.2 compliance must pass:

1) **Hash correctness**
   - recompute every receipt hash and match
2) **Hash chain correctness**
   - verify every `hash_prev` linkage
3) **Schema validation**
   - required fields exist for each event type
4) **Seam join presence**
   - step receipts include `intent_id`, `intent_hash`, `ops[]`
5) **Replay-grade determinism**
   - replay engine can verify and reproduce acceptance decisions from stored gate outcomes and residuals
6) **Thread validity**
   - every step receipt thread is one of the 27 canonical threads

---

## 10) Minimal “truth bundle” for a single step (must be possible)
For any accepted step, the GML stream must allow an auditor to answer:

- What intent was claimed? (`intent_id`, `intent_hash`)
- What actions ran? (`ops[]`, args digests if present)
- Under what clocks? (`t,dt,tau,dtau`)
- Under what gates? (`gates{}` outcomes)
- What evidence was checked? (`residuals{}`, `metrics{}`)
- Why was it accepted? (`status`, acceptance derivable from witness rules)
- Is it chained and untampered? (`hash_prev`, `hash`)

If any of these cannot be answered, the run is not evidence-grade.

---

## 11) Summary (the enforcement point)

- GML is the witness alphabet; it is immutable and hash-chained.
- Its lexicon defines exactly what must be recorded to make claims verifiable.
- PhaseLoom threads (27) are a closed taxonomy in v1.x.
- Evidence-grade truth requires schema validity + hash continuity + seam join keys.

That is “one ledger truth.”