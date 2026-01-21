# GHLL_GLYPH_LIBRARY_DICTIONARY.md
# Triaxis (NPA) — GHLL Glyph Library / Dictionary (v1.2)

**Alphabet:** GHLL (Noetica / Meaning / Intent)  
**Purpose:** Canonical dictionary of GHLL glyph IDs: what each means, how it types, how it lowers into GLLL, and what Aeonica must witness.  
**Truth rule:** GHLL declares; GLLL executes; GML witnesses.

This document is **normative** for v1.2.

---

## 0) Conventions

### 0.1 GHLL ID format (canonical)
- `N:<KIND>.<PATH>`  
Optional semantic version suffix (preferred when meaning changes):
- `N:<KIND>.<PATH>.v<major>`

Examples:
- `N:INV.pde.div_free`
- `N:POLICY.deterministic_replay`
- `N:SPEC.ns_cpu_gold.v1`

### 0.2 GHLL kinds (v1.x closed set)
`TYPE | INV | GOAL | POLICY | DOMAIN | SPEC | MAP | META`

### 0.3 Normalized fields in this dictionary
Each entry below contains (minimum):
- `id`, `kind`, `domain`
- `semantics`
- `formal`
- `type_signature`
- `scope`
- `constraints`
- `lowering.required_map_ids` (or lowering_class)
- `witness.must_record` + `acceptance_rule`

---

## 1) TYPE glyphs (core)

### 1.1 `N:TYPE.scalar`
- **kind:** TYPE  
- **domain:** CORE  
- **semantics:** Scalar numeric value with declared float policy.  
- **formal:** `predicate(is_scalar(x))`  
- **type_signature:** `scalar -> scalar`  
- **scope:** `{ "where":"values", "when":"always" }`  
- **constraints:** `{ "float_policy":"manifest" }`  
- **lowering:** none (compile-time typing only)  
- **witness:** none

### 1.2 `N:TYPE.vector`
- As above; `predicate(is_vector(x))`, requires shape metadata.

### 1.3 `N:TYPE.matrix`
- As above; requires layout metadata.

### 1.4 `N:TYPE.field`
- **semantics:** Spatially indexed tensor field on a lattice/manifold.  
- **formal:** `predicate(is_field(F, domain))`  
- **constraints:** `{ "layout":"manifest", "indexing":"deterministic" }`

### 1.5 `N:TYPE.lattice`
- **semantics:** Discrete grid definition (shape, spacing, boundary conditions).  
- **constraints:** `{ "dx":"manifest", "bc":"manifest" }`

### 1.6 `N:TYPE.clock`
- **semantics:** Clock tuple `(t,dt,tau,dtau)` under a named policy.  
- **constraints:** `{ "clock_policy":"A:CLOCK.policy.*" }`

### 1.7 `N:TYPE.receipt`
- **semantics:** Hash-chained Aeonica receipt under canonical JSON rules.  
- **constraints:** `{ "hash_alg":"SHA-256", "encoding":"canonical_json_v1_2" }`

---

## 2) DOMAIN glyphs (core tags)

### 2.1 `N:DOMAIN.NS`
- **kind:** DOMAIN  
- **domain:** CORE  
- **semantics:** Navier–Stokes / incompressible flow semantics active.  
- **formal:** `predicate(domain_active(NS))`

### 2.2 `N:DOMAIN.GR_NR`
- **semantics:** 3+1 numerical relativity semantics active.

### 2.3 `N:DOMAIN.RFE_UFE`
- **semantics:** Resonant/Universal field equation stack semantics active.

### 2.4 `N:DOMAIN.ZETA`
- **semantics:** Zeta/prime ledger semantics active.

### 2.5 `N:DOMAIN.CONTROL`
- **semantics:** Scheduler + control semantics active.

---

## 3) INV glyphs (invariants: must hold)

### 3.1 `N:INV.pde.div_free`
- **kind:** INV  
- **domain:** NS  
- **semantics:** Velocity field must satisfy divergence-free constraint.  
- **formal:** `{ "kind":"equation", "lhs":"div(v)", "rhs":"0" }`  
- **type_signature:** `constraint over field:v`  
- **scope:** `{ "where":"field:v", "when":"each_step", "region":"entire_domain" }`  
- **constraints:** `{ "tol":{"abs":"0.000000000100","rel":"0.000000010000"}, "norm":"L2" }`  
- **lowering.required_map_ids:** `["N:MAP.inv.div_free.v1"]`  
- **witness.must_record:** `["residual:eps_div"]`  
- **acceptance_rule:** `residual:eps_div <= tol.abs`

### 3.2 `N:INV.pde.energy_nonincreasing`
- **domain:** NS  
- **semantics:** Gated energy must not increase beyond tolerance.  
- **formal:** `{ "kind":"inequality", "lhs":"E(t+dt)-E(t)", "op":"<=", "rhs":"0" }`  
- **constraints:** `{ "tol":{"abs":"0.000000001000","rel":"0.000000010000"}, "functional":"kinetic_energy" }`  
- **lowering.required_map_ids:** `["N:MAP.inv.energy_nonincreasing.v1"]`  
- **witness.must_record:** `["gate_outcome:energy_identity","residual:eps_energy"]`  
- **acceptance_rule:** `gate_outcome:energy_identity==pass && residual:eps_energy <= tol.abs`

### 3.3 `N:INV.clock.stage_coherence`
- **domain:** CORE  
- **semantics:** All sub-operators evaluate at consistent stage times.  
- **formal:** `{ "kind":"predicate", "name":"stage_coherent", "args":["t_stage","operators"] }`  
- **constraints:** `{ "mode":"exact" }`  
- **lowering.required_map_ids:** `["N:MAP.inv.clock.stage_coherence.v1"]`  
- **witness.must_record:** `["metric:delta_stage_t_max"]`  
- **acceptance_rule:** `metric:delta_stage_t_max == 0`

### 3.4 `N:INV.ledger.hash_chain_intact`
- **domain:** CORE  
- **semantics:** Receipt stream hash chain must verify with canonical rules.  
- **formal:** `{ "kind":"predicate", "name":"hash_chain_intact", "args":["hash_prev","hash"] }`  
- **constraints:** `{ "hash_alg":"SHA-256", "encoding":"canonical_json_v1_2" }`  
- **lowering.required_map_ids:** `["N:MAP.inv.ledger.hash_chain_intact.v1"]`  
- **witness.must_record:** `["metric:hash_fail_count"]`  
- **acceptance_rule:** `metric:hash_fail_count == 0`

### 3.5 `N:INV.rails.gate_obligations_met`
- **domain:** CORE  
- **semantics:** If a gate scope is entered, required checks/emits/seal must occur.  
- **formal:** `{ "kind":"predicate", "name":"gate_obligations_met", "args":["gate_scope"] }`  
- **constraints:** `{ "required_ops":["H64:r48","H64:r50","H64:r56","H64:r61"] }`  
- **lowering.required_map_ids:** `["N:MAP.inv.rails.gate_obligations_met.v1"]`  
- **witness.must_record:** `["metric:missing_obligation_count"]`  
- **acceptance_rule:** `metric:missing_obligation_count == 0`

---

## 4) GOAL glyphs (objectives: prefer)

### 4.1 `N:GOAL.min_residual`
- **kind:** GOAL  
- **domain:** CORE  
- **semantics:** Prefer actions that decrease declared residuals fastest without violating invariants.  
- **formal:** `{ "kind":"functional", "name":"minimize", "definition":"sum(residuals)", "variables":["policy_choices"] }`  
- **type_signature:** `objective over residual vectors`  
- **scope:** `{ "when":"each_step" }`  
- **constraints:** `{ "subject_to":["N:INV.*","N:POLICY.*"] }`  
- **lowering:** advisory only (scheduler/optimizer), must not override rails  
- **witness.must_record:** `["metrics:residual_trend"]`  
- **acceptance_rule:** `true` (goals do not accept/reject by themselves)

### 4.2 `N:GOAL.max_stability_margin`
- Prefer larger gate margins (e.g., tail barrier slack).

### 4.3 `N:GOAL.min_wall_time_given_truth`
- Prefer fewer ms without changing acceptance/receipts (truth-preserving optimization).

---

## 5) POLICY glyphs (governance: must follow)

### 5.1 `N:POLICY.rails_only_control`
- **kind:** POLICY  
- **domain:** CORE  
- **semantics:** No control actuation is permitted except through declared rails ops (gates/clamps/filters/projections/budget/rate).  
- **formal:** `{ "kind":"predicate", "name":"rails_only", "args":["control_actions"] }`  
- **constraints:** `{ "allowed_op_classes":["RAILS","TIME","LEDGER"], "forbidden":"undocumented_side_effects" }`  
- **lowering.required_map_ids:** `["N:MAP.policy.rails_only_control.v1"]`  
- **witness.must_record:** `["metric:forbidden_op_count"]`  
- **acceptance_rule:** `metric:forbidden_op_count == 0`

### 5.2 `N:POLICY.deterministic_replay`
- **semantics:** Same inputs + manifest => identical receipts.  
- **constraints:** `{ "float_policy":"manifest", "reduction_order":"fixed", "seeded_random_only":true }`  
- **lowering.required_map_ids:** `["N:MAP.policy.deterministic_replay.v1"]`  
- **witness.must_record:** `["metric:replay_mismatch_count"]`  
- **acceptance_rule:** `metric:replay_mismatch_count == 0`

### 5.3 `N:POLICY.emit_receipts_every_step`
- **semantics:** Every step must emit an accepted/rejected receipt.  
- **constraints:** `{ "required_events":["A:RCPT.step.accepted","A:RCPT.step.rejected"] }`  
- **lowering.required_map_ids:** `["N:MAP.policy.emit_receipts_every_step.v1"]`  
- **witness.must_record:** `["metric:missing_step_receipt_count"]`  
- **acceptance_rule:** `metric:missing_step_receipt_count == 0`

### 5.4 `N:POLICY.rollback_on_gate_fail`
- **semantics:** Any gate failure triggers rollback to last checkpoint.  
- **constraints:** `{ "requires_ops":["H64:r54","H64:r55"], "event":"A:RCPT.rollback.executed" }`  
- **lowering.required_map_ids:** `["N:MAP.policy.rollback_on_gate_fail.v1"]`  
- **witness.must_record:** `["metric:rollback_executed_count"]`  
- **acceptance_rule:** `true` (policy prescribes behavior; failure is witnessed as violation)

### 5.5 `N:POLICY.safe_mode_on_repeat_fail`
- **semantics:** Repeated failures trigger safe-mode entry.  
- **constraints:** `{ "threshold":{"fails":3,"window_steps":20}, "op":"H64:r63" }`  
- **lowering.required_map_ids:** `["N:MAP.policy.safe_mode_on_repeat_fail.v1"]`  
- **witness.must_record:** `["metric:safe_mode_entries"]`  
- **acceptance_rule:** `true`

---

## 6) SPEC glyphs (bundles: named configurations)

### 6.1 `N:SPEC.ns_cpu_gold.v1`
- **kind:** SPEC  
- **domain:** NS  
- **semantics:** CPU-first Navier–Stokes run spec with rails and evidence-grade receipts.  
- **formal:** `{ "kind":"predicate", "name":"spec_bundle", "args":["ns_cpu_gold_v1"] }`  
- **constraints:** {
  "requires_domains":["N:DOMAIN.NS","N:DOMAIN.CONTROL"],
  "requires_policies":["N:POLICY.deterministic_replay","N:POLICY.emit_receipts_every_step","N:POLICY.rollback_on_gate_fail"],
  "requires_invariants":["N:INV.pde.div_free","N:INV.pde.energy_nonincreasing","N:INV.clock.stage_coherence","N:INV.ledger.hash_chain_intact"]
}  
- **lowering:** spec expands into enabling those intents + maps  
- **witness.must_record:** `["metrics:spec_id","metrics:active_intents_count"]`  
- **acceptance_rule:** `true`

---

## 7) MAP glyphs (compiler seam objects — minimum required set)

These are “dictionary entries” for required maps. Full map JSON objects live in your MAP registry file,
but their IDs and roles are defined here.

### 7.1 `N:MAP.inv.div_free.v1`
- **semantics:** enforce divergence-free via projection + check + witness + seal  
- **requires_ops:** `["H64:r48","H64:r53","H64:r50","H64:r56","H64:r61"]`

### 7.2 `N:MAP.inv.energy_nonincreasing.v1`
- **requires_ops:** `["H64:r48","H64:r50","H64:r56","H64:r61"]` (+ optional FILTER)

### 7.3 `N:MAP.inv.clock.stage_coherence.v1`
- **requires_ops:** `["H64:r50","H64:r56","H64:r61"]` (+ TIME recommended)

### 7.4 `N:MAP.inv.ledger.hash_chain_intact.v1`
- **requires_ops:** `["H64:r60","H64:r56"]` (VERIFY + EMIT)

### 7.5 `N:MAP.inv.rails.gate_obligations_met.v1`
- **requires_ops:** `["H64:r48","H64:r49","H64:r50","H64:r56","H64:r61"]`

### 7.6 `N:MAP.policy.rails_only_control.v1`
- **requires_ops:** `["H64:r50","H64:r56"]` (checks + witness forbidden ops count)

### 7.7 `N:MAP.policy.deterministic_replay.v1`
- **requires_ops:** `["H64:r60","H64:r56"]` (VERIFY replay conditions + witness)

### 7.8 `N:MAP.policy.emit_receipts_every_step.v1`
- **requires_ops:** `["H64:r56","H64:r61"]` (EMIT + SEAL always)

### 7.9 `N:MAP.policy.rollback_on_gate_fail.v1`
- **requires_ops:** `["H64:r54","H64:r55","H64:r56"]`

### 7.10 `N:MAP.policy.safe_mode_on_repeat_fail.v1`
- **requires_ops:** `["H64:r62","H64:r63","H64:r56"]` (WARN + SAFE + EMIT)

---

## 8) META glyphs (non-semantic, provenance only)

### 8.1 `N:META.provenance`
- **kind:** META  
- **semantics:** Attach authorship/source metadata; must not affect lowering/execution decisions.

### 8.2 `N:META.citation`
- Citations or references; does not affect truth logic.

---

## 9) Dictionary-level conformance requirements (v1.2)

A Triaxis implementation is GHLL v1.2 compliant if:
1) It can load all entries in this dictionary.
2) It can compute `intent_hash` deterministically (canonical JSON).
3) It can lower executable intents using the required `N:MAP.*` IDs.
4) It emits Aeonica step receipts containing `intent_id` + `intent_hash`.
5) It can replay and verify acceptance decisions from recorded gates/residuals.

---

## 10) Summary
GHLL is the “meaning alphabet” dictionary:
- stable IDs,
- structured formal meaning,
- explicit scope/constraints,
- explicit lowering hooks,
- explicit witness rules.

It stays honest because every claim can be joined to:
- the exact action sequence (GLLL ops),
- and the exact receipts (GML truth).