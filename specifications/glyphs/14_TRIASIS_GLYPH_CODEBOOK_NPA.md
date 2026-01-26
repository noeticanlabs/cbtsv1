# 14_TRIAIS_GLYPH_CODEBOOK_NPA.md
# Triaxis (NPA) — Unified Glyph Codebook (v1.2)

**This document is normative.**  
It defines the canonical IDs for:
- **GLLL:** Hadamard/Praxica-H opcode glyphs
- **GHLL:** Noetica meaning glyphs (core lexicon)
- **GML:** Aeonica thread + receipt glyphs

---

## 1) GLLL — Praxica-H Hadamard Opcode Codebook (H64)

This section defines the stable H64 baseline. For extension mechanisms (H128 opcodes, macro-ops) and full lexicon contracts, see `GLLL_LEXICON_ADDITIONS_AND_CONTRACT.md`.

### 1.1 Hadamard identity rule
- Opcode identity is `H64:rXX`.
- Implementations **MUST** generate the Hadamard matrix by Sylvester construction (defined in 30_PRAXICA_SPEC.md).
- Runtime dispatch is by opcode ID (row index). If a noisy channel is used, decode by correlation and record margin.

### 1.2 Opcode table (H64:r00–r63)

#### Control + flow (r00–r15)
| Op ID | Mnemonic | Semantics |
|---|---|---|
| H64:r00 | NOP | No operation |
| H64:r01 | HALT | Stop execution |
| H64:r02 | JMP | Unconditional jump |
| H64:r03 | BR | Conditional branch |
| H64:r04 | CALL | Call block/proc |
| H64:r05 | RET | Return |
| H64:r06 | LOOP_B | Loop begin |
| H64:r07 | LOOP_E | Loop end |
| H64:r08 | PHI | SSA merge support |
| H64:r09 | SELECT | Branchless select |
| H64:r10 | ASSERT | Assertion (must be witnessed) |
| H64:r11 | TRAP | Hard failure → rollback path |
| H64:r12 | SYNC | Ordering fence |
| H64:r13 | YIELD | Scheduler yield |
| H64:r14 | WAIT | Wait on token/clock |
| H64:r15 | TIME | Read clock snapshot |

#### Memory + data motion (r16–r31)
| Op ID | Mnemonic | Semantics |
|---|---|---|
| H64:r16 | LOAD | mem → reg |
| H64:r17 | STORE | reg → mem |
| H64:r18 | MOV | reg copy |
| H64:r19 | SWAP | swap regs |
| H64:r20 | ALLOC | allocate region |
| H64:r21 | FREE | free region |
| H64:r22 | PUSH | push stack |
| H64:r23 | POP | pop stack |
| H64:r24 | VLOAD | vector load |
| H64:r25 | VSTORE | vector store |
| H64:r26 | GATHER | indexed load |
| H64:r27 | SCATTER | indexed store |
| H64:r28 | PACK | pack lanes |
| H64:r29 | UNPACK | unpack lanes |
| H64:r30 | CAST | bit/type cast |
| H64:r31 | ZERO | zero-fill |

#### Math + linear ops (r32–r47)
| Op ID | Mnemonic | Semantics |
|---|---|---|
| H64:r32 | ADD | add |
| H64:r33 | SUB | subtract |
| H64:r34 | MUL | multiply |
| H64:r35 | DIV | divide |
| H64:r36 | FMA | fused multiply-add |
| H64:r37 | ABS | abs |
| H64:r38 | SQRT | sqrt |
| H64:r39 | INV | reciprocal |
| H64:r40 | DOT | dot product |
| H64:r41 | NORM | norm |
| H64:r42 | MATMUL | matrix multiply |
| H64:r43 | SOLVE | linear solve step |
| H64:r44 | FFT | forward FFT |
| H64:r45 | IFFT | inverse FFT |
| H64:r46 | CONV | convolution |
| H64:r47 | REDUCE | reduction |

#### Rails + gates + ledger hooks (r48–r63)
| Op ID | Mnemonic | Semantics |
|---|---|---|
| H64:r48 | GATE_B | begin gate scope |
| H64:r49 | GATE_E | end gate scope |
| H64:r50 | CHECK | check invariant/residual |
| H64:r51 | CLAMP | clamp bounds |
| H64:r52 | FILTER | filter/dealias |
| H64:r53 | PROJECT | projection (e.g., Leray/constraint) |
| H64:r54 | CKPT | create checkpoint |
| H64:r55 | ROLLBACK | rollback |
| H64:r56 | EMIT | emit receipt packet |
| H64:r57 | TAG | attach thread/scope tags |
| H64:r58 | BUDGET | κ-DM budget op |
| H64:r59 | RATE | set/limit dt |
| H64:r60 | VERIFY | verify continuity |
| H64:r61 | SEAL | commit step |
| H64:r62 | WARN | soft warning (witnessed) |
| H64:r63 | SAFE | force safe-mode |

---

## 2) GHLL — Noetica Core Meaning Lexicon (v1.2)

This section defines the baseline lexicon. For detailed addition contracts and full entry definitions, see `GHLL_LEXICON_ADDITIONS_AND_CONTRACT.md`.

### 2.1 GHLL glyph kinds
| Kind | Prefix | Meaning role |
|---|---|---|
| TYPE | `N:TYPE.*` | type declarations |
| INV | `N:INV.*` | invariants (“must hold”) |
| GOAL | `N:GOAL.*` | objective targets |
| SPEC | `N:SPEC.*` | spec blocks |
| POLICY | `N:POLICY.*` | governance constraints |
| DOMAIN | `N:DOMAIN.*` | semantic domain tags |
| MAP | `N:MAP.*` | lowering maps (compiler seam) |
| META | `N:META.*` | provenance/annotations |

### 2.2 Canonical built-ins (IDs are stable across v1.x)

**Types**
- `N:TYPE.scalar`
- `N:TYPE.vector`
- `N:TYPE.matrix`
- `N:TYPE.field`
- `N:TYPE.lattice`
- `N:TYPE.clock`
- `N:TYPE.receipt`

**Invariants**
- `N:INV.pde.div_free`
- `N:INV.pde.energy_nonincreasing`
- `N:INV.clock.stage_coherence`
- `N:INV.ledger.hash_chain_intact`
- `N:INV.rails.gate_obligations_met`

**Goals**
- `N:GOAL.min_residual`
- `N:GOAL.max_stability_margin`
- `N:GOAL.min_wall_time_given_truth`

**Policies**
- `N:POLICY.rails_only_control`
- `N:POLICY.deterministic_replay`
- `N:POLICY.emit_receipts_every_step`
- `N:POLICY.rollback_on_gate_fail`
- `N:POLICY.safe_mode_on_repeat_fail`

**Domains**
- `N:DOMAIN.NS`
- `N:DOMAIN.GR_NR`
- `N:DOMAIN.RFE_UFE`
- `N:DOMAIN.ZETA`
- `N:DOMAIN.CONTROL`

**Required compiler seam maps (minimum set)**
- `N:MAP.inv.div_free.v1`
- `N:MAP.inv.energy_nonincreasing.v1`
- `N:MAP.inv.clock.stage_coherence.v1`
- `N:MAP.inv.ledger.hash_chain_intact.v1`

---

## 3) GML — Aeonica Thread + Receipt Codebook (v1.2)

This section defines the baseline receipts and threads. For detailed addition contracts, see `GML_LEXICON_ADDITIONS_AND_CONTRACT.md`. For the complete glyph library dictionary, see `GML_GLYPH_LIBRARY_DICTIONARY.md`.

### 3.1 PhaseLoom 27 threads (canonical IDs)
Axes:
- Domain: `PHY | CONS | SEM`
- Scale: `L | M | H`
- Response: `R0 | R1 | R2`

All 27 valid thread IDs:

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

### 3.2 Receipt events (canonical)
- `A:RCPT.step.proposed`
- `A:RCPT.step.accepted`
- `A:RCPT.step.rejected`
- `A:RCPT.gate.pass`
- `A:RCPT.gate.fail`
- `A:RCPT.check.invariant`
- `A:RCPT.ckpt.created`
- `A:RCPT.rollback.executed`
- `A:RCPT.run.summary`

### 3.3 Clock policy IDs (canonical)
- `A:CLOCK.policy.triaxis_v1`
- `A:CLOCK.mode.real_time`
- `A:CLOCK.mode.coherence_time`

---

## 4) Join rule (mandatory)
Every `A:RCPT.step.*` receipt **MUST** include:
- `intent_id` (GHLL join key)
- `ops[]` (GLLL join keys)
- hash chain fields (`hash_prev`, `hash`)

This is what makes the system replayable and forensic.