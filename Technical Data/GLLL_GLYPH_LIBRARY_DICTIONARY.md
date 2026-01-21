# GLLL_GLYPH_LIBRARY_DICTIONARY.md
# Triaxis (NPA) — GLLL Glyph Library / Dictionary (v1.2)

**Alphabet:** GLLL (Hadamard / Praxica-H)  
**Base order:** H64 (stable core)  
**Purpose:** A practical, implementation-ready dictionary of GLLL glyphs: IDs, mnemonics, classes, arguments, effects, determinism envelope, safe-mode status, and receipt obligations.

This document is **normative** for v1.2.

---

## 0) Conventions

### 0.1 Opcode identity
- `op_id` uses Hadamard row identity: `H64:rXX`
- Mnemonics are ASCII uppercase.
- `class` is one of: `CONTROL | MEMORY | MATH | RAILS | TIME | LEDGER`

### 0.2 Effects notation
Effects are declared as strings:
- `reads:<thing>`
- `writes:<thing>`
- `alloc:<thing>`
- `free:<thing>`
- `control:<thing>`
- `gate:<thing>`
- `emit:<thing>`
- `rollback:<thing>`

### 0.3 Receipt obligations
If an opcode has witness obligations, it specifies `receipt.must_record`.
All opcodes executed in a step must appear in `ops[]` of the step receipt.

### 0.4 Safe-mode policy
- `allowed` — permitted in safe-mode
- `restricted` — permitted only with stricter args constraints
- `forbidden` — must reject (TRAP-like behavior)

---

## 1) H64 Core Library Index

### 1.1 CONTROL glyphs (H64:r00–r15)

#### H64:r00 — NOP
- **class:** CONTROL  
- **args:** none  
- **effects:** `control:align`  
- **determinism:** trivial  
- **safe_mode:** allowed  
- **receipt:** none beyond inclusion in `ops[]`

#### H64:r01 — HALT
- **class:** CONTROL  
- **args_schema:** `{"reason":{"type":"string","max_len":256,"optional":true}}`  
- **effects:** `control:stop`  
- **determinism:** must emit final run summary before terminating evidence-grade run  
- **safe_mode:** allowed  
- **receipt.must_record:** `["reason_if_present"]`

#### H64:r02 — JMP
- **class:** CONTROL  
- **args_schema:** `{"target_block":{"type":"string","pattern":"^[A-Za-z0-9_\\.\\-]+$"}}`  
- **effects:** `control:pc_change`  
- **determinism:** target must be resolved deterministically (no dynamic lookup variance)  
- **safe_mode:** allowed  
- **receipt:** none beyond `ops[]`

#### H64:r03 — BR
- **class:** CONTROL  
- **args_schema:** `{"cond_reg":{"type":"string"},"true_block":{"type":"string"},"false_block":{"type":"string"}}`  
- **effects:** `reads:reg:cond`, `control:pc_change`  
- **determinism:** predicate evaluation must be deterministic  
- **safe_mode:** allowed  
- **receipt.must_record:** `["branch_taken"]` (recommended if debugging policy enabled)

#### H64:r04 — CALL
- **class:** CONTROL  
- **args_schema:** `{"callee":{"type":"string"},"argc":{"type":"int","min":0,"max":32}}`  
- **effects:** `control:call`  
- **determinism:** callee resolution must be static or manifest-declared  
- **safe_mode:** allowed  
- **receipt:** none beyond `ops[]`

#### H64:r05 — RET
- **class:** CONTROL  
- **args:** none  
- **effects:** `control:return`  
- **determinism:** return address must be deterministic  
- **safe_mode:** allowed

#### H64:r06 — LOOP_B
- **class:** CONTROL  
- **args_schema:** `{"loop_id":{"type":"string"},"max_iter":{"type":"int","min":1,"max":2147483647}}`  
- **effects:** `control:loop_begin`  
- **determinism:** loop bounds must be fixed or receipt-witnessed if adaptive  
- **safe_mode:** allowed

#### H64:r07 — LOOP_E
- **class:** CONTROL  
- **args_schema:** `{"loop_id":{"type":"string"}}`  
- **effects:** `control:loop_end`  
- **determinism:** loop iteration count must be deterministic  
- **safe_mode:** allowed  
- **receipt.must_record:** `["iters_executed"]` (required if adaptive)

#### H64:r08 — PHI
- **class:** CONTROL  
- **args_schema:** `{"dest":{"type":"string"},"incoming":{"type":"array","items":{"type":"object","keys":["pred","value"]}}}`  
- **effects:** `writes:reg:dest`  
- **determinism:** selection is determined by predecessor block  
- **safe_mode:** allowed

#### H64:r09 — SELECT
- **class:** CONTROL  
- **args_schema:** `{"dest":{"type":"string"},"cond_reg":{"type":"string"},"a":{"type":"string"},"b":{"type":"string"}}`  
- **effects:** `reads:reg:cond`, `reads:reg:a`, `reads:reg:b`, `writes:reg:dest`  
- **determinism:** deterministic predicate evaluation  
- **safe_mode:** allowed

#### H64:r10 — ASSERT
- **class:** CONTROL  
- **args_schema:** `{"predicate":{"type":"string"},"message":{"type":"string","max_len":512}}`  
- **effects:** `control:assert_check`  
- **determinism:** predicate must be fully determined by current state  
- **safe_mode:** allowed  
- **receipt.must_record:** `["assert_pass_fail","message"]`

#### H64:r11 — TRAP
- **class:** CONTROL  
- **args_schema:** `{"reason":{"type":"string","max_len":512}}`  
- **effects:** `control:hard_fail`  
- **determinism:** must emit `A:ALERT.trap` (or equivalent WARN) and terminate/rollback per policy  
- **safe_mode:** allowed  
- **receipt.must_record:** `["reason"]`

#### H64:r12 — SYNC
- **class:** CONTROL  
- **args_schema:** `{"scope":{"type":"string","enum":["thread","step","run"]}}`  
- **effects:** `control:barrier`  
- **determinism:** barrier ordering must be deterministic under scheduler policy  
- **safe_mode:** restricted (must not introduce nondeterministic ordering)

#### H64:r13 — YIELD
- **class:** CONTROL  
- **args_schema:** `{"reason":{"type":"string","max_len":256,"optional":true}}`  
- **effects:** `control:scheduler_yield`  
- **determinism:** must not change execution semantics; only scheduling under deterministic policy  
- **safe_mode:** allowed

#### H64:r14 — WAIT
- **class:** CONTROL  
- **args_schema:** `{"token":{"type":"string"},"timeout_ms":{"type":"int","min":0,"max":600000,"optional":true}}`  
- **effects:** `control:wait`  
- **determinism:** waiting must be bounded/deterministic or witnessed with scheduler policy  
- **safe_mode:** restricted

#### H64:r15 — TIME
- **class:** TIME  
- **args_schema:** `{"dest_t":{"type":"string","optional":true},"dest_tau":{"type":"string","optional":true}}`  
- **effects:** `reads:clock`, `writes:reg:time`  
- **determinism:** clock source must be declared (`A:CLOCK.policy.*`)  
- **safe_mode:** allowed  
- **receipt.must_record:** `["t","dt","tau","dtau"]` (in step receipt; required if TIME used)

---

### 1.2 MEMORY glyphs (H64:r16–r31)

#### H64:r16 — LOAD
- **class:** MEMORY  
- **args_schema:** `{"dest":{"type":"string"},"addr":{"type":"string"},"dtype":{"type":"string"}}`  
- **effects:** `reads:mem`, `writes:reg:dest`  
- **determinism:** address computation deterministic; memory layout declared  
- **safe_mode:** allowed

#### H64:r17 — STORE
- **class:** MEMORY  
- **args_schema:** `{"src":{"type":"string"},"addr":{"type":"string"},"dtype":{"type":"string"}}`  
- **effects:** `reads:reg:src`, `writes:mem`  
- **determinism:** deterministic addressing; no alias ambiguity unless declared  
- **safe_mode:** allowed

#### H64:r18 — MOV
- **class:** MEMORY  
- **args_schema:** `{"dest":{"type":"string"},"src":{"type":"string"}}`  
- **effects:** `reads:reg:src`, `writes:reg:dest`  
- **safe_mode:** allowed

#### H64:r19 — SWAP
- **class:** MEMORY  
- **args_schema:** `{"a":{"type":"string"},"b":{"type":"string"}}`  
- **effects:** `reads:reg:a`, `reads:reg:b`, `writes:reg:a`, `writes:reg:b`  
- **safe_mode:** allowed

#### H64:r20 — ALLOC
- **class:** MEMORY  
- **args_schema:** `{"region":{"type":"string"},"bytes":{"type":"int","min":1},"align":{"type":"int","min":1,"max":4096,"default":64}}`  
- **effects:** `alloc:mem:region`  
- **determinism:** allocator must be deterministic (arena recommended)  
- **safe_mode:** restricted (caps may apply)

#### H64:r21 — FREE
- **class:** MEMORY  
- **args_schema:** `{"region":{"type":"string"}}`  
- **effects:** `free:mem:region`  
- **determinism:** deterministic free order  
- **safe_mode:** restricted

#### H64:r22 — PUSH
- **class:** MEMORY  
- **args_schema:** `{"src":{"type":"string"}}`  
- **effects:** `reads:reg:src`, `writes:stack`  
- **safe_mode:** allowed

#### H64:r23 — POP
- **class:** MEMORY  
- **args_schema:** `{"dest":{"type":"string"}}`  
- **effects:** `reads:stack`, `writes:reg:dest`  
- **safe_mode:** allowed

#### H64:r24 — VLOAD
- **class:** MEMORY  
- **args_schema:** `{"dest":{"type":"string"},"addr":{"type":"string"},"lanes":{"type":"int","min":2,"max":64},"dtype":{"type":"string"}}`  
- **effects:** `reads:mem`, `writes:reg:dest`  
- **determinism:** deterministic alignment/lanes policy  
- **safe_mode:** allowed

#### H64:r25 — VSTORE
- **class:** MEMORY  
- **args_schema:** `{"src":{"type":"string"},"addr":{"type":"string"},"lanes":{"type":"int","min":2,"max":64},"dtype":{"type":"string"}}`  
- **effects:** `reads:reg:src`, `writes:mem`  
- **safe_mode:** allowed

#### H64:r26 — GATHER
- **class:** MEMORY  
- **args_schema:** `{"dest":{"type":"string"},"base_addr":{"type":"string"},"index_vec":{"type":"string"},"dtype":{"type":"string"}}`  
- **effects:** `reads:mem`, `reads:reg:index_vec`, `writes:reg:dest`  
- **determinism:** index ordering fixed; out-of-bounds policy declared  
- **safe_mode:** restricted

#### H64:r27 — SCATTER
- **class:** MEMORY  
- **args_schema:** `{"src":{"type":"string"},"base_addr":{"type":"string"},"index_vec":{"type":"string"},"dtype":{"type":"string"}}`  
- **effects:** `reads:reg:src`, `reads:reg:index_vec`, `writes:mem`  
- **determinism:** no write conflicts unless deterministic resolution declared  
- **safe_mode:** restricted

#### H64:r28 — PACK
- **class:** MEMORY  
- **args_schema:** `{"dest":{"type":"string"},"src":{"type":"string"},"layout":{"type":"string"}}`  
- **effects:** `reads:reg:src`, `writes:reg:dest`  
- **determinism:** layout must be manifest-declared  
- **safe_mode:** allowed

#### H64:r29 — UNPACK
- **class:** MEMORY  
- **args_schema:** `{"dest":{"type":"string"},"src":{"type":"string"},"layout":{"type":"string"}}`  
- **effects:** `reads:reg:src`, `writes:reg:dest`  
- **safe_mode:** allowed

#### H64:r30 — CAST
- **class:** MEMORY  
- **args_schema:** `{"dest":{"type":"string"},"src":{"type":"string"},"to_dtype":{"type":"string"}}`  
- **effects:** `reads:reg:src`, `writes:reg:dest`  
- **determinism:** rounding mode must be declared  
- **safe_mode:** allowed

#### H64:r31 — ZERO
- **class:** MEMORY  
- **args_schema:** `{"target":{"type":"string"},"bytes":{"type":"int","min":1}}`  
- **effects:** `writes:mem_or_reg`  
- **safe_mode:** allowed

---

### 1.3 MATH glyphs (H64:r32–r47)

#### H64:r32 — ADD
- **class:** MATH  
- **args_schema:** `{"dest":{"type":"string"},"a":{"type":"string"},"b":{"type":"string"}}`  
- **effects:** `reads:reg:a`, `reads:reg:b`, `writes:reg:dest`  
- **determinism:** IEEE-754 policy declared  
- **safe_mode:** allowed

#### H64:r33 — SUB
- same as ADD

#### H64:r34 — MUL
- same as ADD

#### H64:r35 — DIV
- **extra determinism:** division-by-zero policy declared  
- **safe_mode:** allowed (often restricted by policy)

#### H64:r36 — FMA
- **args_schema:** `{"dest":{"type":"string"},"a":{"type":"string"},"b":{"type":"string"},"c":{"type":"string"}}`  
- **effects:** `reads:reg:a`, `reads:reg:b`, `reads:reg:c`, `writes:reg:dest`  
- **determinism:** FMA must be either always used or never used (declared)  
- **safe_mode:** allowed

#### H64:r37 — ABS
- **args_schema:** `{"dest":{"type":"string"},"src":{"type":"string"}}`  
- **effects:** `reads:reg:src`, `writes:reg:dest`  
- **safe_mode:** allowed

#### H64:r38 — SQRT
- **args_schema:** `{"dest":{"type":"string"},"src":{"type":"string"}}`  
- **determinism:** domain error policy declared  
- **safe_mode:** allowed

#### H64:r39 — INV
- **args_schema:** `{"dest":{"type":"string"},"src":{"type":"string"}}`  
- **determinism:** same as DIV  
- **safe_mode:** allowed

#### H64:r40 — DOT
- **args_schema:** `{"dest":{"type":"string"},"x":{"type":"string"},"y":{"type":"string"},"n":{"type":"int","min":1}}`  
- **effects:** `reads:vec:x`, `reads:vec:y`, `writes:reg:dest`  
- **determinism:** reduction order must be fixed  
- **safe_mode:** allowed

#### H64:r41 — NORM
- **args_schema:** `{"dest":{"type":"string"},"x":{"type":"string"},"n":{"type":"int","min":1},"p":{"type":"string","enum":["2","inf"]}}`  
- **determinism:** reduction order fixed  
- **safe_mode:** allowed

#### H64:r42 — MATMUL
- **args_schema:** `{"dest":{"type":"string"},"A":{"type":"string"},"B":{"type":"string"},"m":{"type":"int","min":1},"k":{"type":"int","min":1},"n":{"type":"int","min":1},"layout":{"type":"string"}}`  
- **determinism:** block order fixed; parallel schedule deterministic  
- **safe_mode:** allowed

#### H64:r43 — SOLVE
- **args_schema:** `{"dest":{"type":"string"},"A":{"type":"string"},"b":{"type":"string"},"method":{"type":"string","enum":["cg","gmres","lu_small"]}}`  
- **determinism:** iteration caps fixed; convergence check deterministic  
- **safe_mode:** restricted (caps tighter)

#### H64:r44 — FFT
- **args_schema:** `{"dest":{"type":"string"},"src":{"type":"string"},"n":{"type":"int","min":2},"norm":{"type":"string","enum":["none","ortho"]}}`  
- **determinism:** FFT algorithm fixed; twiddle generation deterministic  
- **safe_mode:** allowed

#### H64:r45 — IFFT
- same obligations as FFT

#### H64:r46 — CONV
- **args_schema:** `{"dest":{"type":"string"},"src":{"type":"string"},"kernel":{"type":"string"},"mode":{"type":"string","enum":["valid","same"]}}`  
- **determinism:** kernel fixed and declared  
- **safe_mode:** allowed

#### H64:r47 — REDUCE
- **args_schema:** `{"dest":{"type":"string"},"src":{"type":"string"},"op":{"type":"string","enum":["sum","min","max"]},"n":{"type":"int","min":1}}`  
- **determinism:** reduction order fixed  
- **safe_mode:** allowed

---

### 1.4 RAILS / LEDGER glyphs (H64:r48–r63)

#### H64:r48 — GATE_B
- **class:** RAILS  
- **args_schema:** `{"gate":{"type":"string"},"params":{"type":"object","optional":true}}`  
- **effects:** `gate:enter`  
- **determinism:** params must be manifest-declared or args-digested  
- **safe_mode:** allowed  
- **receipt.must_record:** `["gate_name","params_digest_if_present"]`

#### H64:r49 — GATE_E
- **class:** RAILS  
- **args_schema:** `{"gate":{"type":"string","optional":true}}`  
- **effects:** `gate:exit`  
- **safe_mode:** allowed

#### H64:r50 — CHECK
- **class:** RAILS  
- **args_schema:** `{"check":{"type":"string"},"tol_abs":{"type":"string"},"tol_rel":{"type":"string","optional":true}}`  
- **effects:** `control:check`, `emit:residual`  
- **determinism:** check computation deterministic; norm fixed  
- **safe_mode:** allowed  
- **receipt.must_record:** `["residual_name","residual_value"]` (embedded in step receipt)

#### H64:r51 — CLAMP
- **class:** RAILS  
- **args_schema:** `{"target":{"type":"string"},"min":{"type":"string"},"max":{"type":"string"}}`  
- **effects:** `writes:target`  
- **determinism:** clamp bounds fixed  
- **safe_mode:** allowed  
- **receipt.must_record:** `["clamp_target","min","max"]` (optional)

#### H64:r52 — FILTER
- **class:** RAILS  
- **args_schema:** `{"filter":{"type":"string","enum":["dealias_23","lp_shell","hp_damp"]},"params":{"type":"object","optional":true}}`  
- **effects:** `writes:field`  
- **determinism:** filter implementation fixed; params digested  
- **safe_mode:** allowed  
- **receipt.must_record:** `["filter_id","params_digest_if_present"]`

#### H64:r53 — PROJECT
- **class:** RAILS  
- **args_schema:** `{"project":{"type":"string","enum":["leray","constraint"]},"target":{"type":"string","default":"field:v"}}`  
- **effects:** `reads:target`, `writes:target`  
- **determinism:** operator fixed; solver tolerances fixed  
- **safe_mode:** allowed  
- **receipt.must_record:** `["project_id","target"]`

#### H64:r54 — CKPT
- **class:** LEDGER  
- **args_schema:** `{"ckpt_id":{"type":"string"},"reason":{"type":"string","max_len":256}}`  
- **effects:** `emit:checkpoint`  
- **determinism:** state_digest algorithm fixed  
- **safe_mode:** allowed  
- **receipt.must_record:** `["ckpt_id","reason","state_digest"]` (in ckpt receipt)

#### H64:r55 — ROLLBACK
- **class:** LEDGER  
- **args_schema:** `{"rollback_to":{"type":"string"},"reason":{"type":"string","max_len":256,"optional":true}}`  
- **effects:** `rollback:state_restore`  
- **determinism:** restored_state_digest must match prior ckpt  
- **safe_mode:** allowed  
- **receipt.must_record:** `["rollback_to","restored_state_digest"]`

#### H64:r56 — EMIT
- **class:** LEDGER  
- **args_schema:** `{"emit":{"type":"string","enum":["A:RCPT.check.invariant","A:RCPT.gate.pass","A:RCPT.gate.fail","A:RCPT.step.accepted","A:RCPT.step.rejected"]}}`  
- **effects:** `emit:receipt`  
- **determinism:** canonical JSON rules must be used  
- **safe_mode:** allowed

#### H64:r57 — TAG
- **class:** LEDGER  
- **args_schema:** `{"thread":{"type":"string"},"scope":{"type":"string","enum":["step","gate","run"]}}`  
- **effects:** `control:tag_thread`  
- **determinism:** thread must be one of the 27 canonical PhaseLoom threads  
- **safe_mode:** allowed  
- **receipt.must_record:** `["thread","scope"]`

#### H64:r58 — BUDGET
- **class:** RAILS  
- **args_schema:** `{"consume":{"type":"string"},"recharge":{"type":"string","optional":true},"floor":{"type":"string"}}`  
- **effects:** `reads:kappa`, `writes:kappa`  
- **determinism:** budget update must be pure and recorded  
- **safe_mode:** allowed  
- **receipt.must_record:** `["kappa_used","kappa_floor"]`

#### H64:r59 — RATE
- **class:** TIME  
- **args_schema:** `{"dt":{"type":"string"},"mode":{"type":"string","enum":["set","limit_max","limit_min"]}}`  
- **effects:** `writes:dt_policy`  
- **determinism:** arbitration policy must be declared  
- **safe_mode:** allowed  
- **receipt.must_record:** `["dt_applied","dt_mode"]`

#### H64:r60 — VERIFY
- **class:** LEDGER  
- **args_schema:** `{"what":{"type":"string","enum":["hash_chain","state_digest"]}}`  
- **effects:** `control:verify`  
- **determinism:** sha256 + canonical JSON rules fixed  
- **safe_mode:** allowed  
- **receipt.must_record:** `["verify_target","verify_pass_fail"]`

#### H64:r61 — SEAL
- **class:** LEDGER  
- **args_schema:** `{"commit":{"type":"string","enum":["step"]}}`  
- **effects:** `emit:step_receipt`  
- **determinism:** acceptance must be derivable from recorded gates/residuals  
- **safe_mode:** allowed  
- **receipt.must_record:** `["status","reason_if_rejected"]`

#### H64:r62 — WARN
- **class:** LEDGER  
- **args_schema:** `{"code":{"type":"string"},"message":{"type":"string","max_len":512}}`  
- **effects:** `emit:alert`  
- **determinism:** warn emission must be deterministic given the condition  
- **safe_mode:** allowed  
- **receipt.must_record:** `["code","message"]`

#### H64:r63 — SAFE
- **class:** RAILS  
- **args_schema:** `{"mode":{"type":"string","enum":["enter","exit"]},"reason":{"type":"string","max_len":256}}`  
- **effects:** `control:safe_mode_toggle`  
- **determinism:** safe-mode policy must be declared in manifest  
- **safe_mode:** allowed  
- **receipt.must_record:** `["mode","reason"]`

---

## 2) Dictionary API (how other layers reference GLLL)

### 2.1 Required fields in Aeonica step receipt
Every step receipt that claims execution must include:
- `ops[]` containing the opcode IDs in order

Optional but recommended:
- `op_args_digests[]` aligned with `ops[]` (sha256 of canonical args)

### 2.2 Compiler seam references
`N:MAP.*` lowering maps in GHLL must reference opcodes only by `H64:rXX` IDs (or higher order IDs if enabled).

---

## 3) Minimal runtime support matrix (v1.2)
A conformant runtime must implement at least:
- GATE_B, CHECK, EMIT, SEAL (`H64:r48`, `r50`, `r56`, `r61`)
- CKPT, ROLLBACK (`H64:r54`, `r55`) if any policy uses rollback
- PROJECT, FILTER for PDE/NR engines (`H64:r53`, `r52`) if those domains are enabled

---

## 4) Reserved for expansion
This v1.2 dictionary is the stable H64 set. If/when H128 is enabled, the extension pack is governed by:
- `GLLL_LEXICON_ADDITIONS_AND_CONTRACT.md` §8 (H128 additions)

---

## 5) Summary
This dictionary is the concrete “action alphabet” of Triaxis:
- fixed opcode IDs (Hadamard rows),
- explicit semantics and determinism envelopes,
- explicit witness obligations,
- safe-mode rules.

It is small on purpose: power comes from GHLL meaning + compiler seam maps + GML truth receipts.