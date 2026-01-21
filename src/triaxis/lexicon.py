# Triaxis Lexicon Module (v1.2)
# Canonical glyph codebook bindings for Noetica-Praxica-Aeonica integration

# GHLL - Noetica Core Meaning Lexicon
class GHLL:
    # Types
    TYPE_SCALAR = "N:TYPE.scalar"
    TYPE_VECTOR = "N:TYPE.vector"
    TYPE_MATRIX = "N:TYPE.matrix"
    TYPE_FIELD = "N:TYPE.field"
    TYPE_LATTICE = "N:TYPE.lattice"
    TYPE_CLOCK = "N:TYPE.clock"
    TYPE_RECEIPT = "N:TYPE.receipt"

    # Invariants
    INV_PDE_DIV_FREE = "N:INV.pde.div_free"
    INV_PDE_ENERGY_NONINCREASING = "N:INV.pde.energy_nonincreasing"
    INV_CLOCK_STAGE_COHERENCE = "N:INV.clock.stage_coherence"
    INV_LEDGER_HASH_CHAIN_INTACT = "N:INV.ledger.hash_chain_intact"
    INV_RAILS_GATE_OBLIGATIONS_MET = "N:INV.rails.gate_obligations_met"

    # Goals
    GOAL_MIN_RESIDUAL = "N:GOAL.min_residual"
    GOAL_MAX_STABILITY_MARGIN = "N:GOAL.max_stability_margin"
    GOAL_MIN_WALL_TIME_GIVEN_TRUTH = "N:GOAL.min_wall_time_given_truth"

    # Policies
    POLICY_RAILS_ONLY_CONTROL = "N:POLICY.rails_only_control"
    POLICY_DETERMINISTIC_REPLAY = "N:POLICY.deterministic_replay"
    POLICY_EMIT_RECEIPTS_EVERY_STEP = "N:POLICY.emit_receipts_every_step"
    POLICY_ROLLBACK_ON_GATE_FAIL = "N:POLICY.rollback_on_gate_fail"
    POLICY_SAFE_MODE_ON_REPEAT_FAIL = "N:POLICY.safe_mode_on_repeat_fail"

    # Domains
    DOMAIN_NS = "N:DOMAIN.NS"
    DOMAIN_GR_NR = "N:DOMAIN.GR_NR"
    DOMAIN_RFE_UFE = "N:DOMAIN.RFE_UFE"
    DOMAIN_ZETA = "N:DOMAIN.ZETA"
    DOMAIN_CONTROL = "N:DOMAIN.CONTROL"

    # Maps
    MAP_INV_DIV_FREE_V1 = "N:MAP.inv.div_free.v1"
    MAP_INV_ENERGY_NONINCREASING_V1 = "N:MAP.inv.energy_nonincreasing.v1"
    MAP_INV_CLOCK_STAGE_COHERENCE_V1 = "N:MAP.inv.clock.stage_coherence.v1"
    MAP_INV_LEDGER_HASH_CHAIN_INTACT_V1 = "N:MAP.inv.ledger.hash_chain_intact.v1"

# GML - Aeonica Thread + Receipt Codebook
class GML:
    # PhaseLoom 27 Threads
    # PHY domain
    THREAD_PHY_L_R0 = "A:THREAD.PHY.L.R0"
    THREAD_PHY_L_R1 = "A:THREAD.PHY.L.R1"
    THREAD_PHY_L_R2 = "A:THREAD.PHY.L.R2"
    THREAD_PHY_M_R0 = "A:THREAD.PHY.M.R0"
    THREAD_PHY_M_R1 = "A:THREAD.PHY.M.R1"
    THREAD_PHY_M_R2 = "A:THREAD.PHY.M.R2"
    THREAD_PHY_H_R0 = "A:THREAD.PHY.H.R0"
    THREAD_PHY_H_R1 = "A:THREAD.PHY.H.R1"
    THREAD_PHY_H_R2 = "A:THREAD.PHY.H.R2"

    # CONS domain
    THREAD_CONS_L_R0 = "A:THREAD.CONS.L.R0"
    THREAD_CONS_L_R1 = "A:THREAD.CONS.L.R1"
    THREAD_CONS_L_R2 = "A:THREAD.CONS.L.R2"
    THREAD_CONS_M_R0 = "A:THREAD.CONS.M.R0"
    THREAD_CONS_M_R1 = "A:THREAD.CONS.M.R1"
    THREAD_CONS_M_R2 = "A:THREAD.CONS.M.R2"
    THREAD_CONS_H_R0 = "A:THREAD.CONS.H.R0"
    THREAD_CONS_H_R1 = "A:THREAD.CONS.H.R1"
    THREAD_CONS_H_R2 = "A:THREAD.CONS.H.R2"

    # SEM domain
    THREAD_SEM_L_R0 = "A:THREAD.SEM.L.R0"
    THREAD_SEM_L_R1 = "A:THREAD.SEM.L.R1"
    THREAD_SEM_L_R2 = "A:THREAD.SEM.L.R2"
    THREAD_SEM_M_R0 = "A:THREAD.SEM.M.R0"
    THREAD_SEM_M_R1 = "A:THREAD.SEM.M.R1"
    THREAD_SEM_M_R2 = "A:THREAD.SEM.M.R2"
    THREAD_SEM_H_R0 = "A:THREAD.SEM.H.R0"
    THREAD_SEM_H_R1 = "A:THREAD.SEM.H.R1"
    THREAD_SEM_H_R2 = "A:THREAD.SEM.H.R2"

    # Receipt Events
    RCPT_STEP_PROPOSED = "A:RCPT.step.proposed"
    RCPT_STEP_ACCEPTED = "A:RCPT.step.accepted"
    RCPT_STEP_REJECTED = "A:RCPT.step.rejected"
    RCPT_GATE_PASS = "A:RCPT.gate.pass"
    RCPT_GATE_FAIL = "A:RCPT.gate.fail"
    RCPT_CHECK_INVARIANT = "A:RCPT.check.invariant"
    RCPT_CKPT_CREATED = "A:RCPT.ckpt.created"
    RCPT_ROLLBACK_EXECUTED = "A:RCPT.rollback.executed"
    RCPT_RUN_SUMMARY = "A:RCPT.run.summary"

    # Clock Policies
    CLOCK_POLICY_TRIAXIS_V1 = "A:CLOCK.policy.triaxis_v1"
    CLOCK_MODE_REAL_TIME = "A:CLOCK.mode.real_time"
    CLOCK_MODE_COHERENCE_TIME = "A:CLOCK.mode.coherence_time"

# GLLL - Praxica-H Hadamard Opcode Codebook
class GLLL:
    # Control + flow (r00–r15)
    R00 = "H64:r00"  # NOP
    R01 = "H64:r01"  # HALT
    R02 = "H64:r02"  # JMP
    R03 = "H64:r03"  # BR
    R04 = "H64:r04"  # CALL
    R05 = "H64:r05"  # RET
    R06 = "H64:r06"  # LOOP_B
    R07 = "H64:r07"  # LOOP_E
    R08 = "H64:r08"  # PHI
    R09 = "H64:r09"  # SELECT
    R10 = "H64:r10"  # ASSERT
    R11 = "H64:r11"  # TRAP
    R12 = "H64:r12"  # SYNC
    R13 = "H64:r13"  # YIELD
    R14 = "H64:r14"  # WAIT
    R15 = "H64:r15"  # TIME

    # Memory + data motion (r16–r31)
    R16 = "H64:r16"  # LOAD
    R17 = "H64:r17"  # STORE
    R18 = "H64:r18"  # MOV
    R19 = "H64:r19"  # SWAP
    R20 = "H64:r20"  # ALLOC
    R21 = "H64:r21"  # FREE
    R22 = "H64:r22"  # PUSH
    R23 = "H64:r23"  # POP
    R24 = "H64:r24"  # VLOAD
    R25 = "H64:r25"  # VSTORE
    R26 = "H64:r26"  # GATHER
    R27 = "H64:r27"  # SCATTER
    R28 = "H64:r28"  # PACK
    R29 = "H64:r29"  # UNPACK
    R30 = "H64:r30"  # CAST
    R31 = "H64:r31"  # ZERO

    # Math + linear ops (r32–r47)
    R32 = "H64:r32"  # ADD
    R33 = "H64:r33"  # SUB
    R34 = "H64:r34"  # MUL
    R35 = "H64:r35"  # DIV
    R36 = "H64:r36"  # FMA
    R37 = "H64:r37"  # ABS
    R38 = "H64:r38"  # SQRT
    R39 = "H64:r39"  # INV
    R40 = "H64:r40"  # DOT
    R41 = "H64:r41"  # NORM
    R42 = "H64:r42"  # MATMUL
    R43 = "H64:r43"  # SOLVE
    R44 = "H64:r44"  # FFT
    R45 = "H64:r45"  # IFFT
    R46 = "H64:r46"  # CONV
    R47 = "H64:r47"  # REDUCE

    # Rails + gates + ledger hooks (r48–r63)
    R48 = "H64:r48"  # GATE_B
    R49 = "H64:r49"  # GATE_E
    R50 = "H64:r50"  # CHECK
    R51 = "H64:r51"  # CLAMP
    R52 = "H64:r52"  # FILTER
    R53 = "H64:r53"  # PROJECT
    R54 = "H64:r54"  # CKPT
    R55 = "H64:r55"  # ROLLBACK
    R56 = "H64:r56"  # EMIT
    R57 = "H64:r57"  # TAG
    R58 = "H64:r58"  # BUDGET
    R59 = "H64:r59"  # RATE
    R60 = "H64:r60"  # VERIFY
    R61 = "H64:r61"  # SEAL
    R62 = "H64:r62"  # WARN
    R63 = "H64:r63"  # SAFE

    # Convenience mappings with mnemonics
    NOP = R00
    HALT = R01
    JMP = R02
    BR = R03
    CALL = R04
    RET = R05
    LOOP_B = R06
    LOOP_E = R07
    PHI = R08
    SELECT = R09
    ASSERT = R10
    TRAP = R11
    SYNC = R12
    YIELD = R13
    WAIT = R14
    TIME = R15
    LOAD = R16
    STORE = R17
    MOV = R18
    SWAP = R19
    ALLOC = R20
    FREE = R21
    PUSH = R22
    POP = R23
    VLOAD = R24
    VSTORE = R25
    GATHER = R26
    SCATTER = R27
    PACK = R28
    UNPACK = R29
    CAST = R30
    ZERO = R31
    ADD = R32
    SUB = R33
    MUL = R34
    DIV = R35
    FMA = R36
    ABS = R37
    SQRT = R38
    INV = R39
    DOT = R40
    NORM = R41
    MATMUL = R42
    SOLVE = R43
    FFT = R44
    IFFT = R45
    CONV = R46
    REDUCE = R47
    GATE_B = R48
    GATE_E = R49
    CHECK = R50
    CLAMP = R51
    FILTER = R52
    PROJECT = R53
    CKPT = R54
    ROLLBACK = R55
    EMIT = R56
    TAG = R57
    BUDGET = R58
    RATE = R59
    VERIFY = R60
    SEAL = R61
    WARN = R62
    SAFE = R63