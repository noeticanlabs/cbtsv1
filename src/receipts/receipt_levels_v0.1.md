# Receipt Levels v0.1

## Overview
This document defines the two-speed logging system for performance optimization in the GR solver. Receipts are emitted at three frequencies to balance auditability, performance, and diagnostics.

## Receipt Levels

### M_step (Micro-Step Receipts)
- **Frequency**: Every integration step
- **Purpose**: Must-have sparse canonical data for replay/verify
- **Contents**:
  - Step number, time t, dt
  - Dominant thread
  - Threads summary (minimal)
  - Constraints: eps_pre/post_H/M, d_eps_H/M
  - Geometry: det_gamma_min, R_max, lambda_min/max, cond_gamma (sparse)
  - Damping: mu_H, mu_M
  - Rails: rollback flag, reason (if any)
  - Time audit: t_expected, t_err
  - Commitment: t_prev/next, dt_selected/applied, substeps, commit_ok
  - Consistency_ok
  - Policy_hash
  - Timestamp, lexicon, modules

### M_solve (Solve Receipts)
- **Frequency**: Every nonlinear solve (typically per substep)
- **Purpose**: High-frequency diagnostics, optional snapshots, performance counters
- **Contents**:
  - All M_step data
  - Loom data: C_global, C/D_band stats, D_max, top3_bands
  - Risk_gauge
  - Tight_threads
  - Rails margins, repair_applied/type, lambda_min_pre/post
  - Performance counters: solve iterations, convergence metrics
  - Optional: full snapshots of key fields (if enabled)

### Macro Receipts
- **Frequency**: Every K=100 steps
- **Purpose**: Periodic aggregates with hashes for long-term audit
- **Contents**:
  - Aggregates over last 100 steps: min/max/mean of constraints, geometry metrics
  - Hashes of critical data segments (e.g., SHA256 of receipt chain)
  - Performance aggregates: total solves, avg solve time, rollback counts
  - Milestone markers if reached

## Implementation Notes
- Receipts are stored in memory until flushed periodically.
- M_step and M_solve are JSON-serializable for efficiency.
- Macro receipts include cryptographic hashes for integrity.