# NSC Invariant Registry and Receipt Field Specifications

**Document Version**: 1.0  
**Created**: 2026-01-31  
**Objective**: Define invariant registry entries and receipt field specifications for GR/NS/YM domains

---

## 1. Overview

This document defines the canonical invariant registry entries and Aeonica receipt field specifications for the Noetica/NSC/Triaxis ecosystem. It provides concrete names, IDs, and field keys for:

- **GR Domain**: Hamiltonian/momentum constraints, metric positivity, curvature bounds
- **NS Domain**: Divergence-free, energy dissipation, CFL stability, tail barrier
- **YM Domain**: Gauss law, Bianchi identity, Ward identity, gauge conditions

All entries are compatible with the existing `terminology_registry.json` and `specifications/aeonica/42_AEONICA_RECEIPTS.md`.

---

## 2. Terminology Registry Additions

Add the following entries to `terminology_registry.json` under `ghll.terms`:

### 2.1 Domain Declarations

```json
"DOMAIN_NAVIER_STOKES": {
  "id": "N:DOMAIN.NAVIER_STOKES",
  "category": "domain",
  "status": "reserved",
  "description": "3D incompressible Navier–Stokes domain (LoC/GCAT controlled)"
},
"DOMAIN_YM": {
  "id": "N:DOMAIN.YM",
  "category": "domain",
  "status": "reserved",
  "description": "Yang–Mills domain (classical + Clay-ledger obligations)"
}
```

> **Note**: The existing `DOMAIN_NS` is described as "Numerical Schwarzschild," so `DOMAIN_NAVIER_STOKES` is used for Navier–Stokes to avoid semantic collisions.

### 2.2 GR Invariants (Implemented-Compatible)

These correspond to what the GR host API already computes: `eps_H`, `eps_M`, `R_max`, `H`, `dH`.

```json
"INV_GR_HAMILTONIAN_CONSTRAINT": {
  "id": "N:INV.gr.hamiltonian_constraint",
  "category": "invariant",
  "status": "implemented",
  "description": "Hamiltonian constraint residual bounded (eps_H <= eps_H_max). Source metric: eps_H."
},
"INV_GR_MOMENTUM_CONSTRAINT": {
  "id": "N:INV.gr.momentum_constraint",
  "category": "invariant",
  "status": "implemented",
  "description": "Momentum constraint residual bounded (eps_M <= eps_M_max). Source metric: eps_M."
},
"INV_GR_DET_GAMMA_POSITIVE": {
  "id": "N:INV.gr.det_gamma_positive",
  "category": "invariant",
  "status": "implemented",
  "description": "Spatial metric determinant stays positive (det_gamma_min > 0). Prevents signature/volume-form failure."
},
"INV_GR_CURVATURE_BOUNDED": {
  "id": "N:INV.gr.curvature_bounded",
  "category": "invariant",
  "status": "implemented",
  "description": "Scalar curvature indicator bounded (R_max <= R_max_limit). Source metric: R_max."
},
"INV_GR_ENERGY_DRIFT_BOUNDED": {
  "id": "N:INV.gr.energy_drift_bounded",
  "category": "invariant",
  "status": "implemented",
  "description": "Constraint-energy drift bounded per step (dH <= dH_max). Source metric: dH."
},
"INV_GR_LAPSE_POSITIVE": {
  "id": "N:INV.gr.lapse_positive",
  "category": "invariant",
  "status": "reserved",
  "description": "Lapse remains positive (alpha_min > 0). Recommended PHY hard gate for stable slicing."
}
```

### 2.3 GR Mapping IDs

```json
"MAP_INV_GR_HAMILTONIAN_CONSTRAINT_V1": {
  "id": "N:MAP.inv.gr.hamiltonian_constraint.v1",
  "category": "map",
  "status": "implemented",
  "description": "Maps invariant N:INV.gr.hamiltonian_constraint to receipt residuals.eps_H and gate thresholds."
},
"MAP_INV_GR_MOMENTUM_CONSTRAINT_V1": {
  "id": "N:MAP.inv.gr.momentum_constraint.v1",
  "category": "map",
  "status": "implemented",
  "description": "Maps invariant N:INV.gr.momentum_constraint to receipt residuals.eps_M and gate thresholds."
},
"MAP_INV_GR_DET_GAMMA_POSITIVE_V1": {
  "id": "N:MAP.inv.gr.det_gamma_positive.v1",
  "category": "map",
  "status": "implemented",
  "description": "Maps invariant N:INV.gr.det_gamma_positive to receipt metrics.det_gamma_min and positivity rule."
},
"MAP_INV_GR_CURVATURE_BOUNDED_V1": {
  "id": "N:MAP.inv.gr.curvature_bounded.v1",
  "category": "map",
  "status": "implemented",
  "description": "Maps invariant N:INV.gr.curvature_bounded to receipt metrics.R_max and configured limit."
},
"MAP_INV_GR_ENERGY_DRIFT_BOUNDED_V1": {
  "id": "N:MAP.inv.gr.energy_drift_bounded.v1",
  "category": "map",
  "status": "implemented",
  "description": "Maps invariant N:INV.gr.energy_drift_bounded to receipt metrics.dH and configured drift cap."
}
```

### 2.4 Navier–Stokes Invariants

```json
"INV_NS_DIV_FREE": {
  "id": "N:INV.ns.div_free",
  "category": "invariant",
  "status": "implemented",
  "description": "Incompressibility: ||div u|| bounded. Alias of N:INV.pde.div_free.",
  "aliases": ["N:INV.pde.div_free"]
},
"INV_NS_ENERGY_NONINCREASING": {
  "id": "N:INV.ns.energy_nonincreasing",
  "category": "invariant",
  "status": "implemented",
  "description": "Viscous dissipation: kinetic energy non-increasing (or decreases within budget). Alias of N:INV.pde.energy_nonincreasing.",
  "aliases": ["N:INV.pde.energy_nonincreasing"]
},
"INV_NS_CFL_STABILITY": {
  "id": "N:INV.ns.cfl_stability",
  "category": "invariant",
  "status": "reserved",
  "description": "CFL stability condition holds (dt <= cfl_limit). Gate uses u_max, dx, nu as configured."
},
"INV_NS_TAIL_BARRIER": {
  "id": "N:INV.ns.tail_barrier",
  "category": "invariant",
  "status": "reserved",
  "description": "LoC spectral tail barrier holds (high-k energy tail stays bounded; no forward cascade to infinity)."
},
"INV_NS_ENSTROPHY_BUDGET": {
  "id": "N:INV.ns.enstrophy_budget",
  "category": "invariant",
  "status": "reserved",
  "description": "Enstrophy (||curl u||^2) stays within declared budget on each step/window."
}
```

### 2.5 NS Mapping IDs

```json
"MAP_INV_NS_DIV_FREE_V1": {
  "id": "N:MAP.inv.ns.div_free.v1",
  "category": "map",
  "status": "implemented",
  "description": "Maps N:INV.ns.div_free to receipt residuals.eps_div (or eps_div_L2) and threshold."
},
"MAP_INV_NS_ENERGY_NONINCREASING_V1": {
  "id": "N:MAP.inv.ns.energy_nonincreasing.v1",
  "category": "map",
  "status": "implemented",
  "description": "Maps N:INV.ns.energy_nonincreasing to receipt metrics.E, metrics.dE and tolerance/budget."
},
"MAP_INV_NS_CFL_STABILITY_V1": {
  "id": "N:MAP.inv.ns.cfl_stability.v1",
  "category": "map",
  "status": "reserved",
  "description": "Maps N:INV.ns.cfl_stability to receipt metrics.cfl_ratio (<= 1)."
},
"MAP_INV_NS_TAIL_BARRIER_V1": {
  "id": "N:MAP.inv.ns.tail_barrier.v1",
  "category": "map",
  "status": "reserved",
  "description": "Maps N:INV.ns.tail_barrier to receipt metrics.S_j_max, metrics.j_crit, residuals.eps_tail."
},
"MAP_INV_NS_ENSTROPHY_BUDGET_V1": {
  "id": "N:MAP.inv.ns.enstrophy_budget.v1",
  "category": "map",
  "status": "reserved",
  "description": "Maps N:INV.ns.enstrophy_budget to receipt metrics.enstrophy, metrics.d_enstrophy."
}
```

### 2.6 Yang–Mills Invariants

```json
"INV_YM_GAUSS_LAW": {
  "id": "N:INV.ym.gauss_law",
  "category": "invariant",
  "status": "reserved",
  "description": "Gauss constraint holds: ||D_i E_i|| bounded. Receipt residual: eps_G."
},
"INV_YM_BIANCHI_IDENTITY": {
  "id": "N:INV.ym.bianchi_identity",
  "category": "invariant",
  "status": "reserved",
  "description": "Bianchi identity coherent: ||D_[i F_jk]|| bounded. Receipt residual: eps_BI."
},
"INV_YM_WARD_IDENTITY": {
  "id": "N:INV.ym.ward_identity",
  "category": "invariant",
  "status": "reserved",
  "description": "Ward identity residual bounded (gauge-coherence witness). Receipt residual: eps_W."
},
"INV_YM_ENERGY_BUDGET": {
  "id": "N:INV.ym.energy_budget",
  "category": "invariant",
  "status": "reserved",
  "description": "Yang–Mills energy budget holds: E_total drift stays within declared tolerance/budget."
},
"INV_YM_GAUGE_CONDITION": {
  "id": "N:INV.ym.gauge_condition",
  "category": "invariant",
  "status": "reserved",
  "description": "Chosen gauge condition residual bounded (e.g., Lorenz gauge). Receipt residual: eps_gauge."
}
```

### 2.7 YM Mapping IDs

```json
"MAP_INV_YM_GAUSS_LAW_V1": {
  "id": "N:MAP.inv.ym.gauss_law.v1",
  "category": "map",
  "status": "reserved",
  "description": "Maps N:INV.ym.gauss_law to receipt residuals.eps_G and threshold."
},
"MAP_INV_YM_BIANCHI_IDENTITY_V1": {
  "id": "N:MAP.inv.ym.bianchi_identity.v1",
  "category": "map",
  "status": "reserved",
  "description": "Maps N:INV.ym.bianchi_identity to receipt residuals.eps_BI and threshold."
},
"MAP_INV_YM_WARD_IDENTITY_V1": {
  "id": "N:MAP.inv.ym.ward_identity.v1",
  "category": "map",
  "status": "reserved",
  "description": "Maps N:INV.ym.ward_identity to receipt residuals.eps_W and threshold."
},
"MAP_INV_YM_ENERGY_BUDGET_V1": {
  "id": "N:MAP.inv.ym.energy_budget.v1",
  "category": "map",
  "status": "reserved",
  "description": "Maps N:INV.ym.energy_budget to receipt metrics.E_total, metrics.dE and tolerance/budget."
},
"MAP_INV_YM_GAUGE_CONDITION_V1": {
  "id": "N:MAP.inv.ym.gauge_condition.v1",
  "category": "map",
  "status": "reserved",
  "description": "Maps N:INV.ym.gauge_condition to receipt residuals.eps_gauge and threshold."
}
```

---

## 3. Receipt Field Specifications

This section defines the domain-specific payload for `A:RCPT.step.accepted` and `A:RCPT.step.rejected` receipts, following the Aeonica v1.2 convention.

### 3.1 GR Step Receipt Fields (Domain: `N:DOMAIN.GR_NR`)

#### `gates` Keys

```json
{
  "hamiltonian_constraint": {
    "status": "pass|fail",
    "eps_H": "<decimal>",
    "eps_H_max": "<decimal>",
    "margin": "<decimal>"
  },
  "momentum_constraint": {
    "status": "pass|fail",
    "eps_M": "<decimal>",
    "eps_M_max": "<decimal>",
    "margin": "<decimal>"
  },
  "det_gamma_positive": {
    "status": "pass|fail",
    "det_gamma_min": "<decimal>",
    "limit": "0",
    "margin": "<decimal>"
  },
  "curvature_bounded": {
    "status": "pass|fail",
    "R_max": "<decimal>",
    "R_max_limit": "<decimal>",
    "margin": "<decimal>"
  },
  "energy_drift_bounded": {
    "status": "pass|fail",
    "dH": "<decimal>",
    "dH_max": "<decimal>",
    "margin": "<decimal>"
  },
  "clock_stage_coherence": {
    "status": "pass|fail",
    "delta_stage_t": "<decimal>"
  },
  "ledger_hash_chain": {
    "status": "pass|fail"
  }
}
```

#### `residuals` Keys

```json
{
  "eps_H": "<decimal>",
  "eps_M": "<decimal>"
}
```

#### `metrics` Keys

```json
{
  "R_max": "<decimal>",
  "det_gamma_min": "<decimal>",
  "H": "<decimal>",
  "dH": "<decimal>",
  "dt_CFL": "<decimal>",
  "dt_gauge": "<decimal>",
  "dt_coh": "<decimal>",
  "dt_res": "<decimal>",
  "dt_used": "<decimal>"
}
```

#### `invariants_enforced` List

```json
[
  "N:INV.gr.hamiltonian_constraint",
  "N:INV.gr.momentum_constraint",
  "N:INV.gr.det_gamma_positive",
  "N:INV.gr.curvature_bounded",
  "N:INV.gr.energy_drift_bounded",
  "N:INV.clock.stage_coherence",
  "N:INV.ledger.hash_chain_intact"
]
```

---

### 3.2 NS Step Receipt Fields (Domain: `N:DOMAIN.NAVIER_STOKES`)

#### `gates` Keys

```json
{
  "div_free": {
    "status": "pass|fail",
    "eps_div": "<decimal>",
    "eps_div_max": "<decimal>",
    "margin": "<decimal>"
  },
  "energy_nonincreasing": {
    "status": "pass|fail",
    "E": "<decimal>",
    "dE": "<decimal>",
    "budget": "<decimal>",
    "margin": "<decimal>"
  },
  "cfl_stability": {
    "status": "pass|fail",
    "cfl_ratio": "<decimal>",
    "limit": "1",
    "margin": "<decimal>"
  },
  "tail_barrier": {
    "status": "pass|fail",
    "S_j_max": "<decimal>",
    "c_star": "<decimal>",
    "j_crit": "<decimal>"
  },
  "enstrophy_budget": {
    "status": "pass|fail",
    "enstrophy": "<decimal>",
    "budget": "<decimal>",
    "margin": "<decimal>"
  },
  "clock_stage_coherence": {
    "status": "pass|fail",
    "delta_stage_t": "<decimal>"
  },
  "ledger_hash_chain": {
    "status": "pass|fail"
  }
}
```

#### `residuals` Keys

```json
{
  "eps_div": "<decimal>",
  "eps_tail": "<decimal>"
}
```

#### `metrics` Keys

```json
{
  "E": "<decimal>",
  "dE": "<decimal>",
  "enstrophy": "<decimal>",
  "d_enstrophy": "<decimal>",
  "u_max": "<decimal>",
  "dx_min": "<decimal>",
  "nu": "<decimal>",
  "cfl_ratio": "<decimal>",
  "S_j_max": "<decimal>",
  "j_crit": "<decimal>",
  "dt_used": "<decimal>"
}
```

#### `invariants_enforced` List

```json
[
  "N:INV.ns.div_free",
  "N:INV.ns.energy_nonincreasing",
  "N:INV.ns.cfl_stability",
  "N:INV.ns.tail_barrier",
  "N:INV.ns.enstrophy_budget",
  "N:INV.clock.stage_coherence",
  "N:INV.ledger.hash_chain_intact"
]
```

---

### 3.3 YM Step Receipt Fields (Domain: `N:DOMAIN.YM`)

#### `gates` Keys

```json
{
  "gauss_law": {
    "status": "pass|fail",
    "eps_G": "<decimal>",
    "eps_G_max": "<decimal>",
    "margin": "<decimal>"
  },
  "bianchi_identity": {
    "status": "pass|fail",
    "eps_BI": "<decimal>",
    "eps_BI_max": "<decimal>",
    "margin": "<decimal>"
  },
  "ward_identity": {
    "status": "pass|fail",
    "eps_W": "<decimal>",
    "eps_W_max": "<decimal>",
    "margin": "<decimal>"
  },
  "gauge_condition": {
    "status": "pass|fail",
    "eps_gauge": "<decimal>",
    "eps_gauge_max": "<decimal>",
    "margin": "<decimal>"
  },
  "energy_budget": {
    "status": "pass|fail",
    "E_total": "<decimal>",
    "dE": "<decimal>",
    "budget": "<decimal>",
    "margin": "<decimal>"
  },
  "clock_stage_coherence": {
    "status": "pass|fail",
    "delta_stage_t": "<decimal>"
  },
  "ledger_hash_chain": {
    "status": "pass|fail"
  }
}
```

#### `residuals` Keys

```json
{
  "eps_G": "<decimal>",
  "eps_BI": "<decimal>",
  "eps_W": "<decimal>",
  "eps_gauge": "<decimal>"
}
```

#### `metrics` Keys

```json
{
  "E_total": "<decimal>",
  "dE": "<decimal>",
  "F2_mean": "<decimal>",
  "B2_mean": "<decimal>",
  "E2_mean": "<decimal>",
  "dt_used": "<decimal>"
}
```

#### `invariants_enforced` List

```json
[
  "N:INV.ym.gauss_law",
  "N:INV.ym.bianchi_identity",
  "N:INV.ym.ward_identity",
  "N:INV.ym.gauge_condition",
  "N:INV.ym.energy_budget",
  "N:INV.clock.stage_coherence",
  "N:INV.ledger.hash_chain_intact"
]
```

---

## 4. Implementation Checklist

### 4.1 Terminology Registry Patch

- [ ] Merge `DOMAIN_NAVIER_STOKES` and `DOMAIN_YM` entries
- [ ] Merge GR invariants (hamiltonian_constraint, momentum_constraint, etc.)
- [ ] Merge NS invariants (div_free, energy_nonincreasing, etc.)
- [ ] Merge YM invariants (gauss_law, bianchi_identity, etc.)
- [ ] Merge all mapping IDs (MAP_INV_*)

### 4.2 Receipt Emitter Updates

- [ ] Update `src/core/gr_rhs.py` to emit GR receipt fields
- [ ] Create `src/core/ns_rhs.py` for NS receipt fields
- [ ] Create `src/core/ym_rhs.py` for YM receipt fields
- [ ] Ensure all real values are decimal strings

### 4.3 Receipt Validator Updates

- [ ] Update `src/aeonic/aeonic_receipts.py` to validate domain-specific fields
- [ ] Add GR/NS/YM invariant checkers
- [ ] Verify gate status mapping

---

## 5. Example GR Receipt

```json
{
  "A:KIND": "A:RCPT.step.accepted",
  "A:ID": "step_2026-01-31T19:00:00Z_001",
  "A:TS": "2026-01-31T19:00:00.000Z",
  "N:DOMAIN": "N:DOMAIN.GR_NR",
  "gates": {
    "hamiltonian_constraint": {
      "status": "pass",
      "eps_H": "1.23e-6",
      "eps_H_max": "1.0e-5",
      "margin": "0.877"
    },
    "momentum_constraint": {
      "status": "pass",
      "eps_M": "2.45e-7",
      "eps_M_max": "1.0e-5",
      "margin": "0.975"
    },
    "det_gamma_positive": {
      "status": "pass",
      "det_gamma_min": "0.123",
      "limit": "0",
      "margin": "0.123"
    },
    "curvature_bounded": {
      "status": "pass",
      "R_max": "0.456",
      "R_max_limit": "1.0",
      "margin": "0.544"
    },
    "energy_drift_bounded": {
      "status": "pass",
      "dH": "1.2e-8",
      "dH_max": "1.0e-6",
      "margin": "0.988"
    },
    "clock_stage_coherence": {
      "status": "pass",
      "delta_stage_t": "1.0e-6"
    },
    "ledger_hash_chain": {
      "status": "pass"
    }
  },
  "residuals": {
    "eps_H": "1.23e-6",
    "eps_M": "2.45e-7"
  },
  "metrics": {
    "R_max": "0.456",
    "det_gamma_min": "0.123",
    "H": "0.001",
    "dH": "1.2e-8",
    "dt_CFL": "0.001",
    "dt_gauge": "0.002",
    "dt_coh": "0.0005",
    "dt_res": "0.001",
    "dt_used": "0.001"
  },
  "invariants_enforced": [
    "N:INV.gr.hamiltonian_constraint",
    "N:INV.gr.momentum_constraint",
    "N:INV.gr.det_gamma_positive",
    "N:INV.gr.curvature_bounded",
    "N:INV.gr.energy_drift_bounded",
    "N:INV.clock.stage_coherence",
    "N:INV.ledger.hash_chain_intact"
  ]
}
```

---

## 6. Recommendation: Receipt Format Standardization

The repository currently has two receipt "styles":

1. **Aeonica receipt spec** (`42_AEONICA_RECEIPTS.md`): Requires decimal strings and canonical key ordering
2. **Emitted receipts** (`data/receipts/receipts.json`, `aeonic_receipts.jsonl`): Still use raw floats and ad-hoc keys

**Recommendation**: Standardize on the Aeonica v1.2 spec as the only external receipt format. Treat internal floaty debug receipts as "non-evidence telemetry."

This ensures:
- "Truth" remains crisp and replayable
- GR/NS/YM share a single validator
- No drift between specification and implementation

---

## 7. Related Documents

- [`terminology_registry.json`](terminology_registry.json) — Central terminology registry
- [`specifications/aeonica/42_AEONICA_RECEIPTS.md`](specifications/aeonica/42_AEONICA_RECEIPTS.md) — Aeonica receipt specification
- [`src/core/gr_rhs.py`](src/core/gr_rhs.py) — GR right-hand side computations
- [`src/contracts/host_api.py`](src/contracts/host_api.py) — Host API with computed metrics
- [`plans/nsc_ns_dialect_spec.md`](plans/nsc_ns_dialect_spec.md) — NSC_NS dialect specification
- [`plans/nsc_ym_dialect_spec.md`](plans/nsc_ym_dialect_spec.md) — NSC_YM dialect specification

---

*Document Version: 1.0*  
*Updated: 2026-01-31*
