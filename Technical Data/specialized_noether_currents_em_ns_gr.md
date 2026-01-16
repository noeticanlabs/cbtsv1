# **Specialized Noether Currents: EM, Navier-Stokes, GR (Data-Sheet Style)**

**Version:** SNC-Coherence-v1.0
**Scope:** Physical specializations of UFE/RFE/LoC with explicit conserved currents
**Mapping:** Each current → Ω-receipt schema for ledger verification

---

# 1) Electromagnetism (EM) Specialization

## 1.1 UFE Form

Baseline Maxwell:
[
\partial_t \mathbf E = \nabla \times \mathbf B - \mathbf J,
\qquad
\partial_t \mathbf B = -\nabla \times \mathbf E
]
Coherence term: (\lambda K) enforces charge conservation or gauge fixing.

## 1.2 Noether Currents

### Energy-Momentum (Poynting)

Symmetry: Lorentz boosts (in covariant form).

**Energy density:**
[
T^{00} = \frac12 (\epsilon_0 |\mathbf E|^2 + \mu_0^{-1} |\mathbf B|^2)
]

**Poynting vector (energy flux):**
[
\mathbf S = \frac{1}{\mu_0} \mathbf E \times \mathbf B
]

Continuity:
[
\partial_t T^{00} + \nabla \cdot \mathbf S = -\mathbf J \cdot \mathbf E
]

### Charge Conservation

Symmetry: U(1) gauge.

**Charge density:** \rho, **current:** \mathbf J

Continuity:
[
\partial_t \rho + \nabla \cdot \mathbf J = 0
]

## 1.3 Ω-Receipt Schema

| Receipt Field | Formula                  | Ledger Role          |
| ------------- | ------------------------ | -------------------- |
| Q_energy      | \int T^{00} dV           | Total EM energy      |
| Q_charge      | \int \rho dV             | Total charge         |
| \mathcal R_energy | \Delta Q_energy - \int (-\mathbf J \cdot \mathbf E) dV dt | Residual             |
| \mathcal R_charge  | \Delta Q_charge          | Conservation check   |

**Coherence rail:** If \mathcal R > \varepsilon, inject compensating field via K.

---

# 2) Navier-Stokes (NS) Fluid Dynamics Specialization

## 2.1 UFE Form

Baseline NS:
[
\partial_t \mathbf v + (\mathbf v \cdot \nabla) \mathbf v = -\nabla p / \rho + \nu \Delta \mathbf v + \mathbf f
]
Coherence: (\lambda K) enforces momentum/entropy closure.

## 2.2 Noether Currents

### Energy Conservation

Symmetry: Time translation.

**Energy density:**
[
e = \frac12 \rho |\mathbf v|^2 + \frac{p}{\gamma-1}
]

**Energy flux:**
[
\mathbf S_e = (\frac12 \rho |\mathbf v|^2 + \frac{\gamma p}{\gamma-1}) \mathbf v
]

Continuity:
[
\partial_t e + \nabla \cdot \mathbf S_e = \mathbf f \cdot \mathbf v
]

### Momentum Conservation

Symmetry: Spatial translations.

**Momentum density:** \rho \mathbf v

**Stress tensor:**
[
\sigma_{ij} = -p \delta_{ij} + \rho v_i v_j + \tau_{ij} \quad (\tau for viscous terms)
]

Continuity:
[
\partial_t (\rho v_i) + \partial_j \sigma_{ij} = f_i
]

### Helicity (Topological Invariant)

Symmetry: Rotational in velocity field.

**Helicity density:** \mathbf v \cdot \nabla \times \mathbf v

Continuity (ideal fluid):
[
\partial_t (\mathbf v \cdot \omega) + \nabla \cdot (\omega \times \mathbf v) = 0 \quad (\omega = \nabla \times \mathbf v)
]

## 2.3 Ω-Receipt Schema

| Receipt Field     | Formula                          | Ledger Role              |
| ----------------- | -------------------------------- | ------------------------ |
| Q_energy          | \int e dV                        | Total fluid energy       |
| Q_momentum        | \int \rho \mathbf v dV           | Total momentum           |
| Q_helicity        | \int \mathbf v \cdot \omega dV   | Topological invariant    |
| \mathcal R_energy | \Delta Q_energy - \int \mathbf f \cdot \mathbf v dV dt | Residual                 |
| \mathcal R_momentum| \Delta Q_momentum - \int \mathbf f dV dt          | Conservation check       |
| \mathcal R_helicity| \Delta Q_helicity                 | Closure check            |

**Coherence rail:** Damp instabilities via \lambda K = - \gamma \nabla \cdot \sigma (for divergence damping).

---

# 3) General Relativity (GR) Specialization

## 3.1 UFE Form

Baseline Einstein:
[
R_{\mu\nu} - \frac12 R g_{\mu\nu} = 8\pi G T_{\mu\nu}
]
Coherence: (\lambda K) enforces constraint propagation (e.g., Hamiltonian/lapse).

## 3.2 Noether Currents (Bianchi Identities)

### Bianchi Identity #1: Covariant Conservation

Symmetry: Diffeomorphism.

**Bianchi identity:**
[
\nabla^\mu (R_{\mu\nu} - \frac12 R g_{\mu\nu}) = 0
]

Which implies:
[
\nabla^\mu T_{\mu\nu} = 0 \quad (\text{if Einstein holds})
]

### Bianchi Identity #2: Constraint Propagation

For ADM formalism:
[
\partial_t (\mathcal H) + \mathcal L_\beta \mathcal H = 0
]
\[
\partial_t (\mathcal M_i) + \mathcal L_\beta \mathcal M_i = 0
\]

Where \mathcal H is Hamiltonian constraint, \mathcal M_i momentum constraints.

## 3.3 Covariant Noether Currents

In GR, Noether currents are the conserved quantities from Killing vectors.

For stationary spacetime (Killing vector \xi^\mu):

**Conserved current:**
[
J^\mu = T^\mu_\nu \xi^\nu + \frac{\sqrt{-g}}{16\pi G} (\nabla^\mu \xi^\nu) g_{\nu\sigma} \nabla^\sigma \xi^\rho g_{\rho\lambda} \nabla^\lambda \xi^\sigma
]

**Charge:**
[
Q = \int_\Sigma J^\mu d\Sigma_\mu
]

## 3.4 Ω-Receipt Schema

| Receipt Field     | Formula                          | Ledger Role              |
| ----------------- | -------------------------------- | ------------------------ |
| Q_mass            | ADM mass                         | Total energy             |
| Q_angular_momentum| Komar integral                   | Total angular momentum   |
| \mathcal R_constraints | \max |\mathcal H|, |\mathcal M_i| | Constraint satisfaction  |
| \mathcal R_bianchi   | \nabla^\mu T_{\mu\nu}             | Covariant conservation   |

**Coherence rail:** Constraint damping evolution for \mathcal H, \mathcal M_i.

---

# Universal Mapping to Ω-Receipts

For any specialization, receipts include:

* Declared observables: {Q_i}
* Currents: {J^\mu}
* Residuals: {\mathcal R_i = \Delta Q_i - \int (sources + fluxes) dt}
* Authority spent: \int \lambda |K| dt
* Time dilation: Aeonic \Delta t selections

**Ledger closure:** All \mathcal R_i < \varepsilon with receipts tamper-evident.

---

**End of Specialized Noether Currents Data Sheet**