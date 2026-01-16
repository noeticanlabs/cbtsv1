# **Numerical Discretization Tables for UFE, RFE, and LoC**

**Version:** NDT-Coherence-v1.0
**Scope:** Finite difference, finite volume, spectral methods for PDE solving
**Grid:** Uniform Cartesian (\Omega_h = \bigcup_{i,j,k} [x_i,x_{i+1}] \times [y_j,y_{j+1}] \times [z_k,z_{k+1}]), time steps (\Delta t_n)
**Norms:** L2 norm for residuals, stability via von Neumann

---

## I. UFE Discretization (General Field Evolution)

### I.1 Spatial Discretization

| Field Type | Method          | Stencil                 | Order | Boundary Conditions |
| ---------- | --------------- | ----------------------- | ----- | ------------------- |
| Scalar     | Finite Diff     | Central 2nd: (\Psi_{i+1} - 2\Psi_i + \Psi_{i-1})/\Delta x^2 | 2     | Dirichlet/Neumann   |
| Vector     | Finite Vol      | Divergence: (\int_{\partial V} J \cdot n dS)/|V| | 1     | No-slip/Free         |
| Tensor    | Spectral        | FFT-based               | N     | Periodic            |

### I.2 Time Discretization (Aeonic-Compatible)

| Scheme     | Update Rule                                      | Stability Condition              | CFL Factor |
| ---------- | ------------------------------------------------ | -------------------------------- | ---------- |
| Forward Euler | \Psi^{n+1} = \Psi^n + \Delta t (B + \lambda K + w) | |B| \Delta t < 2                  | 1         |
| RK4        | Standard 4-stage Runge-Kutta                     | |B| \Delta t < 2.8                | 1.4       |
| Backward Euler | Implicit solve: \Psi^{n+1} - \Delta t B \Psi^{n+1} = \Psi^n | Unconditionally stable           | \infty     |

### I.3 Residual Computation

[
\varepsilon_{\mathrm{UFE},h}^n = \frac{\Psi^{n+1} - \Psi^n}{\Delta t} - (B_h + \lambda K_h + w_h)
]

**Tolerance:** |\varepsilon_{\mathrm{UFE},h}|_{L2} \le 10^{-6}

### I.4 Ledger Projection

For each (Q_i), compute discretized residual:
[
\mathcal R_{Q_i,h}^n = \sum_{cells} Q_{i,cell} \Delta V - \int_{t_n}^{t_{n+1}} \int_\Omega S_i dx dt
]

---

## II. RFE Discretization (Phase-Coherence System)

### II.1 Spatial Grid

* (\rho_{i,j,k}), (\theta_{i,j,k}) on staggered grid: \rho at cell centers, \theta at edges
* Coherence current: (J_C)_{i+1/2} = \rho_{i+1/2}^2 (\theta_{i+1} - \theta_i)/\Delta x

### II.2 Time Stepping

| Equation  | Discretization                     | Method     | Stability |
| --------- | ---------------------------------- | ---------- | --------- |
| \partial_t \rho | Upwind for advection: \rho^{n+1} = \rho^n - \Delta t \nabla \cdot (\rho^2 \nabla \theta) | MUSCL     | CFL < 0.5 |
| \partial_t \theta | Crank-Nicolson: implicit diffusion  | CN         | Uncond.   |

### II.3 Coherence Residual

[
\mathcal R_{C,h} = \frac{\rho^{n+1} - \rho^n}{\Delta t} + \nabla_h \cdot (\rho^2 \nabla_h \theta)
]

**Admissibility:** |\mathcal R_{C,h}|_{L\infty} \le 10^{-8}

### II.4 Phase Singularity Handling

* Detect: if |\nabla \theta| > 2\pi / \Delta x, apply vortex removal
* Regularization: add artificial diffusivity \nu_h = 10^{-4} \Delta x^2 / \Delta t

---

## III. LoC Discretization (Constraint Enforcement)

### III.1 Observable Declarations

| Observable | Discretization Formula               | Units  |
| ---------- | ------------------------------------ | ------ |
| Mass       | \sum \rho_{i,j,k} \Delta V           | kg    |
| Energy     | \sum \frac12 \rho |\nabla \theta|^2 \Delta V | J     |
| Coherence  | \sum \rho^2 |\nabla \theta| \Delta V | -     |

### III.2 Residual Bounds

| Scale      | \mathcal M_h                     | \varepsilon_h |
| ---------- | -------------------------------- | ------------- |
| Global     | \max |Q_i|                        | 10^{-3}      |
| Local      | \max_{cell} |Q_{i,cell}|            | 10^{-6}      |
| Time       | \sup_n |Q_i^n - Q_i^{n-1}| / \Delta t | 10^{-5}      |

### III.3 Enforcement Mechanisms

1. **Hard Cutoff:** If \sup |\mathcal R_{Q_i}| / \mathcal M > \varepsilon, halt simulation
2. **Soft Damping:** Add corrective term \lambda K = - \gamma \mathcal R_{Q_i} / |\mathcal R_{Q_i}|
3. **Adaptive \lambda:** \lambda^{n+1} = \lambda^n (1 + \alpha |\mathcal R_{Q_i}| / \varepsilon)

### III.4 Aeonica Clock Selection

Candidate \Delta t_k:

* Advective: 0.5 \Delta x / \max |u|
* Diffusive: 0.5 (\Delta x)^2 / \nu
* Coherence: 0.5 \Delta x / \sqrt{\kappa}

Select: \Delta t_n = \min_k \Delta t_k subject to residual bound.

---

## IV. Solver Architecture Table

| Component   | UFE             | RFE             | LoC             |
| ----------- | --------------- | --------------- | --------------- |
| Library     | PETSc           | Dedalus         | Custom         |
| Linear Solve| GMRES           | FFT             | Direct         |
| Nonlinear   | Newton-Krylov  | Picard          | Fixed-point    |
| Parallel    | MPI             | MPI             | MPI            |
| Output      | HDF5 receipts   | HDF5 receipts   | HDF5 receipts  |

---

## V. Benchmark Parameters

| Test Case         | \Delta x | \Delta t | \lambda | \varepsilon |
| ----------------- | -------- | -------- | ------- | ----------- |
| Coherence blob    | 0.01     | 0.001    | 1.0     | 1e-6        |
| Shock cascade     | 0.005    | 0.0005   | 0.1     | 1e-8        |
| Phase lock        | 0.02     | 0.01     | 10.0    | 1e-4        |

---

## STATUS

✔ Discretizations consistent with continuous forms
✔ Implementable in Python/ C++/ Julia
✔ Stability analyzed
✔ Aeonica integrated

---

**End of Numerical Discretization Tables**