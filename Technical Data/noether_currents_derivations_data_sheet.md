# **Noether Currents — Explicit Derivations for UFE / RFE / LoC (Data-Sheet Style)**

---

# 1) RFE Noether Currents (phase–amplitude field)

## 1.1 Canonical action

Use the **first-order** phase–density action (symplectic form):
[
\boxed{
\mathcal A_{\mathrm{RFE}}[\rho,\theta]
======================================

\int dt\int_\Omega
\Big(
\rho,\partial_t\theta
---------------------

\mathcal H(\rho,\theta,\nabla\theta)
\Big),dx
}
]
Choose Hamiltonian density (minimal):
[
\boxed{
\mathcal H
==========

\frac12,\rho^2|\nabla\theta|^2
+
V(\rho)
}
]
(You can set (V\equiv 0) if you want pure coherence-transport.)

Variations give the Euler–Lagrange equations:
[
\partial_t\rho = -\frac{\delta H}{\delta\theta},
\qquad
\partial_t\theta = \frac{\delta H}{\delta\rho}.
]
For (H=\frac12\int\rho^2|\nabla\theta|^2dx):
[
\boxed{
\frac{\delta H}{\delta\theta}=-\nabla\cdot(\rho^2\nabla\theta),
\qquad
\frac{\delta H}{\delta\rho}=\rho,|\nabla\theta|^2
}
]
so
[
\boxed{
\partial_t\rho + \nabla\cdot(\rho^2\nabla\theta)=0
}
]
and
[
\partial_t\theta = \rho|\nabla\theta|^2 + V'(\rho).
]
The two variational derivative identities above match the project’s derivatives canon.

---

## 1.2 Noether symmetry #1: phase shift (\theta\mapsto \theta+\alpha)

**Symmetry:** (\theta) enters only through (\partial_t\theta) and (\nabla\theta), so constant shifts are a symmetry.

**Noether current (continuity form):**
[
\boxed{
\partial_t(\rho)+\nabla\cdot(\rho^2\nabla\theta)=0
}
]
Thus the Noether charge is:
[
\boxed{
Q_\theta=\int_\Omega \rho,dx
}
]
and the spatial current is:
[
\boxed{
\mathbf j_\theta=\rho^2\nabla\theta.
}
]

> If you prefer (Q=\int \rho^2) as the “charge,” switch to the equivalent complex-field model below (it’s the standard (U(1)) charge).

---

## 1.3 Noether symmetry #2: spatial translations (if (\Omega=\mathbb R^d) or periodic torus)

If (\mathcal H) has no explicit (x)-dependence (and boundary terms vanish or are periodic), spatial translation invariance yields **momentum conservation**.

To state it cleanly, use the **canonical stress tensor** for fields ((\rho,\theta)). The Lagrangian density is
[
\mathcal L = \rho,\partial_t\theta - \mathcal H(\rho,\theta,\nabla\theta).
]
The momentum density is
[
\boxed{
\pi_\theta := \frac{\partial\mathcal L}{\partial(\partial_t\theta)}=\rho,
\qquad
\pi_\rho := \frac{\partial\mathcal L}{\partial(\partial_t\rho)}=0.
}
]
A convenient momentum density is then:
[
\boxed{
\mathbf p = \rho,\nabla\theta
}
]
(up to sign conventions depending on how you identify physical momentum vs canonical momentum).

The conserved form is:
[
\boxed{
\partial_t p_i + \partial_j T_{ij}=0
}
]
with (for (\mathcal H=\frac12\rho^2|\nabla\theta|^2+V(\rho))):
[
\boxed{
T_{ij}
======

## \rho^2,(\partial_i\theta)(\partial_j\theta)

\delta_{ij}\Big(\tfrac12\rho^2|\nabla\theta|^2+V(\rho)\Big).
}
]

---

## 1.4 Noether symmetry #3: time translations (energy)

If (\mathcal L) has no explicit (t)-dependence, energy is conserved (again assuming boundary conditions eliminate flux leakage). The energy density here is essentially (\mathcal H), and the statement is:
[
\boxed{
\partial_t \mathcal H + \nabla\cdot \mathbf S_H = 0
}
]
where (\mathbf S_H) is the corresponding energy flux (computable from the stress tensor and equations of motion). In practice, for implementations, you ledger-check energy via:
[
\Delta \int \mathcal H,dx
\quad\text{vs}\quad
\text{(boundary flux + forcing)}.
]

---

# 2) RFE Noether currents (standard complex-field form, recommended)

If you want the **standard, unambiguous Noether machinery**, define a complex scalar:
[
\psi(x,t)=\rho(x,t)e^{i\theta(x,t)}.
]

Use the common Schrödinger-type Lagrangian (nonrelativistic):
[
\boxed{
\mathcal L(\psi,\psi^*)
=======================

## \frac{i}{2}\big(\psi^*\partial_t\psi-\psi\partial_t\psi^*\big)

## \frac12|\nabla\psi|^2

U(|\psi|).
}
]

## Global (U(1)) symmetry (\psi\mapsto e^{i\alpha}\psi)

Noether current:
[
\boxed{
j^0 = |\psi|^2 = \rho^2,
\qquad
\mathbf j = \operatorname{Im}(\psi^*\nabla\psi)=\rho^2\nabla\theta.
}
]
Continuity:
[
\boxed{
\partial_t(\rho^2)+\nabla\cdot(\rho^2\nabla\theta)=0.
}
]

This matches your “coherence current” intuition exactly, but now it’s a textbook Noether current.

---

# 3) UFE Noether statements (what changes when you add coherence enforcement)

## 3.1 Baseline Lagrangian + coherence term

Assume baseline physics comes from an action:
[
\mathcal A_B[\Psi]=\int \mathcal L_B(\Psi,\partial\Psi),d^{d+1}x.
]
Coherence enforcement can be introduced two ways:

### (A) Lagrangian deformation (soft)

[
\boxed{
\mathcal L_{\text{tot}} = \mathcal L_B + \lambda,\mathcal L_K
}
]
If (\mathcal L_K) respects the same symmetries as (\mathcal L_B), Noether currents remain conserved.

### (B) Non-variational forcing (hard / controller form)

If the UFE includes a non-variational term (e.g., clipping, projection, rollback, discrete rails), then conservation becomes:
[
\boxed{
\partial_\mu J^\mu = \Sigma
}
]
where (\Sigma) is a **controlled source term** representing coherence injection/removal (this is exactly what your ledger should measure).

**Practical rule:**

* “Variational (K)” ⇒ true Noether conservation.
* “Controller (K)” ⇒ balance law with a computable source (\Sigma_K).

---

## 3.2 Noether current formula (generic)

For a continuous symmetry (\delta\Psi^A = \epsilon,\Delta^A(\Psi)) with (\delta\mathcal L = \epsilon,\partial_\mu F^\mu), the Noether current is:
[
\boxed{
J^\mu
=====

\sum_A
\frac{\partial\mathcal L}{\partial(\partial_\mu\Psi^A)},\Delta^A(\Psi)
----------------------------------------------------------------------

F^\mu.
}
]
Conservation holds on-shell:
[
\partial_\mu J^\mu = 0
]
**unless** coherence rails add non-variational terms, in which case:
[
\partial_\mu J^\mu = \Sigma_{\text{rails}}.
]

That (\Sigma_{\text{rails}}) is precisely what your **LoC residual accounting** should expose.

---

# 4) LoC as a constraint layer (Lagrange multipliers) — modified Noether theorem

## 4.1 Constraint-augmented action

Let (\mathcal R_i[\Psi]) be declared residual constraints (ledger closure conditions). Build:
[
\boxed{
\mathcal A_{\mathrm{LoC}}[\Psi,\mu]
===================================

\int\Big(
\mathcal L_B(\Psi,\partial\Psi)
+
\sum_i \mu_i,\mathcal R_i[\Psi]
\Big),d^{d+1}x.
}
]

Euler–Lagrange:
[
\frac{\delta \mathcal A_{\mathrm{LoC}}}{\delta \mu_i}=0
;\Rightarrow;
\boxed{\mathcal R_i[\Psi]=0}
]
so LoC becomes an **enforced closure condition**.

## 4.2 Noether with constraints

If the symmetry leaves both (\mathcal L_B) and each (\mathcal R_i) invariant (up to a divergence), then the Noether current remains conserved.

If the constraints are not invariant, you get:
[
\boxed{
\partial_\mu J^\mu
==================

-\sum_i \mu_i,\delta \mathcal R_i
}
]
So the “violation” of conservation is explicitly tied to (i) **which constraint**, and (ii) **the multiplier (\mu_i)** (authority spent).

That is extremely useful for your ledger: it tells you *why* a conservation law is being bent and *how much authority* did it.

---

# 5) Implementation-ready “currents you should always include” (minimum set)

For any UFE/RFE/LoC solver that wants audit-grade receipts, log these:

1. **U(1) / phase current** (RFE complex form recommended):
   [
   (j^0,\mathbf j)=(\rho^2,\rho^2\nabla\theta)
   ]

2. **Energy density** (\mathcal H) and energy flux (\mathbf S_H) (even if approximate)

3. **Momentum balance**
   [
   \partial_t p_i + \partial_j T_{ij} = \Sigma_i
   ]
   with (\Sigma_i) explicitly attributed to rails/forcing.

4. **Constraint-source attribution**
   [
   \partial_\mu J^\mu = \Sigma_{\text{rails}} \quad\text{(measured)}
   ]

5. **Ledger residuals** (LoC closure)
   [
   \mathcal R_{Q}=\Delta Q - (\text{sources}+\text{flux})
   ]

---

If you want, the next upgrade is to **attach the exact Noether currents for EM (Poynting), Navier–Stokes (energy/helcity), and GR (Bianchi/constraint propagation)** in the same datasheet format, and then map each to a concrete Ω-receipt schema.