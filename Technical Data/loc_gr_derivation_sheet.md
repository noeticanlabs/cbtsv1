# Principle Theory: LoC ⇒ GR (Einstein as a compatibility closure)

**Status:** Canon
**Theme:** Deriving GR from the Law of Coherence (LoC)
**Context:** Why the UFE defaults to Einstein equations in the geometric limit.

---

## Setup (objects + meaning)

Let $(\mathcal M, g_{\mu\nu})$ be a 4-dimensional Lorentzian spacetime, $\nabla$ the Levi-Civita connection of $g$, and "matter" be fields $\Phi$ with action $S_m[g,\Phi]$. Define the stress-energy tensor by
$$
T_{\mu\nu} := -\frac{2}{\sqrt{-g}}\frac{\delta S_m}{\delta g^{\mu\nu}}.
$$

### Definition 1 (LoC admissibility, physics version)

A coupled matter+geometry theory is **LoC-admissible** if:

1.  its laws are **diffeomorphism-covariant** (no privileged coordinates), and
2.  the coupled evolution admits **ledger-compatible closure**: local balance laws do not contradict one another under changes of slicing/coordinates.

In practice this means: the theory must have a local conservation/compatibility identity that survives coupling.

---

## Lemma 1 (Ledger identity from diffeo invariance)

Assume the total action $S[g,\Phi]=S_g[g]+S_m[g,\Phi]$ is diffeomorphism invariant, and the matter fields satisfy their Euler–Lagrange equations (on-shell). Then
$$
\nabla_\mu T^{\mu\nu}=0.
$$

**Proof idea.** Diffeomorphism invariance gives a Noether identity: the variation under an infinitesimal diffeo is zero, yielding a covariant conservation statement for the metric source term when matter EOM hold. (This is the "ledger must close" condition.)

---

## Lemma 2 (Coupling consistency forces a divergence-free geometry tensor)

Suppose the metric field equation is of the form
$$
\mathcal E_{\mu\nu}(g)=\kappa T_{\mu\nu},
$$
for some symmetric rank-2 tensor $\mathcal E_{\mu\nu}(g)$ constructed from $g$. If Lemma 1 holds for generic matter, then LoC-admissibility forces
$$
\nabla_\mu \mathcal E^{\mu\nu}(g)\equiv 0
$$
as an **identity** (off-shell in $g$).

**Reason.** Taking $\nabla_\mu$ of both sides gives $\nabla_\mu \mathcal E^{\mu\nu}=\kappa\nabla_\mu T^{\mu\nu}=0$. For generic matter this must not impose extra constraints on $\Phi$ beyond its EOM; therefore $\nabla_\mu \mathcal E^{\mu\nu}$ must vanish identically.

This is LoC stated as "**the glue can't introduce bookkeeping contradictions**."

---

## Minimality assumptions (the "principle theory closure choices")

To get GR specifically (not an infinite zoo), impose the standard minimal coherence package:

*   **A1 Locality:** $\mathcal E_{\mu\nu}$ depends on $g$ and a finite number of derivatives at a point.
*   **A2 Metric-only gravity:** no additional gravitational fields beyond $g$.
*   **A3 Second-order metric equations:** $\mathcal E_{\mu\nu}$ contains at most second derivatives of $g$.
*   **A4 Symmetry:** $\mathcal E_{\mu\nu}=\mathcal E_{\nu\mu}$.
*   **A5 Dimension:** spacetime dimension is 4.

These are "LoC prefers the simplest closure that won't secretly add contradictions or extra degrees of freedom."

---

## Theorem (LoC-minimal gravity closure = Einstein equation)

Under A1–A5 and Lemma 2, in 4D the only possible choice is
$$
\mathcal E_{\mu\nu}(g)=G_{\mu\nu}+\Lambda g_{\mu\nu}
$$
(up to an overall constant scale), hence the field equation is
$$
G_{\mu\nu}+\Lambda g_{\mu\nu}=\kappa T_{\mu\nu}.
$$

**Proof idea.** Lovelock's classification: in 4D, any symmetric, divergence-free rank-2 tensor depending locally on $g$ and up to its second derivatives is (a constant multiple of) the Einstein tensor plus a cosmological term. (The 4D Gauss–Bonnet density is topological and does not contribute to $\mathcal E_{\mu\nu}$.)

So: **LoC (compatibility) + minimality ⇒ GR**.

---

## Corollary (3+1 constraints are LoC ledgers on slices)

In a 3+1 split with spatial metric $\gamma_{ij}$, extrinsic curvature $K_{ij}$, lapse $\alpha$, shift $\beta^i$:

*   **Hamiltonian constraint**
    $$
    \mathcal H := R^{(3)} + K^2 - K_{ij}K^{ij} - 16\pi\rho = 0
    $$
*   **Momentum constraint**
    $$
    \mathcal M^i := D_j(K^{ij}-\gamma^{ij}K) - 8\pi S^i = 0
    $$

These are the "ledger closure" conditions that must remain compatible with evolution. Analytically, the contracted Bianchi identity is the deep reason constraint propagation is consistent; numerically, it's exactly where incoherence leaks in.

**References:**
*   D. Lovelock, *The Einstein tensor and its generalizations* (1971).
*   R. Wald, *General Relativity*.
*   Misner–Thorne–Wheeler, *Gravitation*.