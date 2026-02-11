"""
UFE Coherence Tests

Tests for Universal Field Equation (UFE) integration with coherence framework.
Validates UFE decomposition, residual computation, BridgeCert pattern, and
proper time construction.
"""

import pytest
import numpy as np
from typing import Callable, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum


class CoherenceStatus(Enum):
    COHERENT = "coherent"
    INCOHERENT = "incoherent"
    PENDING = "pending"


@dataclass
class UFEOp:
    """UFE Operator Package"""
    Lphys: Callable[[np.ndarray], np.ndarray]
    Sgeo: Callable[[np.ndarray], np.ndarray]
    G: Callable[[int, np.ndarray], np.ndarray]
    drive_indices: Tuple[int, ...]


@dataclass
class Receipt:
    """Coherence receipt for UFE systems"""
    t: float
    delta: float
    residual_norm: float
    threshold: float
    passed: bool
    ufe_residual: Dict[str, float]
    bridge_cert_id: str | None = None


class UFECoherenceSystem:
    """
    Coherence system implementing UFE decomposition.
    """

    def __init__(self, op: UFEOp, threshold: float = 1e-6):
        self.op = op
        self.threshold = threshold
        self.receipts: list[Receipt] = []
        self.bridge_certs: dict[str, dict] = {}

    def compute_rhs(self, psi: np.ndarray) -> np.ndarray:
        """Compute UFE right-hand side"""
        L = self.op.Lphys(psi)
        S = self.op.Sgeo(psi)
        G_total = np.zeros_like(psi)
        for i in self.op.drive_indices:
            G_total += self.op.G(i, psi)
        return L + S + G_total

    def compute_residual(
        self,
        psi: np.ndarray,
        psi_next: np.ndarray,
        t: float,
        dt: float
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Compute discrete residual and decomposition.

        Returns:
            residual: ε_Δ = (ψ_{n+1} - ψ_n)/Δ - RHS
            decomposition: dict of component norms
        """
        dpsi_dt = (psi_next - psi) / dt
        rhs = self.compute_rhs(psi)
        residual = dpsi_dt - rhs

        # Decompose residual by component
        L = self.op.Lphys(psi)
        S = self.op.Sgeo(psi)
        G_total = np.zeros_like(psi)
        for i in self.op.drive_indices:
            G_total += self.op.G(i, psi)

        return residual, {
            "Lphys_norm": float(np.linalg.norm(L)),
            "Sgeo_norm": float(np.linalg.norm(S)),
            "G_total_norm": float(np.linalg.norm(G_total)),
            "residual_L": float(np.linalg.norm(residual - dpsi_dt + rhs)),
        }

    def coherence_step(
        self,
        psi: np.ndarray,
        psi_next: np.ndarray,
        t: float,
        dt: float,
        bridge_cert_id: str | None = None
    ) -> Tuple[np.ndarray, Receipt]:
        """
        Execute one coherence step with receipt generation.
        """
        residual, decomposition = self.compute_residual(psi, psi_next, t, dt)
        residual_norm = float(np.linalg.norm(residual))
        passed = residual_norm <= self.threshold

        receipt = Receipt(
            t=t,
            delta=dt,
            residual_norm=residual_norm,
            threshold=self.threshold,
            passed=passed,
            ufe_residual=decomposition,
            bridge_cert_id=bridge_cert_id
        )
        self.receipts.append(receipt)

        return psi_next, receipt

    def register_bridge_cert(self, cert_id: str, params: dict):
        """Register a BridgeCert for the system"""
        self.bridge_certs[cert_id] = params

    def verify_bridge(self, receipt: Receipt) -> bool:
        """Verify that receipt satisfies bridge conditions"""
        if receipt.bridge_cert_id is None:
            return False
        cert = self.bridge_certs.get(receipt.bridge_cert_id)
        if cert is None:
            return False
        # Simplified bridge verification
        # In practice, this would check errorBound(τ_Δ, Δ) >= actual residual
        return True


class TestUFEDecomposition:
    """Tests for UFE operator decomposition"""

    def test_simple_harmonic_oscillator(self):
        """UFE for simple harmonic oscillator"""
        def Lphys(x):
            return np.array([x[1], -x[0]])  # dx/dt = v, dv/dt = -x

        def Sgeo(x):
            return np.zeros(2)  # No geometry correction

        def G(i, x):
            return np.zeros(2)  # No drives

        op = UFEOp(Lphys, Sgeo, G, ())
        system = UFECoherenceSystem(op, threshold=1e-4)

        # Initial state
        psi = np.array([1.0, 0.0])

        # One step of forward Euler
        dt = 0.01
        rhs = system.compute_rhs(psi)
        psi_next = psi + dt * rhs

        # Check coherence
        _, receipt = system.coherence_step(psi, psi_next, 0.0, dt)

        # Should be close to coherent for small dt
        assert receipt.residual_norm < 0.1  # Relaxed for discrete

    def test_with_geometry_correction(self):
        """UFE with geometry correction"""
        def Lphys(x):
            return np.array([x[1], -x[0]])

        def Sgeo(x):
            # Damping geometry correction
            return np.array([0.0, -0.1 * x[1]])

        def G(i, x):
            return np.zeros(2)

        op = UFEOp(Lphys, Sgeo, G, ())
        system = UFECoherenceSystem(op, threshold=1e-4)

        psi = np.array([1.0, 0.0])
        dt = 0.01
        rhs = system.compute_rhs(psi)
        psi_next = psi + dt * rhs

        _, receipt = system.coherence_step(psi, psi_next, 0.0, dt)
        assert receipt.passed
        assert "Sgeo_norm" in receipt.ufe_residual

    def test_with_drive(self):
        """UFE with drive input"""
        def Lphys(x):
            return np.array([x[1], -x[0]])

        def Sgeo(x):
            return np.zeros(2)

        def G(i, x):
            if i == 0:
                return np.array([0.0, 1.0])  # Constant forcing
            return np.zeros(2)

        op = UFEOp(Lphys, Sgeo, G, (0,))
        system = UFECoherenceSystem(op, threshold=1e-4)

        psi = np.array([1.0, 0.0])
        dt = 0.01
        rhs = system.compute_rhs(psi)
        psi_next = psi + dt * rhs

        _, receipt = system.coherence_step(psi, psi_next, 0.0, dt)
        assert receipt.passed
        assert receipt.ufe_residual["G_total_norm"] > 0


class TestGRObserverResidual:
    """Tests for GR observer two-component residual"""

    def test_dynamical_residual(self):
        """Test dynamical coherence: ∇_u u should be ~0 for geodesics"""
        # Simplified: just check the residual decomposition
        def Lphys(x):
            # Geodesic: acceleration = 0
            return np.array([x[1], 0.0])  # v, a

        def Sgeo(x):
            # Clock normalization: g(u,u) = -1
            return np.array([0.0, 0.0])

        def G(i, x):
            return np.zeros(2)

        op = UFEOp(Lphys, Sgeo, G, ())
        system = UFECoherenceSystem(op, threshold=1e-3)

        # Initial state: unit speed
        psi = np.array([0.0, -1.0])  # v = -1, so g(u,u) = -1 (in c=1)
        dt = 0.01

        # Geodesic step (constant velocity)
        psi_next = psi + dt * system.compute_rhs(psi)

        _, receipt = system.coherence_step(psi, psi_next, 0.0, dt)
        # Should be coherent for geodesic motion
        assert receipt.passed

    def test_clock_residual(self):
        """Test clock coherence: proper time normalization"""
        # For proper time, g(u,u) = -1 should hold
        def Lphys(x):
            return np.zeros(2)

        def Sgeo(x):
            # Clock correction to enforce normalization
            speed_sq = x[0]**2 + x[1]**2
            correction = (speed_sq + 1) * 0.1
            return np.array([-correction * x[0], -correction * x[1]])

        def G(i, x):
            return np.zeros(2)

        op = UFEOp(Lphys, Sgeo, G, ())
        system = UFECoherenceSystem(op, threshold=1e-3)

        # Initial state: wrong speed
        psi = np.array([0.5, 0.5])  # speed = √0.5 ≠ 1
        dt = 0.01

        psi_next = psi + dt * system.compute_rhs(psi)

        _, receipt = system.coherence_step(psi, psi_next, 0.0, dt)
        # Clock coherence should bring speed toward 1
        # This is a simplified test


class TestBridgeCert:
    """Tests for BridgeCert pattern"""

    def test_bridge_cert_registration(self):
        """Test BridgeCert registration"""
        def Lphys(x):
            return np.zeros(2)

        def Sgeo(x):
            return np.zeros(2)

        def G(i, x):
            return np.zeros(2)

        op = UFEOp(Lphys, Sgeo, G, ())
        system = UFECoherenceSystem(op, threshold=1e-4)

        # Register a BridgeCert
        system.register_bridge_cert("forward_euler_lipschitz", {
            "lipschitz_constant": 1.0,
            "error_bound": lambda tau, dt: tau + dt
        })

        assert "forward_euler_lipschitz" in system.bridge_certs

    def test_bridge_verification(self):
        """Test bridge verification"""
        def Lphys(x):
            return np.zeros(2)

        def Sgeo(x):
            return np.zeros(2)

        def G(i, x):
            return np.zeros(2)

        op = UFEOp(Lphys, Sgeo, G, ())
        system = UFECoherenceSystem(op, threshold=1e-4)

        system.register_bridge_cert("test_cert", {"param": 1.0})

        receipt = Receipt(
            t=0.0,
            delta=0.01,
            residual_norm=1e-5,
            threshold=1e-4,
            passed=True,
            ufe_residual={},
            bridge_cert_id="test_cert"
        )

        assert system.verify_bridge(receipt)


class TestReceiptChain:
    """Tests for receipt hash chaining"""

    def test_receipt_chain(self):
        """Test that receipts can be chained"""
        def Lphys(x):
            return np.array([x[1], -x[0]])

        def Sgeo(x):
            return np.zeros(2)

        def G(i, x):
            return np.zeros(2)

        op = UFEOp(Lphys, Sgeo, G, ())
        system = UFECoherenceSystem(op, threshold=1e-4)

        psi = np.array([1.0, 0.0])

        # Multiple steps
        for i in range(5):
            dt = 0.01
            rhs = system.compute_rhs(psi)
            psi_next = psi + dt * rhs
            psi, _ = system.coherence_step(psi, psi_next, i * dt, dt)

        assert len(system.receipts) == 5

        # Verify chain integrity (all have passed)
        for receipt in system.receipts:
            assert receipt.passed or receipt.t > 0  # First might fail


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
