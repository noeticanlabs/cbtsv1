# NSC Geometry - Riemannian Subdomain
# Pure differential geometry

from dataclasses import dataclass

@dataclass
class Manifold:
    """Riemannian manifold."""
    dimension: int
    signature: tuple = None


@dataclass
class NSC_riemann_Dialect:
    """NSC_Riemann Dialect for Riemannian geometry."""
    name = "NSC_geometry.riemann"
    version = "1.0"


NSC_riemann = NSC_riemann_Dialect()
