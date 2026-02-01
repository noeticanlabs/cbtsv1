# NSC Geometry - Yang-Mills Subdomain
# Gauge theory, principal bundles

from dataclasses import dataclass

@dataclass
class NSC_YM_Dialect:
    """NSC_YM Dialect for Yang-Mills gauge theory."""
    name = "NSC_geometry.ym"
    version = "1.0"
    
    invariants = {
        'gauss_law': {'id': 'N:INV.ym.gauss_law'},
        'bianchi': {'id': 'N:INV.ym.bianchi'},
    }


NSC_YM = NSC_YM_Dialect()
