"""
Utility functions for GR solver tests.
"""

import numpy as np
from typing import List

def rms(x: np.ndarray) -> float:
    x = np.asarray(x)
    return float(np.sqrt(np.mean(x * x)))

def l2(x: np.ndarray) -> float:
    x = np.asarray(x)
    return float(np.sqrt(np.sum(x * x)))

def estimate_order(errors: List[float], hs: List[float]) -> float:
    """
    Fit log(e) = p log(h) + c using least squares.
    """
    e = np.array(errors, dtype=float)
    h = np.array(hs, dtype=float)
    # protect
    e = np.maximum(e, 1e-300)
    h = np.maximum(h, 1e-300)
    A = np.vstack([np.log(h), np.ones_like(h)]).T
    p, _c = np.linalg.lstsq(A, np.log(e), rcond=None)[0]
    return float(p)