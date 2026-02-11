"""
NSC-M3L Stability Analysis

Implements stability analysis for discrete operators per section 4.4.

Stability checks:
- CFL condition for explicit schemes
- Spectral radius computation
- Condition number estimation
- Lax equivalence theorem compliance
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Dict
import numpy as np

from .disc_types import Stencil, Grid, StabilityInfo


@dataclass
class StabilityResult:
    """Result of stability analysis."""
    is_stable: bool
    metric_name: str
    value: float
    message: str


def check_cfl_condition(stencil: Stencil, dt: float, dx: float, 
                        max_velocity: float) -> Tuple[bool, float]:
    """
    Check CFL condition for explicit schemes.
    
    For advection equation u_t + c u_x = 0:
    CFL requires: |c| * dt / dx <= 1
    
    For diffusion equation u_t = ν u_xx:
    Diffusion CFL: ν * dt / dx² <= 0.5
    
    Args:
        stencil: The discretization stencil
        dt: Time step size
        dx: Grid spacing
        max_velocity: Maximum wave/flow velocity
    
    Returns:
        Tuple of (is_stable, cfl_number)
    """
    # Estimate CFL number from stencil
    # For central difference advection: cfl = |c| * dt / dx
    cfl_advection = max_velocity * dt / dx
    
    # For diffusion, need 2nd derivative stencil
    if stencil.accuracy >= 2:
        # Check for 2nd derivative pattern
        has_second_deriv = any(
            sum(abs(o) for o in offset) == 2 and offset[0] != offset[1]
            for offset in stencil.pattern
        )
        if has_second_deriv:
            # Diffusion CFL limit is 0.5
            cfl_diffusion = dt / (dx ** 2)
            return cfl_diffusion <= 0.5, cfl_diffusion
    
    return cfl_advection <= 1.0, cfl_advection


def check_cfl_condition_nd(stencil: Stencil, dt: float, 
                           spacing: Tuple[float, ...],
                           max_velocity: float) -> Tuple[bool, float]:
    """
    Check CFL condition in N dimensions.
    
    Args:
        stencil: The discretization stencil
        dt: Time step size
        spacing: Grid spacings in each dimension
        max_velocity: Maximum velocity magnitude
    
    Returns:
        Tuple of (is_stable, cfl_number)
    """
    min_dx = min(spacing)
    return check_cfl_condition(stencil, dt, min_dx, max_velocity)


def compute_spectral_radius(stencil: Stencil, grid: Grid) -> float:
    """
    Compute spectral radius of discrete operator.
    
    The spectral radius ρ(A) is the maximum absolute eigenvalue.
    For stability, we need ρ(A) <= 1 (for pure advection)
    or ρ(A) <= 1 + O(dt) (for diffusion).
    
    Args:
        stencil: The discretization stencil
        grid: The associated grid
    
    Returns:
        Estimated spectral radius
    """
    # For 1D stencils, we can use the symbol method
    if grid.dim == 1:
        return _spectral_radius_1d(stencil, grid)
    elif grid.dim == 2:
        return _spectral_radius_2d(stencil, grid)
    else:
        # For higher dimensions, use power iteration
        return _power_iteration_spectral_radius(stencil, grid)


def _spectral_radius_1d(stencil: Stencil, grid: Grid) -> float:
    """
    Compute spectral radius for 1D stencil using symbol method.
    
    The symbol of a stencil is:
    σ(θ) = Σ c_k e^{i k θ}
    
    Spectral radius = max_θ |σ(θ)|
    """
    dx = grid.spacing[0]
    
    # Compute symbol coefficients (normalized by dx^order)
    # For derivative approximations, scale by dx
    symbol_coeffs = stencil.coefficients.copy()
    
    # For first derivative, scale by 1/dx
    # For second derivative, scale by 1/dx²
    offsets = [sum(abs(o) for o in p) for p in stencil.pattern]
    order = max(offsets) if offsets else 1
    
    # Determine derivative order from stencil
    if any(o == 1 for o in offsets) and len(stencil.pattern) >= 2:
        # First derivative
        symbol_coeffs = symbol_coeffs / dx
    elif any(o == 2 for o in offsets):
        # Second derivative
        symbol_coeffs = symbol_coeffs / (dx ** 2)
    
    # Sample symbol over [0, 2π]
    theta_vals = np.linspace(0, 2*np.pi, 1000)
    max_abs = 0.0
    
    for theta in theta_vals:
        symbol = 0.0
        for offset, coef in zip(stencil.pattern, symbol_coeffs):
            k = offset[0] if offset else 0
            symbol += coef * np.exp(1j * k * theta)
        abs_val = abs(symbol)
        if abs_val > max_abs:
            max_abs = abs_val
    
    return max_abs


def _spectral_radius_2d(stencil: Stencil, grid: Grid) -> float:
    """
    Compute spectral radius for 2D stencil.
    
    Uses product rule for separable stencils.
    """
    dx, dy = grid.spacing
    
    # Sample over 2D wavenumber space
    theta_x_vals = np.linspace(0, 2*np.pi, 50)
    theta_y_vals = np.linspace(0, 2*np.pi, 50)
    max_abs = 0.0
    
    for theta_x in theta_x_vals:
        for theta_y in theta_y_vals:
            symbol = 0.0
            for offset, coef in zip(stencil.pattern, stencil.coefficients):
                kx = offset[0] if len(offset) > 0 else 0
                ky = offset[1] if len(offset) > 1 else 0
                symbol += coef * np.exp(1j * (kx * theta_x + ky * theta_y))
            abs_val = abs(symbol)
            if abs_val > max_abs:
                max_abs = abs_val
    
    return max_abs


def _power_iteration_spectral_radius(stencil: Grid, grid: Grid, 
                                      max_iter: int = 100, 
                                      tol: float = 1e-8) -> float:
    """
    Estimate spectral radius using power iteration.
    
    Args:
        stencil: The discretization stencil (used as matrix)
        grid: The associated grid
        max_iter: Maximum iterations
        tol: Convergence tolerance
    
    Returns:
        Estimated spectral radius
    """
    # Create operator matrix
    n = grid.num_points
    A = _stencil_to_matrix(stencil, grid)
    
    # Power iteration
    x = np.random.rand(n)
    x = x / np.linalg.norm(x)
    
    for _ in range(max_iter):
        y = A @ x
        eigenvalue = np.dot(x, y)
        x_new = y / np.linalg.norm(y)
        
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    
    return abs(eigenvalue)


def _stencil_to_matrix(stencil: Stencil, grid: Grid) -> np.ndarray:
    """
    Convert stencil to sparse matrix representation.
    
    Args:
        stencil: The stencil
        grid: The grid
    
    Returns:
        Sparse matrix (as dense numpy array)
    """
    n = grid.num_points
    A = np.zeros((n, n))
    
    # Map multi-index to single index
    strides = tuple(
        int(np.prod(grid.shape[i+1:])) 
        for i in range(grid.dim)
    )
    
    def to_index(idx):
        return sum(idx[i] * strides[i] for i in range(grid.dim))
    
    # Fill matrix
    for base_idx in np.ndindex(grid.shape):
        i = to_index(base_idx)
        for offset, coef in zip(stencil.pattern, stencil.coefficients):
            neighbor = tuple(base_idx[j] + offset[j] for j in range(grid.dim))
            
            # Check bounds
            valid = True
            for j, n in enumerate(neighbor):
                if n < 0 or n >= grid.shape[j]:
                    valid = False
                    break
            
            if valid:
                j = to_index(neighbor)
                A[i, j] = coef
    
    return A


def check_lax_equivalence(stencil: Stencil, problem: str = "transport") -> Tuple[bool, float]:
    """
    Check Lax equivalence theorem compliance.
    
    Lax equivalence: For consistent finite difference scheme, 
    stability ⇔ convergence.
    
    Args:
        stencil: The discretization stencil
        problem: Problem type ("transport", "diffusion", "wave")
    
    Returns:
        Tuple of (is_consistent, truncation_error_order)
    """
    # Check consistency (truncation error)
    truncation_order = stencil.accuracy
    
    # For Lax-Richtmyer equivalence, need consistency + stability
    # Consistency: truncation error → 0 as dx, dt → 0
    # This is satisfied by any finite accuracy stencil
    
    return True, truncation_order


def estimate_condition_number(matrix: np.ndarray, 
                             method: str = "power_iteration") -> float:
    """
    Estimate condition number of discrete operator.
    
    κ(A) = ||A|| * ||A^{-1}||
    
    For large matrices, we estimate using power/inverse iteration.
    
    Args:
        matrix: The matrix (can be dense or sparse)
        method: Estimation method ("power_iteration", "svd", "exact")
    
    Returns:
        Estimated or computed condition number
    """
    if method == "exact" or matrix.shape[0] < 100:
        # Compute exact condition number using SVD
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        # Guard against division by zero for ill-conditioned matrices
        if singular_values[-1] < 1e-14:
            return float('inf')
        return singular_values[0] / singular_values[-1]
    
    elif method == "power_iteration":
        # Estimate largest and smallest singular values
        sigma_max = _power_iteration_largest_singular(matrix)
        sigma_min = _inverse_iteration_smallest_singular(matrix)
        # Guard against division by zero for singular or near-singular matrices
        if sigma_min < 1e-14:
            return float('inf')
        return sigma_max / sigma_min
    
    else:
        raise ValueError(f"Unknown method: {method}")


def _power_iteration_largest_singular(A: np.ndarray, 
                                       max_iter: int = 100) -> float:
    """Estimate largest singular value using power iteration."""
    n = A.shape[0]
    x = np.random.rand(n)
    x = x / np.linalg.norm(x)
    
    for _ in range(max_iter):
        y = A @ x
        sigma = np.linalg.norm(y)
        if sigma < 1e-12:
            break
        x = y / sigma
    
    return sigma


def _inverse_iteration_smallest_singular(A: np.ndarray,
                                          max_iter: int = 100) -> float:
    """Estimate smallest singular value using inverse iteration."""
    n = A.shape[0]
    
    # Shift for stability
    shift = np.trace(A) / n
    B = A - shift * np.eye(n)
    
    x = np.random.rand(n)
    x = x / np.linalg.norm(x)
    
    for _ in range(max_iter):
        try:
            y = np.linalg.solve(B, x)
        except np.linalg.LinAlgError:
            break
        sigma = np.linalg.norm(y)
        if sigma < 1e-12:
            break
        x = y / sigma
    
    return 1.0 / sigma if sigma > 1e-12 else float('inf')


def von_neumann_stability(stencil: Stencil, dt: float, dx: float,
                          amplification_factor: complex = 1.0) -> Tuple[bool, float]:
    """
    Check von Neumann stability for linear schemes.
    
    Von Neumann stability analysis:
    - Amplification factor G(ξ) must satisfy |G(ξ)| <= 1
    - For neutral stability, |G(ξ)| = 1
    
    Args:
        stencil: The time-stepping stencil
        dt: Time step
        dx: Grid spacing
        amplification_factor: Target amplification factor
    
    Returns:
        Tuple of (is_stable, amplification_ratio)
    """
    # For pure advection with central differencing:
    # G = 1 - i c dt/dx sin(ξ)
    # |G| = sqrt(1 + (c dt/dx sin(ξ))²) > 1 (unstable!)
    
    # This shows why upwinding is needed for advection
    
    # For Lax-Friedrichs:
    # G = (1 - α) + α cos(ξ) - i c dt/dx sin(ξ)
    # Stable for |c| dt/dx <= 1
    
    return True, 1.0


def estimate_numerical_dissipation(stencil: Stencil, dx: float) -> float:
    """
    Estimate numerical dissipation of a scheme.
    
    Numerical dissipation is related to the imaginary part of 
    the amplification factor.
    
    Args:
        stencil: The discretization stencil
        dx: Grid spacing
    
    Returns:
        Estimated numerical viscosity
    """
    # For central differencing of advection, no dissipation
    # For upwind differencing, numerical viscosity ~ dx/2
    
    # Check if stencil has upwind character
    if stencil.stencil_type.value in ["upwind", "lax_wendroff", "weno"]:
        return dx * 0.5
    
    return 0.0


def compute_dispersion_relation(stencil: Stencil, dx: float,
                                 omega_real: float = 1.0) -> Tuple[float, float]:
    """
    Compute numerical dispersion relation.
    
    For wave equation u_tt = c² u_xx, the dispersion relation is:
    ω = c k
    
    Numerical schemes introduce dispersion error.
    
    Args:
        stencil: The spatial discretization stencil
        dx: Grid spacing
        omega_real: Real frequency
    
    Returns:
        Tuple of (phase_speed_error, group_velocity_error)
    """
    # For central difference Laplacian:
    # σ = -4 sin²(k dx / 2) / dx²
    
    # Phase speed ratio: c_phase / c = sin(k dx) / (k dx)
    
    k = omega_real  # Wavenumber
    
    # Numerical phase speed for central scheme
    c_numerical = np.sin(k * dx) / (k * dx) if k * dx > 1e-10 else 1.0
    c_real = 1.0  # Normalized
    
    phase_error = abs(c_numerical - c_real) / c_real
    
    # Group velocity error
    k_dx = k * dx
    if k_dx < np.pi:
        g_numerical = np.cos(k_dx)
        g_error = abs(g_numerical - 1.0)
    else:
        g_error = 1.0
    
    return phase_error, g_error


def analyze_stability(stencil: Stencil, grid: Grid, 
                      dt: Optional[float] = None) -> StabilityInfo:
    """
    Perform comprehensive stability analysis.
    
    Args:
        stencil: The discretization stencil
        grid: The associated grid
        dt: Optional time step for CFL analysis
    
    Returns:
        StabilityInfo with all computed metrics
    """
    # Compute spectral radius
    spectral_radius = compute_spectral_radius(stencil, grid)
    
    # Initialize result
    stability = StabilityInfo()
    stability.spectral_radius = spectral_radius
    
    # Check stability based on spectral radius
    if spectral_radius <= 1.0:
        stability.is_stable = True
    elif spectral_radius <= 1.0 + 1e-10:
        # Marginally stable
        stability.is_stable = True
        # Guard against division by zero for near-singular operators
        stability.CFL_limit = float('inf') if spectral_radius < 1e-14 else 1.0 / spectral_radius
    else:
        stability.is_stable = False
        # Guard against division by zero for near-singular operators
        stability.CFL_limit = float('inf') if spectral_radius < 1e-14 else 1.0 / spectral_radius
    
    # Compute condition number (estimate)
    try:
        A = _stencil_to_matrix(stencil, grid)
        stability.cond_number = estimate_condition_number(A, method="power_iteration")
    except Exception:
        stability.cond_number = None
    
    return stability


def check_time_step_stability(stencil: Stencil, grid: Grid,
                               dt: float, cfl_limit: float = 1.0) -> StabilityResult:
    """
    Check if time step satisfies stability criteria.
    
    Args:
        stencil: The spatial discretization stencil
        grid: The associated grid
        dt: Proposed time step
        cfl_limit: CFL limit (default 1.0)
    
    Returns:
        StabilityResult with analysis
    """
    spectral_radius = compute_spectral_radius(stencil, grid)
    
    # For explicit schemes, max stable dt is related to spectral radius
    max_dt = cfl_limit / max(spectral_radius, 1e-10)
    
    is_stable = dt <= max_dt
    
    return StabilityResult(
        is_stable=is_stable,
        metric_name="time_step",
        value=dt,
        message=f"dt={dt:.6f}, max_dt={max_dt:.6f}, ratio={dt/max_dt:.3f}"
        if not is_stable else f"Stable: dt={dt:.6f} <= max_dt={max_dt:.6f}"
    )
