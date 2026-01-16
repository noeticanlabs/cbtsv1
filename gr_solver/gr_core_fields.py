# Lexicon declarations per canon v1.2
LEXICON_IMPORTS = [
    "LoC_axiom",
    "UFE_core",
    "GR_dyn",
    "CTL_time"
]
LEXICON_SYMBOLS = {
    "\\gamma": "GR_field.metric",
    "K": "GR_field.extrinsic",
    "\\alpha": "GR_gauge.lapse",
    "\\beta": "GR_gauge.shift"
}

import numpy as np

SYM6_IDX = {
    "xx": 0,
    "xy": 1,
    "xz": 2,
    "yy": 3,
    "yz": 4,
    "zz": 5,
}

def sym6_to_mat33(sym6: np.ndarray) -> np.ndarray:
    """
    Convert symmetric 3x3 tensor stored as sym6 into full 3x3 matrix.

    sym6 shape: (..., 6)
    return shape: (..., 3, 3)
    """
    assert sym6.shape[-1] == 6

    mat = np.zeros(sym6.shape[:-1] + (3, 3), dtype=sym6.dtype)

    mat[..., 0, 0] = sym6[..., 0]
    mat[..., 0, 1] = sym6[..., 1]
    mat[..., 0, 2] = sym6[..., 2]

    mat[..., 1, 0] = sym6[..., 1]
    mat[..., 1, 1] = sym6[..., 3]
    mat[..., 1, 2] = sym6[..., 4]

    mat[..., 2, 0] = sym6[..., 2]
    mat[..., 2, 1] = sym6[..., 4]
    mat[..., 2, 2] = sym6[..., 5]

    return mat

def mat33_to_sym6(mat: np.ndarray, enforce_symmetry: bool = True) -> np.ndarray:
    """
    Convert full 3x3 matrix to sym6 storage.
    Optionally enforce symmetry by averaging off-diagonals.
    """
    assert mat.shape[-2:] == (3, 3)

    if enforce_symmetry:
        mat = 0.5 * (mat + np.swapaxes(mat, -1, -2))

    sym6 = np.empty(mat.shape[:-2] + (6,), dtype=mat.dtype)

    sym6[..., 0] = mat[..., 0, 0]
    sym6[..., 1] = mat[..., 0, 1]
    sym6[..., 2] = mat[..., 0, 2]
    sym6[..., 3] = mat[..., 1, 1]
    sym6[..., 4] = mat[..., 1, 2]
    sym6[..., 5] = mat[..., 2, 2]

    return sym6

def det_sym6(sym6: np.ndarray) -> np.ndarray:
    """
    Determinant of a symmetric 3x3 matrix in sym6 form.
    """
    xx, xy, xz, yy, yz, zz = np.moveaxis(sym6, -1, 0)

    return (
        xx * (yy * zz - yz * yz)
        - xy * (xy * zz - yz * xz)
        + xz * (xy * yz - yy * xz)
    )

def inv_sym6(sym6: np.ndarray, det_floor: float = 1e-14) -> np.ndarray:
    """
    Inverse of symmetric 3x3 tensor in sym6 form.
    Returns sym6 representation of inverse.
    """
    det = det_sym6(sym6)

    if np.any(det <= det_floor):
        raise ValueError("Metric determinant too small or non-positive")

    xx, xy, xz, yy, yz, zz = np.moveaxis(sym6, -1, 0)

    inv = np.empty_like(sym6)

    inv[..., 0] =  (yy*zz - yz*yz) / det
    inv[..., 1] = -(xy*zz - xz*yz) / det
    inv[..., 2] =  (xy*yz - xz*yy) / det

    inv[..., 3] =  (xx*zz - xz*xz) / det
    inv[..., 4] = -(xx*yz - xy*xz) / det

    inv[..., 5] =  (xx*yy - xy*xy) / det

    return inv

def trace_sym6(sym6: np.ndarray, inv_sym6: np.ndarray) -> np.ndarray:
    """
    Compute trace: gamma^{ij} A_ij
    """
    return (
        inv_sym6[..., 0]*sym6[..., 0]
      + 2.0*inv_sym6[..., 1]*sym6[..., 1]
      + 2.0*inv_sym6[..., 2]*sym6[..., 2]
      + inv_sym6[..., 3]*sym6[..., 3]
      + 2.0*inv_sym6[..., 4]*sym6[..., 4]
      + inv_sym6[..., 5]*sym6[..., 5]
    )

def norm2_sym6(sym6: np.ndarray, inv_sym6: np.ndarray) -> np.ndarray:
    """
    Compute A_ij A^ij
    """
    return trace_sym6(sym6, inv_sym6)

def eigenvalues_sym6(sym6: np.ndarray) -> np.ndarray:
    """
    Compute eigenvalues of symmetric 3x3 matrix in sym6 form.
    Returns eigenvalues sorted in ascending order, shape (..., 3)
    """
    mat = sym6_to_mat33(sym6)
    # Flatten to 2D for batch eigvals: (N, 9) -> (N, 3)
    orig_shape = mat.shape[:-2]
    mat_flat = mat.reshape(-1, 3, 3)
    eigvals = np.linalg.eigvals(mat_flat)
    # Sort ascending
    eigvals = np.sort(eigvals.real, axis=-1)
    eigvals = eigvals.reshape(orig_shape + (3,))
    return eigvals

def cond_sym6(sym6: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Compute condition number: lambda_max / lambda_min, clamped to avoid inf.
    """
    eigvals = eigenvalues_sym6(sym6)
    lambda_min = eigvals[..., 0]
    lambda_max = eigvals[..., 2]
    kappa = np.where(lambda_min > eps, lambda_max / lambda_min, np.inf)
    return kappa

def repair_spd_eigen_clamp(sym6: np.ndarray, lambda_floor: float = 1e-8) -> tuple[np.ndarray, float, float]:
    """
    Repair SPD by eigen-clamping: gamma = Q * diag(max(lambda_i, lambda_floor)) * Q^T
    Returns: (repaired_sym6, lambda_min_pre, lambda_min_post)
    """
    mat = sym6_to_mat33(sym6)
    orig_shape = mat.shape[:-2]
    mat_flat = mat.reshape(-1, 3, 3)

    # Compute pre-repair lambda_min
    eigvals_pre = np.array([np.linalg.eigvals(m).real for m in mat_flat])
    lambda_min_pre = np.min(eigvals_pre)

    repaired_flat = np.empty_like(mat_flat)
    for i in range(mat_flat.shape[0]):
        w, v = np.linalg.eigh(mat_flat[i])  # eigh for symmetric, real eigenvalues
        w_clamped = np.maximum(w, lambda_floor)
        repaired_flat[i] = v @ np.diag(w_clamped) @ v.T

    repaired_mat = repaired_flat.reshape(orig_shape + (3, 3))
    repaired_sym6 = mat33_to_sym6(repaired_mat)

    # Compute post-repair lambda_min
    eigvals_post = np.array([np.linalg.eigvals(m).real for m in repaired_flat])
    lambda_min_post = np.min(eigvals_post)

    return repaired_sym6, lambda_min_pre, lambda_min_post

def symmetry_error(mat: np.ndarray) -> np.ndarray:
    """
    Measure asymmetry magnitude: ||A - A^T||_∞
    """
    return np.max(np.abs(mat - np.swapaxes(mat, -1, -2)))

class GRCoreFields:
    def __init__(self, Nx, Ny, Nz, dx=1.0, dy=1.0, dz=1.0):
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        self.dx, self.dy, self.dz = dx, dy, dz
        # Spatial metric \gamma_{ij}, shape (Nx, Ny, Nz, 6) in sym6 form
        self.gamma_sym6 = np.zeros((Nx, Ny, Nz, 6))
        # Extrinsic curvature K_{ij}, shape (Nx, Ny, Nz, 6) in sym6 form
        self.K_sym6 = np.zeros((Nx, Ny, Nz, 6))
        # Lapse \alpha
        self.alpha = np.ones((Nx, Ny, Nz))
        # Shift \beta^i, vector
        self.beta = np.zeros((Nx, Ny, Nz, 3))
        # Active constraint fields
        # Conformal factor phi (for BSSN)
        self.phi = np.zeros((Nx, Ny, Nz))
        # Z scalar (Hamiltonian constraint evolution)
        self.Z = np.zeros((Nx, Ny, Nz))
        # Z_i vector (momentum constraint evolution)
        self.Z_i = np.zeros((Nx, Ny, Nz, 3))
        # BSSN fields
        # Conformal metric γ̃_ij
        self.gamma_tilde_sym6 = np.zeros((Nx, Ny, Nz, 6))
        # Trace-free extrinsic curvature A_ij
        self.A_sym6 = np.zeros((Nx, Ny, Nz, 6))
        # BSSN Gamma_tilde^i
        self.Gamma_tilde = np.zeros((Nx, Ny, Nz, 3))
        # BSSN gauge variables (optional, for Gamma-driver)
        self.lambda_i = np.zeros((Nx, Ny, Nz, 3))

    def init_minkowski(self):
        """Initialize to Minkowski spacetime."""
        # gamma_sym6: [xx, xy, xz, yy, yz, zz] = [1, 0, 0, 1, 0, 1]
        self.gamma_sym6[..., SYM6_IDX["xx"]] = 1.0
        self.gamma_sym6[..., SYM6_IDX["yy"]] = 1.0
        self.gamma_sym6[..., SYM6_IDX["zz"]] = 1.0
        # K_sym6 all zero
        self.K_sym6.fill(0.0)
        self.alpha.fill(1.0)
        self.beta.fill(0.0)
        # Initialize active constraint fields
        self.phi.fill(0.0)  # ln(psi), for Minkowski psi=1
        self.Z.fill(0.0)
        self.Z_i.fill(0.0)
        # Initialize BSSN fields
        self.gamma_tilde_sym6 = self.gamma_sym6.copy()  # γ̃_ij = γ_ij
        self.A_sym6.fill(0.0)  # A_ij = 0
        self.Gamma_tilde.fill(0.0)  # Gamma_tilde^i = 0
        self.lambda_i.fill(0.0)

    def raise_index(self, tensor, gamma_inv):
        """Raise index using inverse metric. Stub."""
        pass

    def lower_index(self, tensor, gamma):
        """Lower index using metric. Stub."""
        pass

    def bssn_decompose(self):
        """Decompose ADM fields into BSSN variables: γ_ij = e^{4φ} γ̃_ij, A_ij = K_ij - (1/3) γ_ij K"""
        psi4 = np.exp(4 * self.phi)  # ψ^4
        psi4_expanded = psi4[..., np.newaxis]
        self.gamma_tilde_sym6 = self.gamma_sym6 / psi4_expanded

        # K = trace K
        gamma_inv = inv_sym6(self.gamma_sym6)
        K_trace = trace_sym6(self.K_sym6, gamma_inv)

        # A_ij = K_ij - (1/3) γ_ij trK
        A_full = np.zeros((self.Nx, self.Ny, self.Nz, 3, 3))
        gamma_full = np.zeros_like(A_full)
        K_full = np.zeros_like(A_full)
        for i in range(self.Nx):
            for j in range(self.Ny):
                for k in range(self.Nz):
                    gamma_full[i,j,k] = sym6_to_tensor(self.gamma_sym6[i,j,k])
                    K_full[i,j,k] = sym6_to_tensor(self.K_sym6[i,j,k])

        for ii in range(self.Nx):
            for jj in range(self.Ny):
                for kk in range(self.Nz):
                    A_full[ii,jj,kk] = K_full[ii,jj,kk] - (1/3) * K_trace[ii,jj,kk] * gamma_full[ii,jj,kk]

        self.A_sym6 = mat33_to_sym6(A_full)

    def bssn_recompose(self):
        """Recompose ADM fields from BSSN variables: γ_ij = e^{4φ} γ̃_ij, K_ij = A_ij + (1/3) γ_ij K"""
        psi4 = np.exp(4 * self.phi)
        psi4_expanded = psi4[..., np.newaxis]
        self.gamma_sym6 = psi4_expanded * self.gamma_tilde_sym6

        # For K, need to compute K_trace from A and phi evolution, but for now assume it's done in stepper
        # This method assumes gamma_tilde and A are set, recomposes gamma
        # K recomposition needs K_trace, which comes from phi evolution
        pass

    # Other methods as needed