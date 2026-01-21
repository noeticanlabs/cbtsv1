import struct
import numpy as np
from typing import Dict, Any
try:
    from numba import jit
except ImportError:
    jit = lambda f=None, **kwargs: f if f else (lambda g: g)

# Import GR functions
from gr_geometry_nsc import (
    compute_christoffels_compiled,
    compute_ricci_compiled,
    compute_ricci_scalar_compiled,
    second_covariant_derivative_scalar_compiled,
    lie_derivative_gamma_compiled,
    lie_derivative_K_compiled
)
from gr_constraints_nsc import compute_hamiltonian_compiled, compute_momentum_compiled

class HadamardVM:
    def __init__(self, fields: Dict[str, np.ndarray]):
        self.fields = fields  # e.g., {'gamma': np.array(...)}
        self.registers = [0] * 256  # Register-based, init to 0
        self.pc = 0  # Program counter
        self.jit_cache = {}  # Cache for compiled hotspots
        self.jit_tier = 0  # 0: interpreter, 1: basic JIT, 2: optimized JIT
        self.rhs = {}  # For storing computed RHS

    def execute(self, bytecode: bytes):
        self.bytecode = bytecode
        self.pc = 0
        while self.pc < len(bytecode):
            # Check if compiled version exists
            if self.pc in self.jit_cache:
                self.jit_cache[self.pc]()
                break  # Assume compiled handles the rest
            # Fetch 4-byte instruction
            instr = bytecode[self.pc:self.pc+4]
            if len(instr) != 4:
                break  # End
            self.pc += 4
            opcode, arg1, arg2, meta = struct.unpack('BBBB', instr)
            self.dispatch(opcode, arg1, arg2, meta)

    def dispatch(self, opcode: int, arg1: int, arg2: int, meta: int):
        # Register-based dispatch
        if opcode == 0x01:  # ∂ partial deriv
            self.registers[arg1] = self.partial_deriv_op(self.registers[arg2], meta, 1)  # dir=meta
        elif opcode == 0x02:  # ∇ gradient
            self.registers[arg1] = self.gradient_op(self.registers[arg2])
        elif opcode == 0x03:  # ∇² laplacian
            self.registers[arg1] = self.laplacian_op(self.registers[arg2])
        elif opcode == 0x04:  # φ load field to reg
            field_name = list(self.fields.keys())[arg2]
            self.registers[arg1] = self.fields[field_name]
        elif opcode == 0x05:  # ↻ curvature coupling
            self.registers[arg1] = self.curvature_coupling(self.registers[arg2], self.registers[meta])
        elif opcode == 0x06:  # ⊕ addition
            self.registers[arg1] = self.registers[arg2] + meta * self.registers[arg1]  # a + scale*b, but adjust
        elif opcode == 0x07:  # ⊖ subtraction
            self.registers[arg1] = self.registers[arg2] - meta * self.registers[arg1]
        elif opcode == 0x08:  # ◯ diffusion
            self.registers[arg1] = meta * self.laplacian_op(self.registers[arg2])
        elif opcode == 0x09:  # ∆ damping
            self.registers[arg1] = self.registers[arg2] * (1 - meta / 255.0)
        elif opcode == 0x0A:  # □ boundary
            self.apply_boundary(arg1, arg2, meta)
        elif opcode == 0x0B:  # ⇒ step
            self.step_integrate(arg1, arg2, meta)
        elif opcode == 0x0C:  # * multiply
            self.registers[arg1] = self.registers[arg2] * self.registers[meta]
        elif opcode == 0x0D:  # / divide
            self.registers[arg1] = self.registers[arg2] / self.registers[meta]
        elif opcode == 0x0E:  # + add numeric
            self.registers[arg1] = self.registers[arg2] + self.registers[meta]
        elif opcode == 0x0F:  # - sub numeric
            self.registers[arg1] = self.registers[arg2] - self.registers[meta]
        elif opcode == 0x10:  # = assign reg to field
            field_name = list(self.fields.keys())[arg1]
            self.fields[field_name] = self.registers[arg2]
        elif opcode == 0x11:  # ( load const? or push, but register
            self.registers[arg1] = meta  # load const meta into reg arg1
        # Add GR ops
        elif opcode == 0x20:  # ricci
            self.registers[arg1] = self.ricci_tensor(self.registers[arg2])
        elif opcode == 0x21:  # lie
            self.registers[arg1] = self.lie_derivative(self.registers[arg2], self.registers[meta])
        elif opcode == 0x22:  # constraint
            self.constraint_check(arg1, arg2, meta)
        elif opcode == 0x23:  # gauge
            self.gauge_fixing(arg1, arg2, meta)
        # Check for JIT compilation
        if self.pc % 1000 == 0:  # Hotspot threshold
            self.try_jit_compile()

    # Optimized ops
    def partial_deriv_op(self, field: np.ndarray, dir: int, order: int) -> np.ndarray:
        # Finite difference partial derivative
        if dir == 0:  # x
            return np.gradient(field, axis=0)
        elif dir == 1:  # y
            return np.gradient(field, axis=1)
        elif dir == 2:  # z
            return np.gradient(field, axis=2)
        return field  # default

    def gradient_op(self, field: np.ndarray) -> np.ndarray:
        # ∇ as vector of ∂
        grad = np.gradient(field)
        return np.array(grad)  # or stack them

    def laplacian_op(self, field: np.ndarray) -> np.ndarray:
        # ∇² = ∂²/∂x² + ∂²/∂y² + ∂²/∂z²
        grad_xx = np.gradient(np.gradient(field, axis=0), axis=0)
        grad_yy = np.gradient(np.gradient(field, axis=1), axis=1)
        grad_zz = np.gradient(np.gradient(field, axis=2), axis=2)
        return grad_xx + grad_yy + grad_zz

    def curvature_coupling(self, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        # Placeholder: simple coupling
        return src * dst

    def apply_boundary(self, field_reg: int, type_: int, val: int):
        # Apply boundary condition to register field
        field = self.registers[field_reg]
        if type_ == 0:  # none
            pass
        elif type_ == 1:  # dirichlet
            # Set boundaries to val
            field[0,:,:] = val
            field[-1,:,:] = val
            # etc. for all faces
        self.registers[field_reg] = field

    def step_integrate(self, method: int, dt_reg: int, rhs_reg: int):
        # Simple Euler: field += dt * rhs
        dt = self.registers[dt_reg]
        rhs = self.registers[rhs_reg]
        if method == 0:  # Euler
            self.registers[arg1] += dt * rhs  # assuming arg1 is the field reg, but wait, in dispatch arg1 is for step

    def ricci_tensor(self, metric: np.ndarray) -> np.ndarray:
        # Compute full GR RHS
        lambda_val = 0.0
        sources_enabled = False

        # Extract fields
        gamma_sym6 = self.fields['gamma_sym6']
        K_sym6 = self.fields['K_sym6']
        alpha = self.fields['alpha']
        beta = self.fields['beta']
        phi = self.fields['phi']
        gamma_tilde_sym6 = self.fields['gamma_tilde_sym6']
        A_sym6 = self.fields['A_sym6']
        Gamma_tilde = self.fields['Gamma_tilde']
        Z = self.fields['Z']
        Z_i = self.fields['Z_i']
        dx, dy, dz = self.fields['dx'], self.fields['dy'], self.fields['dz']

        Nx, Ny, Nz = gamma_sym6.shape[:3]

        # Initialize outputs
        rhs_gamma_sym6 = np.zeros_like(gamma_sym6)
        rhs_K_sym6 = np.zeros_like(K_sym6)
        rhs_phi = np.zeros_like(phi)
        rhs_gamma_tilde_sym6 = np.zeros_like(gamma_tilde_sym6)
        rhs_A_sym6 = np.zeros_like(A_sym6)
        rhs_Gamma_tilde = np.zeros_like(Gamma_tilde)
        rhs_Z = np.zeros_like(Z)
        rhs_Z_i = np.zeros_like(Z_i)

        # Full GR RHS computation (copied from nsc_runtime_min.py)

        # Set sources
        S_gamma_tilde_sym6 = np.zeros_like(gamma_tilde_sym6)
        S_A_sym6 = np.zeros_like(A_sym6)
        S_phi = np.zeros_like(phi)
        S_Gamma_tilde = np.zeros_like(Gamma_tilde)
        S_Z = np.zeros_like(Z)
        S_Z_i = np.zeros_like(Z_i)

        # Geometry
        christoffels, _ = compute_christoffels_compiled(gamma_sym6, dx, dy, dz)
        ricci_full = compute_ricci_compiled(gamma_sym6, christoffels, dx, dy, dz)
        gamma_inv_sym6 = self.inv_sym6(gamma_sym6)
        gamma_inv_mat = self.sym6_to_mat33(gamma_inv_sym6)
        R = np.einsum('...ij,...ij', gamma_inv_mat, ricci_full)

        gamma_inv_scratch = self.inv_sym6(gamma_sym6)
        K_trace_scratch = self.trace_sym6(K_sym6, gamma_inv_scratch)

        alpha_expanded_scratch = alpha[..., np.newaxis]
        alpha_expanded_33_scratch = alpha[..., np.newaxis, np.newaxis]

        # ADM ∂t gamma_ij
        lie_gamma_scratch = lie_derivative_gamma_compiled(gamma_sym6, beta, dx, dy, dz)
        rhs_gamma_sym6[:] = -2.0 * alpha_expanded_scratch * K_sym6 + lie_gamma_scratch

        # ADM ∂t K_ij
        DD_alpha_scratch = second_covariant_derivative_scalar_compiled(alpha, christoffels, dx, dy, dz)
        DD_alpha_sym6_scratch = self.mat33_to_sym6(DD_alpha_scratch)

        ricci_sym6_scratch = self.mat33_to_sym6(ricci_full)

        K_full_scratch = self.sym6_to_mat33(K_sym6)
        gamma_inv_full_scratch = self.sym6_to_mat33(gamma_inv_scratch)

        K_contracted_full_scratch = np.einsum('...ij,...jk,...kl->...il', K_full_scratch, gamma_inv_full_scratch, K_full_scratch)
        K_contracted_sym6_scratch = self.mat33_to_sym6(K_contracted_full_scratch)

        lie_K_scratch = lie_derivative_K_compiled(K_sym6, beta, dx, dy, dz)

        lambda_term = 2.0 * alpha_expanded_scratch * lambda_val * gamma_sym6

        rhs_K_sym6[:] = (-DD_alpha_sym6_scratch +
                          alpha_expanded_scratch * ricci_sym6_scratch +
                          -2.0 * alpha_expanded_scratch * K_contracted_sym6_scratch +
                          alpha_expanded_scratch * K_trace_scratch[..., np.newaxis] * K_sym6 +
                          lambda_term +
                          lie_K_scratch)

        # BSSN ∂t φ
        div_beta = np.gradient(beta[..., 0], dx, axis=0) + np.gradient(beta[..., 1], dy, axis=1) + np.gradient(beta[..., 2], dz, axis=2)
        rhs_phi[:] = - (alpha / 6.0) * K_trace_scratch + (1.0 / 6.0) * div_beta
        grad_phi = np.array([np.gradient(phi, dx, axis=0),
                             np.gradient(phi, dy, axis=1),
                             np.gradient(phi, dz, axis=2)])
        advection_phi = np.sum(beta * grad_phi.transpose(1,2,3,0), axis=-1)
        rhs_phi[:] += advection_phi

        # BSSN Gamma_tilde (simplified, copy from nsc_runtime_min.py)
        rhs_Gamma_tilde = np.zeros_like(Gamma_tilde)  # Placeholder

        # BSSN ∂t γ̃_ij
        lie_gamma_tilde_scratch = lie_derivative_gamma_compiled(gamma_tilde_sym6, beta, dx, dy, dz)
        rhs_gamma_tilde_sym6[:] = -2.0 * alpha_expanded_scratch * A_sym6 + lie_gamma_tilde_scratch

        # BSSN ∂t A_ij (simplified)
        psi_minus4_scratch = np.exp(-4 * phi)
        psi_minus4_expanded_scratch = psi_minus4_scratch[..., np.newaxis]
        ricci_tf_sym6_scratch = ricci_sym6_scratch - (1/3) * gamma_sym6 * R[..., np.newaxis]

        rhs_A_temp_scratch = psi_minus4_expanded_scratch * alpha_expanded_scratch * ricci_tf_sym6_scratch + alpha_expanded_scratch * K_trace_scratch[..., np.newaxis] * A_sym6

        lie_A_scratch = lie_derivative_K_compiled(A_sym6, beta, dx, dy, dz)
        rhs_A_sym6[:] = rhs_A_temp_scratch + lie_A_scratch

        # Constraint damping
        H = compute_hamiltonian_compiled(R, gamma_sym6, K_sym6, lambda_val)
        M = compute_momentum_compiled(gamma_sym6, K_sym6, christoffels, dx, dy, dz)
        kappa = 1.0
        rhs_Z[:] = -kappa * alpha * H
        rhs_Z_i[:] = -kappa * alpha[..., np.newaxis] * M

        # Add sources
        rhs_gamma_tilde_sym6 += S_gamma_tilde_sym6
        rhs_A_sym6 += S_A_sym6
        rhs_Gamma_tilde += S_Gamma_tilde
        rhs_phi += S_phi
        rhs_Z += S_Z
        rhs_Z_i += S_Z_i

        self.rhs = {
            'rhs_gamma_sym6': rhs_gamma_sym6,
            'rhs_K_sym6': rhs_K_sym6,
            'rhs_phi': rhs_phi,
            'rhs_gamma_tilde_sym6': rhs_gamma_tilde_sym6,
            'rhs_A_sym6': rhs_A_sym6,
            'rhs_Gamma_tilde': rhs_Gamma_tilde,
            'rhs_Z': rhs_Z,
            'rhs_Z_i': rhs_Z_i
        }

        return np.zeros_like(metric)  # Dummy return

    def lie_derivative(self, field: np.ndarray, vector: np.ndarray) -> np.ndarray:
        # Placeholder Lie derivative
        return field

    def constraint_check(self, h_reg: int, m_reg: int, eps: int):
        # Check eps_H < eps, eps_M < eps
        h = np.max(np.abs(self.registers[h_reg]))
        m = np.max(np.abs(self.registers[m_reg]))
        if h > eps / 255.0 or m > eps / 255.0:
            raise ValueError("Constraints violated")

    def gauge_fixing(self, field_reg: int, type_: int, param: int):
        # Placeholder gauge fixing
        pass

    # JIT tier hooks
    def try_jit_compile(self):
        if self.jit_tier < 2:
            self.jit_tier += 1
            # Compile current segment
            start = max(0, self.pc - 4000)
            segment = self.bytecode[start:self.pc]
            self.jit_cache[self.pc] = self.compile_hotspot(segment)

    def compile_hotspot(self, bytecode_segment: bytes) -> callable:
        # Use numba or cython to compile
        # Placeholder: just return lambda that executes segment
        def compiled():
            # Interpret segment faster
            pass
        return compiled

    # Helper functions
    def sym6_to_mat33(self, sym6):
        """Convert symmetric 3x3 tensor stored as sym6 into full 3x3 matrix."""
        shape = sym6.shape[:-1]
        mat = np.zeros(shape + (3, 3), dtype=sym6.dtype)
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

    def mat33_to_sym6(self, mat):
        """Convert full 3x3 matrix to sym6 storage."""
        sym6 = np.empty(mat.shape[:-2] + (6,), dtype=mat.dtype)
        sym6[..., 0] = mat[..., 0, 0]
        sym6[..., 1] = mat[..., 0, 1]
        sym6[..., 2] = mat[..., 0, 2]
        sym6[..., 3] = mat[..., 1, 1]
        sym6[..., 4] = mat[..., 1, 2]
        sym6[..., 5] = mat[..., 2, 2]
        return sym6

    def inv_sym6(self, sym6):
        """Inverse of symmetric 3x3 tensor in sym6 form."""
        xx, xy, xz, yy, yz, zz = np.moveaxis(sym6, -1, 0)
        det = xx * (yy * zz - yz * yz) - xy * (xy * zz - yz * xz) + xz * (xy * yz - yy * xz)
        inv = np.empty_like(sym6)
        inv[..., 0] = (yy*zz - yz*yz) / det
        inv[..., 1] = -(xy*zz - xz*yz) / det
        inv[..., 2] = (xy*yz - xz*yy) / det
        inv[..., 3] = (xx*zz - xz*xz) / det
        inv[..., 4] = -(xx*yz - xy*xz) / det
        inv[..., 5] = (xx*yy - xy*xy) / det
        return inv

    def trace_sym6(self, sym6, inv_sym6):
        """Compute trace: gamma^{ij} A_ij"""
        return (
            inv_sym6[..., 0]*sym6[..., 0]
          + 2.0*inv_sym6[..., 1]*sym6[..., 1]
          + 2.0*inv_sym6[..., 2]*sym6[..., 2]
          + inv_sym6[..., 3]*sym6[..., 3]
          + 2.0*inv_sym6[..., 4]*sym6[..., 4]
          + inv_sym6[..., 5]*sym6[..., 5]
        )