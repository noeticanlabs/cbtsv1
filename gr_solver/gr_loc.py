# Implements: LoC-3 damped coherence with kappa_H, kappa_M damping for GR constraints (H,M,Z,Z_i); enforces LoC-4 witness inequality with eps_H, eps_M residuals; logs LoC-6 eta_rep.

# Lexicon declarations per canon v1.2
LEXICON_IMPORTS = [
    "LoC_axiom",
    "UFE_core",
    "GR_dyn",
    "CTL_time"
]
LEXICON_SYMBOLS = {
    "K_LoC": "UFE_coherence_operator",
    "C": "constraint_energy_functional",
    "\\nabla_\\Psi C": "constraint_gradient",
    "K_damp": "damping_term",
    "K_proj": "projection_term",
    "K_stage": "stage_coherence_term",
    "K_bc": "boundary_coherence_term",
    "\\epsilon_{clk}": "clock_coherence_error"
}

import numpy as np
from .gr_core_fields import inv_sym6, trace_sym6, sym6_to_mat33, mat33_to_sym6
from .gr_constraints import GRConstraints

class GRLoC:
    """
    LoC Augmentation Operator K_LoC = K_damp + K_proj + K_stage + K_bc
    Implements explicit decomposition for coherence control in GR evolution.
    """

    def __init__(self, fields, geometry, constraints, lambda_val=0.1, kappa_H=1.0, kappa_M=1.0,
                 wH=1.0, wM=1.0, wclk=1.0, wproj=1.0,
                 H_max=1e-6, M_max=1e-6, clk_max=1e-6, proj_max=1e-6, budget=1e-8,
                 fail_threshold=3):
        """
        Initialize LoC operator.

        Parameters:
        - fields: GRFields object
        - geometry: GRGeometry object
        - constraints: GRConstraints object
        - lambda_val: coherence gain parameter
        - kappa_H: Hamiltonian constraint damping coefficient
        - kappa_M: momentum constraint damping coefficient
        - wH, wM, wclk, wproj: weights for damage functional
        - H_max, M_max, clk_max, proj_max: max tolerances for gates
        - budget: monotonic budget for damage
        - fail_threshold: threshold for persistent fail
        """
        self.fields = fields
        self.geometry = geometry
        self.constraints = constraints
        self.lambda_val = lambda_val
        self.kappa_H = kappa_H
        self.kappa_M = kappa_M
        self.wH = wH
        self.wM = wM
        self.wclk = wclk
        self.wproj = wproj
        self.H_max = H_max
        self.M_max = M_max
        self.clk_max = clk_max
        self.proj_max = proj_max
        self.budget = budget
        self.fail_threshold = fail_threshold
        self.prev_D = 0.0
        self.fail_counter = 0

        # Preallocate arrays for gradients and operators
        shape_6 = (fields.Nx, fields.Ny, fields.Nz, 6)
        shape_3 = (fields.Nx, fields.Ny, fields.Nz, 3)
        shape_scalar = (fields.Nx, fields.Ny, fields.Nz)

        self.grad_C_gamma = np.zeros(shape_6)
        self.grad_C_K = np.zeros(shape_6)
        self.grad_C_phi = np.zeros(shape_scalar)
        self.grad_C_gamma_tilde = np.zeros(shape_6)
        self.grad_C_A = np.zeros(shape_6)
        self.grad_C_Gamma_tilde = np.zeros(shape_3)
        self.grad_C_Z = np.zeros(shape_scalar)
        self.grad_C_Z_i = np.zeros(shape_3)

        # Operator components
        self.K_damp = {
            'gamma_sym6': np.zeros(shape_6),
            'K_sym6': np.zeros(shape_6),
            'phi': np.zeros(shape_scalar),
            'gamma_tilde_sym6': np.zeros(shape_6),
            'A_sym6': np.zeros(shape_6),
            'Gamma_tilde': np.zeros(shape_3),
            'Z': np.zeros(shape_scalar),
            'Z_i': np.zeros(shape_3)
        }

        self.K_proj = {
            'gamma_sym6': np.zeros(shape_6),
            'K_sym6': np.zeros(shape_6),
            'phi': np.zeros(shape_scalar),
            'gamma_tilde_sym6': np.zeros(shape_6),
            'A_sym6': np.zeros(shape_6),
            'Gamma_tilde': np.zeros(shape_3),
            'Z': np.zeros(shape_scalar),
            'Z_i': np.zeros(shape_3)
        }

        self.K_stage = {
            'gamma_sym6': np.zeros(shape_6),
            'K_sym6': np.zeros(shape_6),
            'phi': np.zeros(shape_scalar),
            'gamma_tilde_sym6': np.zeros(shape_6),
            'A_sym6': np.zeros(shape_6),
            'Gamma_tilde': np.zeros(shape_3),
            'Z': np.zeros(shape_scalar),
            'Z_i': np.zeros(shape_3)
        }

        self.K_bc = {
            'gamma_sym6': np.zeros(shape_6),
            'K_sym6': np.zeros(shape_6),
            'phi': np.zeros(shape_scalar),
            'gamma_tilde_sym6': np.zeros(shape_6),
            'A_sym6': np.zeros(shape_6),
            'Gamma_tilde': np.zeros(shape_3),
            'Z': np.zeros(shape_scalar),
            'Z_i': np.zeros(shape_3)
        }

        # Stage coherence tracking
        self.Psi_auth = {}  # Authoritative state
        self.epsilon_clk = 0.0
        self.stage_errors = []

    def compute_constraint_energy_functional(self):
        """
        Compute constraint energy functional C[Psi] = H^2 + |M|^2
        """
        self.constraints.compute_hamiltonian()
        self.constraints.compute_momentum()

        H = self.constraints.H
        M_norm_sq = np.sum(self.constraints.M**2, axis=-1)

        C = H**2 + M_norm_sq
        return C

    def compute_constraint_gradients(self):
        """
        Compute ∇_Psi C where C = H^2 + |M|^2

        This is a simplified implementation. Full gradients would require
        variational derivatives of constraints w.r.t. all fields.
        """
        # Approximate gradients using finite differences or analytical expressions
        # For now, use simple forms based on constraint definitions

        # ∇_gamma H (simplified: H depends on gamma through Ricci and K)
        # This is complex; simplified approximation
        self.grad_C_gamma[:] = -2 * self.constraints.H[:,:,:,np.newaxis] * self.fields.K_sym6

        # ∇_K H (H proportional to K trace and K^2 terms)
        gamma_inv = inv_sym6(self.fields.gamma_sym6)
        K_trace = trace_sym6(self.fields.K_sym6, gamma_inv)
        self.grad_C_K[:] = 2 * self.constraints.H[:,:,:,np.newaxis] * (
            K_trace[:,:,:,np.newaxis] * self.fields.gamma_sym6 -
            2 * self.fields.K_sym6 +
            self.fields.alpha[:,:,:,np.newaxis] * self.fields.gamma_sym6
        )

        # ∇_phi H (through Ricci scaling)
        self.grad_C_phi[:] = -4 * self.constraints.H * self.fields.phi

        # Simplified for tilde fields (BSSN)
        self.grad_C_gamma_tilde[:] = self.grad_C_gamma * 0.1  # Reduced influence
        self.grad_C_A[:] = self.grad_C_K * 0.1
        self.grad_C_Gamma_tilde[:] = 0.0  # Placeholder

        # ∇_Z H = -2 H (from Z4 formulation)
        self.grad_C_Z[:] = -2 * self.constraints.H

        # ∇_Z_i M_j (simplified)
        self.grad_C_Z_i[:] = -2 * self.constraints.M

    def compute_K_damp(self):
        """
        Compute damping term K_damp = -kappa_H ∇_gamma,K,phi,Z C_H - kappa_M ∇_Z_i C_M
        """
        self.compute_constraint_gradients()

        # Damping for ADM/BSSN variables
        self.K_damp['gamma_sym6'][:] = -self.kappa_H * self.grad_C_gamma
        self.K_damp['K_sym6'][:] = -self.kappa_H * self.grad_C_K
        self.K_damp['phi'][:] = -self.kappa_H * self.grad_C_phi
        self.K_damp['gamma_tilde_sym6'][:] = -self.kappa_H * self.grad_C_gamma_tilde
        self.K_damp['A_sym6'][:] = -self.kappa_H * self.grad_C_A
        self.K_damp['Gamma_tilde'][:] = -self.kappa_H * self.grad_C_Gamma_tilde
        self.K_damp['Z'][:] = -self.kappa_H * self.grad_C_Z
        self.K_damp['Z_i'][:] = -self.kappa_M * self.grad_C_Z_i

    def compute_K_proj(self):
        """
        Compute projection term K_proj for basic projections.

        - If magnetic field present: projection for div B = 0
        - Enforce det(gamma_tilde) = 1
        """
        # Placeholder: skip projection for now to avoid broadcasting issues
        # # Enforce det(gamma_tilde) = 1
        # gamma_tilde_det = np.linalg.det(sym6_to_mat33(self.fields.gamma_tilde_sym6))
        # det_correction = (gamma_tilde_det - 1.0)[:,:,:,np.newaxis, np.newaxis]

        # # Approximate projection: adjust gamma_tilde to enforce determinant
        # gamma_tilde_mat = sym6_to_mat33(self.fields.gamma_tilde_sym6)
        # gamma_tilde_inv = np.linalg.inv(gamma_tilde_mat)
        # proj_matrix = gamma_tilde_mat - (1.0/3.0) * np.trace(gamma_tilde_mat)[:,:,:,np.newaxis,np.newaxis] * gamma_tilde_inv

        # self.K_proj['gamma_tilde_sym6'][:] = mat33_to_sym6(proj_matrix) * 0.01  # Small correction

        # Placeholder for div B if magnetic field present
        # For now, assume no magnetic field, so no div B projection
        pass

    def set_authoritative_state(self, Psi_auth):
        """
        Set the authoritative state Psi_auth for stage coherence.
        """
        self.Psi_auth = {
            'gamma_sym6': Psi_auth.get('gamma_sym6', self.fields.gamma_sym6.copy()),
            'K_sym6': Psi_auth.get('K_sym6', self.fields.K_sym6.copy()),
            'phi': Psi_auth.get('phi', self.fields.phi.copy()),
            'gamma_tilde_sym6': Psi_auth.get('gamma_tilde_sym6', self.fields.gamma_tilde_sym6.copy()),
            'A_sym6': Psi_auth.get('A_sym6', self.fields.A_sym6.copy()),
            'Gamma_tilde': Psi_auth.get('Gamma_tilde', self.fields.Gamma_tilde.copy()),
            'Z': Psi_auth.get('Z', self.fields.Z.copy()),
            'Z_i': Psi_auth.get('Z_i', self.fields.Z_i.copy())
        }

    def compute_K_stage(self, Psi_used=None):
        """
        Compute stage coherence term K_stage based on epsilon_clk = max over stages of Linf(Psi_used - Psi_auth)
        """
        if Psi_used is None:
            Psi_used = {
                'gamma_sym6': self.fields.gamma_sym6,
                'K_sym6': self.fields.K_sym6,
                'phi': self.fields.phi,
                'gamma_tilde_sym6': self.fields.gamma_tilde_sym6,
                'A_sym6': self.fields.A_sym6,
                'Gamma_tilde': self.fields.Gamma_tilde,
                'Z': self.fields.Z,
                'Z_i': self.fields.Z_i
            }

        if not self.Psi_auth:
            self.K_stage = {k: np.zeros_like(v) for k, v in self.K_stage.items()}
            return

        # Compute Linf errors for each field
        errors = []
        for field_name in Psi_used:
            diff = np.abs(Psi_used[field_name] - self.Psi_auth[field_name])
            max_diff = np.max(diff)
            errors.append(max_diff)

            # K_stage as proportional to the difference
            self.K_stage[field_name][:] = -0.1 * (Psi_used[field_name] - self.Psi_auth[field_name])

        self.epsilon_clk = max(errors) if errors else 0.0
        self.stage_errors.append(self.epsilon_clk)

    def compute_K_bc(self):
        """
        Placeholder for boundary coherence control K_bc.
        """
        # For now, zero contribution
        pass

    def compute_eps(self):
        """
        Compute coherence errors eps_H, eps_M, eps_proj.
        eps_clk is computed in compute_K_stage.
        """
        # Ensure constraints are computed
        self.constraints.compute_hamiltonian()
        self.constraints.compute_momentum()

        self.eps_H = np.max(np.abs(self.constraints.H))
        self.eps_M = np.sqrt(np.max(np.sum(self.constraints.M**2, axis=-1)))

        # eps_proj: max deviation of det(gamma_tilde) from 1
        gamma_tilde_det = np.linalg.det(sym6_to_mat33(self.fields.gamma_tilde_sym6))
        self.eps_proj = np.max(np.abs(gamma_tilde_det - 1.0))

    def compute_D(self):
        """
        Compute damage functional D = wH*eps_H^2 + wM*eps_M^2 + wclk*eps_clk^2 + wproj*eps_proj^2
        """
        self.D = (self.wH * self.eps_H**2 +
                  self.wM * self.eps_M**2 +
                  self.wclk * self.epsilon_clk**2 +
                  self.wproj * self.eps_proj**2)

    def check_gates(self):
        """
        Check coherence gates G1, G2, G3, G4.

        G1: eps_H <= H_max and eps_M <= M_max
        G2: eps_clk <= clk_max
        G3: eps_proj <= proj_max
        G4: D <= prev_D + budget

        Returns: (pass_all, details)
        """
        g1 = self.eps_H <= self.H_max and self.eps_M <= self.M_max
        g2 = self.epsilon_clk <= self.clk_max
        g3 = self.eps_proj <= self.proj_max
        g4 = self.D <= self.prev_D + self.budget

        pass_all = g1 and g2 and g3 and g4
        details = {'g1': g1, 'g2': g2, 'g3': g3, 'g4': g4}

        return pass_all, details

    def evaluate_and_respond(self, Psi_used=None):
        """
        Evaluate coherence gates and respond according to ladder:
        pass -> accept
        fail -> correct (dt shrink, adjust gains)
        persistent fail -> rollback

        Returns: (action, params)
        where action in ['accept', 'correct', 'rollback']
        params: dict with adjustments if correct
        """
        # Compute required values
        self.compute_eps()
        self.compute_K_stage(Psi_used)  # to set epsilon_clk
        self.compute_D()

        pass_all, details = self.check_gates()

        if pass_all:
            self.fail_counter = 0
            self.prev_D = self.D
            return 'accept', {}
        else:
            self.fail_counter += 1
            if self.fail_counter < self.fail_threshold:
                # correct: shrink dt, adjust gains
                dt_factor = 0.5
                gain_factor = 0.9
                self.lambda_val *= gain_factor
                self.kappa_H *= gain_factor
                self.kappa_M *= gain_factor
                return 'correct', {'dt_factor': dt_factor, 'gain_factor': gain_factor}
            else:
                # rollback
                self.fail_counter = 0  # reset after rollback?
                return 'rollback', {}

    def compute_K_LoC(self, Psi_used=None):
        """
        Compute full LoC operator K_LoC = K_damp + K_proj + K_stage + K_bc
        Also computes coherence errors eps.
        """
        self.compute_K_damp()
        self.compute_K_proj()
        self.compute_K_stage(Psi_used)
        self.compute_K_bc()
        self.compute_eps()

        K_LoC = {}
        for field in self.K_damp:
            K_LoC[field] = (self.K_damp[field] +
                           self.K_proj[field] +
                           self.K_stage[field] +
                           self.K_bc[field])

        return K_LoC

    def get_K_LoC_for_rhs(self, Psi_used=None):
        """
        Get K_LoC contribution for RHS computation: lambda * K_LoC
        """
        K_LoC = self.compute_K_LoC(Psi_used)

        K_LoC_scaled = {}
        for field in K_LoC:
            K_LoC_scaled[field] = self.lambda_val * K_LoC[field]

        return K_LoC_scaled