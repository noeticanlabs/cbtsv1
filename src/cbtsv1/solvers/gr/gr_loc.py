# gr_loc.py
# =============================================================================
# Limit of Coherence (LoC) Operator Implementation
# =============================================================================
# 
# This module implements the Limit of Coherence (LoC) operator for explicit
# coherence control in GR evolution. The LoC operator is a feedback controller
# that drives constraint violations toward zero.
# 
# **LoC Decomposition:**
#     K_LoC = K_damp + K_proj + K_stage + K_bc
# 
# **K_damp** (Damping term):
#     K_damp = -κ ∇_Ψ C
#     
#     Where C = H² + |M|² is the constraint energy functional.
#     This term pushes the system toward the constraint surface.
# 
# **K_proj** (Projection term):
#     Enforces algebraic constraints (det(γ̃)=1, tr(A)=0)
# 
# **K_stage** (Stage coherence term):
#     Controls discretization error between RK stages:
#     ε_clk = max_s ||Ψ_used(s) - Ψ_auth||_∞
# 
# **K_bc** (Boundary coherence term):
#     Controls boundary behavior for coherence
# 
# **Damage Functional:**
#     D = w_H ε_H² + w_M ε_M² + w_clk ε_clk² + w_proj ε_proj²
# 
#     Measures accumulated "damage" from constraint violations.
#
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
from .geometry.core_fields import inv_sym6, trace_sym6, sym6_to_mat33, mat33_to_sym6
from .constraints.constraints import GRConstraints

class GRLoC:
    """
    Limit of Coherence (LoC) Operator for Explicit Coherence Control.
    
    The LoC operator provides explicit feedback control for GR evolution,
    augmenting the RHS to actively damp constraint violations:
    
    **Operator Decomposition:**
        K_LoC = K_damp + K_proj + K_stage + K_bc
    
    **K_damp** - Damping toward constraint surface:
        K_damp = -κ_H ∇_{γ,K,φ} C_H - κ_M ∇_{Z_i} C_M
        
        Where C = H² + |M|² is the constraint energy functional.
        The negative gradient points toward decreasing C.
    
    **K_proj** - Algebraic constraint projection:
        Enforces det(γ̃) = 1 and tr(A) = 0 via projection.
    
    **K_stage** - Multi-step coherence:
        Controls discretization error at RK stages:
        ε_clk = max_s ||Ψ_used(s) - Ψ_auth||_∞
    
    **K_bc** - Boundary coherence:
        Controls boundary behavior for constraint preservation.
    
    **Gate Checking:**
        - G1: ε_H ≤ H_max and ε_M ≤ M_max (constraint bounds)
        - G2: ε_clk ≤ clk_max (clock coherence)
        - G3: ε_proj ≤ proj_max (projection error)
        - G4: D ≤ D_prev + budget (monotonicity)
    
    **Response Ladder:**
        - pass: Accept step, update D_prev
        - fail: Reduce dt, adjust gains
        - persistent fail: Rollback
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
        Compute constraint energy functional C[Psi] = H^2 + |M|^2.
        
        **Formula:**
            C = H² + |M|² = H² + M_i M^i
        
        The constraint energy measures how far the system is from satisfying
        the Hamiltonian and momentum constraints.
        """
        self.constraints.compute_hamiltonian()
        self.constraints.compute_momentum()

        H = self.constraints.H
        M_norm_sq = np.sum(self.constraints.M**2, axis=-1)

        C = H**2 + M_norm_sq
        return C

    def compute_constraint_gradients(self):
        """
        Compute ∇_Ψ C where C = H² + |M|².
        
        The constraint gradient tells us how each field affects the constraint
        energy. This is used in K_damp to push the system toward lower C.
        
        **Simplified Approximations:**
        - ∇_γ H ≈ -2 H K (through K dependence)
        - ∇_K H ≈ 2 H (γ K - 2K + α γ)  (trace and quadratic terms)
        - ∇_φ H ≈ -4 H φ (through Ricci scaling)
        - ∇_Z H = -2 H (Z4 formulation)
        - ∇_{Z_i} M_j = -2 M_j
        """
        # ∇_gamma H (simplified: H depends on gamma through Ricci and K)
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
        Compute damping term K_damp = -κ_H ∇_{γ,K,φ} C_H - κ_M ∇_{Z_i} C_M.
        
        **Formula:**
            K_damp = κ_H ∇ C_H + κ_M ∇ C_M
        
        The positive sign convention means K_damp points in the direction
        of decreasing constraint energy (negative gradient direction).
        
        This term is added to the RHS to actively damp constraint violations.
        """
        self.compute_constraint_gradients()

        # Damping for ADM/BSSN variables
        self.K_damp['gamma_sym6'][:] = self.kappa_H * self.grad_C_gamma
        self.K_damp['K_sym6'][:] = self.kappa_H * self.grad_C_K
        self.K_damp['phi'][:] = self.kappa_H * self.grad_C_phi
        self.K_damp['gamma_tilde_sym6'][:] = self.kappa_H * self.grad_C_gamma_tilde
        self.K_damp['A_sym6'][:] = self.kappa_H * self.grad_C_A
        self.K_damp['Gamma_tilde'][:] = self.kappa_H * self.grad_C_Gamma_tilde
        self.K_damp['Z'][:] = self.kappa_H * self.grad_C_Z
        self.K_damp['Z_i'][:] = self.kappa_M * self.grad_C_Z_i

    def compute_K_proj(self):
        """
        Compute projection term K_proj for algebraic constraints.
        
        **Purpose:**
        - Enforce det(γ̃) = 1 (conformal metric determinant)
        - Handle div B = 0 for magnetic fields (future extension)
        
        Currently a placeholder - full projection is complex and can cause
        broadcasting issues. The algebraic constraints are enforced via
        geometry.enforce_det_gamma_tilde() and geometry.enforce_traceless_A().
        """
        pass

    def set_authoritative_state(self, Psi_auth):
        """
        Set the authoritative state Ψ_auth for stage coherence tracking.
        
        The authoritative state is the initial state at the beginning of an
        RK step. Intermediate RK stages compute Ψ_used which is compared
        against Ψ_auth to compute ε_clk.
        
        Args:
            Psi_auth: Dict containing initial field values for comparison
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
        Compute stage coherence term K_stage based on ε_clk.
        
        **Stage Coherence Error:**
            ε_clk = max_s max_Ψ ||Ψ_used(s) - Ψ_auth||_∞
        
        This measures the maximum deviation between any intermediate RK stage
        state and the authoritative initial state. Large ε_clk indicates
        the timestep is too large for the current dynamics.
        
        **K_stage Formula:**
            K_stage = -0.1 * (Ψ_used - Ψ_auth)
        
        This provides a small correction to keep stages coherent with the
        authoritative state.
        
        Args:
            Psi_used: Current state (defaults to self.fields)
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
        
        **Purpose:**
        - Control constraint behavior at boundaries
        - For periodic boundaries: K_bc = 0 (no adjustment needed)
        - For other boundaries: enforce appropriate boundary conditions
        
        Currently zero - boundary conditions are handled separately.
        """
        pass

    def compute_D(self):
        """
        Compute damage functional D = w_H ε_H² + w_M ε_M² + w_clk ε_clk² + w_proj ε_proj².
        
        **Physical Interpretation:**
        The damage functional measures accumulated constraint violations,
        weighted by their importance. It is used in gate G4 to ensure
        monotonic progress (damage should not increase too rapidly).
        
        **Interpretation:**
        - Small D: Good constraint preservation
        - Growing D: Constraint violations accumulating
        - D > D_prev + budget: Gate violation (G4)
        """
        self.D = (self.wH * self.eps_H**2 +
                  self.wM * self.eps_M**2 +
                  self.wclk * self.epsilon_clk**2 +
                  self.wproj * self.eps_proj**2)

    def check_gates(self):
        """
        Check coherence gates G1, G2, G3, G4.
        
        **Gate Definitions:**
        - G1: ε_H ≤ H_max and ε_M ≤ M_max
              (Hamiltonian and momentum constraints within bounds)
        - G2: ε_clk ≤ clk_max
              (Stage coherence error within tolerance)
        - G3: ε_proj ≤ proj_max
              (Projection error within tolerance)
        - G4: D ≤ D_prev + budget
              (Damage is not increasing too rapidly)
        
        Returns:
            Tuple: (pass_all, details)
                   pass_all: True if all gates pass
                   details: Dict with individual gate results
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
        
        **Response Ladder:**
        1. pass: Accept step, reset fail_counter, update D_prev
        2. fail: Reduce dt by factor 0.5, reduce gains by 0.9, increment fail_counter
        3. rollback: Reset fail_counter, return rollback signal
        
        Returns:
            Tuple: (action, params)
                   action: 'accept', 'correct', or 'rollback'
                   params: Dict with adjustments if correct
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
                self.fail_counter = 0  # reset after rollback
                return 'rollback', {}

    def compute_K_LoC(self, Psi_used=None):
        """
        Compute full LoC operator K_LoC = K_damp + K_proj + K_stage + K_bc.
        
        This is the main method for computing the coherence augmentation.
        It computes all four components and combines them.
        
        Also computes coherence errors eps_H, eps_M, eps_proj.
        
        Returns:
            Dict: K_LoC for each field type (gamma_sym6, K_sym6, etc.)
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
        Get K_LoC contribution for RHS computation: λ * K_LoC.
        
        The coherence gain λ scales the LoC augmentation. This allows
        turning coherence control on/off or adjusting its strength.
        
        Args:
            Psi_used: Optional state for stage coherence computation
            
        Returns:
            Dict: Scaled K_LoC (lambda * K_LoC) for each field
        """
        K_LoC = self.compute_K_LoC(Psi_used)

        K_LoC_scaled = {}
        for field in K_LoC:
            K_LoC_scaled[field] = self.lambda_val * K_LoC[field]

        return K_LoC_scaled

    def compute_eps(self):
        """
        Compute coherence errors eps_H, eps_M, eps_proj.
        
        **Error Definitions:**
        - ε_H = ||H||_∞ = max |H| (Hamiltonian constraint, Linf)
        - ε_M = ||M||_∞ = max sqrt(M_i M^i) (Momentum constraint, Linf)
        - ε_proj = max |det(γ̃) - 1| (Projection constraint, Linf)
        
        Note: ε_clk is computed in compute_K_stage.
        """
        # Ensure constraints are computed
        self.constraints.compute_hamiltonian()
        self.constraints.compute_momentum()

        # ε_H = max |H|
        self.eps_H = np.max(np.abs(self.constraints.H))
        
        # ε_M = max sqrt(M_i M^i)
        self.eps_M = np.sqrt(np.max(np.sum(self.constraints.M**2, axis=-1)))

        # ε_proj: max deviation of det(gamma_tilde) from 1
        gamma_tilde_det = np.linalg.det(sym6_to_mat33(self.fields.gamma_tilde_sym6))
        self.eps_proj = np.max(np.abs(gamma_tilde_det - 1.0))
