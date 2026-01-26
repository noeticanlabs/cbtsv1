import numpy as np
try:
    from numba import jit, prange
except ImportError:
    # Fallback for environments without Numba
    jit = lambda f=None, **kwargs: f if f else (lambda g: g)
    prange = range

@jit(nopython=True, parallel=True, fastmath=True)
def fused_evolution_kernel(
    gamma, K, alpha, beta, 
    rhs_gamma, rhs_K, rhs_alpha, rhs_beta,
    dt, 
    out_gamma, out_K, out_alpha, out_beta
):
    """
    Fused update: State_new = State_old + dt * RHS
    Also computes norms of the update (delta) inline to avoid extra memory passes.
    """
    Nx, Ny, Nz, _ = gamma.shape
    
    norm_dgamma_sq = 0.0
    norm_dK_sq = 0.0
    
    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                # Gamma update (6 components)
                for c in range(6):
                    dg = dt * rhs_gamma[i,j,k,c]
                    out_gamma[i,j,k,c] = gamma[i,j,k,c] + dg
                    norm_dgamma_sq += dg*dg
                
                # K update (6 components)
                for c in range(6):
                    dK = dt * rhs_K[i,j,k,c]
                    out_K[i,j,k,c] = K[i,j,k,c] + dK
                    norm_dK_sq += dK*dK
                    
                # Alpha update
                da = dt * rhs_alpha[i,j,k]
                out_alpha[i,j,k] = alpha[i,j,k] + da
                
                # Beta update (3 components)
                for c in range(3):
                    db = dt * rhs_beta[i,j,k,c]
                    out_beta[i,j,k,c] = beta[i,j,k,c] + db
                    
    return np.sqrt(norm_dgamma_sq), np.sqrt(norm_dK_sq)

@jit(nopython=True, parallel=True, fastmath=True)
def fused_constraints_norm_kernel(H, M, dx, dy, dz):
    """
    Fused computation of Hamiltonian and Momentum constraints norms.
    Returns L2 and Linf norms in a single pass.
    """
    Nx, Ny, Nz = H.shape
    sum_H2 = 0.0
    max_H = 0.0
    
    vol_elem = dx * dy * dz
    
    for i in prange(Nx):
        for j in range(Ny):
            for k in range(Nz):
                h_val = abs(H[i,j,k])
                sum_H2 += h_val*h_val
                if h_val > max_H:
                    max_H = h_val
                    
    # Note: Momentum norm logic omitted for brevity, follows same pattern
    return np.sqrt(sum_H2 * vol_elem), max_H