# Lexicon declarations per canon v1.2
LEXICON_IMPORTS = [
    "LoC_axiom",
    "UFE_core",
    "GR_dyn",
    "CTL_time"
]
LEXICON_SYMBOLS = {
    "k": "Spectral wavenumber"
}

import numpy as np


def _phi1(z):
    """ETD1 coefficient: (e^z - 1) / z, avoiding division by zero."""
    z = np.asarray(z)
    mask = np.abs(z) > 1e-12
    result = np.zeros_like(z, dtype=np.complex128)
    result[mask] = (np.exp(z[mask]) - 1) / z[mask]
    result[~mask] = 1.0  # limit as z->0
    return result


def _phi2(z):
    """ETD2 coefficient: (e^z - 1 - z) / z^2, avoiding division by zero."""
    z = np.asarray(z)
    mask = np.abs(z) > 1e-12
    result = np.zeros_like(z, dtype=np.complex128)
    result[mask] = (np.exp(z[mask]) - 1 - z[mask]) / (z[mask] ** 2)
    result[~mask] = 0.5  # limit as z->0
    return result


def _phi3(z):
    """ETD3 coefficient: (e^z - 1 - z - z^2/2) / z^3, avoiding division by zero."""
    z = np.asarray(z)
    mask = np.abs(z) > 1e-12
    result = np.zeros_like(z, dtype=np.complex128)
    result[mask] = (np.exp(z[mask]) - 1 - z[mask] - z[mask]**2 / 2) / (z[mask] ** 3)
    result[~mask] = 1/6  # limit as z->0
    return result


class SpectralCache:
    def __init__(self, Nx, Ny, Nz, dx, dy, dz):
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        self.dx, self.dy, self.dz = dx, dy, dz

        # Precompute k-vectors
        self.kx = 2 * np.pi * np.fft.fftfreq(Nx, dx)
        self.ky = 2 * np.pi * np.fft.fftfreq(Ny, dy)
        self.kz = 2 * np.pi * np.fft.rfftfreq(Nz, dz)

        # |k|² for projection
        KX, KY, KZ = np.meshgrid(self.kx, self.ky, self.kz, indexing='ij')
        self.k2 = KX**2 + KY**2 + KZ**2
        self.k_mag = np.sqrt(self.k2)
        self.k_complex = KX + 1j * KY  # generalized for damping operators

        # Dealising mask: remove high k
        k_max = max(self.kx.max(), self.ky.max(), self.kz.max())
        self.dealias_mask = (np.abs(KX) < 2/3 * k_max) & (np.abs(KY) < 2/3 * k_max) & (np.abs(KZ) < 2/3 * k_max)

        # Bin maps for octaves (3x3x3 digitize)
        self.kx_bin = self._digitize_bins(self.kx, 3)
        self.ky_bin = self._digitize_bins(self.ky, 3)
        self.kz_bin = self._digitize_bins(self.kz, 3)

        # ETD factors cache (regime -> factors)
        self.etd_factors = {}  # (viscosity, damping_coef, dt) -> factors dict

    def _digitize_bins(self, k_arr, n_bins):
        """Digitize k into n_bins octaves."""
        k_min, k_max = k_arr.min(), k_arr.max()
        bins = np.logspace(np.log10(max(abs(k_min), 1e-10)), np.log10(k_max), n_bins + 1)
        bin_indices = np.digitize(np.abs(k_arr), bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        return bin_indices

    def get_k_bins(self):
        """Return precomputed bin maps."""
        return self.kx_bin, self.ky_bin, self.kz_bin

    def get_projection_factors(self):
        """Return |k|² for Poisson projection."""
        return self.k2

    def get_dealias_mask(self):
        """Return dealiasing mask."""
        return self.dealias_mask

    def compute_etd_factors(self, viscosity, damping_coef, dt):
        """
        Compute ETD (Exponential Time-Differencing) factors for given parameters.
        
        Args:
            viscosity: Viscosity coefficient ν for hyperbolic damping
            damping_coef: Gauge driver relaxation coefficient γ
            dt: Time step for exponential factors
            
        Returns:
            dict with keys:
                - 'visc_etd1': φ₁(-ν k² dt) for viscosity terms
                - 'visc_etd2': φ₂(-ν k² dt) for second order viscosity
                - 'damp_etd1': φ₁(-γ dt) for gauge damping
                - 'damp_etd2': φ₂(-γ dt) for second order damping
                - 'combined_etd1': φ₁((-ν k² - γ) dt) for combined operator
                - 'combined_etd2': φ₂((-ν k² - γ) dt)
        """
        cache_key = (viscosity, damping_coef, dt)
        if cache_key in self.etd_factors:
            return self.etd_factors[cache_key]
        
        # Compute eigenvalues for viscosity: λ_visc = -ν k²
        # Complex wavenumber handles both real and imaginary components
        k_sq = self.k2
        
        # Viscosity eigenvalues (purely negative real for stable damping)
        lambda_visc = -viscosity * k_sq
        
        # Gauge damping eigenvalues (purely negative real)
        lambda_damp = -damping_coef
        
        # Combined eigenvalues: λ_comb = λ_visc + lambda_damp
        lambda_comb = -viscosity * k_sq - damping_coef
        
        # Compute ETD coefficients
        factors = {
            'visc_etd1': _phi1(lambda_visc * dt),
            'visc_etd2': _phi2(lambda_visc * dt),
            'damp_etd1': _phi1(lambda_damp * dt),
            'damp_etd2': _phi2(lambda_damp * dt),
            'combined_etd1': _phi1(lambda_comb * dt),
            'combined_etd2': _phi2(lambda_comb * dt),
            'visc_exp': np.exp(lambda_visc * dt),  # exp(-ν k² dt)
            'damp_exp': np.exp(lambda_damp * dt),  # exp(-γ dt)
            'combined_exp': np.exp(lambda_comb * dt),  # exp(-(ν k² + γ) dt)
        }
        
        # Also compute for complex wavenumber if damping has imaginary component
        if hasattr(self, 'k_complex'):
            lambda_complex = -viscosity * self.k_complex - damping_coef
            factors['complex_etd1'] = _phi1(lambda_complex * dt)
            factors['complex_etd2'] = _phi2(lambda_complex * dt)
            factors['complex_exp'] = np.exp(lambda_complex * dt)
        
        self.etd_factors[cache_key] = factors
        return factors

    def get_etd_factor(self, k_mag, viscosity, damping_coef, dt, factor_type='visc_etd1'):
        """
        Lookup ETD factor for a specific wavenumber magnitude.
        
        Args:
            k_mag: Magnitude of wavenumber |k|
            viscosity: Viscosity coefficient ν
            damping_coef: Gauge driver relaxation coefficient γ
            dt: Time step
            factor_type: Which factor to return ('visc_etd1', 'visc_etd2', 
                       'damp_etd1', 'combined_etd1', etc.)
            
        Returns:
            ETD coefficient value at given wavenumber
        """
        factors = self.compute_etd_factors(viscosity, damping_coef, dt)
        
        # Find nearest grid point to k_mag
        k_flat = self.k_mag.flatten()
        idx = np.argmin(np.abs(k_flat - k_mag))
        
        return factors[factor_type].flatten()[idx]