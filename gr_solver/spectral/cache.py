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

        # Dealising mask: remove high k
        k_max = max(self.kx.max(), self.ky.max(), self.kz.max())
        self.dealias_mask = (np.abs(KX) < 2/3 * k_max) & (np.abs(KY) < 2/3 * k_max) & (np.abs(KZ) < 2/3 * k_max)

        # Bin maps for octaves (3x3x3 digitize)
        self.kx_bin = self._digitize_bins(self.kx, 3)
        self.ky_bin = self._digitize_bins(self.ky, 3)
        self.kz_bin = self._digitize_bins(self.kz, 3)

        # ETD factors cache (placeholder for viscosity, etc.)
        self.etd_factors = {}  # regime -> factors

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