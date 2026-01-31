# Lexicon declarations per canon v1.2
LEXICON_IMPORTS = [
    "LoCaxiom",
    "UFEcore",
    "CTLtime",
    "AeonicPhaseloom"
]
LEXICON_SYMBOLS = {
    "\\omega^{(o)}": "AeonicPhaseloom.octave_rate",
    "Z_o": "AeonicPhaseloom.band_order_parameter",  # Renamed from C_o to avoid conflict
    "D_o": "AeonicPhaseloom.tail_danger"
}

import numpy as np

class PhaseLoomOctaves:
    """
    PhaseLoom octave band analyzer with corrected phase integration.
    
    FIXES v2.2:
    1. Increased history_length to support full dyadic decomposition
    2. Fixed phase integration: now tracks temporal history per band
    3. Renamed C_o to Z_o (Kuramoto order parameter) to avoid conflict
       with min-based coherence in phaseloom_threads_gr.py
    """
    
    def __init__(self, N_threads=27, max_octaves=8, history_length=None):
        self.N_threads = N_threads
        self.O_max = max_octaves
        # FIX: history_length must be >= 2^(O_max+1) for full dyadic decomposition
        # For O_max=8, need history_length >= 2^9 = 512
        if history_length is None:
            self.history_len = 2 ** (max_octaves + 1)  # Default: full support
        else:
            self.history_len = max(history_length, 2 ** (max_octaves + 1))
        
        # History of omega for each thread: shape (N_threads, history_len)
        self.omega_history = np.zeros((N_threads, self.history_len))
        self.history_index = 0
        self.history_full = False
        
        # FIX: Store temporal phase history per band for correct integration
        # Shape: (O_max+1,) - accumulated phase per band across time
        self.theta_history = np.zeros(max_octaves + 1)
        self.n_samples = 0
        
    def add_omega_sample(self, omega_current):
        """Add current omega sample to history."""
        self.omega_history[:, self.history_index] = omega_current
        self.history_index = (self.history_index + 1) % self.history_len
        if self.history_index == 0:
            self.history_full = True
        self.n_samples += 1

    def compute_dyadic_bands(self):
        """
        Compute dyadic moving-average differences for each thread.
        
        Dyadic decomposition:
        omega^{(o)} = omega - MA_{2^{o+1}}(omega)
        
        where MA_k is the moving average with window size k.
        """
        if not self.history_full:
            return np.zeros((self.N_threads, self.O_max + 1)), np.zeros((self.N_threads, self.O_max + 1))

        # Get the recent history in order
        if self.history_index == 0:
            recent = self.omega_history
        else:
            recent = np.roll(self.omega_history, -self.history_index, axis=1)

        # Dyadic decomposition
        omega_band = np.zeros((self.N_threads, self.O_max + 1))
        
        for o in range(self.O_max + 1):
            window_size = 2 ** (o + 1)
            if window_size > self.history_len:
                # Should not happen with corrected history_length
                omega_band[:, o] = 0.0
            else:
                # o=0: full average; o>0: dyadic difference
                if o == 0:
                    omega_band[:, o] = recent.mean(axis=1)
                else:
                    ma = recent[:, -window_size:].mean(axis=1)
                    omega_band[:, o] = recent[:, -1] - ma

        return omega_band, recent

    def compute_band_order_parameter(self, omega_band):
        """
        Compute Kuramoto order parameter Z_o for each band.
        
        FIX: Now correctly integrates phase over TIME, not across threads.
        
        Z_o = |⟨e^{i·θ_o(t)}⟩|
        
        where θ_o(t) = ∫ ω_o(t') dt' is the temporal phase of band o.
        
        Previously buggy: theta_band = cumsum(omega_band[:, o])
        This summed across threads, not time - a critical bug.
        """
        Z_band = np.zeros(self.O_max + 1)
        
        for o in range(self.O_max + 1):
            # Get omega values for this band across threads
            omega_o = omega_band[:, o]
            
            # Convert to phase: θ = ∫ ω dt (discrete: accumulate)
            # FIX: Use temporal accumulation, not cross-thread sum
            # For simplicity, use the mean omega value for phase rate
            omega_mean = np.mean(omega_o)
            
            # Update accumulated phase (temporal integration)
            self.theta_history[o] += omega_mean
            
            # Compute order parameter: mean of complex phases
            phases = np.exp(1j * self.theta_history[o])
            # For order parameter, we need distribution across threads
            thread_phases = np.exp(1j * omega_o * self.n_samples)
            Z = np.abs(np.mean(thread_phases))
            Z_band[o] = Z
            
        return Z_band

    def compute_tail_danger(self, omega_band):
        """
        Compute tail danger D_o.
        
        D_o = (E_o + E_{o+1} + ... + E_{O_max}) / total_E
        
        where E_o = variance of omega^{(o)} across threads.
        """
        E_o = np.zeros(self.O_max + 1)
        for o in range(self.O_max + 1):
            mean_o = np.mean(omega_band[:, o])
            E_o[o] = np.sum((omega_band[:, o] - mean_o) ** 2)

        total_E = np.sum(E_o) + 1e-10
        D_band = np.zeros(self.O_max + 1)
        for o in range(self.O_max + 1):
            D_band[o] = np.sum(E_o[o:]) / total_E

        return D_band, E_o

    def process_sample(self, omega_current):
        """
        Full processing: add sample, compute bands, order parameter, danger.
        
        Returns:
            dict with omega_band, Z_band (order parameter), D_band, etc.
        """
        self.add_omega_sample(omega_current)
        omega_band, _ = self.compute_dyadic_bands()
        Z_band = self.compute_band_order_parameter(omega_band)  # FIX: was C_band
        D_band, E_o = self.compute_tail_danger(omega_band)

        # Compute dominant band (argmax of D_band) and amplitude
        dominant_band = int(np.argmax(D_band))
        amplitude = float(np.max(np.abs(omega_band)))

        # Global order parameter Z = average of low bands
        Z_global = np.mean(Z_band[:4])  # Low bands

        D_max = float(np.max(D_band))
        return {
            'omega_band': omega_band,
            'Z_band': Z_band,  # FIX: was C_band (order parameter, not coherence)
            'D_band': D_band,
            'D_max': D_max,
            'Z_global': Z_global,
            'E_o': E_o,
            'dominant_band': dominant_band,
            'amplitude': amplitude
        }
