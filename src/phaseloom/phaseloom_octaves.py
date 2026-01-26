# Lexicon declarations per canon v1.2
LEXICON_IMPORTS = [
    "LoC_axiom",
    "UFE_core",
    "CTL_time",
    "Aeonic_Phaseloom"
]
LEXICON_SYMBOLS = {
    "\\omega^{(o)}": "Aeonic_Phaseloom.octave_rate",
    "C_o": "Aeonic_Phaseloom.band_coherence",
    "D_o": "Aeonic_Phaseloom.tail_danger"
}

import numpy as np

class PhaseLoomOctaves:
    def __init__(self, N_threads=27, max_octaves=8, history_length=2):
        self.N_threads = N_threads
        self.O_max = max_octaves
        self.history_len = history_length
        # History of omega for each thread: shape (N_threads, history_len)
        self.omega_history = np.zeros((N_threads, self.history_len))
        self.history_index = 0
        self.history_full = False

    def add_omega_sample(self, omega_current):
        """Add current omega sample to history."""
        self.omega_history[:, self.history_index] = omega_current
        self.history_index = (self.history_index + 1) % self.history_len
        if self.history_index == 0:
            self.history_full = True

    def compute_dyadic_bands(self):
        """Compute dyadic moving-average differences for each thread."""
        if not self.history_full:
            return np.zeros((self.N_threads, self.O_max + 1)), np.zeros((self.N_threads, self.O_max + 1))

        # Get the recent history in order
        if self.history_index == 0:
            recent = self.omega_history
        else:
            recent = np.roll(self.omega_history, -self.history_index, axis=1)

        # Dyadic decomposition: omega^{(o)} = omega - MA_2^{o+1}(omega)
        # Simplified: use cumulative averages
        omega_band = np.zeros((self.N_threads, self.O_max + 1))
        omega_band[:, 0] = recent.mean(axis=1)  # o=0: full average

        for o in range(1, self.O_max + 1):
            window_size = 2 ** (o + 1)
            if window_size > self.history_len:
                omega_band[:, o] = 0.0
            else:
                # Optimization: The last point of a valid convolution with a uniform kernel 
                # is simply the mean of the last 'window_size' elements.
                ma = recent[:, -window_size:].mean(axis=1)
                omega_band[:, o] = recent[:, -1] - ma

        return omega_band, recent

    def compute_band_coherence(self, omega_band):
        """Compute coherence C_o for each band."""
        C_band = np.zeros(self.O_max + 1)
        for o in range(self.O_max + 1):
            # Phase from omega_band (assume small angles for simplicity)
            theta_band = np.cumsum(omega_band[:, o])  # Integrate to get phase
            # Coherence: |mean exp(i theta)|
            phases = np.exp(1j * theta_band)
            Z = np.mean(phases)
            C_band[o] = np.abs(Z)
        return C_band

    def compute_tail_danger(self, omega_band):
        """Compute tail danger D_o."""
        # E_o = sum_i (omega_i^{(o)} - mean)^2
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
        """Full processing: add sample, compute bands, coherence, danger."""
        self.add_omega_sample(omega_current)
        omega_band, _ = self.compute_dyadic_bands()
        C_band = self.compute_band_coherence(omega_band)
        D_band, E_o = self.compute_tail_danger(omega_band)

        # Debug logs
        print(f"omega_band sum per band: {np.sum(np.abs(omega_band), axis=0)}")
        print(f"D_band: {D_band}")

        # Compute dominant band (argmax of D_band) and amplitude
        dominant_band = int(np.argmax(D_band))
        amplitude = float(np.max(np.abs(omega_band)))

        # Global coherence C = C_0 or average
        C_global = np.mean(C_band[:4])  # Low bands

        D_max = float(np.max(D_band))
        return {
            'omega_band': omega_band,
            'C_band': C_band,
            'D_band': D_band,
            'D_max': D_max,
            'C_global': C_global,
            'E_o': E_o,
            'dominant_band': dominant_band,
            'amplitude': amplitude
        }