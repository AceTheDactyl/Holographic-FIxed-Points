"""
Kuramoto model of coupled phase oscillators.

The Kuramoto model describes synchronization in populations of coupled
oscillators. It exhibits a phase transition at critical coupling K_c,
above which oscillators synchronize.

Fixed Point: r → 1 (full synchronization) when K > K_c
Critical threshold: K_c = sqrt(8/π) * σ_ω for Gaussian frequency distribution

Physical interpretations:
- Firefly synchronization
- Neural oscillations
- Power grid stability
- Circadian rhythms
- Josephson junction arrays

References:
    Kuramoto, Y. (1984). Chemical Oscillations, Waves, and Turbulence.
    Strogatz, S. H. (2000). From Kuramoto to Crawford. Physica D.
"""

import numpy as np
from .base import HolographicSolver


class KuramotoSolver(HolographicSolver):
    """
    Kuramoto model solver for synchronization dynamics.

    The model describes N coupled phase oscillators with dynamics:
        dθ_i/dt = ω_i + (K/N) * Σ_j sin(θ_j - θ_i)

    where:
        θ_i: phase of oscillator i
        ω_i: natural frequency of oscillator i
        K: coupling strength
        N: number of oscillators

    The order parameter r = |⟨e^{iθ}⟩| measures phase coherence:
        r = 0: incoherent (uniformly distributed phases)
        r = 1: fully synchronized (all phases equal)

    Parameters:
        n_oscillators (int): Number of oscillators (default: 100)
        coupling (float): Coupling strength K (default: 1.0)
        frequency_std (float): Std dev of natural frequencies (default: 1.0)
        frequency_mean (float): Mean natural frequency (default: 0.0)
        seed (int, optional): Random seed for reproducibility

    Example:
        >>> solver = KuramotoSolver({
        ...     "n_oscillators": 100,
        ...     "coupling": 2.0,
        ...     "frequency_std": 1.0
        ... })
        >>> solver.setup()
        >>> for _ in range(1000):
        ...     solver.step(0.01)
        >>> r = solver.compute_order_parameter()
        >>> print(f"Synchronization: {r:.3f}")
    """

    def setup(self) -> None:
        """Initialize oscillator phases and natural frequencies."""
        # Random seed for reproducibility
        seed = self.params.get("seed")
        if seed is not None:
            np.random.seed(seed)

        n_oscillators = self.params.get("n_oscillators", 100)
        freq_std = self.params.get("frequency_std", 1.0)
        freq_mean = self.params.get("frequency_mean", 0.0)

        # Natural frequencies (Gaussian distribution)
        self.natural_freqs = np.random.normal(freq_mean, freq_std, n_oscillators)

        # Initial phases (uniform random on circle)
        self._state = np.random.uniform(0, 2 * np.pi, n_oscillators)

        # Coupling strength
        self.K = self.params.get("coupling", 1.0)
        self._freq_std = freq_std
        self._n = n_oscillators
        self._time = 0.0

    def step(self, dt: float) -> None:
        """
        Euler integration of Kuramoto dynamics.

        Uses mean-field formulation for O(N) complexity instead of O(N²).

        Args:
            dt: Time step size.
        """
        N = self._n
        phases = self._state

        # Mean-field decomposition: compute r and ψ
        # r * e^{iψ} = (1/N) * Σ e^{iθ_j}
        complex_order = np.mean(np.exp(1j * phases))
        r = np.abs(complex_order)
        psi = np.angle(complex_order)

        # Mean-field dynamics: dθ_i/dt = ω_i + K * r * sin(ψ - θ_i)
        dphases = self.natural_freqs + self.K * r * np.sin(psi - phases)

        # Euler update
        self._state = phases + dt * dphases
        self._state = np.mod(self._state, 2 * np.pi)  # Keep in [0, 2π]
        self._time += dt

    def step_full(self, dt: float) -> None:
        """
        Full O(N²) Euler integration (for small N or verification).

        Args:
            dt: Time step size.
        """
        N = self._n
        phases = self._state

        # Compute pairwise interactions: (K/N) * Σ sin(θ_j - θ_i)
        phase_diff = phases[np.newaxis, :] - phases[:, np.newaxis]
        interaction = np.sum(np.sin(phase_diff), axis=1) / N

        # Update: dθ/dt = ω + K * interaction
        self._state = phases + dt * (self.natural_freqs + self.K * interaction)
        self._state = np.mod(self._state, 2 * np.pi)
        self._time += dt

    def compute_order_parameter(self) -> float:
        """
        Kuramoto order parameter r ∈ [0, 1].

        r = |⟨e^{iθ}⟩| measures phase coherence.
        r = 0: completely incoherent
        r = 1: fully synchronized

        Returns:
            Order parameter r.
        """
        return float(np.abs(np.mean(np.exp(1j * self._state))))

    def compute_mean_phase(self) -> float:
        """
        Mean phase ψ of the oscillator population.

        Returns:
            Mean phase ψ in [−π, π].
        """
        return float(np.angle(np.mean(np.exp(1j * self._state))))

    def get_critical_threshold(self) -> float:
        """
        Critical coupling K_c for Gaussian frequency distribution.

        K_c = sqrt(8/π) * σ ≈ 1.596 * σ

        Below K_c: incoherent (r → 0)
        Above K_c: partial/full synchronization (r > 0)

        Returns:
            Critical coupling strength K_c.
        """
        return float(np.sqrt(8 / np.pi) * self._freq_std)

    def set_coupling(self, K: float) -> None:
        """
        Dynamically adjust coupling strength.

        Args:
            K: New coupling strength.
        """
        self.K = K
        self.params["coupling"] = K

    def get_phases(self) -> np.ndarray:
        """Return current oscillator phases."""
        return self._state.copy()

    def get_frequencies(self) -> np.ndarray:
        """Return natural frequencies."""
        return self.natural_freqs.copy()

    def get_instantaneous_frequencies(self) -> np.ndarray:
        """
        Compute instantaneous frequencies (rate of phase change).

        Returns:
            Array of instantaneous frequencies.
        """
        complex_order = np.mean(np.exp(1j * self._state))
        r = np.abs(complex_order)
        psi = np.angle(complex_order)
        return self.natural_freqs + self.K * r * np.sin(psi - self._state)

    def locked_oscillators(self, tolerance: float = 0.1) -> np.ndarray:
        """
        Find oscillators that are phase-locked (near mean phase).

        Args:
            tolerance: Phase tolerance for being considered locked.

        Returns:
            Boolean array indicating locked oscillators.
        """
        psi = self.compute_mean_phase()
        phase_diffs = np.abs(np.angle(np.exp(1j * (self._state - psi))))
        return phase_diffs < tolerance
