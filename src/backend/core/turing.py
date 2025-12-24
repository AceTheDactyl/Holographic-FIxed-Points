"""
FitzHugh-Nagumo reaction-diffusion system for Turing patterns.

Turing patterns arise from diffusion-driven instability in reaction-diffusion
systems. When the inhibitor diffuses faster than the activator, spatially
periodic patterns can emerge from homogeneous initial conditions.

Fixed Point: Pattern emergence when D_v/D_u exceeds critical ratio
Turing instability creates spatial order from homogeneous state.

Physical interpretations:
- Animal coat patterns (zebra stripes, leopard spots)
- Chemical reaction patterns (Belousov-Zhabotinsky)
- Vegetation patterns in arid environments
- Morphogenesis in developmental biology
- Cardiac tissue wave patterns

References:
    Turing, A. M. (1952). Chemical basis of morphogenesis. Phil. Trans. R. Soc.
    FitzHugh, R. (1961). Impulses and physiological states in theoretical
        models of nerve membrane. Biophysical Journal.
    Murray, J. D. (2003). Mathematical Biology.
"""

import numpy as np
from .base import HolographicSolver


class TuringPatternSolver(HolographicSolver):
    """
    FitzHugh-Nagumo reaction-diffusion solver for pattern formation.

    The system models two species U (activator) and V (inhibitor):
        ∂u/∂t = D_u∇²u + f(u,v)
        ∂v/∂t = D_v∇²v + g(u,v)

    With FitzHugh-Nagumo kinetics:
        f(u,v) = u - u³ - v + k
        g(u,v) = (u - v)/τ

    Turing patterns form when D_v/D_u is sufficiently large and
    the homogeneous fixed point becomes unstable to spatial perturbations.

    Parameters:
        grid_size (int): Side length of square grid (default: 100)
        D_u (float): Activator diffusion coefficient (default: 2.8e-4)
        D_v (float): Inhibitor diffusion coefficient (default: 5e-3)
        tau (float): Time scale ratio (default: 0.1)
        k (float): Reaction parameter (default: -0.005)
        seed (int, optional): Random seed for reproducibility

    Example:
        >>> solver = TuringPatternSolver({
        ...     "grid_size": 128,
        ...     "D_u": 2.8e-4,
        ...     "D_v": 5e-3,
        ... })
        >>> solver.setup()
        >>> for _ in range(10000):
        ...     solver.step(0.1)
        >>> pattern = solver.get_pattern()
    """

    def setup(self) -> None:
        """Initialize concentration fields with random perturbation."""
        seed = self.params.get("seed")
        if seed is not None:
            np.random.seed(seed)

        size = self.params.get("grid_size", 100)

        # Diffusion coefficients
        self.D_u = self.params.get("D_u", 2.8e-4)
        self.D_v = self.params.get("D_v", 5e-3)

        # Reaction parameters
        self.tau = self.params.get("tau", 0.1)
        self.k = self.params.get("k", -0.005)

        # Grid spacing (domain is [-1, 1] × [-1, 1])
        self.dx = 2.0 / size

        # Initialize with small random perturbation from homogeneous state
        # Homogeneous fixed point: u = v = k/(1-k) for simple kinetics
        perturbation_scale = self.params.get("perturbation", 0.05)
        self.U = perturbation_scale * (np.random.rand(size, size) - 0.5)
        self.V = perturbation_scale * (np.random.rand(size, size) - 0.5)

        self._state = np.stack([self.U, self.V])
        self._size = size
        self._time = 0.0

    def _laplacian(self, Z: np.ndarray) -> np.ndarray:
        """
        5-point stencil discrete Laplacian with periodic boundary.

        ∇²Z ≈ (Z[i+1,j] + Z[i-1,j] + Z[i,j+1] + Z[i,j-1] - 4Z[i,j]) / dx²

        Args:
            Z: 2D field array.

        Returns:
            Discrete Laplacian of Z.
        """
        return (
            np.roll(Z, 1, axis=0) + np.roll(Z, -1, axis=0) +
            np.roll(Z, 1, axis=1) + np.roll(Z, -1, axis=1) -
            4 * Z
        ) / self.dx**2

    def step(self, dt: float) -> None:
        """
        Euler integration of FitzHugh-Nagumo reaction-diffusion dynamics.

        Args:
            dt: Time step size.
        """
        lap_U = self._laplacian(self.U)
        lap_V = self._laplacian(self.V)

        # Reaction terms
        # f(u,v) = u - u³ - v + k
        # g(u,v) = (u - v) / τ
        f = self.U - self.U**3 - self.V + self.k
        g = (self.U - self.V) / self.tau

        # Reaction-diffusion update
        dU = self.D_u * lap_U + f
        dV = self.D_v * lap_V + g

        self.U += dU * dt
        self.V += dV * dt

        self._state = np.stack([self.U, self.V])
        self._time += dt

    def step_semi_implicit(self, dt: float) -> None:
        """
        Semi-implicit integration for improved stability.

        Uses implicit diffusion, explicit reaction.
        Requires FFT for efficient solution.

        Args:
            dt: Time step size.
        """
        # This would use spectral methods for the diffusion part
        # For now, fall back to explicit
        self.step(dt)

    def compute_order_parameter(self) -> float:
        """
        Pattern contrast: std(U) measures spatial heterogeneity.

        High values indicate pattern formation (spatial structure).
        Near-zero values indicate homogeneous state.

        Returns:
            Standard deviation of U field.
        """
        return float(np.std(self.U))

    def get_critical_threshold(self) -> float:
        """
        Critical diffusion ratio for Turing instability.

        Pattern formation requires D_v/D_u > critical_ratio.
        The exact threshold depends on reaction kinetics.

        For FitzHugh-Nagumo, instability requires:
        D_v/D_u > ((1 + √(1 - 4αβ))/(2α))²

        where α and β are linearized kinetics coefficients.

        Returns:
            Approximate critical ratio sqrt(D_v/D_u).
        """
        # Simplified criterion - actual analysis requires
        # Jacobian eigenvalue computation at fixed point
        return float(np.sqrt(self.D_v / self.D_u))

    def get_pattern(self) -> np.ndarray:
        """Return current activator (U) field for visualization."""
        return self.U.copy()

    def get_inhibitor(self) -> np.ndarray:
        """Return current inhibitor (V) field."""
        return self.V.copy()

    def get_pattern_wavelength(self) -> float:
        """
        Estimate dominant wavelength via FFT analysis.

        Returns:
            Estimated wavelength in grid units.
        """
        # 2D FFT
        fft_U = np.fft.fft2(self.U)
        power = np.abs(fft_U)**2

        # Zero out DC component
        power[0, 0] = 0

        # Find dominant frequency
        size = self._size
        freqs = np.fft.fftfreq(size, d=self.dx)
        freq_x, freq_y = np.meshgrid(freqs, freqs)
        freq_magnitude = np.sqrt(freq_x**2 + freq_y**2)

        # Radially average power spectrum
        max_idx = np.unravel_index(np.argmax(power), power.shape)
        dominant_freq = freq_magnitude[max_idx]

        if dominant_freq > 0:
            return float(1.0 / dominant_freq)
        return float(self._size * self.dx)

    def pattern_energy(self) -> float:
        """
        Total pattern energy (variance) of the system.

        Returns:
            Sum of variances of U and V fields.
        """
        return float(np.var(self.U) + np.var(self.V))

    def is_patterned(self, threshold: float = 0.1) -> bool:
        """
        Determine if significant pattern has formed.

        Args:
            threshold: Minimum contrast for pattern detection.

        Returns:
            True if pattern contrast exceeds threshold.
        """
        return self.compute_order_parameter() > threshold

    def set_diffusion_coefficients(self, D_u: float, D_v: float) -> None:
        """
        Update diffusion coefficients.

        Args:
            D_u: New activator diffusion coefficient.
            D_v: New inhibitor diffusion coefficient.
        """
        self.D_u = D_u
        self.D_v = D_v
        self.params["D_u"] = D_u
        self.params["D_v"] = D_v
