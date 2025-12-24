"""
Bekenstein entropy bound and holographic entropy calculations.

The Bekenstein bound is the maximum entropy that can be contained in a
region of space with given size and energy. For black holes, this equals
the Bekenstein-Hawking entropy S = A/(4l_P²).

Fixed Point: S = A/(4l_P²) (Bekenstein-Hawking entropy)
The maximum entropy in a region equals its boundary area in Planck units.

Physical interpretation: Information in a volume is bounded by its surface
area—the holographic principle. This is the foundation of holographic
duality (AdS/CFT correspondence).

References:
    Bekenstein, J. D. (1973). Black holes and entropy. Phys. Rev. D.
    Hawking, S. W. (1975). Particle creation by black holes. CMP.
    't Hooft, G. (1993). Dimensional reduction in quantum gravity.
    Susskind, L. (1995). The world as a hologram.
"""

import numpy as np
from .base import HolographicSolver


# Physical constants (SI units)
class PhysicalConstants:
    """Standard physical constants."""
    G = 6.67430e-11      # Gravitational constant (m³/kg/s²)
    c = 299792458.0      # Speed of light (m/s)
    hbar = 1.054571817e-34  # Reduced Planck constant (J·s)
    k_B = 1.380649e-23   # Boltzmann constant (J/K)

    # Derived Planck units
    l_planck = np.sqrt(hbar * G / c**3)  # ~1.616e-35 m
    t_planck = l_planck / c              # ~5.391e-44 s
    m_planck = np.sqrt(hbar * c / G)     # ~2.176e-8 kg
    E_planck = m_planck * c**2           # ~1.956e9 J

    # Astronomical units
    M_sun = 1.989e30     # Solar mass (kg)


class BekensteinSolver(HolographicSolver):
    """
    Bekenstein entropy bound and black hole thermodynamics solver.

    Models the holographic entropy bound for gravitational systems:
    - Bekenstein bound: S_max ≤ 2πRE/(ℏc)
    - Black hole entropy: S_BH = A/(4l_P²) = πr_s²/l_P²

    The solver can model static black holes or include Hawking radiation
    for dynamical evolution.

    Parameters:
        mass_kg (float): System mass in kg (default: 1.0)
        radius_m (float): Bounding radius in meters (default: 1.0)
        include_hawking (bool): Model Hawking evaporation (default: False)
        hawking_rate (float): Evaporation rate coefficient

    Example:
        >>> # Solar mass black hole
        >>> solver = BekensteinSolver({"mass_kg": 1.989e30})
        >>> solver.setup()
        >>> entropy_bits = solver.holographic_bits()
        >>> print(f"Holographic entropy: {entropy_bits:.2e} bits")
    """

    def setup(self) -> None:
        """Initialize black hole / bounded region properties."""
        self._constants = PhysicalConstants()

        # System parameters
        self.mass = self.params.get("mass_kg", 1.0)
        self.radius = self.params.get("radius_m", 1.0)

        # Hawking radiation settings
        self._include_hawking = self.params.get("include_hawking", False)
        self._hawking_rate = self.params.get("hawking_rate", 1e-20)

        # State: [entropy, horizon_area, hawking_temperature]
        self._state = np.array([0.0, 0.0, 0.0])
        self._compute_state()
        self._time = 0.0

    def _compute_state(self) -> None:
        """Compute horizon properties from current mass."""
        C = self._constants

        # Schwarzschild radius: r_s = 2GM/c²
        r_s = 2 * C.G * self.mass / C.c**2

        # Event horizon area: A = 4πr_s²
        area = 4 * np.pi * r_s**2

        # Bekenstein-Hawking entropy: S = A/(4l_P²)
        # In natural units where k_B = 1
        entropy = area / (4 * C.l_planck**2)

        # Hawking temperature: T_H = ℏc³/(8πGMk_B)
        if self.mass > 0:
            hawking_temp = C.hbar * C.c**3 / (8 * np.pi * C.G * self.mass * C.k_B)
        else:
            hawking_temp = np.inf

        self._state = np.array([entropy, area, hawking_temp])
        self._r_schwarzschild = r_s

    def step(self, dt: float) -> None:
        """
        Advance simulation (optionally including Hawking evaporation).

        For static black holes, entropy is constant. With Hawking radiation
        enabled, mass decreases according to:
            dM/dt ∝ -1/M²

        Args:
            dt: Time step in seconds.
        """
        if self._include_hawking and self.mass > 0:
            # Stefan-Boltzmann law for black hole emission
            # P = (ℏc⁶)/(15360πG²M²) ≈ σA_H T_H⁴
            # Simplified: dM/dt ∝ -1/M²
            C = self._constants

            # More accurate Hawking rate
            power = C.hbar * C.c**6 / (15360 * np.pi * C.G**2 * self.mass**2)
            dm = -power / C.c**2 * dt

            self.mass = max(self.mass + dm * self._hawking_rate, 1e-50)
            self._compute_state()

        self._time += dt

    def compute_order_parameter(self) -> float:
        """
        Return entropy in Planck units (dimensionless).

        This is S/k_B, representing the number of microstates.

        Returns:
            Bekenstein-Hawking entropy in natural units.
        """
        return float(self._state[0])

    def get_critical_threshold(self) -> float:
        """
        Bekenstein bound: S_max = 2πRE/(ℏc).

        For a black hole filling its Schwarzschild radius,
        this equals the Bekenstein-Hawking entropy.

        Returns:
            Bekenstein bound in natural units.
        """
        C = self._constants
        E = self.mass * C.c**2  # Total energy
        R = max(self.radius, self._r_schwarzschild)  # Use larger of given radius or r_s
        return float(2 * np.pi * R * E / (C.hbar * C.c))

    def holographic_bits(self) -> float:
        """
        Number of bits on the holographic boundary.

        Converts entropy from natural units (nats) to bits.

        Returns:
            Information content in bits.
        """
        return float(self._state[1] / (4 * self._constants.l_planck**2 * np.log(2)))

    def get_horizon_area(self) -> float:
        """Return event horizon area in m²."""
        return float(self._state[1])

    def get_schwarzschild_radius(self) -> float:
        """Return Schwarzschild radius in meters."""
        return float(self._r_schwarzschild)

    def get_hawking_temperature(self) -> float:
        """Return Hawking temperature in Kelvin."""
        return float(self._state[2])

    def get_evaporation_time(self) -> float:
        """
        Estimated time for complete Hawking evaporation.

        t_evap ≈ 5120πG²M³/(ℏc⁴)

        Returns:
            Evaporation time in seconds.
        """
        C = self._constants
        return float(5120 * np.pi * C.G**2 * self.mass**3 / (C.hbar * C.c**4))

    def saturation_ratio(self) -> float:
        """
        How close the system is to saturating the Bekenstein bound.

        Returns:
            Ratio S/S_max ∈ [0, 1]. Equal to 1 for black holes.
        """
        current = self.compute_order_parameter()
        maximum = self.get_critical_threshold()
        return float(current / maximum) if maximum > 0 else 0.0

    def entropy_density(self) -> float:
        """
        Entropy per unit horizon area (should be 1/4 in Planck units).

        Returns:
            S/A in units of 1/l_P².
        """
        area = self._state[1]
        entropy = self._state[0]
        if area > 0:
            return float(entropy * self._constants.l_planck**2 / area)
        return 0.0

    def set_mass(self, mass_kg: float) -> None:
        """
        Update the system mass.

        Args:
            mass_kg: New mass in kilograms.
        """
        self.mass = mass_kg
        self.params["mass_kg"] = mass_kg
        self._compute_state()
