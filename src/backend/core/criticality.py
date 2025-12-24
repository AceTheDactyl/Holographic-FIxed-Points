"""
Nuclear criticality and point reactor kinetics model.

The point reactor kinetics model describes neutron population dynamics
in nuclear systems. Criticality occurs when k_eff = 1, representing
a self-sustaining chain reaction.

Fixed Point: k_eff = 1 (critical state - sustained chain reaction)
k < 1: subcritical (exponential decay)
k > 1: supercritical (exponential growth)

Physical interpretations:
- Nuclear reactor control
- Nuclear weapons physics
- Criticality safety analysis
- Research reactor operation

References:
    Keepin, G. R. (1965). Physics of Nuclear Kinetics.
    Duderstadt, J. J. & Hamilton, L. J. (1976). Nuclear Reactor Analysis.
    Henry, A. F. (1975). Nuclear-Reactor Analysis.
"""

import numpy as np
from .base import HolographicSolver


class NuclearCriticalitySolver(HolographicSolver):
    """
    Point reactor kinetics model for nuclear criticality analysis.

    Models neutron population N and delayed neutron precursors C:
        dN/dt = ((ρ - β)/Λ) * N + λ * C
        dC/dt = (β/Λ) * N - λ * C

    where:
        ρ = (k - 1)/k: reactivity
        β: delayed neutron fraction (~0.0065 for U-235)
        Λ = l/k: mean generation time
        λ: precursor decay constant
        l: prompt neutron lifetime

    The critical state corresponds to ρ = 0, or equivalently k_eff = 1.

    Parameters:
        initial_neutrons (float): Initial neutron population (default: 1e6)
        k_infinity (float): Infinite medium k (default: 1.03)
        leakage_factor (float): Geometric leakage (default: 0.02)
        delayed_fraction (float): Delayed neutron fraction β (default: 0.0065)
        decay_constant (float): Precursor decay λ (default: 0.08)
        prompt_lifetime (float): Prompt neutron lifetime (default: 1e-4)
        control_rod_worth (float): Control rod reactivity (default: 0.0)

    Example:
        >>> solver = NuclearCriticalitySolver({
        ...     "k_infinity": 1.03,
        ...     "leakage_factor": 0.03,
        ... })
        >>> solver.setup()
        >>> print(f"k_eff = {solver.compute_order_parameter():.4f}")
    """

    def setup(self) -> None:
        """Initialize neutron population and precursor concentration."""
        # Initial neutron population
        self.N = self.params.get("initial_neutrons", 1e6)

        # Multiplication factor components
        self.k_inf = self.params.get("k_infinity", 1.03)  # Infinite medium
        self.leakage = self.params.get("leakage_factor", 0.02)  # Geometric leakage

        # Delayed neutron parameters (one-group approximation)
        self.beta = self.params.get("delayed_fraction", 0.0065)
        self.lambda_d = self.params.get("decay_constant", 0.08)  # s^-1

        # Prompt neutron lifetime
        self.l_prompt = self.params.get("prompt_lifetime", 1e-4)  # seconds

        # Control rod worth (negative = inserted, absorbing)
        self.rho_control = self.params.get("control_rod_worth", 0.0)

        # Initialize delayed neutron precursor concentration
        # At equilibrium: C = (β/λ) * (N/Λ) = β*k_eff*N / (λ*l)
        k_eff = self._compute_k_eff()
        Lambda = self.l_prompt / k_eff if k_eff > 0 else self.l_prompt
        self.C = self.beta * self.N / (self.lambda_d * Lambda)

        self._state = np.array([self.N, self.C])
        self._time = 0.0

    def _compute_k_eff(self) -> float:
        """
        Compute effective multiplication factor.

        k_eff = k_∞ * (1 - leakage) + control_rods

        Returns:
            Effective multiplication factor.
        """
        return self.k_inf * (1 - self.leakage) + self.rho_control

    def _compute_reactivity(self) -> float:
        """
        Compute reactivity ρ = (k - 1)/k.

        ρ > 0: supercritical
        ρ = 0: critical
        ρ < 0: subcritical

        Returns:
            Reactivity in absolute units (not pcm or %).
        """
        k_eff = self._compute_k_eff()
        if k_eff > 0:
            return (k_eff - 1) / k_eff
        return -1.0

    def step(self, dt: float) -> None:
        """
        Point kinetics equations with one delayed group.

        Uses simplified Euler integration. For stiff systems (large β/Λ),
        consider implicit methods.

        Args:
            dt: Time step in seconds.
        """
        k_eff = self._compute_k_eff()
        rho = self._compute_reactivity()

        # Mean generation time Λ = l/k
        Lambda = self.l_prompt / k_eff if k_eff > 0 else self.l_prompt

        # Point kinetics equations
        # dN/dt = ((ρ - β)/Λ) * N + λ * C
        # dC/dt = (β/Λ) * N - λ * C
        dN = ((rho - self.beta) / Lambda) * self.N + self.lambda_d * self.C
        dC = (self.beta / Lambda) * self.N - self.lambda_d * self.C

        # Update with non-negativity constraint
        self.N = max(self.N + dN * dt, 0)
        self.C = max(self.C + dC * dt, 0)

        self._state = np.array([self.N, self.C])
        self._time += dt

    def step_six_group(self, dt: float) -> None:
        """
        Six-group delayed neutron kinetics (more accurate).

        Not implemented - uses one-group approximation.

        Args:
            dt: Time step in seconds.
        """
        # Would require 6 precursor groups with different β_i and λ_i
        # For U-235 thermal fission
        self.step(dt)

    def compute_order_parameter(self) -> float:
        """
        Effective multiplication factor k_eff.

        k_eff = 1: critical (sustained reaction)
        k_eff < 1: subcritical (decaying)
        k_eff > 1: supercritical (growing)

        Returns:
            k_eff value.
        """
        return float(self._compute_k_eff())

    def get_critical_threshold(self) -> float:
        """
        Critical state: k_eff = 1.0.

        Returns:
            Critical k value (always 1.0).
        """
        return 1.0

    def get_neutron_population(self) -> float:
        """Return current neutron population."""
        return float(self.N)

    def get_precursor_concentration(self) -> float:
        """Return delayed neutron precursor concentration."""
        return float(self.C)

    def get_reactivity(self) -> float:
        """Return current reactivity."""
        return float(self._compute_reactivity())

    def get_reactivity_pcm(self) -> float:
        """Return reactivity in pcm (percent mille)."""
        return float(self._compute_reactivity() * 1e5)

    def get_reactivity_dollars(self) -> float:
        """Return reactivity in dollars (ρ/β)."""
        return float(self._compute_reactivity() / self.beta)

    def get_period(self) -> float:
        """
        Reactor period T (time for e-fold change in power).

        For ρ > β: T ≈ Λ/(ρ - β)  (prompt jump)
        For ρ < β: T ≈ (β - ρ)/λρ  (delayed kinetics)

        Returns:
            Reactor period in seconds.
        """
        rho = self._compute_reactivity()
        k_eff = self._compute_k_eff()
        Lambda = self.l_prompt / k_eff if k_eff > 0 else self.l_prompt

        if abs(rho) < 1e-10:
            return float('inf')  # Critical = infinite period
        elif rho > self.beta:
            # Prompt supercritical
            return float(Lambda / (rho - self.beta))
        else:
            # Delayed kinetics dominated
            return float((self.beta - rho) / (self.lambda_d * abs(rho)))

    def set_control_rods(self, worth: float) -> None:
        """
        Adjust control rod insertion.

        Positive worth increases k_eff (withdrawal).
        Negative worth decreases k_eff (insertion).

        Args:
            worth: Control rod reactivity change.
        """
        self.rho_control = worth
        self.params["control_rod_worth"] = worth

    def insert_control_rods(self, amount: float = 0.01) -> None:
        """Insert control rods (decrease k_eff)."""
        self.set_control_rods(self.rho_control - amount)

    def withdraw_control_rods(self, amount: float = 0.01) -> None:
        """Withdraw control rods (increase k_eff)."""
        self.set_control_rods(self.rho_control + amount)

    def scram(self) -> None:
        """Emergency shutdown - full rod insertion."""
        self.set_control_rods(-0.1)  # -10% reactivity

    def is_critical(self, tolerance: float = 0.001) -> bool:
        """Check if reactor is critical within tolerance."""
        return abs(self._compute_k_eff() - 1.0) < tolerance

    def is_prompt_critical(self) -> bool:
        """Check if reactor is prompt critical (ρ > β)."""
        return self._compute_reactivity() > self.beta
