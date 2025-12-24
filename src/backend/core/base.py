"""
Base class for holographic fixed point solvers.

This module defines the abstract interface that all physics solvers
must implement, providing a unified API for fixed point analysis
across different physical systems.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional
import numpy as np


@dataclass
class FixedPointResult:
    """
    Result from fixed point analysis.

    Attributes:
        order_parameter: The system's order parameter (e.g., r for Kuramoto,
            k_eff for nuclear, S for entropy bounds).
        critical_threshold: The theoretical critical value where phase
            transition or fixed point behavior occurs.
        is_at_fixed_point: Whether the system is within tolerance of
            the critical threshold.
        state_vector: Current system state as a numpy array.
        metadata: Additional diagnostic information including time,
            parameters, and solver-specific data.
    """
    order_parameter: float
    critical_threshold: float
    is_at_fixed_point: bool
    state_vector: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to JSON-serializable dictionary."""
        return {
            "order_parameter": float(self.order_parameter),
            "critical_threshold": float(self.critical_threshold),
            "is_at_fixed_point": bool(self.is_at_fixed_point),
            "state_vector": self.state_vector.tolist() if self.state_vector.size < 1000 else [],
            "metadata": {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in self.metadata.items()
                if not isinstance(v, np.ndarray)
            },
        }


class HolographicSolver(ABC):
    """
    Abstract base class for all holographic fixed point solvers.

    Each solver represents a different physical system exhibiting
    critical behavior: synchronization, phase transitions, or
    information-theoretic bounds.

    The solver maintains internal state and provides methods for:
    - setup(): Initialize from parameters
    - step(dt): Advance simulation by one timestep
    - compute_order_parameter(): Calculate system's order parameter
    - get_critical_threshold(): Return theoretical critical value
    - analyze_fixed_point(): Determine if system is near critical point
    - iterate(steps, dt): Generator for trajectory analysis

    Example usage:
        >>> solver = KuramotoSolver({"n_oscillators": 100, "coupling": 2.0})
        >>> solver.setup()
        >>> for _ in range(1000):
        ...     solver.step(0.01)
        >>> result = solver.analyze_fixed_point()
        >>> print(f"Order parameter: {result.order_parameter:.3f}")
    """

    def __init__(self, params: dict[str, Any]):
        """
        Initialize solver with parameters.

        Args:
            params: Dictionary of solver-specific parameters.
        """
        self.params = params.copy()
        self._state: Optional[np.ndarray] = None
        self._time: float = 0.0

    @abstractmethod
    def setup(self) -> None:
        """
        Initialize solver state from parameters.

        Must be called before step() or any analysis methods.
        Implementations should set self._state to the initial state.
        """
        pass

    @abstractmethod
    def step(self, dt: float) -> None:
        """
        Advance simulation by one timestep.

        Args:
            dt: Time step size.
        """
        pass

    @abstractmethod
    def compute_order_parameter(self) -> float:
        """
        Calculate the system's order parameter.

        The order parameter characterizes the collective behavior:
        - Kuramoto: r ∈ [0,1] phase coherence
        - Nuclear: k_eff multiplication factor
        - Bekenstein: S entropy in Planck units
        - Turing: spatial pattern contrast

        Returns:
            The current order parameter value.
        """
        pass

    @abstractmethod
    def get_critical_threshold(self) -> float:
        """
        Return the theoretical critical threshold.

        This is the value of the order parameter (or control parameter)
        at which the system exhibits phase transition or fixed point behavior.

        Returns:
            The critical threshold value.
        """
        pass

    @property
    def time(self) -> float:
        """Current simulation time."""
        return self._time

    @property
    def state(self) -> Optional[np.ndarray]:
        """Current system state."""
        return self._state

    def analyze_fixed_point(self, tolerance: float = 1e-6) -> FixedPointResult:
        """
        Determine if system is at/near a holographic fixed point.

        Args:
            tolerance: Threshold for determining if at fixed point.

        Returns:
            FixedPointResult containing order parameter, threshold,
            and fixed point status.
        """
        order = self.compute_order_parameter()
        critical = self.get_critical_threshold()
        return FixedPointResult(
            order_parameter=order,
            critical_threshold=critical,
            is_at_fixed_point=abs(order - critical) < tolerance,
            state_vector=self._state.copy() if self._state is not None else np.array([]),
            metadata={"time": self._time, "params": self.params.copy()},
        )

    def iterate(self, steps: int, dt: float) -> Iterator[FixedPointResult]:
        """
        Generator yielding fixed point analysis at each step.

        Args:
            steps: Number of simulation steps.
            dt: Time step size.

        Yields:
            FixedPointResult at each step.
        """
        for _ in range(steps):
            self.step(dt)
            yield self.analyze_fixed_point()

    def run_to_equilibrium(
        self,
        dt: float = 0.01,
        max_steps: int = 10000,
        tolerance: float = 1e-4,
        window: int = 100,
    ) -> FixedPointResult:
        """
        Run simulation until order parameter stabilizes.

        Args:
            dt: Time step size.
            max_steps: Maximum number of steps before giving up.
            tolerance: Variance threshold for equilibrium detection.
            window: Number of samples for variance calculation.

        Returns:
            Final FixedPointResult after equilibration.
        """
        history = []
        for _ in range(max_steps):
            self.step(dt)
            order = self.compute_order_parameter()
            history.append(order)

            if len(history) >= window:
                recent = history[-window:]
                if np.std(recent) < tolerance:
                    break
                history = recent  # Keep only recent history

        return self.analyze_fixed_point()

    def reset(self) -> None:
        """Reset solver to initial state (requires re-calling setup())."""
        self._state = None
        self._time = 0.0
