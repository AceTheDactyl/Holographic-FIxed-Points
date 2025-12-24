"""
APL-inspired operator grammar for physics expressions.

Enables tacit/point-free composition of physics operations using
operator overloading. This allows expressing physics equations in
a notation closer to mathematical formalism.

Examples:
    # Kuramoto order parameter: r = |⟨e^{iθ}⟩|
    >>> OrderParameter = Abs @ Mean @ Exp
    >>> phases = np.array([0.1, 0.2, 0.15, 0.12])
    >>> r = OrderParameter(1j * phases)

    # Composition: (f ∘ g)(x) = f(g(x))
    >>> composed = Gradient @ Laplacian
    >>> result = composed(field)

    # Fork: (f | g)(x) = (f(x), g(x))
    >>> both = Sum | Mean
    >>> total, average = both(data)

References:
    Iverson, K. E. (1962). A Programming Language.
    Hui, R. K. W. (1987). Some Uses of { and }.
"""

from typing import Callable, Any, Tuple, Union
import numpy as np


class PhysicsOperator:
    """
    Base class for composable physics operators.

    Supports:
    - Function application: op(x) or op(x, y)
    - Composition: f @ g means f(g(x))
    - Fork: f | g means (f(x), g(x))
    - Power: f ** n means apply f n times
    """

    def __init__(
        self,
        func: Callable,
        name: str,
        symbol: str = None,
        arity: int = 1,
    ):
        """
        Create a physics operator.

        Args:
            func: The underlying function.
            name: Human-readable name.
            symbol: Mathematical symbol (optional, defaults to name).
            arity: Number of arguments (1 for unary, 2 for binary).
        """
        self.func = func
        self.name = name
        self.symbol = symbol or name
        self.arity = arity

    def __call__(self, *args, **kwargs) -> Any:
        """Apply the operator to arguments."""
        return self.func(*args, **kwargs)

    def __matmul__(self, other: "PhysicsOperator") -> "PhysicsOperator":
        """
        Composition: (f @ g)(x) = f(g(x))

        This is mathematical function composition.
        """
        if isinstance(other, PhysicsOperator):
            return PhysicsOperator(
                lambda *args, **kwargs: self.func(other.func(*args, **kwargs)),
                f"({self.name} ∘ {other.name})",
                f"({self.symbol} ∘ {other.symbol})",
            )
        raise TypeError(f"Cannot compose {type(self)} with {type(other)}")

    def __or__(self, other: "PhysicsOperator") -> "PhysicsOperator":
        """
        Fork: (f | g)(x) = (f(x), g(x))

        Applies both operators to the same input.
        """
        if isinstance(other, PhysicsOperator):
            return PhysicsOperator(
                lambda *args, **kwargs: (
                    self.func(*args, **kwargs),
                    other.func(*args, **kwargs),
                ),
                f"({self.name} ⊕ {other.name})",
                f"({self.symbol} | {other.symbol})",
            )
        raise TypeError(f"Cannot fork {type(self)} with {type(other)}")

    def __pow__(self, n: int) -> "PhysicsOperator":
        """
        Power: f ** n means apply f n times.
        """
        if not isinstance(n, int) or n < 0:
            raise ValueError("Power must be non-negative integer")

        if n == 0:
            return PhysicsOperator(lambda x: x, "id", "I")

        def repeated(x):
            result = x
            for _ in range(n):
                result = self.func(result)
            return result

        return PhysicsOperator(
            repeated,
            f"({self.name})^{n}",
            f"({self.symbol})^{n}",
        )

    def __repr__(self) -> str:
        return f"PhysicsOperator({self.symbol})"

    def __str__(self) -> str:
        return self.symbol


def create_operator(
    func: Callable,
    name: str,
    symbol: str = None,
) -> PhysicsOperator:
    """
    Create a physics operator from a function.

    Args:
        func: The function to wrap.
        name: Operator name.
        symbol: Mathematical symbol.

    Returns:
        PhysicsOperator wrapping the function.
    """
    return PhysicsOperator(func, name, symbol)


# =============================================================================
# Primitive Operators
# =============================================================================

# Differential operators
def _gradient(u: np.ndarray, axis: int = None) -> np.ndarray:
    """Compute gradient using central differences."""
    if axis is not None:
        return np.gradient(u, axis=axis)
    return np.gradient(u)


def _laplacian(u: np.ndarray) -> np.ndarray:
    """Compute discrete Laplacian using 5-point stencil."""
    if u.ndim == 1:
        return np.gradient(np.gradient(u))
    elif u.ndim == 2:
        return (
            np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) +
            np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) -
            4 * u
        )
    else:
        raise ValueError(f"Laplacian not implemented for {u.ndim}D arrays")


def _divergence(v: np.ndarray) -> np.ndarray:
    """Compute divergence of vector field."""
    if v.ndim < 2:
        raise ValueError("Divergence requires at least 2D array (vector field)")
    return sum(np.gradient(v[..., i], axis=i) for i in range(v.shape[-1]))


def _curl_2d(v: np.ndarray) -> np.ndarray:
    """Compute curl of 2D vector field (returns scalar)."""
    dvx_dy = np.gradient(v[..., 0], axis=0)
    dvy_dx = np.gradient(v[..., 1], axis=1)
    return dvy_dx - dvx_dy


Gradient = PhysicsOperator(_gradient, "gradient", "∇")
Laplacian = PhysicsOperator(_laplacian, "laplacian", "∇²")
Divergence = PhysicsOperator(_divergence, "divergence", "∇·")
Curl = PhysicsOperator(_curl_2d, "curl", "∇×")

# Reduction operators
Sum = PhysicsOperator(np.sum, "sum", "Σ")
Mean = PhysicsOperator(np.mean, "mean", "⟨·⟩")
Prod = PhysicsOperator(np.prod, "product", "∏")
Max = PhysicsOperator(np.max, "max", "max")
Min = PhysicsOperator(np.min, "min", "min")

# Element-wise operators
Exp = PhysicsOperator(np.exp, "exp", "exp")
Log = PhysicsOperator(np.log, "log", "ln")
Sin = PhysicsOperator(np.sin, "sin", "sin")
Cos = PhysicsOperator(np.cos, "cos", "cos")
Abs = PhysicsOperator(np.abs, "abs", "|·|")
Sqrt = PhysicsOperator(np.sqrt, "sqrt", "√")
Square = PhysicsOperator(np.square, "square", "·²")

# Norm operators
def _norm(x: np.ndarray) -> float:
    """Compute L2 norm."""
    return float(np.linalg.norm(x))


def _normalize(x: np.ndarray) -> np.ndarray:
    """Normalize to unit length."""
    n = np.linalg.norm(x)
    return x / n if n > 0 else x


Norm = PhysicsOperator(_norm, "norm", "‖·‖")
Normalize = PhysicsOperator(_normalize, "normalize", "·/‖·‖")

# Complex operators
Real = PhysicsOperator(np.real, "real", "Re")
Imag = PhysicsOperator(np.imag, "imag", "Im")
Conj = PhysicsOperator(np.conj, "conjugate", "·*")
Phase = PhysicsOperator(np.angle, "phase", "arg")


# =============================================================================
# Composed Operators for Physics
# =============================================================================

# Kuramoto order parameter: r = |⟨e^{iθ}⟩|
def _kuramoto_order(phases: np.ndarray) -> float:
    """Compute Kuramoto order parameter from phases."""
    return float(np.abs(np.mean(np.exp(1j * phases))))


OrderParameter = PhysicsOperator(_kuramoto_order, "order_parameter", "r")

# Phase coherence (same as order parameter for oscillators)
PhaseCoherence = Abs @ Mean @ Exp  # |⟨e^{iθ}⟩|

# Energy operators
KineticEnergy = PhysicsOperator(
    lambda v: 0.5 * np.sum(v**2),
    "kinetic_energy",
    "½v²",
)

# Field energy: ∫|∇φ|² dx
def _field_energy(phi: np.ndarray) -> float:
    """Compute field gradient energy."""
    grad = np.gradient(phi)
    if isinstance(grad, np.ndarray):
        return float(np.sum(grad**2))
    return float(sum(np.sum(g**2) for g in grad))


FieldEnergy = PhysicsOperator(_field_energy, "field_energy", "∫|∇φ|²")


# =============================================================================
# Higher-Order Operators (Operators that create operators)
# =============================================================================

def Scan(op: PhysicsOperator) -> PhysicsOperator:
    """
    Create running/cumulative version of reduction operator.

    Scan(Sum)([1,2,3,4]) = [1, 3, 6, 10]
    """
    if op.func == np.sum:
        return PhysicsOperator(np.cumsum, f"scan({op.name})", f"\\{op.symbol}")
    elif op.func == np.prod:
        return PhysicsOperator(np.cumprod, f"scan({op.name})", f"\\{op.symbol}")
    else:
        def cumulative(x):
            result = []
            for i in range(len(x)):
                result.append(op.func(x[:i+1]))
            return np.array(result)
        return PhysicsOperator(cumulative, f"scan({op.name})", f"\\{op.symbol}")


def Each(op: PhysicsOperator) -> PhysicsOperator:
    """
    Apply operator to each element (map/vectorize).
    """
    return PhysicsOperator(
        lambda x: np.array([op.func(xi) for xi in x]),
        f"each({op.name})",
        f"{op.symbol}¨",
    )


def Reduce(binary_op: Callable) -> PhysicsOperator:
    """
    Create reduction operator from binary function.

    Reduce(np.add) is equivalent to Sum
    """
    from functools import reduce
    return PhysicsOperator(
        lambda x: reduce(binary_op, x),
        "reduce",
        "/",
    )


def Outer(op: Callable) -> PhysicsOperator:
    """
    Create outer product operator.

    Outer(np.multiply)(a, b) computes a ⊗ b
    """
    return PhysicsOperator(
        lambda a, b: np.outer(a, b) if op == np.multiply else
                     np.array([[op(ai, bj) for bj in b] for ai in a]),
        "outer",
        "∘.",
        arity=2,
    )


# =============================================================================
# Utility: Expression Building
# =============================================================================

class PhysicsExpression:
    """
    Build complex physics expressions using operator algebra.

    Example:
        >>> expr = PhysicsExpression()
        >>> kuramoto = expr.abs(expr.mean(expr.exp(1j * expr.var("theta"))))
    """

    def __init__(self):
        self._vars = {}

    def var(self, name: str) -> np.ndarray:
        """Create or retrieve a variable."""
        if name not in self._vars:
            self._vars[name] = None
        return self._vars.get(name)

    def bind(self, name: str, value: np.ndarray) -> "PhysicsExpression":
        """Bind a value to a variable."""
        self._vars[name] = value
        return self

    # Delegate to operators
    def __getattr__(self, name: str) -> PhysicsOperator:
        operators = {
            "grad": Gradient,
            "laplacian": Laplacian,
            "div": Divergence,
            "curl": Curl,
            "sum": Sum,
            "mean": Mean,
            "exp": Exp,
            "log": Log,
            "sin": Sin,
            "cos": Cos,
            "abs": Abs,
            "norm": Norm,
        }
        if name in operators:
            return operators[name]
        raise AttributeError(f"Unknown operator: {name}")
