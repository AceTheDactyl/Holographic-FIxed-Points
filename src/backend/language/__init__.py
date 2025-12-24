"""
APL-inspired operator grammar for physics expressions.

This module provides composable operators that enable tacit/point-free
expression of physics operations, inspired by APL's array programming model.
"""

from .operators import (
    PhysicsOperator,
    Gradient,
    Laplacian,
    Divergence,
    Curl,
    Sum,
    Mean,
    Exp,
    Sin,
    Cos,
    Abs,
    Norm,
    OrderParameter,
    PhaseCoherence,
    create_operator,
)

__all__ = [
    "PhysicsOperator",
    "Gradient",
    "Laplacian",
    "Divergence",
    "Curl",
    "Sum",
    "Mean",
    "Exp",
    "Sin",
    "Cos",
    "Abs",
    "Norm",
    "OrderParameter",
    "PhaseCoherence",
    "create_operator",
]
