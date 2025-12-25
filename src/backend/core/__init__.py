"""
Core physics solvers for holographic fixed point analysis.
"""

from .base import HolographicSolver, FixedPointResult
from .kuramoto import KuramotoSolver
from .bekenstein import BekensteinSolver
from .turing import TuringPatternSolver
from .criticality import NuclearCriticalitySolver
from .rosetta import RosettaSolver, RosettaConstants

__all__ = [
    "HolographicSolver",
    "FixedPointResult",
    "KuramotoSolver",
    "BekensteinSolver",
    "TuringPatternSolver",
    "NuclearCriticalitySolver",
    "RosettaSolver",
    "RosettaConstants",
]
