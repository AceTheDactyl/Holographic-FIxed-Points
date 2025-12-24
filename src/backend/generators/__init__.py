"""
Static data generators for GitHub Pages deployment.

These generators create pre-computed simulation data that can be
bundled with the static frontend and served without a backend.
"""

from .static_data import (
    generate_kuramoto_phase_diagram,
    generate_turing_patterns,
    generate_holographic_bounds,
    generate_criticality_response,
    generate_manifest,
    generate_all,
)

__all__ = [
    "generate_kuramoto_phase_diagram",
    "generate_turing_patterns",
    "generate_holographic_bounds",
    "generate_criticality_response",
    "generate_manifest",
    "generate_all",
]
