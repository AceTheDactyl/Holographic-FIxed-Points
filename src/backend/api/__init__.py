"""
Flask API for holographic fixed points simulations.

Provides REST endpoints for:
- Running individual simulations
- Parameter sweeps and phase diagrams
- Pre-computed data access
"""

from .app import create_app

__all__ = ["create_app"]
