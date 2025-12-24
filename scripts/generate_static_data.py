#!/usr/bin/env python3
"""
Generate pre-computed simulation data for GitHub Pages deployment.

This script generates static JSON files containing pre-computed
simulation results that can be loaded by the static frontend.

Usage:
    python scripts/generate_static_data.py

Output:
    public/data/kuramoto_phase_diagram.json
    public/data/turing_patterns.json
    public/data/bekenstein_entropy.json
    public/data/criticality_response.json
    public/data/manifest.json
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from backend.generators.static_data import generate_all


def main():
    """Entry point for data generation."""
    generate_all()


if __name__ == "__main__":
    main()
