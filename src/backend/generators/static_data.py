"""
Generate pre-computed simulation data for GitHub Pages deployment.

This module creates static JSON files containing simulation results
that the frontend can load without requiring a backend server.

Run during GitHub Actions build phase:
    python -m backend.generators.static_data

Output files are written to public/data/
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.core import (
    KuramotoSolver,
    BekensteinSolver,
    TuringPatternSolver,
    NuclearCriticalitySolver,
)

# Output directory - use environment variable or default to repo root
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", Path.cwd() / "public" / "data"))


def save_json(data: Any, filename: str) -> Path:
    """Save data to JSON file with NumPy handling."""
    output_path = OUTPUT_DIR / filename

    def numpy_handler(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(output_path, "w") as f:
        json.dump(data, f, default=numpy_handler)

    return output_path


def generate_kuramoto_phase_diagram() -> Path:
    """
    Generate Kuramoto order parameter vs coupling strength data.

    Creates phase diagram showing the synchronization transition
    for different numbers of oscillators.
    """
    print("Generating Kuramoto phase diagram...")

    K_values = np.linspace(0.1, 5.0, 100)
    n_oscillators_values = [50, 100, 200, 500]

    results = {
        "K": K_values.tolist(),
        "data": {},
        "description": "Kuramoto model phase transition: order parameter r vs coupling K",
    }

    for N in n_oscillators_values:
        print(f"  N = {N}...")
        r_values = []
        r_std_values = []

        for K in K_values:
            solver = KuramotoSolver({
                "n_oscillators": N,
                "coupling": float(K),
                "frequency_std": 1.0,
                "seed": 42,
            })
            solver.setup()

            # Equilibrate
            for _ in range(2000):
                solver.step(0.01)

            # Sample order parameter
            samples = []
            for _ in range(100):
                solver.step(0.01)
                samples.append(solver.compute_order_parameter())

            r_values.append(float(np.mean(samples)))
            r_std_values.append(float(np.std(samples)))

        results["data"][f"N={N}"] = {
            "r_mean": r_values,
            "r_std": r_std_values,
        }

    # Critical threshold
    results["K_critical"] = float(np.sqrt(8 / np.pi))

    output_path = save_json(results, "kuramoto_phase_diagram.json")
    print(f"  Saved to {output_path}")
    return output_path


def generate_turing_patterns() -> Path:
    """
    Generate Turing pattern evolution snapshots.

    Creates time series of pattern formation showing
    emergence of spatial structure from homogeneous initial conditions.
    """
    print("Generating Turing patterns...")

    grid_size = 128
    solver = TuringPatternSolver({
        "grid_size": grid_size,
        "D_u": 2.8e-4,
        "D_v": 5e-3,
        "tau": 0.1,
        "k": -0.005,
        "seed": 42,
    })
    solver.setup()

    snapshots = []
    times = [0, 1, 2, 5, 10, 20, 50, 100]

    current_time = 0.0
    dt = 0.01  # Smaller timestep for numerical stability

    for target_time in times:
        while current_time < target_time:
            solver.step(dt)
            current_time += dt

        # Downsample pattern for JSON size
        pattern = solver.get_pattern()
        if grid_size > 64:
            # Simple downsampling
            factor = grid_size // 64
            pattern = pattern[::factor, ::factor]

        snapshots.append({
            "time": target_time,
            "pattern": pattern.tolist(),
            "contrast": float(solver.compute_order_parameter()),
            "wavelength": float(solver.get_pattern_wavelength()),
        })
        print(f"  t = {target_time}, contrast = {solver.compute_order_parameter():.4f}")

    results = {
        "snapshots": snapshots,
        "grid_size": 64,  # Downsampled size
        "original_size": grid_size,
        "parameters": {
            "D_u": 2.8e-4,
            "D_v": 5e-3,
            "tau": 0.1,
            "k": -0.005,
        },
        "critical_ratio": float(solver.get_critical_threshold()),
        "description": "Turing pattern formation in FitzHugh-Nagumo system",
    }

    output_path = save_json(results, "turing_patterns.json")
    print(f"  Saved to {output_path}")
    return output_path


def generate_holographic_bounds() -> Path:
    """
    Generate Bekenstein entropy data for various masses.

    Creates dataset showing entropy vs mass relationship
    from laboratory scales to supermassive black holes.
    """
    print("Generating holographic entropy bounds...")

    # Logarithmic mass range: 1 kg to 10^40 kg
    log_masses = np.linspace(0, 40, 100)
    masses = 10.0 ** log_masses

    results = []
    M_sun = 1.989e30  # Solar mass

    for mass in masses:
        solver = BekensteinSolver({"mass_kg": float(mass)})
        solver.setup()

        results.append({
            "mass_kg": float(mass),
            "mass_log10": float(np.log10(mass)),
            "mass_solar": float(mass / M_sun),
            "entropy_planck": float(solver.compute_order_parameter()),
            "entropy_log10": float(np.log10(max(solver.compute_order_parameter(), 1e-300))),
            "entropy_bits": float(solver.holographic_bits()),
            "bekenstein_bound": float(solver.get_critical_threshold()),
            "schwarzschild_radius_m": float(solver.get_schwarzschild_radius()),
            "hawking_temp_K": float(solver.get_hawking_temperature()),
            "evaporation_time_s": float(solver.get_evaporation_time()),
        })

    # Notable objects
    notable = [
        {"name": "1 kg mass", "mass_solar": 1.0 / M_sun},
        {"name": "Earth", "mass_solar": 3.003e-6},
        {"name": "Sun", "mass_solar": 1.0},
        {"name": "Stellar BH (10 M☉)", "mass_solar": 10.0},
        {"name": "Sgr A* (4M M☉)", "mass_solar": 4e6},
        {"name": "M87* (6.5B M☉)", "mass_solar": 6.5e9},
    ]

    output = {
        "data": results,
        "notable_objects": notable,
        "description": "Bekenstein-Hawking entropy for objects of various masses",
        "units": {
            "mass": "kg",
            "entropy_planck": "dimensionless (k_B = 1)",
            "entropy_bits": "bits",
            "temperature": "Kelvin",
            "radius": "meters",
            "time": "seconds",
        },
    }

    output_path = save_json(output, "bekenstein_entropy.json")
    print(f"  Saved to {output_path}")
    return output_path


def generate_criticality_response() -> Path:
    """
    Generate nuclear reactor criticality response curves.

    Shows neutron population dynamics for various reactivity insertions.
    """
    print("Generating criticality response curves...")

    # Different reactivity scenarios
    reactivities = [
        {"name": "subcritical", "k_inf": 0.99, "leakage": 0.02},
        {"name": "critical", "k_inf": 1.02, "leakage": 0.02},
        {"name": "supercritical", "k_inf": 1.05, "leakage": 0.02},
        {"name": "prompt_critical", "k_inf": 1.10, "leakage": 0.02},
    ]

    results = {"curves": [], "description": "Point reactor kinetics response curves"}

    for scenario in reactivities:
        print(f"  Scenario: {scenario['name']}...")

        solver = NuclearCriticalitySolver({
            "initial_neutrons": 1e6,
            "k_infinity": scenario["k_inf"],
            "leakage_factor": scenario["leakage"],
        })
        solver.setup()

        trajectory = []
        dt = 0.001
        duration = 10.0
        current_time = 0.0

        while current_time < duration:
            solver.step(dt)
            current_time += dt

            if int(current_time / dt) % 100 == 0:  # Sample every 100 steps
                trajectory.append({
                    "time": float(current_time),
                    "neutrons": float(solver.get_neutron_population()),
                    "k_eff": float(solver.compute_order_parameter()),
                    "reactivity_dollars": float(solver.get_reactivity_dollars()),
                })

        results["curves"].append({
            "name": scenario["name"],
            "k_infinity": scenario["k_inf"],
            "k_eff": float(solver.compute_order_parameter()),
            "trajectory": trajectory,
        })

    output_path = save_json(results, "criticality_response.json")
    print(f"  Saved to {output_path}")
    return output_path


def generate_manifest() -> Path:
    """Create index of all available datasets."""
    print("Generating manifest...")

    manifest = {
        "generated": datetime.now().isoformat(),
        "version": "0.1.0",
        "datasets": [
            {
                "id": "kuramoto_phase_diagram",
                "name": "Kuramoto Phase Transition",
                "file": "kuramoto_phase_diagram.json",
                "description": "Order parameter vs coupling strength for synchronization",
                "category": "synchronization",
            },
            {
                "id": "turing_patterns",
                "name": "Turing Pattern Formation",
                "file": "turing_patterns.json",
                "description": "Reaction-diffusion pattern evolution snapshots",
                "category": "pattern_formation",
            },
            {
                "id": "bekenstein_entropy",
                "name": "Bekenstein-Hawking Entropy",
                "file": "bekenstein_entropy.json",
                "description": "Holographic entropy bounds for various masses",
                "category": "entropy_bounds",
            },
            {
                "id": "criticality_response",
                "name": "Nuclear Criticality Response",
                "file": "criticality_response.json",
                "description": "Reactor dynamics for various reactivity states",
                "category": "criticality",
            },
        ],
        "fixed_points": [
            {
                "name": "Kuramoto K_c",
                "type": "synchronization",
                "formula": "K_c = √(8/π) × σ",
                "value": float(np.sqrt(8 / np.pi)),
                "unit": "dimensionless",
            },
            {
                "name": "Nuclear k_eff",
                "type": "criticality",
                "formula": "k_eff = 1",
                "value": 1.0,
                "unit": "dimensionless",
            },
            {
                "name": "Bekenstein-Hawking S",
                "type": "entropy_bound",
                "formula": "S = A/(4ℓ_P²)",
                "value": None,
                "unit": "k_B",
            },
            {
                "name": "Turing instability",
                "type": "pattern_formation",
                "formula": "D_v/D_u > critical_ratio",
                "value": None,
                "unit": "dimensionless",
            },
        ],
        "categories": {
            "synchronization": {
                "name": "Synchronization",
                "color": "#00d9ff",
                "description": "Coupled oscillator phenomena",
            },
            "entropy_bounds": {
                "name": "Entropy Bounds",
                "color": "#ff6b35",
                "description": "Holographic information limits",
            },
            "pattern_formation": {
                "name": "Pattern Formation",
                "color": "#9d4edd",
                "description": "Spatial structure emergence",
            },
            "criticality": {
                "name": "Criticality",
                "color": "#e94560",
                "description": "Phase transition thresholds",
            },
        },
    }

    output_path = save_json(manifest, "manifest.json")
    print(f"  Saved to {output_path}")
    return output_path


def generate_all() -> None:
    """Generate all static data files."""
    print("=" * 60)
    print("APL Holographic Fixed Points - Static Data Generation")
    print("=" * 60)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate all datasets
    generate_kuramoto_phase_diagram()
    generate_turing_patterns()
    generate_holographic_bounds()
    generate_criticality_response()
    generate_manifest()

    print("=" * 60)
    print("All static data generated successfully!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    generate_all()
