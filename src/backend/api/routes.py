"""
API routes for holographic fixed points simulations.

Provides endpoints for:
- Running simulations with custom parameters
- Parameter sweeps for phase diagrams
- Accessing pre-computed data
"""

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from flask import Blueprint, jsonify, request

from ..core import (
    KuramotoSolver,
    BekensteinSolver,
    TuringPatternSolver,
    NuclearCriticalitySolver,
)

# Blueprints
simulations_bp = Blueprint("simulations", __name__)
precomputed_bp = Blueprint("precomputed", __name__)

# Solver registry
SOLVERS = {
    "kuramoto": KuramotoSolver,
    "bekenstein": BekensteinSolver,
    "turing": TuringPatternSolver,
    "criticality": NuclearCriticalitySolver,
}


# === Simulation Endpoints ===


@simulations_bp.route("/", methods=["GET"])
def list_solvers():
    """List available solvers and their parameters."""
    return jsonify({
        "solvers": {
            "kuramoto": {
                "description": "Kuramoto coupled oscillator model",
                "parameters": {
                    "n_oscillators": "Number of oscillators (default: 100)",
                    "coupling": "Coupling strength K (default: 1.0)",
                    "frequency_std": "Frequency distribution std (default: 1.0)",
                },
                "order_parameter": "r ∈ [0,1] phase coherence",
            },
            "bekenstein": {
                "description": "Bekenstein-Hawking entropy bounds",
                "parameters": {
                    "mass_kg": "System mass in kg",
                    "radius_m": "Bounding radius in meters",
                    "include_hawking": "Model Hawking evaporation",
                },
                "order_parameter": "S entropy in Planck units",
            },
            "turing": {
                "description": "Turing pattern reaction-diffusion",
                "parameters": {
                    "grid_size": "Simulation grid size (default: 100)",
                    "D_u": "Activator diffusion coefficient",
                    "D_v": "Inhibitor diffusion coefficient",
                },
                "order_parameter": "Pattern contrast (std)",
            },
            "criticality": {
                "description": "Nuclear reactor point kinetics",
                "parameters": {
                    "k_infinity": "Infinite medium multiplication",
                    "leakage_factor": "Geometric leakage",
                    "control_rod_worth": "Control rod reactivity",
                },
                "order_parameter": "k_eff multiplication factor",
            },
        }
    })


@simulations_bp.route("/<solver_type>", methods=["POST"])
def run_simulation(solver_type: str):
    """
    Execute a simulation and return results.

    POST /api/v1/simulations/<solver_type>

    Body (JSON):
        {
            "steps": 1000,      // Number of simulation steps
            "dt": 0.01,         // Time step size
            ...                 // Solver-specific parameters
        }

    Returns:
        {
            "solver": "kuramoto",
            "critical_threshold": 1.596,
            "final_state": {...},
            "trajectory": [...]
        }
    """
    if solver_type not in SOLVERS:
        return jsonify({"error": f"Unknown solver: {solver_type}"}), 400

    try:
        params = request.json or {}
        steps = params.pop("steps", 1000)
        dt = params.pop("dt", 0.01)
        sample_rate = params.pop("sample_rate", 100)  # Number of samples to return

        solver = SOLVERS[solver_type](params)
        solver.setup()

        # Collect trajectory with sampling
        trajectory = []
        sample_interval = max(1, steps // sample_rate)

        for i, result in enumerate(solver.iterate(steps, dt)):
            if i % sample_interval == 0:
                trajectory.append({
                    "time": result.metadata["time"],
                    "order_parameter": result.order_parameter,
                    "is_critical": result.is_at_fixed_point,
                })

        final = solver.analyze_fixed_point()

        return jsonify({
            "solver": solver_type,
            "critical_threshold": solver.get_critical_threshold(),
            "final_state": final.to_dict(),
            "trajectory": trajectory,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@simulations_bp.route("/kuramoto/sweep", methods=["POST"])
def kuramoto_coupling_sweep():
    """
    Sweep coupling strength to find critical transition.

    POST /api/v1/simulations/kuramoto/sweep

    Body:
        {
            "K_min": 0.1,
            "K_max": 5.0,
            "K_steps": 50,
            "n_oscillators": 100,
            ...
        }
    """
    try:
        params = request.json or {}
        K_min = params.pop("K_min", 0.1)
        K_max = params.pop("K_max", 5.0)
        K_steps = params.pop("K_steps", 50)
        equilibration_steps = params.pop("equilibration_steps", 1000)
        sample_steps = params.pop("sample_steps", 100)

        results = []
        K_values = np.linspace(K_min, K_max, K_steps)

        for K in K_values:
            solver = KuramotoSolver({**params, "coupling": float(K)})
            solver.setup()

            # Equilibrate
            for _ in range(equilibration_steps):
                solver.step(0.01)

            # Sample order parameter
            r_samples = []
            for _ in range(sample_steps):
                solver.step(0.01)
                r_samples.append(solver.compute_order_parameter())

            results.append({
                "K": float(K),
                "r_mean": float(np.mean(r_samples)),
                "r_std": float(np.std(r_samples)),
            })

        K_c = float(np.sqrt(8 / np.pi) * params.get("frequency_std", 1.0))

        return jsonify({
            "sweep": results,
            "K_critical": K_c,
            "parameters": params,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@simulations_bp.route("/turing/evolve", methods=["POST"])
def turing_evolution():
    """
    Evolve Turing pattern and return snapshots.

    POST /api/v1/simulations/turing/evolve

    Body:
        {
            "grid_size": 64,
            "total_time": 100,
            "snapshot_times": [0, 10, 50, 100],
            ...
        }
    """
    try:
        params = request.json or {}
        grid_size = params.pop("grid_size", 64)
        dt = params.pop("dt", 0.1)
        total_time = params.pop("total_time", 100.0)
        snapshot_times = params.pop("snapshot_times", [0, 10, 50, 100])

        solver = TuringPatternSolver({**params, "grid_size": grid_size})
        solver.setup()

        snapshots = []
        snapshot_idx = 0
        current_time = 0.0

        # Initial snapshot
        if 0 in snapshot_times:
            snapshots.append({
                "time": 0,
                "pattern": solver.get_pattern().tolist(),
                "contrast": solver.compute_order_parameter(),
            })
            snapshot_idx = 1

        while current_time < total_time and snapshot_idx < len(snapshot_times):
            solver.step(dt)
            current_time += dt

            if snapshot_idx < len(snapshot_times) and current_time >= snapshot_times[snapshot_idx]:
                snapshots.append({
                    "time": snapshot_times[snapshot_idx],
                    "pattern": solver.get_pattern().tolist(),
                    "contrast": solver.compute_order_parameter(),
                })
                snapshot_idx += 1

        return jsonify({
            "snapshots": snapshots,
            "grid_size": grid_size,
            "critical_ratio": solver.get_critical_threshold(),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@simulations_bp.route("/criticality/transient", methods=["POST"])
def criticality_transient():
    """
    Simulate reactor transient (rod insertion/withdrawal).

    POST /api/v1/simulations/criticality/transient

    Body:
        {
            "initial_k_infinity": 1.03,
            "rod_changes": [
                {"time": 1.0, "worth": -0.005},
                {"time": 5.0, "worth": 0.01}
            ],
            "duration": 10.0
        }
    """
    try:
        params = request.json or {}
        duration = params.pop("duration", 10.0)
        dt = params.pop("dt", 0.001)
        rod_changes = params.pop("rod_changes", [])

        solver = NuclearCriticalitySolver(params)
        solver.setup()

        trajectory = []
        current_time = 0.0
        rod_idx = 0

        # Sort rod changes by time
        rod_changes.sort(key=lambda x: x["time"])

        while current_time < duration:
            # Apply rod changes
            while rod_idx < len(rod_changes) and current_time >= rod_changes[rod_idx]["time"]:
                solver.set_control_rods(rod_changes[rod_idx]["worth"])
                rod_idx += 1

            solver.step(dt)
            current_time += dt

            # Sample every 100 steps
            if int(current_time / dt) % 100 == 0:
                trajectory.append({
                    "time": float(current_time),
                    "k_eff": solver.compute_order_parameter(),
                    "neutrons": solver.get_neutron_population(),
                    "reactivity_dollars": solver.get_reactivity_dollars(),
                })

        return jsonify({
            "trajectory": trajectory,
            "final_k_eff": solver.compute_order_parameter(),
            "final_neutrons": solver.get_neutron_population(),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# === Pre-computed Data Endpoints ===


DATA_DIR = Path(__file__).parent.parent.parent.parent.parent / "public" / "data"


@precomputed_bp.route("/manifest", methods=["GET"])
def get_manifest():
    """Get manifest of available pre-computed datasets."""
    manifest_path = DATA_DIR / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Manifest not found"}), 404


@precomputed_bp.route("/<dataset_id>", methods=["GET"])
def get_dataset(dataset_id: str):
    """Get a pre-computed dataset by ID."""
    file_path = DATA_DIR / f"{dataset_id}.json"
    if file_path.exists():
        with open(file_path) as f:
            return jsonify(json.load(f))
    return jsonify({"error": f"Dataset {dataset_id} not found"}), 404
