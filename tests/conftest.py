"""Pytest configuration and fixtures for holographic solvers testing."""

import pytest
import numpy as np


@pytest.fixture
def kuramoto_params():
    """Standard Kuramoto solver parameters."""
    return {
        "n_oscillators": 50,
        "coupling": 2.0,
        "frequency_std": 1.0,
        "frequency_mean": 0.0,
        "seed": 42,
    }


@pytest.fixture
def turing_params():
    """Standard Turing pattern solver parameters."""
    return {
        "grid_size": 32,
        "D_u": 2.8e-4,
        "D_v": 5e-3,
        "tau": 0.1,
        "k": -0.005,
        "seed": 42,
    }


@pytest.fixture
def bekenstein_params():
    """Standard Bekenstein solver parameters for solar mass black hole."""
    return {
        "mass_kg": 1.989e30,  # Solar mass
        "radius_m": 1.0,
        "include_hawking": False,
    }


@pytest.fixture
def criticality_params():
    """Standard nuclear criticality solver parameters."""
    return {
        "initial_neutrons": 1e6,
        "k_infinity": 1.03,
        "leakage_factor": 0.02,
        "delayed_fraction": 0.0065,
        "decay_constant": 0.08,
        "prompt_lifetime": 1e-4,
        "control_rod_worth": 0.0,
    }


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset numpy random seed before each test for reproducibility."""
    np.random.seed(42)
    yield
