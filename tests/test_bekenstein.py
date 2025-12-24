"""Tests for Bekenstein entropy bound solver."""

import numpy as np
import pytest

from backend.core.bekenstein import BekensteinSolver, PhysicalConstants


class TestPhysicalConstants:
    """Test physical constants values."""

    def test_planck_length_order(self):
        """Test Planck length is correct order of magnitude."""
        l_p = PhysicalConstants.l_planck
        assert 1e-36 < l_p < 1e-34  # ~1.616e-35 m

    def test_planck_mass_order(self):
        """Test Planck mass is correct order of magnitude."""
        m_p = PhysicalConstants.m_planck
        assert 1e-9 < m_p < 1e-7  # ~2.176e-8 kg

    def test_solar_mass(self):
        """Test solar mass value."""
        assert np.isclose(PhysicalConstants.M_sun, 1.989e30, rtol=0.01)


class TestBekensteinSolver:
    """Test suite for BekensteinSolver."""

    def test_setup_initializes_state(self, bekenstein_params):
        """Test that setup creates valid state."""
        solver = BekensteinSolver(bekenstein_params)
        solver.setup()

        assert solver.state is not None
        assert len(solver.state) == 3  # [entropy, area, temperature]
        assert solver.time == 0.0

    def test_schwarzschild_radius_solar_mass(self, bekenstein_params):
        """Test Schwarzschild radius for solar mass black hole."""
        solver = BekensteinSolver(bekenstein_params)
        solver.setup()

        r_s = solver.get_schwarzschild_radius()
        # Solar mass Schwarzschild radius ≈ 2.95 km
        expected = 2 * PhysicalConstants.G * bekenstein_params["mass_kg"] / PhysicalConstants.c**2

        assert np.isclose(r_s, expected)
        assert 2900 < r_s < 3000  # ≈ 2.95 km

    def test_horizon_area_from_radius(self, bekenstein_params):
        """Test horizon area A = 4πr²."""
        solver = BekensteinSolver(bekenstein_params)
        solver.setup()

        r_s = solver.get_schwarzschild_radius()
        area = solver.get_horizon_area()
        expected = 4 * np.pi * r_s**2

        assert np.isclose(area, expected)

    def test_entropy_positive(self, bekenstein_params):
        """Test entropy is positive."""
        solver = BekensteinSolver(bekenstein_params)
        solver.setup()

        entropy = solver.compute_order_parameter()
        assert entropy > 0

    def test_entropy_proportional_to_area(self, bekenstein_params):
        """Test S = A/(4l_P²) relation."""
        solver = BekensteinSolver(bekenstein_params)
        solver.setup()

        area = solver.get_horizon_area()
        entropy = solver.compute_order_parameter()
        l_p = PhysicalConstants.l_planck

        expected_ratio = area / (4 * l_p**2)
        assert np.isclose(entropy, expected_ratio)

    def test_entropy_density_quarter(self, bekenstein_params):
        """Test entropy density is 1/4 in Planck units."""
        solver = BekensteinSolver(bekenstein_params)
        solver.setup()

        density = solver.entropy_density()
        assert np.isclose(density, 0.25)

    def test_holographic_bits_positive(self, bekenstein_params):
        """Test holographic bits count is positive."""
        solver = BekensteinSolver(bekenstein_params)
        solver.setup()

        bits = solver.holographic_bits()
        assert bits > 0

    def test_bekenstein_bound(self, bekenstein_params):
        """Test Bekenstein bound S_max = 2πRE/(ℏc)."""
        solver = BekensteinSolver(bekenstein_params)
        solver.setup()

        threshold = solver.get_critical_threshold()

        C = PhysicalConstants
        E = bekenstein_params["mass_kg"] * C.c**2
        R = max(bekenstein_params["radius_m"], solver.get_schwarzschild_radius())
        expected = 2 * np.pi * R * E / (C.hbar * C.c)

        assert np.isclose(threshold, expected)

    def test_saturation_ratio_black_hole(self, bekenstein_params):
        """Test saturation ratio for black hole."""
        solver = BekensteinSolver(bekenstein_params)
        solver.setup()

        ratio = solver.saturation_ratio()
        # Black hole should nearly saturate the bound
        assert 0 < ratio <= 1

    def test_hawking_temperature_formula(self, bekenstein_params):
        """Test Hawking temperature T_H = ℏc³/(8πGMk_B)."""
        solver = BekensteinSolver(bekenstein_params)
        solver.setup()

        T_H = solver.get_hawking_temperature()
        C = PhysicalConstants
        M = bekenstein_params["mass_kg"]

        expected = C.hbar * C.c**3 / (8 * np.pi * C.G * M * C.k_B)
        assert np.isclose(T_H, expected)

    def test_hawking_temperature_inverse_mass(self):
        """Test that Hawking temperature increases with decreasing mass."""
        solver1 = BekensteinSolver({"mass_kg": 1e30})
        solver1.setup()

        solver2 = BekensteinSolver({"mass_kg": 1e29})  # 10x smaller
        solver2.setup()

        # Smaller black hole should be hotter
        assert solver2.get_hawking_temperature() > solver1.get_hawking_temperature()

    def test_evaporation_time_positive(self, bekenstein_params):
        """Test evaporation time is positive."""
        solver = BekensteinSolver(bekenstein_params)
        solver.setup()

        t_evap = solver.get_evaporation_time()
        assert t_evap > 0

    def test_hawking_evaporation_decreases_mass(self):
        """Test that Hawking evaporation decreases mass over time."""
        params = {
            "mass_kg": 1e20,  # Small black hole
            "include_hawking": True,
            "hawking_rate": 1.0,
        }
        solver = BekensteinSolver(params)
        solver.setup()

        initial_entropy = solver.compute_order_parameter()

        # Run for many steps
        for _ in range(1000):
            solver.step(1e10)  # Large time steps

        final_entropy = solver.compute_order_parameter()

        # Entropy should decrease (mass evaporating)
        assert final_entropy < initial_entropy

    def test_step_without_hawking_preserves_entropy(self, bekenstein_params):
        """Test that step without Hawking radiation preserves entropy."""
        solver = BekensteinSolver(bekenstein_params)
        solver.setup()

        initial_entropy = solver.compute_order_parameter()

        for _ in range(100):
            solver.step(1.0)

        final_entropy = solver.compute_order_parameter()

        assert np.isclose(initial_entropy, final_entropy)

    def test_set_mass_updates_state(self, bekenstein_params):
        """Test that set_mass updates all derived quantities."""
        solver = BekensteinSolver(bekenstein_params)
        solver.setup()

        initial_entropy = solver.compute_order_parameter()

        new_mass = bekenstein_params["mass_kg"] * 2
        solver.set_mass(new_mass)

        new_entropy = solver.compute_order_parameter()

        # Entropy ∝ M² (via A = 4π(2GM/c²)² ∝ M²)
        assert new_entropy > initial_entropy

    def test_analyze_fixed_point_result(self, bekenstein_params):
        """Test analyze_fixed_point returns valid result."""
        solver = BekensteinSolver(bekenstein_params)
        solver.setup()

        result = solver.analyze_fixed_point()

        assert hasattr(result, "order_parameter")
        assert hasattr(result, "critical_threshold")
        assert hasattr(result, "is_at_fixed_point")
        assert result.order_parameter > 0

    def test_to_dict_serializable(self, bekenstein_params):
        """Test that result can be serialized to dict."""
        solver = BekensteinSolver(bekenstein_params)
        solver.setup()

        result = solver.analyze_fixed_point()
        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "order_parameter" in result_dict
        assert isinstance(result_dict["order_parameter"], float)
