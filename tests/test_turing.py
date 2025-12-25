"""Tests for Turing pattern reaction-diffusion solver."""

import numpy as np
import pytest

from backend.core.turing import TuringPatternSolver


class TestTuringPatternSolver:
    """Test suite for TuringPatternSolver."""

    def test_setup_initializes_grids(self, turing_params):
        """Test that setup creates valid U and V grids."""
        solver = TuringPatternSolver(turing_params)
        solver.setup()

        assert solver.U is not None
        assert solver.V is not None
        assert solver.U.shape == (turing_params["grid_size"], turing_params["grid_size"])
        assert solver.V.shape == (turing_params["grid_size"], turing_params["grid_size"])

    def test_state_contains_uv_stack(self, turing_params):
        """Test that state is stacked [U, V]."""
        solver = TuringPatternSolver(turing_params)
        solver.setup()

        assert solver.state.shape == (2, turing_params["grid_size"], turing_params["grid_size"])
        np.testing.assert_array_equal(solver.state[0], solver.U)
        np.testing.assert_array_equal(solver.state[1], solver.V)

    def test_step_updates_time(self, turing_params):
        """Test that step advances simulation time."""
        solver = TuringPatternSolver(turing_params)
        solver.setup()

        dt = 0.1
        solver.step(dt)

        assert np.isclose(solver.time, dt)

    def test_order_parameter_is_std(self, turing_params):
        """Test that order parameter is std(U)."""
        solver = TuringPatternSolver(turing_params)
        solver.setup()

        expected = np.std(solver.U)
        actual = solver.compute_order_parameter()

        assert np.isclose(actual, expected)

    def test_critical_threshold_diffusion_ratio(self, turing_params):
        """Test critical threshold based on diffusion ratio."""
        solver = TuringPatternSolver(turing_params)
        solver.setup()

        expected = np.sqrt(turing_params["D_v"] / turing_params["D_u"])
        actual = solver.get_critical_threshold()

        assert np.isclose(actual, expected)

    def test_pattern_formation_with_turing_instability(self, turing_params):
        """Test that patterns form when diffusion ratio is sufficient."""
        # Ensure Turing instability condition is met
        turing_params["D_v"] = 5e-3  # Fast inhibitor
        turing_params["D_u"] = 2.8e-4  # Slow activator

        solver = TuringPatternSolver(turing_params)
        solver.setup()

        initial_contrast = solver.compute_order_parameter()

        # Run simulation
        for _ in range(500):
            solver.step(0.1)

        final_contrast = solver.compute_order_parameter()

        # Pattern should develop (contrast increases)
        assert final_contrast > initial_contrast

    def test_laplacian_periodic_boundary(self, turing_params):
        """Test that Laplacian respects periodic boundaries."""
        turing_params["grid_size"] = 8
        solver = TuringPatternSolver(turing_params)
        solver.setup()

        # Create simple test pattern
        test_grid = np.zeros((8, 8))
        test_grid[0, 0] = 1.0

        lap = solver._laplacian(test_grid)

        # Check corners are connected via periodic BC
        assert lap[0, 0] != 0  # Center point
        assert lap[-1, 0] != 0  # Connected via periodic BC
        assert lap[0, -1] != 0  # Connected via periodic BC

    def test_get_pattern_returns_copy(self, turing_params):
        """Test that get_pattern returns a copy."""
        solver = TuringPatternSolver(turing_params)
        solver.setup()

        pattern = solver.get_pattern()
        pattern[0, 0] = 999.0

        # Original should be unchanged
        assert solver.U[0, 0] != 999.0

    def test_get_inhibitor_returns_v(self, turing_params):
        """Test that get_inhibitor returns V field."""
        solver = TuringPatternSolver(turing_params)
        solver.setup()

        inhibitor = solver.get_inhibitor()
        np.testing.assert_array_equal(inhibitor, solver.V)

    def test_pattern_energy_computation(self, turing_params):
        """Test pattern energy (total variance) calculation."""
        solver = TuringPatternSolver(turing_params)
        solver.setup()

        expected = np.var(solver.U) + np.var(solver.V)
        actual = solver.pattern_energy()

        assert np.isclose(actual, expected)

    def test_is_patterned_threshold(self, turing_params):
        """Test is_patterned with different thresholds."""
        solver = TuringPatternSolver(turing_params)
        solver.setup()

        contrast = solver.compute_order_parameter()

        # Should be patterned if threshold is below contrast
        assert solver.is_patterned(threshold=contrast - 0.01)
        # Should not be patterned if threshold is above contrast
        assert not solver.is_patterned(threshold=contrast + 0.1)

    def test_set_diffusion_coefficients(self, turing_params):
        """Test updating diffusion coefficients."""
        solver = TuringPatternSolver(turing_params)
        solver.setup()

        new_d_u = 1e-3
        new_d_v = 1e-2
        solver.set_diffusion_coefficients(new_d_u, new_d_v)

        assert solver.D_u == new_d_u
        assert solver.D_v == new_d_v
        assert solver.params["D_u"] == new_d_u
        assert solver.params["D_v"] == new_d_v

    def test_get_pattern_wavelength_positive(self, turing_params):
        """Test that wavelength estimation returns positive value."""
        solver = TuringPatternSolver(turing_params)
        solver.setup()

        # Run to develop pattern
        for _ in range(100):
            solver.step(0.1)

        wavelength = solver.get_pattern_wavelength()
        assert wavelength > 0

    def test_seed_reproducibility(self, turing_params):
        """Test that same seed produces identical results."""
        solver1 = TuringPatternSolver(turing_params)
        solver1.setup()
        for _ in range(50):
            solver1.step(0.1)
        pattern1 = solver1.get_pattern()

        solver2 = TuringPatternSolver(turing_params)
        solver2.setup()
        for _ in range(50):
            solver2.step(0.1)
        pattern2 = solver2.get_pattern()

        np.testing.assert_array_almost_equal(pattern1, pattern2)

    def test_numerical_stability(self, turing_params):
        """Test that simulation remains numerically stable."""
        solver = TuringPatternSolver(turing_params)
        solver.setup()

        # Run many steps
        for _ in range(1000):
            solver.step(0.1)

        # Check for NaN or Inf
        assert not np.any(np.isnan(solver.U))
        assert not np.any(np.isinf(solver.U))
        assert not np.any(np.isnan(solver.V))
        assert not np.any(np.isinf(solver.V))
