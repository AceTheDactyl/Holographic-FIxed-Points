"""Tests for Kuramoto synchronization solver."""

import numpy as np
import pytest

from backend.core.kuramoto import KuramotoSolver


class TestKuramotoSolver:
    """Test suite for KuramotoSolver."""

    def test_setup_initializes_state(self, kuramoto_params):
        """Test that setup creates valid initial state."""
        solver = KuramotoSolver(kuramoto_params)
        solver.setup()

        assert solver.state is not None
        assert len(solver.state) == kuramoto_params["n_oscillators"]
        assert solver.time == 0.0
        assert solver.K == kuramoto_params["coupling"]

    def test_phases_in_valid_range(self, kuramoto_params):
        """Test that phases remain in [0, 2π]."""
        solver = KuramotoSolver(kuramoto_params)
        solver.setup()

        for _ in range(100):
            solver.step(0.01)

        phases = solver.get_phases()
        assert np.all(phases >= 0)
        assert np.all(phases <= 2 * np.pi)

    def test_order_parameter_bounds(self, kuramoto_params):
        """Test that order parameter r ∈ [0, 1]."""
        solver = KuramotoSolver(kuramoto_params)
        solver.setup()

        for _ in range(100):
            solver.step(0.01)
            r = solver.compute_order_parameter()
            assert 0 <= r <= 1

    def test_critical_threshold_formula(self, kuramoto_params):
        """Test K_c = sqrt(8/π) * σ formula."""
        solver = KuramotoSolver(kuramoto_params)
        solver.setup()

        expected_kc = np.sqrt(8 / np.pi) * kuramoto_params["frequency_std"]
        actual_kc = solver.get_critical_threshold()

        assert np.isclose(actual_kc, expected_kc)

    def test_synchronization_above_critical_coupling(self):
        """Test that oscillators synchronize when K > K_c."""
        params = {
            "n_oscillators": 100,
            "coupling": 3.0,  # Well above K_c ≈ 1.596
            "frequency_std": 1.0,
            "seed": 42,
        }
        solver = KuramotoSolver(params)
        solver.setup()

        # Run to equilibrium
        result = solver.run_to_equilibrium(dt=0.01, max_steps=5000)

        # Should have significant synchronization
        assert result.order_parameter > 0.5

    def test_no_synchronization_below_critical_coupling(self):
        """Test that oscillators don't synchronize when K < K_c."""
        params = {
            "n_oscillators": 100,
            "coupling": 0.5,  # Well below K_c ≈ 1.596
            "frequency_std": 1.0,
            "seed": 42,
        }
        solver = KuramotoSolver(params)
        solver.setup()

        # Run simulation
        for _ in range(1000):
            solver.step(0.01)

        r = solver.compute_order_parameter()
        # Order parameter should be low (with some finite-size fluctuations)
        assert r < 0.5

    def test_set_coupling_updates_k(self, kuramoto_params):
        """Test that set_coupling correctly updates K."""
        solver = KuramotoSolver(kuramoto_params)
        solver.setup()

        new_k = 5.0
        solver.set_coupling(new_k)

        assert solver.K == new_k
        assert solver.params["coupling"] == new_k

    def test_iterate_yields_results(self, kuramoto_params):
        """Test that iterate generator yields FixedPointResult objects."""
        solver = KuramotoSolver(kuramoto_params)
        solver.setup()

        results = list(solver.iterate(steps=10, dt=0.01))

        assert len(results) == 10
        for result in results:
            assert hasattr(result, "order_parameter")
            assert hasattr(result, "is_at_fixed_point")

    def test_reset_clears_state(self, kuramoto_params):
        """Test that reset clears solver state."""
        solver = KuramotoSolver(kuramoto_params)
        solver.setup()

        for _ in range(100):
            solver.step(0.01)

        solver.reset()

        assert solver.state is None
        assert solver.time == 0.0

    def test_mean_phase_computation(self, kuramoto_params):
        """Test mean phase is in valid range."""
        solver = KuramotoSolver(kuramoto_params)
        solver.setup()

        psi = solver.compute_mean_phase()
        assert -np.pi <= psi <= np.pi

    def test_locked_oscillators_detection(self):
        """Test locked oscillators detection for synchronized system."""
        params = {
            "n_oscillators": 50,
            "coupling": 4.0,  # Strong coupling
            "frequency_std": 0.5,
            "seed": 42,
        }
        solver = KuramotoSolver(params)
        solver.setup()

        # Run to equilibrium
        solver.run_to_equilibrium(dt=0.01, max_steps=5000)

        locked = solver.locked_oscillators(tolerance=0.3)
        # Should have many locked oscillators with strong coupling
        assert np.sum(locked) > params["n_oscillators"] * 0.5

    def test_seed_reproducibility(self, kuramoto_params):
        """Test that same seed produces identical results."""
        solver1 = KuramotoSolver(kuramoto_params)
        solver1.setup()
        for _ in range(100):
            solver1.step(0.01)
        r1 = solver1.compute_order_parameter()

        solver2 = KuramotoSolver(kuramoto_params)
        solver2.setup()
        for _ in range(100):
            solver2.step(0.01)
        r2 = solver2.compute_order_parameter()

        assert np.isclose(r1, r2)

    def test_full_step_matches_mean_field(self, kuramoto_params):
        """Test that full O(N²) step gives similar results to mean-field."""
        kuramoto_params["n_oscillators"] = 20  # Small N for O(N²)

        solver_mf = KuramotoSolver(kuramoto_params)
        solver_mf.setup()

        solver_full = KuramotoSolver(kuramoto_params)
        solver_full.setup()
        # Copy initial state
        solver_full._state = solver_mf._state.copy()
        solver_full.natural_freqs = solver_mf.natural_freqs.copy()

        # Run both
        for _ in range(50):
            solver_mf.step(0.01)
            solver_full.step_full(0.01)

        # Results should be similar (not exact due to numerical differences)
        r_mf = solver_mf.compute_order_parameter()
        r_full = solver_full.compute_order_parameter()
        assert np.abs(r_mf - r_full) < 0.2
