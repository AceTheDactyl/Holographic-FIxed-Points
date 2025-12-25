"""Tests for nuclear criticality solver."""

import numpy as np
import pytest

from backend.core.criticality import NuclearCriticalitySolver


class TestNuclearCriticalitySolver:
    """Test suite for NuclearCriticalitySolver."""

    def test_setup_initializes_state(self, criticality_params):
        """Test that setup creates valid initial state."""
        solver = NuclearCriticalitySolver(criticality_params)
        solver.setup()

        assert solver.state is not None
        assert len(solver.state) == 2  # [N, C]
        assert solver.time == 0.0
        assert solver.N == criticality_params["initial_neutrons"]

    def test_k_eff_computation(self, criticality_params):
        """Test k_eff = k_∞ * (1 - leakage) + control."""
        solver = NuclearCriticalitySolver(criticality_params)
        solver.setup()

        k_eff = solver.compute_order_parameter()
        expected = criticality_params["k_infinity"] * (1 - criticality_params["leakage_factor"])

        assert np.isclose(k_eff, expected)

    def test_critical_threshold_is_one(self, criticality_params):
        """Test critical threshold is always 1.0."""
        solver = NuclearCriticalitySolver(criticality_params)
        solver.setup()

        assert solver.get_critical_threshold() == 1.0

    def test_reactivity_formula(self, criticality_params):
        """Test reactivity ρ = (k - 1)/k."""
        solver = NuclearCriticalitySolver(criticality_params)
        solver.setup()

        k_eff = solver.compute_order_parameter()
        rho = solver.get_reactivity()
        expected = (k_eff - 1) / k_eff

        assert np.isclose(rho, expected)

    def test_reactivity_pcm_conversion(self, criticality_params):
        """Test reactivity in pcm (10^5 units)."""
        solver = NuclearCriticalitySolver(criticality_params)
        solver.setup()

        rho = solver.get_reactivity()
        rho_pcm = solver.get_reactivity_pcm()

        assert np.isclose(rho_pcm, rho * 1e5)

    def test_reactivity_dollars_conversion(self, criticality_params):
        """Test reactivity in dollars (ρ/β)."""
        solver = NuclearCriticalitySolver(criticality_params)
        solver.setup()

        rho = solver.get_reactivity()
        rho_dollars = solver.get_reactivity_dollars()
        beta = criticality_params["delayed_fraction"]

        assert np.isclose(rho_dollars, rho / beta)

    def test_supercritical_growth(self, criticality_params):
        """Test neutron population grows when k > 1."""
        # Make system supercritical
        criticality_params["k_infinity"] = 1.05
        criticality_params["leakage_factor"] = 0.01

        solver = NuclearCriticalitySolver(criticality_params)
        solver.setup()

        initial_n = solver.get_neutron_population()

        for _ in range(1000):
            solver.step(0.001)

        final_n = solver.get_neutron_population()

        assert final_n > initial_n

    def test_subcritical_decay(self, criticality_params):
        """Test neutron population decays when k < 1."""
        # Make system subcritical
        criticality_params["k_infinity"] = 0.95
        criticality_params["leakage_factor"] = 0.01

        solver = NuclearCriticalitySolver(criticality_params)
        solver.setup()

        initial_n = solver.get_neutron_population()

        for _ in range(1000):
            solver.step(0.001)

        final_n = solver.get_neutron_population()

        assert final_n < initial_n

    def test_is_critical_detection(self, criticality_params):
        """Test is_critical correctly identifies critical state."""
        # Set to exactly critical
        criticality_params["k_infinity"] = 1.0 / (1 - criticality_params["leakage_factor"])

        solver = NuclearCriticalitySolver(criticality_params)
        solver.setup()

        assert solver.is_critical(tolerance=0.001)

    def test_is_prompt_critical(self, criticality_params):
        """Test prompt criticality detection."""
        # Make highly supercritical (ρ > β)
        criticality_params["k_infinity"] = 1.1
        criticality_params["leakage_factor"] = 0.0

        solver = NuclearCriticalitySolver(criticality_params)
        solver.setup()

        # ρ = (1.1 - 1)/1.1 ≈ 0.091 > β ≈ 0.0065
        assert solver.is_prompt_critical()

    def test_control_rod_insertion(self, criticality_params):
        """Test control rod insertion decreases k_eff."""
        solver = NuclearCriticalitySolver(criticality_params)
        solver.setup()

        initial_k = solver.compute_order_parameter()

        solver.insert_control_rods(amount=0.02)

        final_k = solver.compute_order_parameter()

        assert final_k < initial_k

    def test_control_rod_withdrawal(self, criticality_params):
        """Test control rod withdrawal increases k_eff."""
        solver = NuclearCriticalitySolver(criticality_params)
        solver.setup()

        solver.set_control_rods(-0.02)  # Start with rods inserted
        initial_k = solver.compute_order_parameter()

        solver.withdraw_control_rods(amount=0.02)

        final_k = solver.compute_order_parameter()

        assert final_k > initial_k

    def test_scram_reduces_reactivity(self, criticality_params):
        """Test SCRAM emergency shutdown reduces reactivity."""
        solver = NuclearCriticalitySolver(criticality_params)
        solver.setup()

        initial_rho = solver.get_reactivity()

        solver.scram()

        final_rho = solver.get_reactivity()

        assert final_rho < initial_rho
        assert final_rho < 0  # Should be subcritical

    def test_reactor_period_infinite_at_critical(self, criticality_params):
        """Test period is infinite at exactly critical."""
        # Set to exactly critical
        criticality_params["k_infinity"] = 1.0 / (1 - criticality_params["leakage_factor"])

        solver = NuclearCriticalitySolver(criticality_params)
        solver.setup()

        period = solver.get_period()
        assert period == float("inf")

    def test_precursor_concentration(self, criticality_params):
        """Test delayed neutron precursor concentration is positive."""
        solver = NuclearCriticalitySolver(criticality_params)
        solver.setup()

        C = solver.get_precursor_concentration()
        assert C > 0

    def test_step_preserves_positivity(self, criticality_params):
        """Test neutron population remains positive."""
        solver = NuclearCriticalitySolver(criticality_params)
        solver.setup()

        for _ in range(1000):
            solver.step(0.001)

        assert solver.get_neutron_population() >= 0
        assert solver.get_precursor_concentration() >= 0

    def test_set_control_rods_updates_params(self, criticality_params):
        """Test set_control_rods updates both attribute and params."""
        solver = NuclearCriticalitySolver(criticality_params)
        solver.setup()

        new_worth = -0.05
        solver.set_control_rods(new_worth)

        assert solver.rho_control == new_worth
        assert solver.params["control_rod_worth"] == new_worth

    def test_analyze_fixed_point_includes_metadata(self, criticality_params):
        """Test analyze_fixed_point includes time and params."""
        solver = NuclearCriticalitySolver(criticality_params)
        solver.setup()

        for _ in range(10):
            solver.step(0.001)

        result = solver.analyze_fixed_point()

        assert "time" in result.metadata
        assert "params" in result.metadata
        assert result.metadata["time"] > 0

    def test_run_to_equilibrium(self, criticality_params):
        """Test run_to_equilibrium terminates."""
        # Set near critical
        criticality_params["k_infinity"] = 1.02

        solver = NuclearCriticalitySolver(criticality_params)
        solver.setup()

        result = solver.run_to_equilibrium(dt=0.001, max_steps=5000)

        assert result is not None
        assert result.order_parameter > 0
