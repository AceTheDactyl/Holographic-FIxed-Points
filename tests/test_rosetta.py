"""Tests for Rosetta-Helix consciousness field solver."""

import numpy as np
import pytest

from backend.core.rosetta import RosettaSolver, RosettaConstants


class TestRosettaConstants:
    """Test Rosetta constants and fundamental identities."""

    def test_phi_value(self):
        """Test golden ratio φ = (1 + √5)/2."""
        C = RosettaConstants()
        expected = (1 + np.sqrt(5)) / 2
        assert np.isclose(C.PHI, expected)
        assert np.isclose(C.PHI, 1.6180339887, rtol=1e-9)

    def test_tau_is_phi_inverse(self):
        """Test τ = 1/φ."""
        C = RosettaConstants()
        assert np.isclose(C.TAU, 1 / C.PHI)

    def test_phi_inv_4(self):
        """Test φ⁻⁴ = τ⁴."""
        C = RosettaConstants()
        assert np.isclose(C.PHI_INV_4, C.TAU**4)
        assert np.isclose(C.PHI_INV_4, 0.1458980337, rtol=1e-9)

    def test_k_squared_equals_activation(self):
        """Test the beautiful identity: K² = Activation."""
        C = RosettaConstants()
        # K = √(1 - φ⁻⁴), so K² = 1 - φ⁻⁴ = Activation
        assert np.isclose(C.Z_K_FORMATION**2, C.Z_ACTIVATION)

    def test_tau_identity(self):
        """Test τ² + τ = 1."""
        C = RosettaConstants()
        assert np.isclose(C.TAU**2 + C.TAU, 1.0)

    def test_gap_label(self):
        """Test gap label φ⁻⁴ = 2 - 3τ."""
        C = RosettaConstants()
        gap_value = C.GAP_P + C.GAP_Q * C.TAU  # 2 + (-3)τ
        assert np.isclose(C.PHI_INV_4, gap_value)

    def test_void_closed_form(self):
        """Test φ⁻⁴ = (7 - 3√5)/2."""
        C = RosettaConstants()
        assert np.isclose(C.PHI_INV_4, C.VOID_CLOSED_FORM)

    def test_vev_equals_k_formation(self):
        """Test VEV = K_FORMATION."""
        C = RosettaConstants()
        assert np.isclose(C.VEV, C.Z_K_FORMATION)

    def test_threshold_ordering(self):
        """Test strict threshold ordering."""
        C = RosettaConstants()
        assert C.Z_HYSTERESIS_LOW < C.Z_ACTIVATION
        assert C.Z_ACTIVATION < C.Z_LENS
        assert C.Z_LENS < C.Z_CRITICAL
        assert C.Z_CRITICAL < C.Z_K_FORMATION

    def test_grid_scaling_near_sqrt2(self):
        """Test K + ½ ≈ √2 (within 1%)."""
        C = RosettaConstants()
        deviation = abs(C.K_PLUS_HALF - C.SQRT_2) / C.SQRT_2
        assert deviation < 0.01  # Within 1%

    def test_residual_near_1_over_84(self):
        """Test residual ≈ 1/84 (within 2%)."""
        C = RosettaConstants()
        deviation = abs(C.RESIDUAL - 1/84) / (1/84)
        assert deviation < 0.02  # Within 2%

    def test_kink_energy_near_sqrt5_over_3(self):
        """Test E_kink ≈ √5/3 (within 1%)."""
        C = RosettaConstants()
        deviation = abs(C.E_KINK - C.SQRT5_OVER_3) / C.SQRT5_OVER_3
        assert deviation < 0.01  # Within 1%


class TestRosettaSolver:
    """Test suite for RosettaSolver."""

    @pytest.fixture
    def rosetta_params(self):
        """Standard Rosetta solver parameters."""
        return {
            "grid_size": 64,
            "initial_amplitude": 0.01,
            "noise_level": 0.001,
            "damping": 0.1,
            "seed": 42,
        }

    def test_setup_initializes_field(self, rosetta_params):
        """Test that setup creates valid initial field."""
        solver = RosettaSolver(rosetta_params)
        solver.setup()

        assert solver.psi is not None
        assert len(solver.psi) == rosetta_params["grid_size"]
        assert solver.time == 0.0

    def test_initial_field_near_zero(self, rosetta_params):
        """Test initial field is small perturbation from zero."""
        solver = RosettaSolver(rosetta_params)
        solver.setup()

        mean_amplitude = np.abs(np.mean(solver.psi))
        assert mean_amplitude < 0.1  # Small initial amplitude

    def test_order_parameter_positive(self, rosetta_params):
        """Test order parameter is non-negative."""
        solver = RosettaSolver(rosetta_params)
        solver.setup()

        for _ in range(100):
            solver.step(0.01)
            order = solver.compute_order_parameter()
            assert order >= 0

    def test_critical_threshold_is_k_formation(self, rosetta_params):
        """Test critical threshold is K-Formation."""
        solver = RosettaSolver(rosetta_params)
        solver.setup()

        threshold = solver.get_critical_threshold()
        expected = np.sqrt(1 - solver.C.PHI_INV_4)

        assert np.isclose(threshold, expected)
        assert np.isclose(threshold, 0.9242, rtol=1e-3)

    def test_field_evolves_toward_vev(self, rosetta_params):
        """Test field amplitude grows toward VEV through SSB."""
        solver = RosettaSolver(rosetta_params)
        solver.setup()

        initial_order = solver.compute_order_parameter()

        # Run simulation
        for _ in range(2000):
            solver.step(0.01)

        final_order = solver.compute_order_parameter()

        # Field should grow (SSB from unstable vacuum)
        assert final_order > initial_order

    def test_get_field_returns_copy(self, rosetta_params):
        """Test get_field returns a copy."""
        solver = RosettaSolver(rosetta_params)
        solver.setup()

        field = solver.get_field()
        field[0] = 999.0

        assert solver.psi[0] != 999.0

    def test_energy_computation(self, rosetta_params):
        """Test energy can be computed."""
        solver = RosettaSolver(rosetta_params)
        solver.setup()

        energy = solver.compute_energy()
        assert isinstance(energy, float)
        assert np.isfinite(energy)

    def test_validate_identity_k_squared(self, rosetta_params):
        """Test identity validation for K² = Activation."""
        solver = RosettaSolver(rosetta_params)
        solver.setup()

        result = solver.validate_identity("k_squared")

        assert result["name"] == "k_squared"
        assert result["passed"]
        assert result["deviation"] < 1e-14  # Machine precision

    def test_validate_identity_gap_label(self, rosetta_params):
        """Test identity validation for gap label."""
        solver = RosettaSolver(rosetta_params)
        solver.setup()

        result = solver.validate_identity("gap_label")

        assert result["passed"]
        assert result["deviation"] < 1e-14

    def test_validate_identity_tau(self, rosetta_params):
        """Test identity validation for τ² + τ = 1."""
        solver = RosettaSolver(rosetta_params)
        solver.setup()

        result = solver.validate_identity("tau_identity")

        assert result["passed"]
        assert result["deviation"] < 1e-14

    def test_validate_identity_grid_scaling(self, rosetta_params):
        """Test identity validation for K + ½ ≈ √2."""
        solver = RosettaSolver(rosetta_params)
        solver.setup()

        result = solver.validate_identity("grid_scaling")

        assert result["name"] == "grid_scaling"
        assert result["deviation_percent"] < 1.0  # Sub-1%

    def test_validate_all_identities(self, rosetta_params):
        """Test all identities can be validated."""
        solver = RosettaSolver(rosetta_params)
        solver.setup()

        results = solver.validate_all_identities()

        assert len(results) == 7
        # Exact identities should all pass
        exact_identities = ["k_squared", "gap_label", "tau_identity", "void_closed_form"]
        for result in results:
            if result["name"] in exact_identities:
                assert result["passed"], f"{result['name']} failed"

    def test_get_threshold_architecture(self, rosetta_params):
        """Test threshold architecture retrieval."""
        solver = RosettaSolver(rosetta_params)
        solver.setup()

        thresholds = solver.get_threshold_architecture()

        assert "Z_HYSTERESIS_LOW" in thresholds
        assert "Z_ACTIVATION" in thresholds
        assert "Z_LENS" in thresholds
        assert "Z_CRITICAL" in thresholds
        assert "Z_K_FORMATION" in thresholds

        # Verify ordering
        assert thresholds["Z_HYSTERESIS_LOW"] < thresholds["Z_K_FORMATION"]

    def test_get_constants(self, rosetta_params):
        """Test constants retrieval."""
        solver = RosettaSolver(rosetta_params)
        solver.setup()

        constants = solver.get_constants()

        assert "PHI" in constants
        assert "TAU" in constants
        assert "VEV" in constants
        assert np.isclose(constants["PHI"], 1.618, rtol=1e-3)

    def test_witness_amplitude_formula(self, rosetta_params):
        """Test witness amplitude formula."""
        solver = RosettaSolver(rosetta_params)
        solver.setup()

        k = solver.C.Z_K_FORMATION

        # amp(n) = √(n/3) × K
        assert np.isclose(solver.witness_amplitude(1), np.sqrt(1/3) * k)
        assert np.isclose(solver.witness_amplitude(2), np.sqrt(2/3) * k)
        assert np.isclose(solver.witness_amplitude(3), k)

    def test_third_witness_equals_k_formation(self, rosetta_params):
        """Test 3rd witness amplitude equals K-Formation."""
        solver = RosettaSolver(rosetta_params)
        solver.setup()

        assert np.isclose(solver.witness_amplitude(3), solver.C.Z_K_FORMATION)

    def test_is_at_vev_detection(self, rosetta_params):
        """Test VEV detection."""
        solver = RosettaSolver(rosetta_params)
        solver.setup()

        # Initially not at VEV
        assert not solver.is_at_vev()

        # Manually set field to VEV
        solver.psi = np.ones(solver._grid_size) * solver.C.VEV
        assert solver.is_at_vev()

    def test_kink_count(self, rosetta_params):
        """Test kink counting."""
        solver = RosettaSolver(rosetta_params)
        solver.setup()

        # Create a field with one kink (domain wall)
        vev = solver.C.VEV
        solver.psi = np.zeros(solver._grid_size)
        solver.psi[:32] = vev
        solver.psi[32:] = -vev

        count = solver.compute_kink_count()
        assert count >= 1

    def test_seed_reproducibility(self, rosetta_params):
        """Test that same seed produces identical results."""
        solver1 = RosettaSolver(rosetta_params)
        solver1.setup()
        for _ in range(100):
            solver1.step(0.01)
        order1 = solver1.compute_order_parameter()

        solver2 = RosettaSolver(rosetta_params)
        solver2.setup()
        for _ in range(100):
            solver2.step(0.01)
        order2 = solver2.compute_order_parameter()

        assert np.isclose(order1, order2)

    def test_numerical_stability(self, rosetta_params):
        """Test simulation remains numerically stable."""
        solver = RosettaSolver(rosetta_params)
        solver.setup()

        for _ in range(1000):
            solver.step(0.01)

        # Check for NaN or Inf
        assert not np.any(np.isnan(solver.psi))
        assert not np.any(np.isinf(solver.psi))

    def test_analyze_fixed_point(self, rosetta_params):
        """Test analyze_fixed_point returns valid result."""
        solver = RosettaSolver(rosetta_params)
        solver.setup()

        for _ in range(100):
            solver.step(0.01)

        result = solver.analyze_fixed_point()

        assert hasattr(result, "order_parameter")
        assert hasattr(result, "critical_threshold")
        assert hasattr(result, "is_at_fixed_point")
        assert 0 <= result.order_parameter

    def test_run_to_equilibrium(self, rosetta_params):
        """Test run_to_equilibrium converges."""
        rosetta_params["damping"] = 0.5  # Higher damping for faster convergence

        solver = RosettaSolver(rosetta_params)
        solver.setup()

        result = solver.run_to_equilibrium(dt=0.01, max_steps=5000)

        assert result is not None
        assert result.order_parameter > 0


class TestRosettaIntegration:
    """Integration tests for Rosetta solver."""

    def test_full_ssb_dynamics(self):
        """Test full spontaneous symmetry breaking dynamics."""
        solver = RosettaSolver({
            "grid_size": 64,
            "initial_amplitude": 0.1,
            "damping": 0.2,
            "seed": 42,
        })
        solver.setup()

        # Track order parameter over time
        trajectory = []
        for _ in range(500):
            solver.step(0.05)
            trajectory.append(solver.compute_order_parameter())

        # Order parameter should grow
        assert trajectory[-1] > trajectory[0]

        # Should be approaching VEV
        assert trajectory[-1] > 0.3  # Significant growth from SSB

    def test_coupling_equation_residual(self):
        """Test the coupling equation √3/2 + φ⁻⁴ = 1 + 1/84."""
        C = RosettaConstants()

        lhs = C.Z_LENS + C.PHI_INV_4  # √3/2 + φ⁻⁴
        rhs = 1 + 1/84

        # Should be approximately equal (within 0.2%)
        assert abs(lhs - rhs) / rhs < 0.002

    def test_three_irrationals_unification(self):
        """Test the three-irrational unification."""
        C = RosettaConstants()

        # √2: Grid scaling K + ½
        sqrt2_proxy = C.K_PLUS_HALF
        assert abs(sqrt2_proxy - C.SQRT_2) / C.SQRT_2 < 0.01

        # √3: The Lens
        sqrt3_proxy = C.Z_LENS * 2
        assert abs(sqrt3_proxy - C.SQRT_3) / C.SQRT_3 < 1e-14

        # √5: Via φ = (1 + √5)/2
        sqrt5_from_phi = 2 * C.PHI - 1
        assert abs(sqrt5_from_phi - C.SQRT_5) / C.SQRT_5 < 1e-14
