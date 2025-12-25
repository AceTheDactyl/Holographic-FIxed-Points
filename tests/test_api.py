"""Tests for Flask API endpoints."""

import json
import pytest
from pathlib import Path

from backend.api.app import create_app


@pytest.fixture
def app():
    """Create test application."""
    app = create_app()
    app.config["TESTING"] = True
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


class TestSimulationsAPI:
    """Test simulation endpoints."""

    def test_list_solvers(self, client):
        """Test GET /api/v1/simulations/ returns solver list."""
        response = client.get("/api/v1/simulations/")
        assert response.status_code == 200

        data = response.get_json()
        assert "solvers" in data
        assert "kuramoto" in data["solvers"]
        assert "bekenstein" in data["solvers"]
        assert "turing" in data["solvers"]
        assert "criticality" in data["solvers"]

    def test_run_kuramoto_simulation(self, client):
        """Test POST /api/v1/simulations/kuramoto runs simulation."""
        response = client.post(
            "/api/v1/simulations/kuramoto",
            json={
                "steps": 100,
                "dt": 0.01,
                "n_oscillators": 20,
                "coupling": 2.0,
                "seed": 42,
            },
            content_type="application/json",
        )
        assert response.status_code == 200

        data = response.get_json()
        assert data["solver"] == "kuramoto"
        assert "critical_threshold" in data
        assert "trajectory" in data
        assert "final_state" in data

    def test_run_bekenstein_simulation(self, client):
        """Test POST /api/v1/simulations/bekenstein."""
        response = client.post(
            "/api/v1/simulations/bekenstein",
            json={
                "steps": 10,
                "dt": 1.0,
                "mass_kg": 1e30,
            },
            content_type="application/json",
        )
        assert response.status_code == 200

        data = response.get_json()
        assert data["solver"] == "bekenstein"
        assert data["critical_threshold"] > 0

    def test_run_turing_simulation(self, client):
        """Test POST /api/v1/simulations/turing."""
        response = client.post(
            "/api/v1/simulations/turing",
            json={
                "steps": 50,
                "dt": 0.1,
                "grid_size": 16,
                "seed": 42,
            },
            content_type="application/json",
        )
        assert response.status_code == 200

        data = response.get_json()
        assert data["solver"] == "turing"

    def test_run_criticality_simulation(self, client):
        """Test POST /api/v1/simulations/criticality."""
        response = client.post(
            "/api/v1/simulations/criticality",
            json={
                "steps": 100,
                "dt": 0.001,
                "k_infinity": 1.03,
            },
            content_type="application/json",
        )
        assert response.status_code == 200

        data = response.get_json()
        assert data["solver"] == "criticality"

    def test_unknown_solver_returns_400(self, client):
        """Test unknown solver returns 400 error."""
        response = client.post(
            "/api/v1/simulations/unknown_solver",
            json={},
            content_type="application/json",
        )
        assert response.status_code == 400

        data = response.get_json()
        assert "error" in data

    def test_kuramoto_sweep(self, client):
        """Test POST /api/v1/simulations/kuramoto/sweep."""
        response = client.post(
            "/api/v1/simulations/kuramoto/sweep",
            json={
                "K_min": 0.5,
                "K_max": 2.5,
                "K_steps": 5,
                "equilibration_steps": 100,
                "sample_steps": 10,
                "n_oscillators": 20,
                "seed": 42,
            },
            content_type="application/json",
        )
        assert response.status_code == 200

        data = response.get_json()
        assert "sweep" in data
        assert len(data["sweep"]) == 5
        assert "K_critical" in data

    def test_turing_evolve(self, client):
        """Test POST /api/v1/simulations/turing/evolve."""
        response = client.post(
            "/api/v1/simulations/turing/evolve",
            json={
                "grid_size": 8,
                "total_time": 10.0,
                "snapshot_times": [0, 5, 10],
                "dt": 0.1,
                "seed": 42,
            },
            content_type="application/json",
        )
        assert response.status_code == 200

        data = response.get_json()
        assert "snapshots" in data
        assert len(data["snapshots"]) >= 2  # At least initial and final

    def test_criticality_transient(self, client):
        """Test POST /api/v1/simulations/criticality/transient."""
        response = client.post(
            "/api/v1/simulations/criticality/transient",
            json={
                "duration": 1.0,
                "dt": 0.01,
                "k_infinity": 1.03,
                "rod_changes": [{"time": 0.5, "worth": -0.01}],
            },
            content_type="application/json",
        )
        assert response.status_code == 200

        data = response.get_json()
        assert "trajectory" in data
        assert "final_k_eff" in data

    def test_trajectory_has_order_parameter(self, client):
        """Test trajectory includes order parameter values."""
        response = client.post(
            "/api/v1/simulations/kuramoto",
            json={
                "steps": 100,
                "dt": 0.01,
                "sample_rate": 10,
                "n_oscillators": 20,
                "seed": 42,
            },
            content_type="application/json",
        )
        assert response.status_code == 200

        data = response.get_json()
        for point in data["trajectory"]:
            assert "time" in point
            assert "order_parameter" in point
            assert 0 <= point["order_parameter"] <= 1


class TestPrecomputedDataAPI:
    """Test pre-computed data endpoints."""

    def test_get_manifest_not_found(self, client):
        """Test manifest returns 404 when not present."""
        response = client.get("/api/v1/data/manifest")
        # Will be 404 if no data generated, 200 if data exists
        assert response.status_code in [200, 404]

    def test_get_dataset_not_found(self, client):
        """Test unknown dataset returns 404."""
        response = client.get("/api/v1/data/nonexistent_dataset")
        assert response.status_code == 404

        data = response.get_json()
        assert "error" in data
