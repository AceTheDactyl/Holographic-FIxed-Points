"""Pydantic models for API request/response validation."""

from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator


class SimulationRequest(BaseModel):
    """Base request for running a simulation."""

    steps: int = Field(default=1000, ge=1, le=100000, description="Number of simulation steps")
    dt: float = Field(default=0.01, gt=0, le=1.0, description="Time step size")
    sample_rate: int = Field(default=100, ge=1, le=10000, description="Number of trajectory samples")
    seed: Optional[int] = Field(default=None, ge=0, description="Random seed for reproducibility")


class KuramotoRequest(SimulationRequest):
    """Request for Kuramoto oscillator simulation."""

    n_oscillators: int = Field(default=100, ge=2, le=10000, description="Number of oscillators")
    coupling: float = Field(default=1.0, ge=0, le=100.0, description="Coupling strength K")
    frequency_std: float = Field(default=1.0, gt=0, le=10.0, description="Frequency distribution std")
    frequency_mean: float = Field(default=0.0, ge=-10.0, le=10.0, description="Mean frequency")


class KuramotoSweepRequest(BaseModel):
    """Request for Kuramoto coupling sweep."""

    K_min: float = Field(default=0.1, ge=0, le=10.0, description="Minimum coupling")
    K_max: float = Field(default=5.0, gt=0, le=100.0, description="Maximum coupling")
    K_steps: int = Field(default=50, ge=2, le=1000, description="Number of K values")
    equilibration_steps: int = Field(default=1000, ge=100, le=100000, description="Steps to equilibrate")
    sample_steps: int = Field(default=100, ge=10, le=10000, description="Steps to sample")
    n_oscillators: int = Field(default=100, ge=2, le=10000, description="Number of oscillators")
    frequency_std: float = Field(default=1.0, gt=0, le=10.0, description="Frequency std")
    seed: Optional[int] = Field(default=None, ge=0)

    @field_validator("K_max")
    @classmethod
    def k_max_greater_than_min(cls, v: float, info) -> float:
        if "K_min" in info.data and v <= info.data["K_min"]:
            raise ValueError("K_max must be greater than K_min")
        return v


class TuringRequest(SimulationRequest):
    """Request for Turing pattern simulation."""

    grid_size: int = Field(default=100, ge=8, le=512, description="Grid side length")
    D_u: float = Field(default=2.8e-4, gt=0, le=1.0, description="Activator diffusion")
    D_v: float = Field(default=5e-3, gt=0, le=1.0, description="Inhibitor diffusion")
    tau: float = Field(default=0.1, gt=0, le=10.0, description="Time scale ratio")
    k: float = Field(default=-0.005, ge=-1.0, le=1.0, description="Reaction parameter")
    perturbation: float = Field(default=0.05, ge=0, le=1.0, description="Initial perturbation scale")


class TuringEvolveRequest(BaseModel):
    """Request for Turing pattern evolution with snapshots."""

    grid_size: int = Field(default=64, ge=8, le=256, description="Grid side length")
    dt: float = Field(default=0.1, gt=0, le=1.0, description="Time step")
    total_time: float = Field(default=100.0, gt=0, le=10000.0, description="Total simulation time")
    snapshot_times: list[float] = Field(default=[0, 10, 50, 100], description="Times to capture snapshots")
    D_u: float = Field(default=2.8e-4, gt=0, le=1.0)
    D_v: float = Field(default=5e-3, gt=0, le=1.0)
    seed: Optional[int] = Field(default=None, ge=0)


class BekensteinRequest(SimulationRequest):
    """Request for Bekenstein entropy simulation."""

    mass_kg: float = Field(default=1.0, gt=0, le=1e45, description="System mass in kg")
    radius_m: float = Field(default=1.0, gt=0, le=1e30, description="Bounding radius in m")
    include_hawking: bool = Field(default=False, description="Include Hawking evaporation")
    hawking_rate: float = Field(default=1e-20, ge=0, le=1.0, description="Evaporation rate coefficient")


class CriticalityRequest(SimulationRequest):
    """Request for nuclear criticality simulation."""

    initial_neutrons: float = Field(default=1e6, gt=0, le=1e20, description="Initial neutron population")
    k_infinity: float = Field(default=1.03, gt=0.5, le=2.0, description="Infinite medium k")
    leakage_factor: float = Field(default=0.02, ge=0, le=0.5, description="Geometric leakage")
    delayed_fraction: float = Field(default=0.0065, gt=0, le=0.1, description="Delayed neutron fraction β")
    decay_constant: float = Field(default=0.08, gt=0, le=1.0, description="Precursor decay λ")
    prompt_lifetime: float = Field(default=1e-4, gt=0, le=1.0, description="Prompt neutron lifetime")
    control_rod_worth: float = Field(default=0.0, ge=-0.5, le=0.5, description="Control rod reactivity")


class RodChange(BaseModel):
    """Control rod change event."""

    time: float = Field(ge=0, description="Time of rod change")
    worth: float = Field(ge=-0.5, le=0.5, description="New rod worth")


class CriticalityTransientRequest(BaseModel):
    """Request for criticality transient simulation."""

    duration: float = Field(default=10.0, gt=0, le=1000.0, description="Simulation duration")
    dt: float = Field(default=0.001, gt=0, le=0.1, description="Time step")
    k_infinity: float = Field(default=1.03, gt=0.5, le=2.0, description="Infinite medium k")
    leakage_factor: float = Field(default=0.02, ge=0, le=0.5, description="Geometric leakage")
    delayed_fraction: float = Field(default=0.0065, gt=0, le=0.1, description="Delayed neutron fraction")
    rod_changes: list[RodChange] = Field(default=[], description="List of rod change events")


class RosettaRequest(SimulationRequest):
    """Request for Rosetta-Helix consciousness field simulation."""

    grid_size: int = Field(default=100, ge=16, le=512, description="Number of spatial points")
    initial_amplitude: float = Field(default=0.01, gt=0, le=1.0, description="Initial field amplitude")
    noise_level: float = Field(default=0.001, ge=0, le=0.1, description="Random perturbation strength")
    damping: float = Field(default=0.1, gt=0, le=1.0, description="Dissipation coefficient")


class RosettaValidateRequest(BaseModel):
    """Request for Rosetta identity validation."""

    identities: list[str] = Field(
        default=["k_squared", "gap_label", "tau_identity", "void_closed_form",
                 "grid_scaling", "residual_84", "kink_energy"],
        description="List of identity names to validate"
    )


# Response models


class TrajectoryPoint(BaseModel):
    """Single point in simulation trajectory."""

    time: float
    order_parameter: float
    is_critical: bool = False


class SimulationResponse(BaseModel):
    """Response from simulation endpoint."""

    solver: str
    critical_threshold: float
    final_state: dict[str, Any]
    trajectory: list[TrajectoryPoint]


class SweepResult(BaseModel):
    """Result from parameter sweep."""

    K: float
    r_mean: float
    r_std: float


class KuramotoSweepResponse(BaseModel):
    """Response from Kuramoto sweep endpoint."""

    sweep: list[SweepResult]
    K_critical: float
    parameters: dict[str, Any]


class TuringSnapshot(BaseModel):
    """Single snapshot of Turing pattern."""

    time: float
    pattern: list[list[float]]
    contrast: float


class TuringEvolveResponse(BaseModel):
    """Response from Turing evolve endpoint."""

    snapshots: list[TuringSnapshot]
    grid_size: int
    critical_ratio: float


class CriticalityTrajectoryPoint(BaseModel):
    """Point in criticality trajectory."""

    time: float
    k_eff: float
    neutrons: float
    reactivity_dollars: float


class CriticalityTransientResponse(BaseModel):
    """Response from criticality transient endpoint."""

    trajectory: list[CriticalityTrajectoryPoint]
    final_k_eff: float
    final_neutrons: float


class RosettaIdentityResult(BaseModel):
    """Result of a Rosetta identity validation."""

    name: str
    formula: str
    expected: float
    actual: float
    deviation: float
    deviation_percent: float
    passed: bool


class RosettaValidateResponse(BaseModel):
    """Response from Rosetta validation endpoint."""

    identities: list[RosettaIdentityResult]
    all_passed: bool
    thresholds: dict[str, float]
    constants: dict[str, float]


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
