/**
 * Shared type definitions for APL Holographic Fixed Points.
 *
 * These types are used by both the Python backend (via JSON) and
 * the TypeScript frontend for type-safe data handling.
 */

// =============================================================================
// Core Data Types
// =============================================================================

/**
 * Result from fixed point analysis.
 */
export interface FixedPointResult {
  order_parameter: number;
  critical_threshold: number;
  is_at_fixed_point: boolean;
  state_vector: number[];
  metadata: Record<string, unknown>;
}

/**
 * Trajectory point for time series visualization.
 */
export interface TrajectoryPoint {
  time: number;
  order_parameter: number;
  is_critical: boolean;
}

/**
 * Generic simulation result.
 */
export interface SimulationResult {
  solver: SolverType;
  critical_threshold: number;
  final_state: FixedPointResult;
  trajectory: TrajectoryPoint[];
}

// =============================================================================
// Solver Types
// =============================================================================

export type SolverType = 'kuramoto' | 'bekenstein' | 'turing' | 'criticality' | 'rosetta';

export interface KuramotoParams {
  n_oscillators?: number;
  coupling?: number;
  frequency_std?: number;
  frequency_mean?: number;
  seed?: number;
}

export interface BekensteinParams {
  mass_kg?: number;
  radius_m?: number;
  include_hawking?: boolean;
  hawking_rate?: number;
}

export interface TuringParams {
  grid_size?: number;
  D_u?: number;
  D_v?: number;
  tau?: number;
  k?: number;
  seed?: number;
}

export interface CriticalityParams {
  initial_neutrons?: number;
  k_infinity?: number;
  leakage_factor?: number;
  delayed_fraction?: number;
  decay_constant?: number;
  prompt_lifetime?: number;
  control_rod_worth?: number;
}

export interface RosettaParams {
  grid_size?: number;
  initial_amplitude?: number;
  noise_level?: number;
  damping?: number;
  seed?: number;
}

export type SolverParams =
  | KuramotoParams
  | BekensteinParams
  | TuringParams
  | CriticalityParams
  | RosettaParams;

// =============================================================================
// Pre-computed Data Types
// =============================================================================

/**
 * Kuramoto phase diagram data.
 */
export interface KuramotoPhaseDiagram {
  K: number[];
  data: Record<string, { r_mean: number[]; r_std: number[] }>;
  K_critical: number;
  description: string;
}

/**
 * Turing pattern snapshot.
 */
export interface TuringSnapshot {
  time: number;
  pattern: number[][];
  contrast: number;
  wavelength?: number;
}

/**
 * Turing patterns dataset.
 */
export interface TuringPatternsData {
  snapshots: TuringSnapshot[];
  grid_size: number;
  original_size: number;
  parameters: TuringParams;
  critical_ratio: number;
  description: string;
}

/**
 * Bekenstein entropy data point.
 */
export interface BekensteinDataPoint {
  mass_kg: number;
  mass_log10: number;
  mass_solar: number;
  entropy_planck: number;
  entropy_log10: number;
  entropy_bits: number;
  bekenstein_bound: number;
  schwarzschild_radius_m: number;
  hawking_temp_K: number;
  evaporation_time_s: number;
}

/**
 * Bekenstein entropy dataset.
 */
export interface BekensteinEntropyData {
  data: BekensteinDataPoint[];
  notable_objects: { name: string; mass_solar: number }[];
  description: string;
  units: Record<string, string>;
}

/**
 * Criticality response curve.
 */
export interface CriticalityCurve {
  name: string;
  k_infinity: number;
  k_eff: number;
  trajectory: {
    time: number;
    neutrons: number;
    k_eff: number;
    reactivity_dollars: number;
  }[];
}

/**
 * Criticality response dataset.
 */
export interface CriticalityResponseData {
  curves: CriticalityCurve[];
  description: string;
}

// =============================================================================
// Manifest Types
// =============================================================================

export interface DatasetInfo {
  id: string;
  name: string;
  file: string;
  description: string;
  category: string;
}

export interface FixedPointInfo {
  name: string;
  type: string;
  formula: string;
  value: number | null;
  unit: string;
}

export interface CategoryInfo {
  name: string;
  color: string;
  description: string;
}

export interface Manifest {
  generated: string;
  version: string;
  datasets: DatasetInfo[];
  fixed_points: FixedPointInfo[];
  categories: Record<string, CategoryInfo>;
}

// =============================================================================
// Visualization Types
// =============================================================================

export interface VisualizationConfig {
  colorScale: string;
  showPhaseSpace: boolean;
  animationSpeed: number;
  darkMode: boolean;
}

export interface Point2D {
  x: number;
  y: number;
}

export interface Point3D extends Point2D {
  z: number;
}

/**
 * Color definition with optional alpha.
 */
export interface Color {
  r: number;
  g: number;
  b: number;
  a?: number;
}

// =============================================================================
// API Types
// =============================================================================

export interface ApiError {
  error: string;
}

export interface ApiResponse<T> {
  data?: T;
  error?: string;
}

/**
 * Kuramoto sweep result.
 */
export interface KuramotoSweepResult {
  sweep: { K: number; r_mean: number; r_std: number }[];
  K_critical: number;
  parameters: KuramotoParams;
}

/**
 * Turing evolution result.
 */
export interface TuringEvolutionResult {
  snapshots: TuringSnapshot[];
  grid_size: number;
  critical_ratio: number;
}

// =============================================================================
// Rosetta-Helix Types
// =============================================================================

/**
 * Rosetta constants derived from φ, √2, √3, √5.
 */
export interface RosettaConstants {
  PHI: number;
  TAU: number;
  PHI_INV_4: number;
  PHI_INV_7: number;
  SQRT_2: number;
  SQRT_3: number;
  SQRT_5: number;
  M_SQUARED: number;
  VEV: number;
  E_KINK: number;
}

/**
 * Rosetta threshold architecture.
 */
export interface RosettaThresholds {
  Z_HYSTERESIS_LOW: number;
  Z_ACTIVATION: number;
  Z_LENS: number;
  Z_CRITICAL: number;
  Z_K_FORMATION: number;
}

/**
 * Rosetta identity validation result.
 */
export interface RosettaIdentityResult {
  name: string;
  formula: string;
  expected: number;
  actual: number;
  deviation: number;
  deviation_percent: number;
  passed: boolean;
}

/**
 * Rosetta validation response.
 */
export interface RosettaValidationResult {
  identities: RosettaIdentityResult[];
  all_passed: boolean;
  thresholds: RosettaThresholds;
  constants: RosettaConstants;
}

/**
 * Rosetta equation metadata.
 */
export interface RosettaEquation {
  lagrangian: string;
  vev: number;
  m_squared: number;
  coupling: string;
}

/**
 * Rosetta constants endpoint response.
 */
export interface RosettaConstantsResponse {
  constants: RosettaConstants;
  thresholds: RosettaThresholds;
  witnesses: Record<string, number>;
  equation: RosettaEquation;
}
