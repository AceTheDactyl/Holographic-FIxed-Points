/**
 * Zustand state management for simulation parameters and data.
 *
 * This store manages:
 * - Active visualization/dataset selection
 * - Solver parameters for each physics model
 * - Loaded pre-computed data cache
 * - Visualization settings
 */

import { create } from 'zustand';
import type {
  SolverType,
  KuramotoParams,
  TuringParams,
  BekensteinParams,
  CriticalityParams,
  KuramotoPhaseDiagram,
  TuringPatternsData,
  BekensteinEntropyData,
  CriticalityResponseData,
  Manifest,
} from '@shared/types';

// =============================================================================
// State Interface
// =============================================================================

interface SimulationState {
  // Active view
  activeView: SolverType;
  isLoading: boolean;
  error: string | null;

  // Kuramoto parameters
  kuramoto: KuramotoParams;

  // Turing parameters
  turing: TuringParams;

  // Bekenstein parameters
  bekenstein: BekensteinParams;

  // Criticality parameters
  criticality: CriticalityParams;

  // Visualization settings
  colorScale: string;
  showPhaseSpace: boolean;
  animationSpeed: number;
  darkMode: boolean;

  // Loaded data cache
  manifest: Manifest | null;
  kuramotoData: KuramotoPhaseDiagram | null;
  turingData: TuringPatternsData | null;
  bekensteinData: BekensteinEntropyData | null;
  criticalityData: CriticalityResponseData | null;

  // Current simulation results
  currentOrderParameter: number;
  currentCriticalThreshold: number;
  isAtFixedPoint: boolean;

  // Actions
  setActiveView: (view: SolverType) => void;
  setKuramotoParams: (params: Partial<KuramotoParams>) => void;
  setTuringParams: (params: Partial<TuringParams>) => void;
  setBekensteinParams: (params: Partial<BekensteinParams>) => void;
  setCriticalityParams: (params: Partial<CriticalityParams>) => void;
  setVisualizationSettings: (settings: Partial<VisualizationSettings>) => void;
  loadManifest: () => Promise<void>;
  loadDataset: (datasetId: string) => Promise<void>;
  setSimulationResults: (order: number, threshold: number, atFixed: boolean) => void;
  setError: (error: string | null) => void;
  reset: () => void;
}

interface VisualizationSettings {
  colorScale: string;
  showPhaseSpace: boolean;
  animationSpeed: number;
  darkMode: boolean;
}

// =============================================================================
// Default Values
// =============================================================================

const defaultKuramotoParams: KuramotoParams = {
  n_oscillators: 100,
  coupling: 1.5,
  frequency_std: 1.0,
  frequency_mean: 0.0,
};

const defaultTuringParams: TuringParams = {
  grid_size: 64,
  D_u: 2.8e-4,
  D_v: 5e-3,
  tau: 0.1,
  k: -0.005,
};

const defaultBekensteinParams: BekensteinParams = {
  mass_kg: 1e30,
  radius_m: 1e6,
  include_hawking: false,
};

const defaultCriticalityParams: CriticalityParams = {
  initial_neutrons: 1e6,
  k_infinity: 1.03,
  leakage_factor: 0.02,
  delayed_fraction: 0.0065,
  control_rod_worth: 0.0,
};

// =============================================================================
// Store Implementation
// =============================================================================

export const useSimulationStore = create<SimulationState>((set, get) => ({
  // Initial state
  activeView: 'kuramoto',
  isLoading: false,
  error: null,

  kuramoto: defaultKuramotoParams,
  turing: defaultTuringParams,
  bekenstein: defaultBekensteinParams,
  criticality: defaultCriticalityParams,

  colorScale: 'viridis',
  showPhaseSpace: true,
  animationSpeed: 1.0,
  darkMode: true,

  manifest: null,
  kuramotoData: null,
  turingData: null,
  bekensteinData: null,
  criticalityData: null,

  currentOrderParameter: 0,
  currentCriticalThreshold: Math.sqrt(8 / Math.PI),
  isAtFixedPoint: false,

  // Actions
  setActiveView: (view) => set({ activeView: view }),

  setKuramotoParams: (params) =>
    set((state) => ({
      kuramoto: { ...state.kuramoto, ...params },
    })),

  setTuringParams: (params) =>
    set((state) => ({
      turing: { ...state.turing, ...params },
    })),

  setBekensteinParams: (params) =>
    set((state) => ({
      bekenstein: { ...state.bekenstein, ...params },
    })),

  setCriticalityParams: (params) =>
    set((state) => ({
      criticality: { ...state.criticality, ...params },
    })),

  setVisualizationSettings: (settings) =>
    set((state) => ({
      colorScale: settings.colorScale ?? state.colorScale,
      showPhaseSpace: settings.showPhaseSpace ?? state.showPhaseSpace,
      animationSpeed: settings.animationSpeed ?? state.animationSpeed,
      darkMode: settings.darkMode ?? state.darkMode,
    })),

  loadManifest: async () => {
    set({ isLoading: true, error: null });
    try {
      const response = await fetch('./data/manifest.json');
      if (!response.ok) throw new Error('Failed to load manifest');
      const manifest = await response.json();
      set({ manifest, isLoading: false });
    } catch (error) {
      set({
        error: error instanceof Error ? error.message : 'Unknown error',
        isLoading: false,
      });
    }
  },

  loadDataset: async (datasetId: string) => {
    const state = get();
    set({ isLoading: true, error: null });

    try {
      const response = await fetch(`./data/${datasetId}.json`);
      if (!response.ok) throw new Error(`Failed to load ${datasetId}`);
      const data = await response.json();

      // Store in appropriate cache based on dataset ID
      if (datasetId === 'kuramoto_phase_diagram') {
        set({ kuramotoData: data, isLoading: false });
      } else if (datasetId === 'turing_patterns') {
        set({ turingData: data, isLoading: false });
      } else if (datasetId === 'bekenstein_entropy') {
        set({ bekensteinData: data, isLoading: false });
      } else if (datasetId === 'criticality_response') {
        set({ criticalityData: data, isLoading: false });
      } else {
        set({ isLoading: false });
      }
    } catch (error) {
      set({
        error: error instanceof Error ? error.message : 'Unknown error',
        isLoading: false,
      });
    }
  },

  setSimulationResults: (order, threshold, atFixed) =>
    set({
      currentOrderParameter: order,
      currentCriticalThreshold: threshold,
      isAtFixedPoint: atFixed,
    }),

  setError: (error) => set({ error }),

  reset: () =>
    set({
      kuramoto: defaultKuramotoParams,
      turing: defaultTuringParams,
      bekenstein: defaultBekensteinParams,
      criticality: defaultCriticalityParams,
      currentOrderParameter: 0,
      isAtFixedPoint: false,
      error: null,
    }),
}));

// =============================================================================
// Selectors
// =============================================================================

export const selectActiveParams = (state: SimulationState): Record<string, unknown> => {
  switch (state.activeView) {
    case 'kuramoto':
      return state.kuramoto;
    case 'turing':
      return state.turing;
    case 'bekenstein':
      return state.bekenstein;
    case 'criticality':
      return state.criticality;
    default:
      return {};
  }
};

export const selectCriticalThreshold = (state: SimulationState): number => {
  switch (state.activeView) {
    case 'kuramoto':
      return Math.sqrt(8 / Math.PI) * (state.kuramoto.frequency_std ?? 1.0);
    case 'criticality':
      return 1.0;
    case 'turing':
      return Math.sqrt((state.turing.D_v ?? 5e-3) / (state.turing.D_u ?? 2.8e-4));
    default:
      return state.currentCriticalThreshold;
  }
};
