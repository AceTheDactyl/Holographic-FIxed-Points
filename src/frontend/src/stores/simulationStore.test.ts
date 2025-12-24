/**
 * Tests for simulation store.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { useSimulationStore, selectActiveParams, selectCriticalThreshold } from './simulationStore';

describe('SimulationStore', () => {
  beforeEach(() => {
    // Reset the store before each test
    useSimulationStore.getState().reset();
  });

  describe('Initial State', () => {
    it('should have correct initial active view', () => {
      const state = useSimulationStore.getState();
      expect(state.activeView).toBe('kuramoto');
    });

    it('should have correct initial Kuramoto params', () => {
      const state = useSimulationStore.getState();
      expect(state.kuramoto.n_oscillators).toBe(100);
      expect(state.kuramoto.coupling).toBe(1.5);
      expect(state.kuramoto.frequency_std).toBe(1.0);
    });

    it('should have correct initial Turing params', () => {
      const state = useSimulationStore.getState();
      expect(state.turing.grid_size).toBe(64);
      expect(state.turing.D_u).toBe(2.8e-4);
      expect(state.turing.D_v).toBe(5e-3);
    });

    it('should have correct initial criticality params', () => {
      const state = useSimulationStore.getState();
      expect(state.criticality.k_infinity).toBe(1.03);
      expect(state.criticality.leakage_factor).toBe(0.02);
    });

    it('should not be loading initially', () => {
      const state = useSimulationStore.getState();
      expect(state.isLoading).toBe(false);
    });

    it('should have no error initially', () => {
      const state = useSimulationStore.getState();
      expect(state.error).toBeNull();
    });
  });

  describe('Actions', () => {
    it('setActiveView should update the active view', () => {
      useSimulationStore.getState().setActiveView('turing');
      expect(useSimulationStore.getState().activeView).toBe('turing');

      useSimulationStore.getState().setActiveView('bekenstein');
      expect(useSimulationStore.getState().activeView).toBe('bekenstein');
    });

    it('setKuramotoParams should update Kuramoto params partially', () => {
      useSimulationStore.getState().setKuramotoParams({ coupling: 3.0 });
      const state = useSimulationStore.getState();
      expect(state.kuramoto.coupling).toBe(3.0);
      // Other params should remain unchanged
      expect(state.kuramoto.n_oscillators).toBe(100);
    });

    it('setTuringParams should update Turing params partially', () => {
      useSimulationStore.getState().setTuringParams({ grid_size: 128 });
      const state = useSimulationStore.getState();
      expect(state.turing.grid_size).toBe(128);
      expect(state.turing.D_u).toBe(2.8e-4);
    });

    it('setBekensteinParams should update Bekenstein params', () => {
      useSimulationStore.getState().setBekensteinParams({ mass_kg: 1e35 });
      const state = useSimulationStore.getState();
      expect(state.bekenstein.mass_kg).toBe(1e35);
    });

    it('setCriticalityParams should update criticality params', () => {
      useSimulationStore.getState().setCriticalityParams({ k_infinity: 1.1 });
      const state = useSimulationStore.getState();
      expect(state.criticality.k_infinity).toBe(1.1);
    });

    it('setVisualizationSettings should update settings', () => {
      useSimulationStore.getState().setVisualizationSettings({
        darkMode: false,
        animationSpeed: 2.0,
      });
      const state = useSimulationStore.getState();
      expect(state.darkMode).toBe(false);
      expect(state.animationSpeed).toBe(2.0);
    });

    it('setSimulationResults should update simulation results', () => {
      useSimulationStore.getState().setSimulationResults(0.75, 1.596, true);
      const state = useSimulationStore.getState();
      expect(state.currentOrderParameter).toBe(0.75);
      expect(state.currentCriticalThreshold).toBe(1.596);
      expect(state.isAtFixedPoint).toBe(true);
    });

    it('setError should update error state', () => {
      useSimulationStore.getState().setError('Test error');
      expect(useSimulationStore.getState().error).toBe('Test error');

      useSimulationStore.getState().setError(null);
      expect(useSimulationStore.getState().error).toBeNull();
    });

    it('reset should restore default values', () => {
      // Modify state
      useSimulationStore.getState().setKuramotoParams({ coupling: 5.0 });
      useSimulationStore.getState().setError('Some error');

      // Reset
      useSimulationStore.getState().reset();

      const state = useSimulationStore.getState();
      expect(state.kuramoto.coupling).toBe(1.5);
      expect(state.error).toBeNull();
    });
  });

  describe('Selectors', () => {
    it('selectActiveParams returns Kuramoto params when active', () => {
      const state = useSimulationStore.getState();
      const params = selectActiveParams(state);
      expect(params).toEqual(state.kuramoto);
    });

    it('selectActiveParams returns Turing params when active', () => {
      useSimulationStore.getState().setActiveView('turing');
      const state = useSimulationStore.getState();
      const params = selectActiveParams(state);
      expect(params).toEqual(state.turing);
    });

    it('selectActiveParams returns Bekenstein params when active', () => {
      useSimulationStore.getState().setActiveView('bekenstein');
      const state = useSimulationStore.getState();
      const params = selectActiveParams(state);
      expect(params).toEqual(state.bekenstein);
    });

    it('selectActiveParams returns criticality params when active', () => {
      useSimulationStore.getState().setActiveView('criticality');
      const state = useSimulationStore.getState();
      const params = selectActiveParams(state);
      expect(params).toEqual(state.criticality);
    });

    it('selectCriticalThreshold returns correct Kuramoto threshold', () => {
      const state = useSimulationStore.getState();
      const threshold = selectCriticalThreshold(state);
      // K_c = sqrt(8/π) * σ ≈ 1.596 * 1.0
      expect(threshold).toBeCloseTo(Math.sqrt(8 / Math.PI), 5);
    });

    it('selectCriticalThreshold returns 1.0 for criticality', () => {
      useSimulationStore.getState().setActiveView('criticality');
      const state = useSimulationStore.getState();
      const threshold = selectCriticalThreshold(state);
      expect(threshold).toBe(1.0);
    });

    it('selectCriticalThreshold returns correct Turing threshold', () => {
      useSimulationStore.getState().setActiveView('turing');
      const state = useSimulationStore.getState();
      const threshold = selectCriticalThreshold(state);
      // sqrt(D_v / D_u)
      expect(threshold).toBeCloseTo(Math.sqrt(5e-3 / 2.8e-4), 5);
    });
  });

  describe('Async Actions', () => {
    it('loadManifest should handle fetch errors', async () => {
      // Mock fetch to fail
      global.fetch = vi.fn().mockRejectedValue(new Error('Network error'));

      await useSimulationStore.getState().loadManifest();

      const state = useSimulationStore.getState();
      expect(state.error).toBe('Network error');
      expect(state.isLoading).toBe(false);
    });

    it('loadDataset should handle fetch errors', async () => {
      global.fetch = vi.fn().mockRejectedValue(new Error('Failed to fetch'));

      await useSimulationStore.getState().loadDataset('kuramoto_phase_diagram');

      const state = useSimulationStore.getState();
      expect(state.error).toBe('Failed to fetch');
      expect(state.isLoading).toBe(false);
    });
  });
});
