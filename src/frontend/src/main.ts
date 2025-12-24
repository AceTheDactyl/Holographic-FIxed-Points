/**
 * Main application entry point for APL Holographic Fixed Points.
 *
 * Initializes the dashboard, loads pre-computed data, and sets up
 * interactive visualizations for physics fixed point analysis.
 */

import { useSimulationStore } from './stores/simulationStore';
import { PhaseSpaceVisualizer } from './visualizers/PhaseSpaceVisualizer';
import { createPhaseDiagram } from './visualizers/PhaseDiagramD3';
import { createTuringVisualization } from './visualizers/TuringPatternPlotly';
import type { KuramotoPhaseDiagram, TuringPatternsData, SolverType } from '@shared/types';

// =============================================================================
// Application State
// =============================================================================

let phaseSpaceViz: PhaseSpaceVisualizer | null = null;
let phaseDiagram: { update: (k: number) => void; destroy: () => void } | null = null;
let turingViz: { goToFrame: (i: number) => void; destroy: () => void } | null = null;

// Animation state
let animationRunning = false;
let phases: number[] = [];

// =============================================================================
// Initialization
// =============================================================================

async function init(): Promise<void> {
  console.log('Initializing APL Holographic Fixed Points...');

  // Get store
  const store = useSimulationStore.getState();

  // Setup UI event listeners
  setupNavigationListeners();
  setupControlListeners();
  setupActionButtons();

  // Load manifest and initial data
  try {
    await store.loadManifest();
    await loadInitialData();
    await initializeVisualizations();
    updateUI();
  } catch (error) {
    console.error('Initialization error:', error);
    showError('Failed to initialize application. Please refresh.');
  }

  console.log('Initialization complete.');
}

async function loadInitialData(): Promise<void> {
  const store = useSimulationStore.getState();

  // Load datasets in parallel
  await Promise.all([
    store.loadDataset('kuramoto_phase_diagram'),
    store.loadDataset('turing_patterns'),
    store.loadDataset('bekenstein_entropy'),
    store.loadDataset('criticality_response'),
  ]);
}

async function initializeVisualizations(): Promise<void> {
  const store = useSimulationStore.getState();

  // Initialize phase space visualizer
  const phaseSpaceContainer = document.getElementById('phase-space-container');
  if (phaseSpaceContainer) {
    phaseSpaceViz = new PhaseSpaceVisualizer(phaseSpaceContainer, {
      containerWidth: phaseSpaceContainer.clientWidth,
      containerHeight: phaseSpaceContainer.clientHeight,
    });

    // Generate initial random phases
    phases = phaseSpaceViz.generateRandomPhases(store.kuramoto.n_oscillators ?? 100, 0);
    phaseSpaceViz.update(phases);
  }

  // Initialize phase diagram
  const phaseDiagramContainer = document.getElementById('phase-diagram-container');
  if (phaseDiagramContainer && store.kuramotoData) {
    phaseDiagram = createPhaseDiagram(
      phaseDiagramContainer,
      store.kuramotoData as KuramotoPhaseDiagram,
      {
        width: phaseDiagramContainer.clientWidth,
        height: phaseDiagramContainer.clientHeight,
      }
    );
    phaseDiagram.update(store.kuramoto.coupling ?? 1.5);
  }
}

// =============================================================================
// Navigation
// =============================================================================

function setupNavigationListeners(): void {
  const navButtons = document.querySelectorAll('.nav-btn');

  navButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      const view = btn.getAttribute('data-view') as SolverType;
      if (view) {
        switchView(view);
      }
    });
  });
}

async function switchView(view: SolverType): Promise<void> {
  const store = useSimulationStore.getState();
  store.setActiveView(view);

  // Update navigation buttons
  document.querySelectorAll('.nav-btn').forEach((btn) => {
    btn.classList.toggle('active', btn.getAttribute('data-view') === view);
  });

  // Show/hide control sections
  document.querySelectorAll('.control-section').forEach((section) => {
    section.classList.add('hidden');
  });
  document.getElementById(`${view}-controls`)?.classList.remove('hidden');

  // Update fixed point highlighting
  document.querySelectorAll('.fixed-point-card').forEach((card) => {
    card.classList.remove('active');
  });

  // Highlight relevant fixed point
  const fpMap: Record<SolverType, string> = {
    kuramoto: 'fp-kuramoto',
    turing: 'fp-turing',
    bekenstein: 'fp-bekenstein',
    criticality: 'fp-nuclear',
  };
  document.getElementById(fpMap[view])?.classList.add('active');

  // Reinitialize visualizations for the new view
  await reinitializeVisualizationsForView(view);

  updateUI();
}

async function reinitializeVisualizationsForView(view: SolverType): Promise<void> {
  const store = useSimulationStore.getState();

  // Clean up existing visualizations
  if (phaseDiagram) {
    phaseDiagram.destroy();
    phaseDiagram = null;
  }
  if (turingViz) {
    turingViz.destroy();
    turingViz = null;
  }

  const primaryContainer = document.getElementById('phase-diagram-container');
  const secondaryContainer = document.getElementById('phase-space-container');

  if (!primaryContainer) return;

  // Update visualization titles
  const primaryCard = document.getElementById('primary-viz');
  const secondaryCard = document.getElementById('secondary-viz');

  switch (view) {
    case 'kuramoto':
      if (primaryCard) primaryCard.querySelector('h3')!.textContent = 'Phase Diagram';
      if (secondaryCard) secondaryCard.querySelector('h3')!.textContent = 'Phase Space';

      if (store.kuramotoData) {
        phaseDiagram = createPhaseDiagram(
          primaryContainer,
          store.kuramotoData as KuramotoPhaseDiagram,
          { width: primaryContainer.clientWidth, height: primaryContainer.clientHeight }
        );
        phaseDiagram.update(store.kuramoto.coupling ?? 1.5);
      }
      break;

    case 'turing':
      if (primaryCard) primaryCard.querySelector('h3')!.textContent = 'Pattern Evolution';
      if (secondaryCard) secondaryCard.querySelector('h3')!.textContent = 'Current Pattern';

      if (store.turingData) {
        primaryContainer.innerHTML = '';
        turingViz = await createTuringVisualization(
          primaryContainer,
          store.turingData as TuringPatternsData
        );
      }
      break;

    case 'bekenstein':
      if (primaryCard) primaryCard.querySelector('h3')!.textContent = 'Entropy vs Mass';
      if (secondaryCard) secondaryCard.querySelector('h3')!.textContent = 'Horizon Properties';
      // Would add Bekenstein visualization here
      break;

    case 'criticality':
      if (primaryCard) primaryCard.querySelector('h3')!.textContent = 'Neutron Population';
      if (secondaryCard) secondaryCard.querySelector('h3')!.textContent = 'Reactivity Response';
      // Would add criticality visualization here
      break;
  }
}

// =============================================================================
// Control Listeners
// =============================================================================

function setupControlListeners(): void {
  const store = useSimulationStore.getState();

  // Kuramoto controls
  setupSlider('coupling-slider', 'coupling-value', (value) => {
    store.setKuramotoParams({ coupling: value });
    phaseDiagram?.update(value);
  });

  setupSlider('oscillators-slider', 'oscillators-value', (value) => {
    store.setKuramotoParams({ n_oscillators: Math.round(value) });
    if (phaseSpaceViz) {
      phases = phaseSpaceViz.generateRandomPhases(Math.round(value), 0);
      phaseSpaceViz.update(phases);
    }
  }, (v) => Math.round(v).toString());

  setupSlider('freq-std-slider', 'freq-std-value', (value) => {
    store.setKuramotoParams({ frequency_std: value });
    updateCriticalThreshold();
  });

  // Turing controls
  setupSlider('du-slider', 'du-value', (value) => {
    store.setTuringParams({ D_u: value });
  }, (v) => v.toExponential(1));

  setupSlider('dv-slider', 'dv-value', (value) => {
    store.setTuringParams({ D_v: value });
  }, (v) => v.toExponential(1));

  setupSlider('time-slider', 'time-value', (value) => {
    turingViz?.goToFrame(Math.round(value));
  }, (v) => `t=${[0, 1, 2, 5, 10, 20, 50, 100][Math.round(v)] ?? 0}`);

  // Bekenstein controls
  setupSlider('mass-slider', 'mass-value', (value) => {
    store.setBekensteinParams({ mass_kg: Math.pow(10, value) });
  }, (v) => `10^${v.toFixed(0)} kg`);

  // Criticality controls
  setupSlider('k-inf-slider', 'k-inf-value', (value) => {
    store.setCriticalityParams({ k_infinity: value });
  });

  setupSlider('leakage-slider', 'leakage-value', (value) => {
    store.setCriticalityParams({ leakage_factor: value });
  });
}

function setupSlider(
  sliderId: string,
  displayId: string,
  onChange: (value: number) => void,
  formatValue?: (value: number) => string
): void {
  const slider = document.getElementById(sliderId) as HTMLInputElement;
  const display = document.getElementById(displayId);

  if (!slider || !display) return;

  const updateDisplay = () => {
    const value = parseFloat(slider.value);
    display.textContent = formatValue ? formatValue(value) : value.toFixed(2);
    onChange(value);
  };

  slider.addEventListener('input', updateDisplay);
}

// =============================================================================
// Action Buttons
// =============================================================================

function setupActionButtons(): void {
  const runBtn = document.getElementById('run-btn');
  const resetBtn = document.getElementById('reset-btn');

  runBtn?.addEventListener('click', runSimulation);
  resetBtn?.addEventListener('click', resetSimulation);
}

function runSimulation(): void {
  const store = useSimulationStore.getState();

  if (store.activeView === 'kuramoto' && phaseSpaceViz) {
    animationRunning = true;
    animateKuramoto();
  } else if (store.activeView === 'turing' && turingViz) {
    turingViz.goToFrame(0);
  }

  updateStatus('Running...');
}

function resetSimulation(): void {
  animationRunning = false;
  const store = useSimulationStore.getState();
  store.reset();

  if (phaseSpaceViz) {
    phases = phaseSpaceViz.generateRandomPhases(100, 0);
    phaseSpaceViz.update(phases);
  }

  // Reset sliders to defaults
  resetSliders();
  updateStatus('Ready');
  updateUI();
}

function resetSliders(): void {
  const defaults: Record<string, number> = {
    'coupling-slider': 1.5,
    'oscillators-slider': 100,
    'freq-std-slider': 1.0,
    'du-slider': 0.00028,
    'dv-slider': 0.005,
    'time-slider': 0,
    'mass-slider': 30,
    'k-inf-slider': 1.03,
    'leakage-slider': 0.02,
  };

  Object.entries(defaults).forEach(([id, value]) => {
    const slider = document.getElementById(id) as HTMLInputElement;
    if (slider) {
      slider.value = String(value);
      slider.dispatchEvent(new Event('input'));
    }
  });
}

// =============================================================================
// Kuramoto Animation
// =============================================================================

function animateKuramoto(): void {
  if (!animationRunning || !phaseSpaceViz) return;

  const store = useSimulationStore.getState();
  const K = store.kuramoto.coupling ?? 1.5;
  const N = phases.length;
  const dt = 0.05;

  // Generate random natural frequencies if not set
  const frequencies = phases.map(() => (Math.random() - 0.5) * 2);

  // Kuramoto dynamics
  const sumCos = phases.reduce((sum, p) => sum + Math.cos(p), 0);
  const sumSin = phases.reduce((sum, p) => sum + Math.sin(p), 0);
  const r = Math.sqrt(sumCos * sumCos + sumSin * sumSin) / N;
  const psi = Math.atan2(sumSin / N, sumCos / N);

  // Update phases
  phases = phases.map((phase, i) => {
    const dphase = frequencies[i] + K * r * Math.sin(psi - phase);
    return (phase + dphase * dt + 2 * Math.PI) % (2 * Math.PI);
  });

  phaseSpaceViz.update(phases);

  // Update order parameter display
  const currentR = document.getElementById('current-r');
  if (currentR) {
    currentR.textContent = `r = ${r.toFixed(3)}`;
  }

  // Update simulation results in store
  store.setSimulationResults(r, Math.sqrt(8 / Math.PI), Math.abs(r - 0.5) < 0.1);

  requestAnimationFrame(animateKuramoto);
}

// =============================================================================
// UI Updates
// =============================================================================

function updateUI(): void {
  const store = useSimulationStore.getState();

  // Update critical threshold display
  updateCriticalThreshold();
}

function updateCriticalThreshold(): void {
  const store = useSimulationStore.getState();
  const sigma = store.kuramoto.frequency_std ?? 1.0;
  const Kc = Math.sqrt(8 / Math.PI) * sigma;

  const fpValue = document.querySelector('#fp-kuramoto .fixed-point-value');
  if (fpValue) {
    fpValue.textContent = `√(8/π) × ${sigma.toFixed(1)} ≈ ${Kc.toFixed(3)}`;
  }
}

function updateStatus(message: string): void {
  const statusEl = document.getElementById('simulation-status');
  if (statusEl) {
    statusEl.textContent = message;
  }
}

function showError(message: string): void {
  console.error(message);
  updateStatus(`Error: ${message}`);
}

// =============================================================================
// Start Application
// =============================================================================

document.addEventListener('DOMContentLoaded', init);
