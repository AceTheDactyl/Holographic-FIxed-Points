/**
 * Plotly.js visualization for Turing patterns.
 *
 * Creates animated heatmaps showing pattern formation evolution
 * with playback controls and time slider.
 */

import Plotly from 'plotly.js-dist-min';
import type { TuringSnapshot, TuringPatternsData } from '@shared/types';

export interface TuringVisualizationConfig {
  colorscale: string;
  showColorbar: boolean;
  animationDuration: number;
  paperBgColor: string;
  plotBgColor: string;
  fontColor: string;
}

const defaultConfig: TuringVisualizationConfig = {
  colorscale: 'Viridis',
  showColorbar: true,
  animationDuration: 500,
  paperBgColor: '#1a1a2e',
  plotBgColor: '#1a1a2e',
  fontColor: '#eaeaea',
};

export async function createTuringVisualization(
  container: HTMLElement,
  data: TuringPatternsData,
  config: Partial<TuringVisualizationConfig> = {}
): Promise<{
  goToFrame: (index: number) => void;
  play: () => void;
  pause: () => void;
  destroy: () => void;
}> {
  const cfg = { ...defaultConfig, ...config };
  const { snapshots } = data;

  // Create frames for animation
  const frames: Plotly.Frame[] = snapshots.map((snap, i) => ({
    name: `frame${i}`,
    data: [
      {
        z: snap.pattern,
        type: 'heatmap' as const,
        colorscale: cfg.colorscale,
        showscale: cfg.showColorbar,
        zmin: -0.5,
        zmax: 0.5,
      },
    ],
  }));

  // Initial data
  const initialData: Partial<Plotly.PlotData>[] = [
    {
      z: snapshots[0].pattern,
      type: 'heatmap',
      colorscale: cfg.colorscale,
      showscale: cfg.showColorbar,
      colorbar: {
        title: { text: 'Activator U', font: { color: cfg.fontColor } },
        tickfont: { color: cfg.fontColor },
      },
      zmin: -0.5,
      zmax: 0.5,
    },
  ];

  // Layout
  const layout: Partial<Plotly.Layout> = {
    title: {
      text: 'Turing Pattern Formation',
      font: { color: cfg.fontColor, size: 16 },
    },
    paper_bgcolor: cfg.paperBgColor,
    plot_bgcolor: cfg.plotBgColor,
    font: { color: cfg.fontColor },
    xaxis: {
      title: 'x',
      showgrid: false,
      zeroline: false,
      tickfont: { color: cfg.fontColor },
    },
    yaxis: {
      title: 'y',
      showgrid: false,
      zeroline: false,
      scaleanchor: 'x',
      tickfont: { color: cfg.fontColor },
    },
    margin: { t: 50, r: 80, b: 50, l: 50 },
    updatemenus: [
      {
        type: 'buttons',
        showactive: false,
        y: 0,
        x: 0.1,
        xanchor: 'right',
        yanchor: 'top',
        pad: { t: 60 },
        buttons: [
          {
            label: '▶ Play',
            method: 'animate',
            args: [
              null,
              {
                fromcurrent: true,
                frame: { duration: cfg.animationDuration },
                transition: { duration: cfg.animationDuration / 2 },
              },
            ],
          },
          {
            label: '⏸ Pause',
            method: 'animate',
            args: [
              [null],
              {
                mode: 'immediate',
                frame: { duration: 0 },
                transition: { duration: 0 },
              },
            ],
          },
        ],
      },
    ],
    sliders: [
      {
        active: 0,
        steps: snapshots.map((snap, i) => ({
          label: `t=${snap.time}`,
          method: 'animate',
          args: [
            [`frame${i}`],
            {
              mode: 'immediate',
              frame: { duration: 0 },
              transition: { duration: 0 },
            },
          ],
        })),
        x: 0.1,
        len: 0.8,
        currentvalue: {
          prefix: 'Time: ',
          xanchor: 'center',
          font: { color: cfg.fontColor },
        },
        pad: { t: 40 },
        font: { color: cfg.fontColor },
        tickcolor: cfg.fontColor,
      },
    ],
    annotations: [
      {
        text: `Contrast: ${snapshots[0].contrast.toFixed(4)}`,
        xref: 'paper',
        yref: 'paper',
        x: 0.02,
        y: 0.98,
        showarrow: false,
        font: { color: cfg.fontColor, size: 12 },
      },
    ],
  };

  // Plot configuration
  const plotConfig: Partial<Plotly.Config> = {
    responsive: true,
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['lasso2d', 'select2d'],
  };

  // Create plot
  await Plotly.newPlot(container, initialData as Plotly.Data[], layout, plotConfig);

  // Add frames
  await Plotly.addFrames(container, frames);

  // Control functions
  function goToFrame(index: number): void {
    if (index >= 0 && index < snapshots.length) {
      Plotly.animate(container, [`frame${index}`], {
        mode: 'immediate',
        frame: { duration: 0 },
        transition: { duration: 0 },
      });

      // Update contrast annotation
      Plotly.relayout(container, {
        'annotations[0].text': `Contrast: ${snapshots[index].contrast.toFixed(4)}`,
      });
    }
  }

  function play(): void {
    Plotly.animate(container, null, {
      fromcurrent: true,
      frame: { duration: cfg.animationDuration },
      transition: { duration: cfg.animationDuration / 2 },
    });
  }

  function pause(): void {
    Plotly.animate(container, [null], {
      mode: 'immediate',
      frame: { duration: 0 },
      transition: { duration: 0 },
    });
  }

  function destroy(): void {
    Plotly.purge(container);
  }

  return { goToFrame, play, pause, destroy };
}

/**
 * Create a static comparison view of Turing patterns at different times.
 */
export async function createTuringComparison(
  container: HTMLElement,
  data: TuringPatternsData,
  times: number[] = [0, 10, 50, 100],
  config: Partial<TuringVisualizationConfig> = {}
): Promise<void> {
  const cfg = { ...defaultConfig, ...config };
  const { snapshots } = data;

  // Filter snapshots to requested times
  const selectedSnapshots = times
    .map((t) => snapshots.find((s) => s.time === t))
    .filter((s): s is TuringSnapshot => s !== undefined);

  if (selectedSnapshots.length === 0) {
    console.warn('No matching snapshots found');
    return;
  }

  // Create subplots
  const nCols = Math.min(selectedSnapshots.length, 4);
  const nRows = Math.ceil(selectedSnapshots.length / nCols);

  const traces: Partial<Plotly.PlotData>[] = selectedSnapshots.map((snap, i) => ({
    z: snap.pattern,
    type: 'heatmap',
    colorscale: cfg.colorscale,
    showscale: i === selectedSnapshots.length - 1,
    xaxis: `x${i + 1}`,
    yaxis: `y${i + 1}`,
    zmin: -0.5,
    zmax: 0.5,
  }));

  // Calculate subplot positions
  const layout: Partial<Plotly.Layout> = {
    title: {
      text: 'Turing Pattern Evolution',
      font: { color: cfg.fontColor, size: 16 },
    },
    paper_bgcolor: cfg.paperBgColor,
    plot_bgcolor: cfg.plotBgColor,
    font: { color: cfg.fontColor },
    grid: { rows: nRows, columns: nCols, pattern: 'independent' },
    annotations: selectedSnapshots.map((snap, i) => ({
      text: `t = ${snap.time}`,
      xref: 'paper' as const,
      yref: 'paper' as const,
      x: ((i % nCols) + 0.5) / nCols,
      y: 1 - Math.floor(i / nCols) / nRows + 0.02,
      showarrow: false,
      font: { color: cfg.fontColor, size: 12 },
    })),
  };

  // Configure each subplot axis
  for (let i = 0; i < selectedSnapshots.length; i++) {
    const xKey = `xaxis${i + 1}` as keyof Plotly.Layout;
    const yKey = `yaxis${i + 1}` as keyof Plotly.Layout;

    (layout as Record<string, unknown>)[xKey] = {
      showticklabels: false,
      showgrid: false,
      zeroline: false,
    };
    (layout as Record<string, unknown>)[yKey] = {
      showticklabels: false,
      showgrid: false,
      zeroline: false,
      scaleanchor: `x${i + 1}`,
    };
  }

  await Plotly.newPlot(container, traces as Plotly.Data[], layout, {
    responsive: true,
    displayModeBar: false,
  });
}
