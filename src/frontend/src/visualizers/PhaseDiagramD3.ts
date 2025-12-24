/**
 * D3.js phase diagram visualization for Kuramoto model.
 *
 * Creates an interactive line plot showing order parameter r
 * as a function of coupling strength K, with critical threshold marker.
 */

import * as d3 from 'd3';
import type { KuramotoPhaseDiagram } from '@shared/types';

export interface PhaseDiagramConfig {
  width: number;
  height: number;
  margin: { top: number; right: number; bottom: number; left: number };
  backgroundColor: string;
  textColor: string;
  criticalColor: string;
  lineColors: string[];
}

const defaultConfig: PhaseDiagramConfig = {
  width: 600,
  height: 400,
  margin: { top: 40, right: 120, bottom: 60, left: 70 },
  backgroundColor: '#1a1a2e',
  textColor: '#eaeaea',
  criticalColor: '#e94560',
  lineColors: ['#00d9ff', '#00ff88', '#ff6b35', '#9d4edd'],
};

export function createPhaseDiagram(
  container: HTMLElement,
  data: KuramotoPhaseDiagram,
  config: Partial<PhaseDiagramConfig> = {}
): { update: (currentK: number) => void; destroy: () => void } {
  const cfg = { ...defaultConfig, ...config };
  const { margin } = cfg;
  const width = cfg.width - margin.left - margin.right;
  const height = cfg.height - margin.top - margin.bottom;

  // Clear existing content
  d3.select(container).selectAll('*').remove();

  // Create SVG
  const svg = d3
    .select(container)
    .append('svg')
    .attr('width', cfg.width)
    .attr('height', cfg.height)
    .attr('viewBox', `0 0 ${cfg.width} ${cfg.height}`)
    .style('background-color', cfg.backgroundColor);

  const g = svg
    .append('g')
    .attr('transform', `translate(${margin.left},${margin.top})`);

  // Scales
  const xScale = d3
    .scaleLinear()
    .domain([d3.min(data.K)!, d3.max(data.K)!])
    .range([0, width]);

  const yScale = d3.scaleLinear().domain([0, 1]).range([height, 0]);

  const colorScale = d3.scaleOrdinal<string>().range(cfg.lineColors);

  // Axes
  const xAxis = d3.axisBottom(xScale).ticks(10);
  const yAxis = d3.axisLeft(yScale).ticks(10);

  g.append('g')
    .attr('class', 'x-axis')
    .attr('transform', `translate(0,${height})`)
    .call(xAxis)
    .selectAll('text')
    .style('fill', cfg.textColor);

  g.append('g')
    .attr('class', 'y-axis')
    .call(yAxis)
    .selectAll('text')
    .style('fill', cfg.textColor);

  // Style axis lines
  g.selectAll('.domain, .tick line').style('stroke', cfg.textColor);

  // Axis labels
  g.append('text')
    .attr('class', 'x-label')
    .attr('x', width / 2)
    .attr('y', height + 45)
    .attr('fill', cfg.textColor)
    .attr('text-anchor', 'middle')
    .style('font-size', '14px')
    .text('Coupling Strength K');

  g.append('text')
    .attr('class', 'y-label')
    .attr('transform', 'rotate(-90)')
    .attr('x', -height / 2)
    .attr('y', -50)
    .attr('fill', cfg.textColor)
    .attr('text-anchor', 'middle')
    .style('font-size', '14px')
    .text('Order Parameter r');

  // Critical threshold line
  const criticalX = xScale(data.K_critical);

  g.append('line')
    .attr('class', 'critical-line')
    .attr('x1', criticalX)
    .attr('x2', criticalX)
    .attr('y1', 0)
    .attr('y2', height)
    .attr('stroke', cfg.criticalColor)
    .attr('stroke-width', 2)
    .attr('stroke-dasharray', '8,4');

  g.append('text')
    .attr('class', 'critical-label')
    .attr('x', criticalX + 8)
    .attr('y', 20)
    .attr('fill', cfg.criticalColor)
    .style('font-size', '12px')
    .style('font-weight', '500')
    .text(`Kc = ${data.K_critical.toFixed(3)}`);

  // Line generator
  const line = d3
    .line<number>()
    .x((_, i) => xScale(data.K[i]))
    .y((d) => yScale(d))
    .curve(d3.curveMonotoneX);

  // Draw lines for each N
  const lineGroups = Object.entries(data.data);

  lineGroups.forEach(([label, values], i) => {
    const lineData = values.r_mean;

    // Confidence band
    if (values.r_std) {
      const areaGenerator = d3
        .area<number>()
        .x((_, j) => xScale(data.K[j]))
        .y0((_, j) => yScale(Math.max(0, lineData[j] - values.r_std[j])))
        .y1((_, j) => yScale(Math.min(1, lineData[j] + values.r_std[j])))
        .curve(d3.curveMonotoneX);

      g.append('path')
        .datum(lineData)
        .attr('class', `confidence-band-${i}`)
        .attr('fill', colorScale(String(i)))
        .attr('fill-opacity', 0.15)
        .attr('d', areaGenerator);
    }

    // Main line
    g.append('path')
      .datum(lineData)
      .attr('class', `line-${i}`)
      .attr('fill', 'none')
      .attr('stroke', colorScale(String(i)))
      .attr('stroke-width', 2.5)
      .attr('d', line);

    // Legend
    g.append('line')
      .attr('x1', width + 15)
      .attr('x2', width + 35)
      .attr('y1', 20 + i * 25)
      .attr('y2', 20 + i * 25)
      .attr('stroke', colorScale(String(i)))
      .attr('stroke-width', 2.5);

    g.append('text')
      .attr('x', width + 40)
      .attr('y', 25 + i * 25)
      .attr('fill', cfg.textColor)
      .style('font-size', '12px')
      .text(label);
  });

  // Current K indicator
  const indicator = g
    .append('line')
    .attr('class', 'current-k-indicator')
    .attr('y1', 0)
    .attr('y2', height)
    .attr('stroke', '#ffffff')
    .attr('stroke-width', 1.5)
    .attr('stroke-dasharray', '4,4')
    .style('opacity', 0);

  const indicatorLabel = g
    .append('text')
    .attr('class', 'current-k-label')
    .attr('y', -10)
    .attr('fill', '#ffffff')
    .attr('text-anchor', 'middle')
    .style('font-size', '11px')
    .style('opacity', 0);

  // Title
  g.append('text')
    .attr('class', 'title')
    .attr('x', width / 2)
    .attr('y', -15)
    .attr('fill', cfg.textColor)
    .attr('text-anchor', 'middle')
    .style('font-size', '16px')
    .style('font-weight', '600')
    .text('Kuramoto Phase Transition');

  // Update function for current K
  function update(currentK: number): void {
    const x = xScale(currentK);

    indicator.attr('x1', x).attr('x2', x).style('opacity', 1);

    indicatorLabel.attr('x', x).text(`K = ${currentK.toFixed(2)}`).style('opacity', 1);
  }

  // Destroy function
  function destroy(): void {
    d3.select(container).selectAll('*').remove();
  }

  return { update, destroy };
}

/**
 * Create an interactive phase diagram with tooltips and hover effects.
 */
export function createInteractivePhaseDiagram(
  container: HTMLElement,
  data: KuramotoPhaseDiagram,
  onKChange?: (K: number) => void
): { update: (currentK: number) => void; destroy: () => void } {
  const diagram = createPhaseDiagram(container, data);

  // Add mouse interaction
  const svg = d3.select(container).select('svg');
  const g = svg.select('g');
  const margin = { left: 70 };

  // Invisible overlay for mouse tracking
  const overlay = g
    .append('rect')
    .attr('class', 'overlay')
    .attr('width', container.clientWidth - margin.left - 120)
    .attr('height', container.clientHeight - 100)
    .style('fill', 'none')
    .style('pointer-events', 'all');

  const xScale = d3
    .scaleLinear()
    .domain([d3.min(data.K)!, d3.max(data.K)!])
    .range([0, container.clientWidth - margin.left - 120]);

  overlay.on('mousemove', (event: MouseEvent) => {
    const [mouseX] = d3.pointer(event);
    const K = xScale.invert(mouseX);

    if (K >= data.K[0] && K <= data.K[data.K.length - 1]) {
      diagram.update(K);
      if (onKChange) {
        onKChange(K);
      }
    }
  });

  overlay.on('mouseleave', () => {
    // Could hide indicator here
  });

  return diagram;
}
