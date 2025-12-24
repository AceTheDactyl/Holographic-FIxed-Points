/**
 * Three.js phase space visualizer for Kuramoto oscillators.
 *
 * Visualizes oscillator phases as particles on a circle with
 * color representing phase and the order parameter as a central arrow.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

export interface PhaseSpaceConfig {
  containerWidth: number;
  containerHeight: number;
  backgroundColor: number;
  circleRadius: number;
  particleSize: number;
}

const defaultConfig: PhaseSpaceConfig = {
  containerWidth: 400,
  containerHeight: 400,
  backgroundColor: 0x1a1a2e,
  circleRadius: 2,
  particleSize: 0.08,
};

export class PhaseSpaceVisualizer {
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private renderer: THREE.WebGLRenderer;
  private controls: OrbitControls;
  private particles: THREE.Points | null = null;
  private orderArrow: THREE.ArrowHelper | null = null;
  private circle: THREE.Line | null = null;
  private config: PhaseSpaceConfig;
  private container: HTMLElement;
  private animationId: number | null = null;

  constructor(container: HTMLElement, config: Partial<PhaseSpaceConfig> = {}) {
    this.container = container;
    this.config = { ...defaultConfig, ...config };

    // Scene setup
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(this.config.backgroundColor);

    // Camera
    const aspect = this.config.containerWidth / this.config.containerHeight;
    this.camera = new THREE.PerspectiveCamera(60, aspect, 0.1, 1000);
    this.camera.position.set(0, 5, 0);
    this.camera.lookAt(0, 0, 0);

    // Renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(this.config.containerWidth, this.config.containerHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(this.renderer.domElement);

    // Controls
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.minDistance = 3;
    this.controls.maxDistance = 15;

    // Add reference geometry
    this.addReferenceGeometry();

    // Handle resize
    this.handleResize = this.handleResize.bind(this);
    window.addEventListener('resize', this.handleResize);

    // Start animation loop
    this.animate();
  }

  private addReferenceGeometry(): void {
    // Unit circle
    const circleGeometry = new THREE.BufferGeometry();
    const circlePoints: number[] = [];
    const segments = 64;

    for (let i = 0; i <= segments; i++) {
      const theta = (i / segments) * Math.PI * 2;
      circlePoints.push(
        Math.cos(theta) * this.config.circleRadius,
        0,
        Math.sin(theta) * this.config.circleRadius
      );
    }

    circleGeometry.setAttribute(
      'position',
      new THREE.Float32BufferAttribute(circlePoints, 3)
    );

    const circleMaterial = new THREE.LineBasicMaterial({
      color: 0x444466,
      transparent: true,
      opacity: 0.5,
    });

    this.circle = new THREE.Line(circleGeometry, circleMaterial);
    this.scene.add(this.circle);

    // Axes
    const axesHelper = new THREE.AxesHelper(1);
    axesHelper.position.set(-3, 0, -3);
    this.scene.add(axesHelper);

    // Grid
    const gridHelper = new THREE.GridHelper(6, 12, 0x333344, 0x222233);
    this.scene.add(gridHelper);

    // Ambient light
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    this.scene.add(ambientLight);
  }

  /**
   * Visualize oscillator phases as points on the unit circle.
   */
  visualizePhases(phases: number[]): void {
    // Remove existing particles
    if (this.particles) {
      this.scene.remove(this.particles);
      this.particles.geometry.dispose();
      (this.particles.material as THREE.PointsMaterial).dispose();
    }

    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(phases.length * 3);
    const colors = new Float32Array(phases.length * 3);

    phases.forEach((phase, i) => {
      // Position on circle in XZ plane
      positions[i * 3] = Math.cos(phase) * this.config.circleRadius;
      positions[i * 3 + 1] = 0;
      positions[i * 3 + 2] = Math.sin(phase) * this.config.circleRadius;

      // Color by phase (HSL color wheel)
      const hue = phase / (2 * Math.PI);
      const color = new THREE.Color().setHSL(hue, 0.8, 0.5);
      colors[i * 3] = color.r;
      colors[i * 3 + 1] = color.g;
      colors[i * 3 + 2] = color.b;
    });

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
      size: this.config.particleSize,
      vertexColors: true,
      transparent: true,
      opacity: 0.9,
      sizeAttenuation: true,
    });

    this.particles = new THREE.Points(geometry, material);
    this.scene.add(this.particles);
  }

  /**
   * Visualize the order parameter as a central arrow.
   */
  visualizeOrderParameter(r: number, psi: number): void {
    // Remove existing arrow
    if (this.orderArrow) {
      this.scene.remove(this.orderArrow);
    }

    if (r < 0.001) return; // Don't show arrow for very small r

    // Arrow direction
    const direction = new THREE.Vector3(
      Math.cos(psi),
      0,
      Math.sin(psi)
    ).normalize();

    // Arrow length proportional to r
    const length = r * this.config.circleRadius;

    // Color: green for high r, red for low r
    const color = new THREE.Color().setHSL(r * 0.3, 0.9, 0.5);

    this.orderArrow = new THREE.ArrowHelper(
      direction,
      new THREE.Vector3(0, 0.1, 0),
      length,
      color.getHex(),
      0.2,
      0.1
    );

    this.scene.add(this.orderArrow);
  }

  /**
   * Update visualization with new phase data.
   */
  update(phases: number[]): void {
    this.visualizePhases(phases);

    // Compute order parameter
    const sumCos = phases.reduce((sum, p) => sum + Math.cos(p), 0);
    const sumSin = phases.reduce((sum, p) => sum + Math.sin(p), 0);
    const N = phases.length;

    const r = Math.sqrt(sumCos * sumCos + sumSin * sumSin) / N;
    const psi = Math.atan2(sumSin / N, sumCos / N);

    this.visualizeOrderParameter(r, psi);
  }

  /**
   * Generate random phases for demonstration.
   */
  generateRandomPhases(n: number, coherence: number = 0): number[] {
    const phases: number[] = [];
    const meanPhase = Math.random() * 2 * Math.PI;

    for (let i = 0; i < n; i++) {
      if (coherence > 0) {
        // Concentrated around mean phase
        const spread = (1 - coherence) * Math.PI;
        phases.push(meanPhase + (Math.random() - 0.5) * spread * 2);
      } else {
        // Uniform distribution
        phases.push(Math.random() * 2 * Math.PI);
      }
    }

    return phases.map(p => ((p % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI));
  }

  private animate = (): void => {
    this.animationId = requestAnimationFrame(this.animate);
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  };

  private handleResize(): void {
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;

    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  /**
   * Set camera view.
   */
  setView(view: 'top' | 'side' | 'perspective'): void {
    switch (view) {
      case 'top':
        this.camera.position.set(0, 8, 0);
        break;
      case 'side':
        this.camera.position.set(8, 2, 0);
        break;
      case 'perspective':
        this.camera.position.set(5, 5, 5);
        break;
    }
    this.camera.lookAt(0, 0, 0);
  }

  /**
   * Clean up resources.
   */
  dispose(): void {
    if (this.animationId !== null) {
      cancelAnimationFrame(this.animationId);
    }

    window.removeEventListener('resize', this.handleResize);

    if (this.particles) {
      this.particles.geometry.dispose();
      (this.particles.material as THREE.PointsMaterial).dispose();
    }

    this.controls.dispose();
    this.renderer.dispose();

    if (this.renderer.domElement.parentNode) {
      this.renderer.domElement.parentNode.removeChild(this.renderer.domElement);
    }
  }
}
