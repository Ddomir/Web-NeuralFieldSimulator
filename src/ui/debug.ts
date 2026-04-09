import type { FieldParams } from "../sim/field.ts";

export class DebugOverlay {
  private el: HTMLElement;
  private visible = false;
  private frameCount = 0;
  private lastFpsTime = performance.now();
  private fps = 0;

  private staging: GPUBuffer | null = null;
  private stagingSize = 0;
  private sampling = false;

  private stats = { min: 0, mean: 0, max: 0, variance: 0, fresh: false };

  private device: GPUDevice;
  private getFieldBuffer: () => GPUBuffer | null;
  private fieldCells: number;
  private getParams: () => FieldParams;

  constructor(
    device: GPUDevice,
    getFieldBuffer: () => GPUBuffer | null,
    fieldCells: number,
    getParams: () => FieldParams,
  ) {
    this.device = device;
    this.getFieldBuffer = getFieldBuffer;
    this.fieldCells = fieldCells;
    this.getParams = getParams;
    this.el = document.createElement("div");
    this.el.id = "debug-overlay";
    this.el.style.display = "none";
    document.body.appendChild(this.el);

    window.addEventListener("keydown", (e) => {
      if (e.key === "`") {
        this.visible = !this.visible;
        this.el.style.display = this.visible ? "block" : "none";
        if (this.visible) this.sample();
      }
    });

    setInterval(() => { if (this.visible) this.sample(); }, 1000);
  }

  tick(): void {
    this.frameCount++;
    const now     = performance.now();
    const elapsed = now - this.lastFpsTime;
    if (elapsed >= 500) {
      this.fps         = Math.round(this.frameCount / (elapsed / 1000));
      this.frameCount  = 0;
      this.lastFpsTime = now;
      if (this.visible) this.render();
    }
  }

  private render(): void {
    const s = this.stats;
    const p = this.getParams();

    const W_e   = p.A_e * 2 * Math.PI * p.sigma_e * p.sigma_e;
    const W_i   = p.A_i * 2 * Math.PI * p.sigma_i * p.sigma_i;
    const W_net = W_e - W_i;

    const f0        = 1 / (1 + Math.exp(-p.beta * (0 - p.theta)));
    const fprime    = p.beta * f0 * (1 - f0);
    const tauRatio  = p.tau_v / p.tau;
    const waveThreshold = 4 / (W_net * p.beta * p.beta);
    const wavesOk   = p.tau_v < 100 && tauRatio > waveThreshold;
    const turingOn  = (p.beta * p.beta * W_e) / 4 > 1;
    const adaptOn   = p.tau_v < 100;

    let regime = "?";
    if (!adaptOn) {
      regime = W_net < 0.05 ? "Balanced rest" : "Noise-driven rest";
    } else if (turingOn && W_net > 0.6) {
      regime = tauRatio < 3 ? "Tonic-clonic / bursting" : "Spatial Turing patterns";
    } else if (wavesOk) {
      regime = tauRatio > 15 ? "Slow delta waves" : "Excitable traveling waves";
    } else {
      regime = "Sub-wave";
    }

    const n = (x: number, d = 2) => x.toFixed(d);

    this.el.innerHTML =
      `<div class="dbg-section">` +
        `<div>${this.fps} fps</div>` +
        (s.fresh
          ? `<div>u  [${n(s.min)}, ${n(s.mean)}, ${n(s.max)}]</div>`
          : `<div>sampling…</div>`) +
        `<div>W_net: ${n(W_net, 3)}  &nbsp; u*≈${n((W_net * f0) / (1 - W_net * fprime), 3)}</div>` +
        `<div>Regime: ${regime}</div>` +
      `</div>` +
      `<div class="dbg-hint">[ \` ] toggle</div>`;
  }

  private async sample(): Promise<void> {
    if (this.sampling) return;
    const buf = this.getFieldBuffer();
    if (!buf) return;
    this.sampling = true;

    const byteSize = this.fieldCells * 4;

    if (!this.staging || this.stagingSize !== byteSize) {
      this.staging?.destroy();
      this.staging = this.device.createBuffer({
        size:  byteSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      this.stagingSize = byteSize;
    }

    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(buf, 0, this.staging, 0, byteSize);
    this.device.queue.submit([encoder.finish()]);

    await this.staging.mapAsync(GPUMapMode.READ);
    const data = new Float32Array(this.staging.getMappedRange().slice(0));
    this.staging.unmap();

    let min = Infinity, max = -Infinity, sum = 0;
    for (let i = 0; i < data.length; i++) {
      if (data[i] < min) min = data[i];
      if (data[i] > max) max = data[i];
      sum += data[i];
    }
    const mean = sum / data.length;

    let varSum = 0;
    for (let i = 0; i < data.length; i++) {
      const d = data[i] - mean;
      varSum += d * d;
    }
    const variance = varSum / data.length;

    this.stats = { min, max, mean, variance, fresh: true };
    this.sampling = false;
    this.render();
  }
}
