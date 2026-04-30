import type { FieldParams } from "../sim/field.ts";

export class KernelOverlay {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private params: FieldParams;
  private enabled = false;
  private mouseX = -1;
  private mouseY = -1;

  // offscreen ImageData cache — rebuilt when params change
  private ringCache: ImageData | null = null;
  private cacheKey = "";

  constructor(private simCanvas: HTMLCanvasElement, params: FieldParams) {
    this.params = { ...params };

    this.canvas = document.createElement("canvas");
    this.canvas.style.cssText = `
      position: absolute;
      top: 0; left: 0;
      width: 100%; height: 100%;
      pointer-events: none;
      display: none;
    `;
    this.ctx = this.canvas.getContext("2d")!;

    // sit on top of the sim canvas
    simCanvas.parentElement!.style.position = "relative";
    simCanvas.insertAdjacentElement("afterend", this.canvas);

    this.syncSize();
    new ResizeObserver(() => this.syncSize()).observe(simCanvas);

    simCanvas.addEventListener("mousemove", (e) => {
      if (!this.enabled) return;
      const rect = simCanvas.getBoundingClientRect();
      // position in CSS pixels relative to the canvas
      this.mouseX = e.clientX - rect.left;
      this.mouseY = e.clientY - rect.top;
      this.draw();
    });

    simCanvas.addEventListener("mouseleave", () => {
      this.mouseX = -1;
      this.clear();
    });
  }

  enable(): void {
    this.enabled = true;
    this.canvas.style.display = "block";
  }

  disable(): void {
    this.enabled = false;
    this.canvas.style.display = "none";
    this.clear();
  }

  update(params: FieldParams): void {
    this.params = { ...params };
    this.ringCache = null; // invalidate cache
    if (this.enabled && this.mouseX >= 0) this.draw();
  }

  private syncSize(): void {
    const r = this.simCanvas.getBoundingClientRect();
    this.canvas.width  = r.width;
    this.canvas.height = r.height;
    this.ringCache = null;
    if (this.enabled && this.mouseX >= 0) this.draw();
  }

  private buildRingCache(): ImageData {
    const { A_e, A_i, sigma_e, sigma_i, kernelRadius } = this.params;

    // Radius in CSS pixels — scale kernel_radius (grid cells) to canvas pixels
    const rect = this.simCanvas.getBoundingClientRect();
    const pxPerCell = rect.width / this.params.width;
    const radiusPx = kernelRadius * pxPerCell;
    const diam = Math.ceil(radiusPx * 2) + 2;

    const img = new ImageData(diam, diam);
    const cx = diam / 2;
    const cy = diam / 2;

    for (let py = 0; py < diam; py++) {
      for (let px = 0; px < diam; px++) {
        const dx = px - cx;
        const dy = py - cy;
        const distPx = Math.sqrt(dx * dx + dy * dy);
        if (distPx > radiusPx) continue;

        // convert pixel distance back to grid-cell distance for kernel eval
        const r = distPx / pxPerCell;
        const r2 = r * r;
        const w = A_e * Math.exp(-r2 / (2 * sigma_e * sigma_e))
                - A_i * Math.exp(-r2 / (2 * sigma_i * sigma_i));

        // alpha proportional to |w|, capped
        const maxW = A_e; // peak excitatory value
        const alpha = Math.min(Math.abs(w) / maxW, 1) * 180;

        const i = (py * diam + px) * 4;
        if (w > 0) {
          img.data[i]     = 109; // #6dd6a0 green
          img.data[i + 1] = 214;
          img.data[i + 2] = 160;
        } else {
          img.data[i]     = 244; // #f47c7c red
          img.data[i + 1] = 124;
          img.data[i + 2] = 124;
        }
        img.data[i + 3] = alpha;
      }
    }

    return img;
  }

  private draw(): void {
    const W = this.canvas.width;
    const H = this.canvas.height;
    this.ctx.clearRect(0, 0, W, H);

    if (this.mouseX < 0) return;

    const key = `${this.params.A_e},${this.params.A_i},${this.params.sigma_e},${this.params.sigma_i},${this.params.kernelRadius},${W}`;
    if (this.cacheKey !== key || !this.ringCache) {
      this.ringCache = this.buildRingCache();
      this.cacheKey = key;
    }

    const diam = this.ringCache.width;
    const offscreen = new OffscreenCanvas(diam, diam);
    const octx = offscreen.getContext("2d")!;
    octx.putImageData(this.ringCache, 0, 0);

    const drawX = this.mouseX - diam / 2;
    const drawY = this.mouseY - diam / 2;
    this.ctx.drawImage(offscreen, drawX, drawY);

    // ring outline at kernel boundary
    const { kernelRadius } = this.params;
    const pxPerCell = W / this.params.width;
    const radiusPx = kernelRadius * pxPerCell;

    this.ctx.beginPath();
    this.ctx.arc(this.mouseX, this.mouseY, radiusPx, 0, Math.PI * 2);
    this.ctx.strokeStyle = "rgba(255,255,255,0.15)";
    this.ctx.lineWidth = 1;
    this.ctx.stroke();

    // zero-crossing ring (where w(r) = 0, between excit and inhib zones)
    const zeroCrossR = this.findZeroCrossing();
    if (zeroCrossR > 0) {
      this.ctx.beginPath();
      this.ctx.arc(this.mouseX, this.mouseY, zeroCrossR * pxPerCell, 0, Math.PI * 2);
      this.ctx.strokeStyle = "rgba(255,255,255,0.3)";
      this.ctx.lineWidth = 1;
      this.ctx.setLineDash([3, 3]);
      this.ctx.stroke();
      this.ctx.setLineDash([]);
    }
  }

  private findZeroCrossing(): number {
    const { A_e, A_i, sigma_e, sigma_i, kernelRadius } = this.params;
    // binary search for r where w(r) = 0 between sigma_e and kernelRadius
    let lo = sigma_e, hi = kernelRadius;
    for (let i = 0; i < 32; i++) {
      const mid = (lo + hi) / 2;
      const r2 = mid * mid;
      const w = A_e * Math.exp(-r2 / (2 * sigma_e * sigma_e))
              - A_i * Math.exp(-r2 / (2 * sigma_i * sigma_i));
      if (w > 0) lo = mid; else hi = mid;
    }
    return (lo + hi) / 2;
  }

  private clear(): void {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }
}
