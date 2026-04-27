// Equation panel: renders the two Amari equations with hoverable terms.
// Each term shows a tooltip explaining what it represents and which param controls it.

import type { FieldParams } from "../sim/field.ts";

interface Term {
  text: string;
  tip: string;
  highlight?: string; // css color class key
}

// Discrete Euler update (matches field_update.wgsl exactly):
//   u_new = u + (dt/τ) * (−u + ∑ w(r)·f(u) − v + tonic + η)
//   v_new = v + (dt/τᵥ) * (u − v)

const EQ1: (Term | string)[] = [
  { text: "u_new", tip: "Next membrane potential — the field state after this timestep.", highlight: "decay" },
  " = ",
  { text: "u", tip: "Current membrane potential at this cell.", highlight: "decay" },
  " + ",
  { text: "(dt/τ)", tip: "Euler step size — dt is the timestep, τ is the membrane time constant controlling how fast u responds. Smaller τ → faster dynamics.", highlight: "tau" },
  " · ( ",
  { text: "−u", tip: "Passive decay — without any input the field relaxes back to zero.", highlight: "decay" },
  " + ",
  { text: "∑ w(r)·f(u)", tip: "Lateral input: sums over all neighbors within kernel_radius. w(r) is the Mexican hat kernel (excitatory near, inhibitory far); f(u) is the sigmoid firing rate gated by β and θ.", highlight: "conv" },
  " − ",
  { text: "v", tip: "Spike-frequency adaptation — slow negative feedback that prevents sustained firing. Enable and tune with the Adaptation toggle.", highlight: "adapt" },
  " + ",
  { text: "tonic", tip: "Constant background drive added every step. Raises resting excitability across the whole field.", highlight: "tonic" },
  " + ",
  { text: "η", tip: "Per-cell noise drawn each step. Amplitude set by the Noise slider.", highlight: "noise" },
  " )",
];

const EQ2: (Term | string)[] = [
  { text: "v_new", tip: "Next adaptation state.", highlight: "adapt" },
  " = ",
  { text: "v", tip: "Current adaptation value.", highlight: "adapt" },
  " + ",
  { text: "(dt/τᵥ)", tip: "Adaptation step size — τᵥ controls how slowly v chases u. Long τᵥ → adaptation can't keep up → runaway excitation (seizure mode).", highlight: "tau" },
  " · ( ",
  { text: "u", tip: "Membrane potential drives v upward — the more excited the cell, the faster adaptation builds.", highlight: "decay" },
  " − ",
  { text: "v", tip: "Passive decay of the adaptation variable back to zero.", highlight: "adapt" },
  " )",
];

const EQ_KERNEL: (Term | string)[] = [
  { text: "w(r)", tip: "Mexican hat kernel — the connectivity weight between two cells at distance r.", highlight: "conv" },
  " = ",
  { text: "Aₑ", tip: "Excitatory amplitude — scales the peak of the central excitatory Gaussian. Controlled by 'Excit. amp' slider.", highlight: "excit_amp" },
  "·exp(−r²/",
  { text: "2σₑ²", tip: "Excitatory spread — width of the central excitatory lobe. Controlled by 'Excit. spread' slider.", highlight: "excit_sig" },
  ") − ",
  { text: "Aᵢ", tip: "Inhibitory amplitude — scales the surround inhibitory Gaussian. Controlled by 'Inhib. amp' slider.", highlight: "inhib_amp" },
  "·exp(−r²/",
  { text: "2σᵢ²", tip: "Inhibitory spread — width of the surround inhibitory lobe. Controlled by 'Inhib. spread' slider. Must be wider than σₑ for a stable Mexican hat shape.", highlight: "inhib_sig" },
  ")",
];

const HIGHLIGHT_COLORS: Record<string, string> = {
  tau:       "#7eb8f7",
  decay:     "#aaa",
  conv:      "#f7c948",
  adapt:     "#e07030",
  tonic:     "#88d8a0",
  noise:     "#c792ea",
  excit_amp: "#6dd6a0",  // excitatory amplitude — green
  excit_sig: "#3ecfcf",  // excitatory spread — teal
  inhib_amp: "#f47c7c",  // inhibitory amplitude — red
  inhib_sig: "#c97bf7",  // inhibitory spread — violet
};

export class EquationPanel {
  private kernelCanvas!: HTMLCanvasElement;
  private kernelCtx!: CanvasRenderingContext2D;

  constructor(container: HTMLElement, initialParams: FieldParams) {
    const panel = document.createElement("div");
    panel.className = "eq-panel";

    const title = document.createElement("div");
    title.className = "eq-title";
    title.textContent = "Amari Neural Field";
    panel.appendChild(title);

    panel.appendChild(this.buildEquation(EQ1, "eq-line"));
    panel.appendChild(this.buildEquation(EQ2, "eq-line eq-line--sub"));

    const kernelTitle = document.createElement("div");
    kernelTitle.className = "eq-title";
    kernelTitle.style.marginTop = "6px";
    kernelTitle.textContent = "Mexican Hat Kernel";
    panel.appendChild(kernelTitle);

    panel.appendChild(this.buildEquation(EQ_KERNEL, "eq-line"));
    panel.appendChild(this.buildKernelCanvas(initialParams));

    const hint = document.createElement("div");
    hint.className = "eq-hint";
    hint.textContent = "hover terms for details";
    panel.appendChild(hint);

    container.appendChild(panel);
  }

  update(params: FieldParams): void {
    this.drawKernel(params);
  }

  private buildKernelCanvas(params: FieldParams): HTMLElement {
    const wrap = document.createElement("div");
    wrap.className = "kernel-canvas-wrap";

    const canvas = document.createElement("canvas");
    canvas.width  = 240;
    canvas.height = 80;
    canvas.className = "kernel-canvas";
    this.kernelCanvas = canvas;
    this.kernelCtx = canvas.getContext("2d")!;

    wrap.appendChild(canvas);
    this.drawKernel(params);
    return wrap;
  }

  private drawKernel(params: FieldParams): void {
    const ctx = this.kernelCtx;
    const W = this.kernelCanvas.width;
    const H = this.kernelCanvas.height;
    const { A_e, A_i, sigma_e, sigma_i, kernelRadius } = params;

    ctx.clearRect(0, 0, W, H);

    // sample w(r) over [0, kernelRadius]
    const samples = W;
    const values: number[] = [];
    for (let i = 0; i < samples; i++) {
      const r = (i / (samples - 1)) * kernelRadius;
      const r2 = r * r;
      values.push(A_e * Math.exp(-r2 / (2 * sigma_e * sigma_e))
                - A_i * Math.exp(-r2 / (2 * sigma_i * sigma_i)));
    }

    const maxAbs = Math.max(...values.map(Math.abs), 1e-6);
    const pad = 6;
    const midY = H / 2;
    const scaleY = (H / 2 - pad) / maxAbs;

    // individual gaussian curves
    const excitValues: number[] = [];
    const inhibValues: number[] = [];
    for (let i = 0; i < samples; i++) {
      const r = (i / (samples - 1)) * kernelRadius;
      const r2 = r * r;
      excitValues.push( A_e * Math.exp(-r2 / (2 * sigma_e * sigma_e)));
      inhibValues.push(-A_i * Math.exp(-r2 / (2 * sigma_i * sigma_i)));
    }

    const drawCurve = (vals: number[], stroke: string, fill: string) => {
      ctx.beginPath();
      for (let i = 0; i < samples; i++) {
        const x = (i / (samples - 1)) * W;
        const y = midY - vals[i] * scaleY;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.lineTo(W, midY);
      ctx.lineTo(0, midY);
      ctx.closePath();
      ctx.fillStyle = fill;
      ctx.fill();
      ctx.beginPath();
      for (let i = 0; i < samples; i++) {
        const x = (i / (samples - 1)) * W;
        const y = midY - vals[i] * scaleY;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.strokeStyle = stroke;
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.stroke();
      ctx.setLineDash([]);
    };

    // zero line
    ctx.strokeStyle = "#2a2a2a";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, midY);
    ctx.lineTo(W, midY);
    ctx.stroke();

    // excitatory gaussian — green/teal
    drawCurve(excitValues, "#3ecfcf", "rgba(109,214,160,0.10)");
    // inhibitory gaussian — red/violet
    drawCurve(inhibValues, "#c97bf7", "rgba(244,124,124,0.10)");

    // net w(r) curve — white/bright on top
    ctx.beginPath();
    for (let i = 0; i < samples; i++) {
      const x = (i / (samples - 1)) * W;
      const y = midY - values[i] * scaleY;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.strokeStyle = "#ddd";
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // r axis label
    ctx.fillStyle = "#444";
    ctx.font = "9px monospace";
    ctx.fillText("0", 2, midY - 2);
    ctx.fillText(`r = ${kernelRadius}`, W - 36, midY - 2);
  }

  private buildEquation(parts: (Term | string)[], lineClass: string): HTMLElement {
    const line = document.createElement("div");
    line.className = lineClass;

    const expr = document.createElement("div");
    expr.className = "eq-expr";

    // Tooltip rendered after expr so it overlays below without shifting layout
    const tooltip = document.createElement("div");
    tooltip.className = "eq-tooltip";

    for (const part of parts) {
      if (typeof part === "string") {
        const span = document.createElement("span");
        span.className = "eq-op";
        span.textContent = part;
        expr.appendChild(span);
      } else {
        const span = document.createElement("span");
        span.className = "eq-term";
        span.textContent = part.text;
        if (part.highlight) {
          span.style.color = HIGHLIGHT_COLORS[part.highlight] ?? "#ccc";
        }

        span.addEventListener("mouseenter", () => {
          tooltip.textContent = part.tip;
          tooltip.classList.add("eq-tooltip--visible");
        });
        span.addEventListener("mouseleave", () => {
          tooltip.classList.remove("eq-tooltip--visible");
        });

        expr.appendChild(span);
      }
    }

    line.appendChild(expr);
    line.appendChild(tooltip);
    return line;
  }
}
