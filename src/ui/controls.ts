// Controls panel — builds sliders for every tunable FieldParam.
// onChange fires whenever any slider moves, passing the full updated params object.

import type { FieldParams } from "../sim/field.ts";

interface SliderDef {
  key: keyof FieldParams;
  label: string;
  min: number;
  max: number;
  step: number;
}

const SLIDERS: SliderDef[] = [
  { key: "dt",      label: "Timestep (dt)",         min: 0.01, max: 0.2,  step: 0.01 },
  { key: "tau",     label: "Time constant (τ)",      min: 0.1,  max: 5.0,  step: 0.1  },
  { key: "beta",    label: "Sigmoid steepness (β)",  min: 1.0,  max: 20.0, step: 0.5  },
  { key: "theta",   label: "Firing threshold (θ)",   min: -1.0, max: 1.0,  step: 0.05 },
  { key: "A_e",     label: "Excitatory amplitude",   min: 0.0,  max: 0.1,  step: 0.001 },
  { key: "A_i",     label: "Inhibitory amplitude",   min: 0.0,  max: 0.1,  step: 0.001 },
  { key: "sigma_e", label: "Excitatory spread (σₑ)", min: 1.0,  max: 20.0, step: 0.5  },
  { key: "sigma_i", label: "Inhibitory spread (σᵢ)", min: 1.0,  max: 40.0, step: 0.5  },
];

export class ControlsPanel {
  private params: FieldParams;
  private onChange: (p: FieldParams) => void;

  constructor(container: HTMLElement, initialParams: FieldParams, onChange: (p: FieldParams) => void) {
    this.params   = { ...initialParams };
    this.onChange = onChange;
    this.build(container);
  }

  private build(container: HTMLElement): void {
    const panel = document.createElement("div");
    panel.id = "controls-panel";

    for (const def of SLIDERS) {
      const row = document.createElement("div");
      row.className = "slider-row";

      const label = document.createElement("label");
      label.textContent = def.label;

      const valueDisplay = document.createElement("span");
      valueDisplay.className = "slider-value";
      valueDisplay.textContent = this.formatValue(this.params[def.key] as number);

      const slider = document.createElement("input");
      slider.type  = "range";
      slider.min   = String(def.min);
      slider.max   = String(def.max);
      slider.step  = String(def.step);
      slider.value = String(this.params[def.key]);

      slider.addEventListener("input", () => {
        const val = parseFloat(slider.value);
        (this.params as unknown as Record<string, number>)[def.key] = val;
        valueDisplay.textContent = this.formatValue(val);
        this.onChange({ ...this.params });
      });

      row.appendChild(label);
      row.appendChild(slider);
      row.appendChild(valueDisplay);
      panel.appendChild(row);
    }

    container.appendChild(panel);
  }

  private formatValue(v: number): string {
    // Show enough decimal places based on magnitude
    return v < 0.01 ? v.toExponential(2) : v.toFixed(3);
  }
}
