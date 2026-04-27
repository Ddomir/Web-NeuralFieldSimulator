import type { FieldParams } from "../sim/field.ts";
import { PRESETS } from "../sim/field.ts";
import { EquationPanel } from "./equation.ts";

interface SliderDef {
  key: keyof FieldParams;
  label: string;
  min: number;
  max: number;
  step: number;
  color?: string;
}

const SLIDERS: SliderDef[] = [
  { key: "noise",   label: "Noise",             min: 0.0,    max: 0.5,   step: 0.001,  color: "#c792ea" },
  { key: "dt",      label: "Timestep (dt)",      min: 0.005,  max: 0.1,   step: 0.005  },
  { key: "tau",     label: "Membrane τ",         min: 0.1,    max: 5.0,   step: 0.1,    color: "#7eb8f7" },
  { key: "beta",    label: "Sigmoid gain (β)",   min: 1.0,    max: 30.0,  step: 0.5,    color: "#f7c948" },
  { key: "theta",   label: "Threshold (θ)",      min: -1.0,   max: 1.0,   step: 0.05,   color: "#f7c948" },
  { key: "A_e",     label: "Excit. amp (Aₑ)",   min: 0.0005, max: 0.025, step: 0.0005, color: "#6dd6a0" },
  { key: "A_i",     label: "Inhib. amp (Aᵢ)",   min: 0.0005, max: 0.025, step: 0.0005, color: "#f47c7c" },
  { key: "sigma_e", label: "Excit. spread (σₑ)", min: 2.0,    max: 20.0,  step: 1.0,    color: "#3ecfcf" },
  { key: "sigma_i", label: "Inhib. spread (σᵢ)", min: 4.0,    max: 30.0,  step: 1.0,    color: "#c97bf7" },
];

export class ControlsPanel {
  private params: FieldParams;
  private onChange: (p: FieldParams) => void;
  private onPreset: (p: FieldParams) => void;
  private equationPanel!: EquationPanel;
  private sliderEls = new Map<keyof FieldParams, { slider: HTMLInputElement; display: HTMLSpanElement }>();
  private tauVSlider!: HTMLInputElement;
  private tauVDisplay!: HTMLSpanElement;
  private adaptCheckbox!: HTMLInputElement;

  constructor(
    container: HTMLElement,
    initialParams: FieldParams,
    onChange: (p: FieldParams) => void,
    onPreset: (p: FieldParams) => void,
  ) {
    this.params   = { ...initialParams };
    this.onChange = onChange;
    this.onPreset = onPreset;
    this.build(container);
  }

  private build(container: HTMLElement): void {
    const panel = document.createElement("div");
    panel.id = "controls-panel";

    this.equationPanel = new EquationPanel(panel, this.params);
    panel.appendChild(this.buildPresetRow());

    for (const def of SLIDERS) {
      panel.appendChild(this.buildSliderRow(def));
    }

    panel.appendChild(this.buildAdaptationRow());
    container.appendChild(panel);
  }

  private buildPresetRow(): HTMLElement {
    const row = document.createElement("div");
    row.className = "preset-row";

    for (const preset of PRESETS) {
      const btn = document.createElement("button");
      btn.textContent = preset.name;
      btn.className = "preset-btn";
      btn.addEventListener("click", () => {
        Object.assign(this.params, preset.params);
        this.syncSliders();
        this.onPreset({ ...this.params });
        this.equationPanel.update(this.params);
      });
      row.appendChild(btn);
    }

    return row;
  }

  private buildSliderRow(def: SliderDef): HTMLElement {
    const row = document.createElement("div");
    row.className = "slider-row";
    if (def.color) row.style.setProperty("--slider-color", def.color);

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
      this.equationPanel.update(this.params);
    });

    this.sliderEls.set(def.key, { slider, display: valueDisplay });
    row.appendChild(label);
    row.appendChild(slider);
    row.appendChild(valueDisplay);
    return row;
  }

  private syncSliders(): void {
    for (const [key, { slider, display }] of this.sliderEls) {
      const val = this.params[key] as number;
      slider.value = String(val);
      display.textContent = this.formatValue(val);
    }
    const adaptOn = this.params.tau_v < 100;
    this.adaptCheckbox.checked = adaptOn;
    this.tauVSlider.disabled = !adaptOn;
    const effectiveTauV = adaptOn ? this.params.tau_v : 10;
    this.tauVSlider.value = String(effectiveTauV);
    this.tauVDisplay.textContent = this.formatValue(effectiveTauV);
  }

  private buildAdaptationRow(): HTMLElement {
    const section = document.createElement("div");
    section.className = "adapt-section";
    section.style.setProperty("--slider-color", "#e07030");

    const header = document.createElement("div");
    header.className = "slider-row adapt-header";

    const label = document.createElement("label");
    label.textContent = "Adaptation (τᵥ)";

    this.adaptCheckbox = document.createElement("input");
    this.adaptCheckbox.type    = "checkbox";
    this.adaptCheckbox.checked = this.params.tau_v < 100;

    this.tauVDisplay = document.createElement("span");
    this.tauVDisplay.className = "slider-value";

    this.tauVSlider = document.createElement("input");
    this.tauVSlider.type = "range";
    this.tauVSlider.min  = "0.5";
    this.tauVSlider.max  = "30";
    this.tauVSlider.step = "0.5";

    const effectiveTauV = this.params.tau_v < 100 ? this.params.tau_v : 10;
    this.tauVSlider.value        = String(effectiveTauV);
    this.tauVDisplay.textContent = this.formatValue(effectiveTauV);
    this.tauVSlider.disabled     = !this.adaptCheckbox.checked;

    this.adaptCheckbox.addEventListener("change", () => {
      const on = this.adaptCheckbox.checked;
      this.tauVSlider.disabled = !on;
      this.params.tau_v = on ? parseFloat(this.tauVSlider.value) : 999;
      this.onChange({ ...this.params });
    });

    this.tauVSlider.addEventListener("input", () => {
      const val = parseFloat(this.tauVSlider.value);
      this.params.tau_v = val;
      this.tauVDisplay.textContent = this.formatValue(val);
      this.onChange({ ...this.params });
    });

    header.appendChild(label);
    header.appendChild(this.adaptCheckbox);
    header.appendChild(this.tauVDisplay);
    section.appendChild(header);
    section.appendChild(this.tauVSlider);
    return section;
  }

  private formatValue(v: number): string {
    if (v < 0.01 && v > 0) return v.toExponential(2);
    if (Number.isInteger(v)) return v.toFixed(0);
    return v.toFixed(3);
  }
}
