import { initGPU } from "./gpu/device.ts";
import { DEFAULT_PARAMS, createFieldBuffers, updateParamBuffer } from "./sim/field.ts";
import { ComputePipeline } from "./gpu/compute.ts";
import { RenderPipeline } from "./gpu/render.ts";
import { ControlsPanel } from "./ui/controls.ts";

const statusEl = document.getElementById("status")!;
const canvas   = document.getElementById("webgpu-canvas") as HTMLCanvasElement;

function showError(msg: string): void {
  statusEl.textContent = msg;
  statusEl.classList.add("error");
  canvas.style.display = "none";
}

async function main(): Promise<void> {
  // Check for GPU compatibility.
  const gpu = await initGPU(canvas);
  if (!gpu) { showError("WebGPU is not supported or unavailable."); return; }

  const { device, format, context } = gpu;
  // params is mutable — sliders write into this object then push to GPU
  const params = { ...DEFAULT_PARAMS };

  // Allocate ping/pong/params buffers + ping init data
  const buffers = createFieldBuffers(device, params);

  // Build compute (field update) and render (colorized display) pipelines to be dispatched
  const compute = new ComputePipeline(device, buffers, params);
  const render  = new RenderPipeline(device, format, params);

  // Controls panel — on any slider change, push updated params to both GPU buffers
  new ControlsPanel(document.body, params, (updated) => {
    Object.assign(params, updated);
    updateParamBuffer(device, buffers, params);
    render.updateSigmoidParams(params.beta, params.theta);
  });

  statusEl.textContent = "Running";

  // Main loop: one compute step + one render pass per animation frame
  function frame(): void {
    const encoder  = device.createCommandEncoder();
    const fieldNow = compute.dispatch(encoder, buffers);
    const swapView = context.getCurrentTexture().createView();
    render.draw(encoder, fieldNow, swapView);
    device.queue.submit([encoder.finish()]);
    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}

main();
