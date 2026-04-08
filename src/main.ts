import { initGPU } from "./gpu/device.ts";
import { DEFAULT_PARAMS, createFieldBuffers } from "./sim/field.ts";
import { ComputePipeline } from "./gpu/compute.ts";
import { RenderPipeline } from "./gpu/render.ts";

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
  const params = DEFAULT_PARAMS;

  // Allocate ping/pong/params buffers + ping init data
  const buffers = createFieldBuffers(device, params);

  // Build compute (field update) and render (colorized display) pipelines to be dispatched
  const compute = new ComputePipeline(device, buffers, params);
  const render  = new RenderPipeline(device, format, params);

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
