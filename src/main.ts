import { initGPU } from "./gpu/device.ts";
import { DEFAULT_PARAMS, createFieldBuffers, updateParamBuffer, injectExcitation } from "./sim/field.ts";
import { ComputePipeline } from "./gpu/compute.ts";
import { RenderPipeline } from "./gpu/render.ts";
import { ControlsPanel } from "./ui/controls.ts";
import { DebugOverlay } from "./ui/debug.ts";

const statusEl = document.getElementById("status")!;
const canvas   = document.getElementById("webgpu-canvas") as HTMLCanvasElement;

function showError(msg: string): void {
  statusEl.textContent = msg;
  statusEl.classList.add("error");
  canvas.style.display = "none";
}

async function main(): Promise<void> {
  const gpu = await initGPU(canvas);
  if (!gpu) { showError("WebGPU is not supported or unavailable."); return; }

  const { device, format, context } = gpu;
  const params = { ...DEFAULT_PARAMS };

  const buffers = createFieldBuffers(device, params);
  const compute = new ComputePipeline(device, buffers, params);
  const render  = new RenderPipeline(device, format, params);
  render.updateSigmoidParams(params.beta, params.theta); // sync render buffer to initial params

  new ControlsPanel(
    document.body,
    params,
    (updated) => {
      Object.assign(params, updated);
      updateParamBuffer(device, buffers, params);
      render.updateSigmoidParams(params.beta, params.theta);
    },
  );

  function handlePointer(e: PointerEvent): void {
    if (!(e.buttons & 1)) return;
    const buf = compute.currentBuffer;
    if (!buf) return;
    const rect = canvas.getBoundingClientRect();
    const cx = Math.floor(((e.clientX - rect.left) / rect.width)  * params.width);
    const cy = Math.floor(((e.clientY - rect.top)  / rect.height) * params.height);
    injectExcitation(device, buf, params, cx, cy);
  }

  canvas.addEventListener("pointerdown", handlePointer);
  canvas.addEventListener("pointermove", handlePointer);

  const debug = new DebugOverlay(
    device,
    () => compute.currentBuffer,
    params.width * params.height,
    () => params,
  );

  statusEl.textContent = "Running";

  // Seed a central wave on startup.
  setTimeout(() => {
    const buf = compute.currentBuffer;
    if (buf) injectExcitation(device, buf, params, params.width / 2 | 0, params.height / 2 | 0);
  }, 200);

  const STEPS_PER_FRAME = 12;

  function frame(): void {
    params.frame = (params.frame + 1) & 0xffffffff;
    updateParamBuffer(device, buffers, params);

    const encoder = device.createCommandEncoder();
    let fieldNow!: GPUBuffer;
    for (let i = 0; i < STEPS_PER_FRAME; i++) {
      fieldNow = compute.dispatch(encoder, buffers);
    }

    render.draw(encoder, fieldNow, context.getCurrentTexture().createView());
    device.queue.submit([encoder.finish()]);
    debug.tick();
    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}

main();
