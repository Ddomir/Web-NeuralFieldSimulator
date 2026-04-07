const statusEl = document.getElementById("status")!;
const canvas = document.getElementById("webgpu-canvas") as HTMLCanvasElement;

function showError(msg: string): void {
  statusEl.textContent = msg;
  statusEl.classList.add("error");
  canvas.style.display = "none";
}

async function main(): Promise<void> {
  // Check API exists
  if (!navigator.gpu) {
    showError("WebGPU is not supported in this browser.");
    return;
  }

  // Request adapter
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    showError("No WebGPU adapter found (GPU unavailable or blocked).");
    return;
  }

  // Request device
  const device = await adapter.requestDevice();

  const context = canvas.getContext("webgpu") as GPUCanvasContext;
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format });

  statusEl.textContent = "WebGPU ready";

}

main();
