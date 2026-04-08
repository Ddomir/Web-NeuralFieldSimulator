// Initializes the WebGPU adapter and device.
// Returns null if WebGPU is unavailable so callers can show a friendly error.
export interface GPUContext {
  adapter: GPUAdapter;
  device: GPUDevice;
  format: GPUTextureFormat;
  context: GPUCanvasContext;
}

/**
 * Initializes webGPU onto user.
 * 
 * @param canvas 
 * @returns (In one object): {adapter, device, format, context}
 */
export async function initGPU(canvas: HTMLCanvasElement): Promise<GPUContext | null> {
  if (!navigator.gpu) return null;

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return null;

  const device = await adapter.requestDevice();

  const context = canvas.getContext("webgpu") as GPUCanvasContext;
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format });

  return { adapter, device, format, context };
}
