export interface FieldParams {
  width: number;
  height: number;
  dt: number;
  tau: number;
  tau_v: number;
  beta: number;
  theta: number;
  A_e: number;
  A_i: number;
  sigma_e: number;
  sigma_i: number;
  kernelRadius: number;
  noise: number;
  frame: number;
  tonic: number;  // constant baseline drive added to each cell each step
}

const W = 256;
const H = 256;

export interface Preset {
  name: string;
  params: Partial<FieldParams>;
}

export const PRESETS: Preset[] = [
  {
    name: "Evoked Wave",
    params: {
      dt: 0.02, tau: 1.0, tau_v: 5.0,
      beta: 20.0, theta: 0.20,
      A_e: 1.4 / (2 * Math.PI * 4 * 4),
      A_i: 0.9 / (2 * Math.PI * 8 * 8),
      sigma_e: 4, sigma_i: 8, kernelRadius: 16, noise: 0.001,
    },
  },
  {
    name: "Alpha Rest",
    params: {
      dt: 0.02, tau: 1.0, tau_v: 10.0,
      beta: 8.0, theta: 0.20,
      A_e: 1.0 / (2 * Math.PI * 4 * 4),
      A_i: 0.4 / (2 * Math.PI * 8 * 8),
      sigma_e: 4, sigma_i: 8, kernelRadius: 16, noise: 0.050,
      tonic: 0.10,
    },
  },
  {
    name: "Seizure",
    params: {
      dt: 0.02, tau: 1.0, tau_v: 40.0,
      beta: 12.0, theta: 0.20,
      A_e: 1.2 / (2 * Math.PI * 4 * 4),
      A_i: 0.4 / (2 * Math.PI * 8 * 8),
      sigma_e: 4, sigma_i: 8, kernelRadius: 16, noise: 0.001,
      tonic: 0.0,
    },
  },
];

// Evoked wave
export const DEFAULT_PARAMS: FieldParams = {
  width:  W,
  height: H,
  dt:       0.02,
  tau:      1.0,
  tau_v:    5.0,
  beta:     20.0,
  theta:    0.20,
  A_e:      1.4 / (2 * Math.PI * 4 * 4),
  A_i:      0.9 / (2 * Math.PI * 8 * 8),
  sigma_e:  4,
  sigma_i:  8,
  kernelRadius: 16,
  noise:    0.001,
  frame:    0,
  tonic:    0.0,
};

export interface FieldBuffers {
  ping: GPUBuffer;
  pong: GPUBuffer;
  vPing: GPUBuffer;
  vPong: GPUBuffer;
  params: GPUBuffer;
}

export function encodeParams(p: FieldParams): ArrayBuffer {
  const buf = new ArrayBuffer(15 * 4);
  const view = new DataView(buf);
  view.setUint32( 0,  p.width,        true);
  view.setUint32( 4,  p.height,       true);
  view.setFloat32( 8, p.dt,           true);
  view.setFloat32(12, p.tau,          true);
  view.setFloat32(16, p.tau_v,        true);
  view.setFloat32(20, p.beta,         true);
  view.setFloat32(24, p.theta,        true);
  view.setFloat32(28, p.A_e,          true);
  view.setFloat32(32, p.A_i,          true);
  view.setFloat32(36, p.sigma_e,      true);
  view.setFloat32(40, p.sigma_i,      true);
  view.setUint32( 44, p.kernelRadius, true);
  view.setFloat32(48, p.noise,        true);
  view.setUint32( 52, p.frame,        true);
  view.setFloat32(56, p.tonic,        true);
  return buf;
}

export function createFieldBuffers(device: GPUDevice, p: FieldParams): FieldBuffers {
  const cellCount = p.width * p.height;
  const byteSize = cellCount * 4;

  const initData = new Float32Array(cellCount);
  for (let i = 0; i < cellCount; i++) {
    initData[i] = (Math.random() - 0.5) * 0.2;
  }

  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

  const ping = device.createBuffer({ size: byteSize, usage, mappedAtCreation: true });
  new Float32Array(ping.getMappedRange()).set(initData);
  ping.unmap();

  const pong  = device.createBuffer({ size: byteSize, usage });
  const vPing = device.createBuffer({ size: byteSize, usage });
  const vPong = device.createBuffer({ size: byteSize, usage });

  const paramData = encodeParams(p);
  const params = device.createBuffer({
    size: paramData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint8Array(params.getMappedRange()).set(new Uint8Array(paramData));
  params.unmap();

  return { ping, pong, vPing, vPong, params };
}

export function resetField(device: GPUDevice, buffers: FieldBuffers, p: FieldParams): void {
  const cellCount = p.width * p.height;
  const uData = new Float32Array(cellCount);
  const vData = new Float32Array(cellCount);
  for (let i = 0; i < cellCount; i++) {
    uData[i] = (Math.random() - 0.5) * 0.2;
  }
  device.queue.writeBuffer(buffers.ping,  0, uData);
  device.queue.writeBuffer(buffers.pong,  0, uData);
  device.queue.writeBuffer(buffers.vPing, 0, vData);
  device.queue.writeBuffer(buffers.vPong, 0, vData);
}

export async function injectExcitation(
  device: GPUDevice,
  fieldBuffer: GPUBuffer,
  p: FieldParams,
  cx: number,
  cy: number,
  radius = 18,
  strength = 3.0,
): Promise<void> {
  const sigma2 = (radius * 0.6) ** 2;

  const x0 = Math.max(0, cx - radius);
  const y0 = Math.max(0, cy - radius);
  const x1 = Math.min(p.width  - 1, cx + radius);
  const y1 = Math.min(p.height - 1, cy + radius);

  const patchW = x1 - x0 + 1;
  const patchH = y1 - y0 + 1;
  const rowBytes = patchW * 4;

  const staging = device.createBuffer({
    size:  patchH * rowBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const encoder = device.createCommandEncoder();
  for (let row = 0; row < patchH; row++) {
    encoder.copyBufferToBuffer(
      fieldBuffer, ((y0 + row) * p.width + x0) * 4,
      staging,     row * rowBytes,
      rowBytes,
    );
  }
  device.queue.submit([encoder.finish()]);

  await staging.mapAsync(GPUMapMode.READ);
  const readback = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();

  for (let row = 0; row < patchH; row++) {
    for (let col = 0; col < patchW; col++) {
      const dx = (x0 + col) - cx;
      const dy = (y0 + row) - cy;
      if (dx * dx + dy * dy > radius * radius) continue; // clip to circle
      readback[row * patchW + col] += strength * Math.exp(-(dx * dx + dy * dy) / sigma2);
    }
  }

  for (let row = 0; row < patchH; row++) {
    device.queue.writeBuffer(
      fieldBuffer, ((y0 + row) * p.width + x0) * 4,
      readback.buffer, row * rowBytes, rowBytes,
    );
  }
}

export function updateParamBuffer(device: GPUDevice, buffers: FieldBuffers, p: FieldParams): void {
  device.queue.writeBuffer(buffers.params, 0, encodeParams(p));
}
