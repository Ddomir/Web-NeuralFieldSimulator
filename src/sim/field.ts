export interface FieldParams {
  width: number;
  height: number;
  dt: number;          // Euler timestep (s)
  tau: number;         // membrane time constant — controls decay speed
  beta: number;        // sigmoid steepness
  theta: number;       // sigmoid threshold (resting = 0)
  A_e: number;         // excitatory kernel amplitude
  A_i: number;         // inhibitory kernel amplitude
  sigma_e: number;     // excitatory spread (px)
  sigma_i: number;     // inhibitory spread (px)
  kernelRadius: number; // convolution truncation radius (px)
}

// Parameters derived from Amari (1977) pattern formation conditions.
export const DEFAULT_PARAMS: FieldParams = {
  width: 512,
  height: 512,
  dt: 0.05,
  tau: 1.0,
  beta: 10.0,
  theta: 0.25,
  A_e: 1.5 / (2 * Math.PI * 4  * 4),
  A_i: 1.0 / (2 * Math.PI * 10 * 10),
  sigma_e: 4.0,
  sigma_i: 10.0,
  kernelRadius: 20,
};

export interface FieldBuffers {
  // Two storage buffers ping-ponged each simulation step
  ping: GPUBuffer;
  pong: GPUBuffer;
  // Uniform buffer holding FieldParams laid out as packed f32/u32
  params: GPUBuffer;
}

/**
 * Build the packed ArrayBuffer that matches the WGSL Params struct layout.
 */
export function encodeParams(p: FieldParams): ArrayBuffer {
  const buf = new ArrayBuffer(11 * 4); // 11 fields × 4 bytes
  const view = new DataView(buf);
  view.setUint32(0,  p.width,        true);
  view.setUint32(4,  p.height,       true);
  view.setFloat32(8,  p.dt,          true);
  view.setFloat32(12, p.tau,         true);
  view.setFloat32(16, p.beta,        true);
  view.setFloat32(20, p.theta,       true);
  view.setFloat32(24, p.A_e,         true);
  view.setFloat32(28, p.A_i,         true);
  view.setFloat32(32, p.sigma_e,     true);
  view.setFloat32(36, p.sigma_i,     true);
  view.setUint32(40,  p.kernelRadius, true);
  return buf;
}

/**
 * Holds all neural field simulation parameters and GPU buffer references. 
 * The field is a flat WIDTHxHEIGHT grid of f32 membrane potentials.
 * 
 * Two buffers (ping/pong) are swapped each step so the compute shader can
 * read the current state and write the next without aliasing.

 * @param device GPUDevice
 * @param p field params
 * @returns Ping/pong storage buffers and param uniform buffer.
 */
export function createFieldBuffers(device: GPUDevice, p: FieldParams): FieldBuffers {
  const cellCount = p.width * p.height; // total cell count
  const byteSize = cellCount * 4; // one f32 per cell

  // Random initial membrane potentials [-0.5, 0.5] — enough to seed spatial variation
  // without immediately saturating the sigmoid
  const initData = new Float32Array(cellCount);
  for (let i = 0; i < cellCount; i++) {
    initData[i] = (Math.random() - 0.5) * 1.0;
  }

  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

  const ping = device.createBuffer({ size: byteSize, usage, mappedAtCreation: true }); // already mapped
  new Float32Array(ping.getMappedRange()).set(initData); // set init data into ping
  ping.unmap(); // unlocked back to GPU

  // pong starts empty (zeroed); will be fully written before first read
  const pong = device.createBuffer({ size: byteSize, usage });

  const paramData = encodeParams(p); // packs data into ArrayBuffer

  // create params uniform buffer
  const params = device.createBuffer({
    size: paramData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint8Array(params.getMappedRange()).set(new Uint8Array(paramData));
  params.unmap();

  return { ping, pong, params };
}
