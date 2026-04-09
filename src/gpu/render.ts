import renderWGSL from "./render.wgsl?raw";
import type { FieldParams } from "../sim/field.ts";

/**
 * Fullscreen-quad render pipeline.
 * 
 * Reads the latest field storage buffer and colorizes each cell.
 */
export class RenderPipeline {
  private pipeline: GPURenderPipeline;
  private paramsBuffer: GPUBuffer;
  private bindGroupCache = new Map<GPUBuffer, GPUBindGroup>();
  private readonly device: GPUDevice;
  private readonly layout: GPUBindGroupLayout;

  constructor(device: GPUDevice, format: GPUTextureFormat, params: FieldParams) {
    this.device = device;

    const module = device.createShaderModule({ code: renderWGSL });

    this.pipeline = device.createRenderPipeline({
      layout: "auto",
      vertex:   { module, entryPoint: "vs_main" },
      fragment: { module, entryPoint: "fs_main", targets: [{ format }] },
      primitive: { topology: "triangle-list" },
    });

    this.layout = this.pipeline.getBindGroupLayout(0);

    // Uniform: width (u32), height (u32), beta (f32), theta (f32) = 16 bytes
    this.paramsBuffer = device.createBuffer({
      size:  16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    this.writeParams(params.width, params.height, params.beta, params.theta, true);
    this.paramsBuffer.unmap();
  }

  private writeParams(width: number, height: number, beta: number, theta: number, mapped = false): void {
    const buf = new ArrayBuffer(16);
    const dv  = new DataView(buf);
    dv.setUint32 ( 0, width,  true);
    dv.setUint32 ( 4, height, true);
    dv.setFloat32( 8, beta,   true);
    dv.setFloat32(12, theta,  true);
    if (mapped) {
      new Uint8Array(this.paramsBuffer.getMappedRange()).set(new Uint8Array(buf));
    } else {
      this.device.queue.writeBuffer(this.paramsBuffer, 0, buf);
    }
  }

  updateSigmoidParams(beta: number, theta: number): void {
    const buf = new ArrayBuffer(8);
    const dv  = new DataView(buf);
    dv.setFloat32(0, beta,  true);
    dv.setFloat32(4, theta, true);
    this.device.queue.writeBuffer(this.paramsBuffer, 8, buf);
  }

  draw(encoder: GPUCommandEncoder, fieldBuffer: GPUBuffer, view: GPUTextureView): void {
    if (!this.bindGroupCache.has(fieldBuffer)) {
      this.bindGroupCache.set(fieldBuffer, this.device.createBindGroup({
        layout: this.layout,
        entries: [
          { binding: 0, resource: { buffer: this.paramsBuffer } },
          { binding: 1, resource: { buffer: fieldBuffer       } },
        ],
      }));
    }

    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view,
        clearValue: { r: 0.02, g: 0.02, b: 0.1, a: 1 },
        loadOp:  "clear",
        storeOp: "store",
      }],
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroupCache.get(fieldBuffer)!);
    pass.draw(6);
    pass.end();
  }
}
