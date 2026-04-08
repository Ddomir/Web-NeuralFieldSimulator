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
  // Cache one bind group per field buffer so we don't recreate every frame
  private bindGroupCache = new Map<GPUBuffer, GPUBindGroup>();
  private readonly device: GPUDevice;
  private readonly layout: GPUBindGroupLayout;

  constructor(device: GPUDevice, format: GPUTextureFormat, params: FieldParams) {
    this.device = device;

    const module = device.createShaderModule({ code: renderWGSL });

    this.pipeline = device.createRenderPipeline({
      layout: "auto",  // Inferred expected layout
      vertex:   { module, entryPoint: "vs_main" }, // vertex stage (positioning)
      fragment: { module, entryPoint: "fs_main", targets: [{ format }] }, // fragment stage (coloring)
      primitive: { topology: "triangle-list" },
    });

    this.layout = this.pipeline.getBindGroupLayout(0);

    // Uniform: width (u32), height (u32), beta (f32), theta (f32) = 16 bytes
    // Mixed types require DataView to write each field at the correct byte offset.
    const paramBuf = new ArrayBuffer(16);
    const dv = new DataView(paramBuf);
    dv.setUint32(0,  params.width, true);
    dv.setUint32(4,  params.height, true);
    dv.setFloat32(8,  params.beta, true);
    dv.setFloat32(12, params.theta, true);
    this.paramsBuffer = device.createBuffer({
      size: paramBuf.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Uint8Array(this.paramsBuffer.getMappedRange()).set(new Uint8Array(paramBuf));
    this.paramsBuffer.unmap();
  }

  /**
   * Encodes a render pass that draws the field onto the swap-chain texture.
   * 
   * fieldBuffer: the buffer that holds the CURRENT (just-written) field state.
   */  
  draw(encoder: GPUCommandEncoder, fieldBuffer: GPUBuffer, view: GPUTextureView): void {
    // Lazily create a bind group for this buffer
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
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp:  "clear",
        storeOp: "store",
      }],
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroupCache.get(fieldBuffer)!);
    pass.draw(6); // 2 triangles = 6 vertices, no vertex buffer
    pass.end();
  }

  // Update beta/theta in the render uniform buffer when sliders change.
  updateSigmoidParams(beta: number, theta: number): void {
    const buf = new ArrayBuffer(8);
    const dv = new DataView(buf);
    dv.setFloat32(0, beta,  true);
    dv.setFloat32(4, theta, true);
    this.device.queue.writeBuffer(this.paramsBuffer, 8, buf);
  }
}
