import fieldUpdateWGSL from "./field_update.wgsl?raw";
import type { FieldBuffers, FieldParams } from "../sim/field.ts";

/**
 * Wraps the field_update compute pipeline.
 * 
 * Call step() each frame; it dispatches the compute shader and swaps ping/pong.
 */
export class ComputePipeline {
  private pipeline: GPUComputePipeline;
  // Two bind groups ping-ponging between each other
  private bindGroupA: GPUBindGroup; // ping-in, pong-out
  private bindGroupB: GPUBindGroup; // pong-in, ping-out
  private step = 0; // keeps track of frame number (for ping/pong swapping)

  // how many & which workgroups
  private readonly dispatchX: number;
  private readonly dispatchY: number;

  constructor(device: GPUDevice, buffers: FieldBuffers, params: FieldParams) {
    this.pipeline = device.createComputePipeline({
      layout: "auto", // Inferred expected layout
      compute: {
        module: device.createShaderModule({ code: fieldUpdateWGSL }),
        entryPoint: "main",
      },
    });

    const layout = this.pipeline.getBindGroupLayout(0); // get schema/signature of pipeline resources at each binding
    
    // Put actual buffer into the bindGroups
    this.bindGroupA = ComputePipeline.makeBindGroup(device, layout, buffers.params, buffers.ping, buffers.pong);
    this.bindGroupB = ComputePipeline.makeBindGroup(device, layout, buffers.params, buffers.pong, buffers.ping);

    // Each workgroup covers 16x16 cells; round up to cover whole grid
    this.dispatchX = Math.ceil(params.width  / 16);
    this.dispatchY = Math.ceil(params.height / 16);
  }


  /**
   * Create bind groups for buffers
   * 
   * @param device GPU
   * @param layout bind group layout
   * @param paramsBuffer uniform buffer
   * @param inBuffer storage buffer
   * @param outBuffer storage buffer
   * @returns Bind group of buffers
   */
  private static makeBindGroup(
    device: GPUDevice,
    layout: GPUBindGroupLayout,
    paramsBuffer: GPUBuffer,
    inBuffer: GPUBuffer,
    outBuffer: GPUBuffer,
  ): GPUBindGroup {
    return device.createBindGroup({
      layout,
      entries: [
        { binding: 0, resource: { buffer: paramsBuffer } },
        { binding: 1, resource: { buffer: inBuffer  } },
        { binding: 2, resource: { buffer: outBuffer } },
      ],
    });
  }

  /**
   * Encodes one simulation step into the provided command encoder.
   * 
   * Returns the buffer that now holds the latest field state (for the renderer).
  */
  dispatch(encoder: GPUCommandEncoder, buffers: FieldBuffers): GPUBuffer {
    const bindGroup = this.step % 2 === 0 ? this.bindGroupA : this.bindGroupB; // either ping (even) or pong (odd)
    const outBuffer = this.step % 2 === 0 ? buffers.pong   : buffers.ping; // either ping (odd) or pong (even)
    this.step++; // add to frame counter

    const pass = encoder.beginComputePass(); // use WGSL code
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(this.dispatchX, this.dispatchY); // sets off needed amount of workgroups
    pass.end(); // finish

    return outBuffer; // caller gives this to the render pipeline
  }
}
