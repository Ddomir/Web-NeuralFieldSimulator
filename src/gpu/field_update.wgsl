// Neural field update compute shader (Amari model, Euler integration)
//
// Each workgroup thread handles one cell (x, y).
// Reads from `field_in`, writes to `field_out`.

// Matches field.ts encodeParams ArrayBuffer
struct Params {
  width        : u32,
  height       : u32,
  dt           : f32,
  tau          : f32,
  beta         : f32,
  theta        : f32,
  A_e          : f32,
  A_i          : f32,
  sigma_e      : f32,
  sigma_i      : f32,
  kernel_radius: u32,
}

@group(0) @binding(0) var<uniform>           params    : Params;
@group(0) @binding(1) var<storage, read>     field_in  : array<f32>;
@group(0) @binding(2) var<storage, read_write> field_out : array<f32>;

// Sigmoid firing rate
// f(u) = 1 / (1 + exp(-beta * (u - theta)))
fn sigmoid(u: f32) -> f32 {
  return 1.0 / (1.0 + exp(-params.beta * (u - params.theta)));
}

// Mexican hat connectivity kernel evaluated at squared distance r²
// w(r) = A_e * exp(-r² / (2*sigma_e²)) - A_i * exp(-r² / (2*sigma_i²))
fn mexican_hat(r2: f32) -> f32 {
  let inv_2se2 = 1.0 / (2.0 * params.sigma_e * params.sigma_e);
  let inv_2si2 = 1.0 / (2.0 * params.sigma_i * params.sigma_i);
  return params.A_e * exp(-r2 * inv_2se2) - params.A_i * exp(-r2 * inv_2si2);
}

// Update rule:
// tau * du/dt = -u + integral(w(r) * f(u)) + I
// => u_new = u + dt/tau * (-u + conv + I)
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  let W = params.width;
  let H = params.height;

  if (x >= W || y >= H) { return; } // OOB check

  let idx   = y * W + x;
  let u     = field_in[idx];
  let R     = i32(params.kernel_radius);

  // Convolve field_in with the Mexican hat kernel over a (2R+1)² neighbourhood.
  // Wrap edges with torus topology so activity can't leak at boundaries.
  // Kernel amplitudes (A_e, A_i) are pre-scaled in field.ts so conv stays in ~[-1,1].
  var conv: f32 = 0.0;
  for (var dy = -R; dy <= R; dy++) {
    for (var dx = -R; dx <= R; dx++) {
      let nx = (i32(x) + dx + i32(W)) % i32(W);
      let ny = (i32(y) + dy + i32(H)) % i32(H);
      let r2 = f32(dx * dx + dy * dy);
      let w  = mexican_hat(r2);
      conv  += w * sigmoid(field_in[u32(ny) * W + u32(nx)]);
    }
  }

  // Euler step: tau * du/dt = -u + conv  (no external input this pass)
  let du = (params.dt / params.tau) * (-u + conv);
  field_out[idx] = clamp(u + du, -3.0, 3.0); // clamp prevents runaway blow-up
}
