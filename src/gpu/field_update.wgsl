// Neural field update — Amari model with optional spike-frequency adaptation + noise
//
// Equations (from Amari 1977, Mexican hat kernel form from Coombes et al.):
//   tau * du/dt = -u + Σ w(|x-y|) * f(u(y)) + eta            [membrane potential]
//   tau_v * dv/dt = u - v                                      [adaptation, optional]

struct Params {
  width        : u32,
  height       : u32,
  dt           : f32,
  tau          : f32,
  tau_v        : f32,
  beta         : f32,
  theta        : f32,
  A_e          : f32,
  A_i          : f32,
  sigma_e      : f32,  // in grid cells
  sigma_i      : f32,  // in grid cells
  kernel_radius: u32,  // in grid cells
  noise        : f32,
  frame        : u32,
  tonic        : f32,  // constant baseline drive
}

@group(0) @binding(0) var<uniform>             params : Params;
@group(0) @binding(1) var<storage, read>       u_in   : array<f32>;
@group(0) @binding(2) var<storage, read_write> u_out  : array<f32>;
@group(0) @binding(3) var<storage, read>       v_in   : array<f32>;
@group(0) @binding(4) var<storage, read_write> v_out  : array<f32>;

fn sigmoid(u: f32) -> f32 {
  return 1.0 / (1.0 + exp(-params.beta * (u - params.theta)));
}

// Mexican hat kernel in physical distance r (normalized units)
fn mexican_hat(r2_phys: f32) -> f32 {
  let inv_2se2 = 1.0 / (2.0 * params.sigma_e * params.sigma_e);
  let inv_2si2 = 1.0 / (2.0 * params.sigma_i * params.sigma_i);
  return params.A_e * exp(-r2_phys * inv_2se2)
       - params.A_i * exp(-r2_phys * inv_2si2);
}

// per-cell noise
fn pcg(v: u32) -> u32 {
  let state = v * 747796405u + 2891336453u;
  let word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn rand_signed(seed: u32) -> f32 {
  return (f32(pcg(seed)) / 4294967295.0) * 2.0 - 1.0;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  let W = params.width;
  let H = params.height;

  if (x >= W || y >= H) { return; }

  let idx = y * W + x;
  let u   = u_in[idx];
  let v   = v_in[idx];
  let R   = i32(params.kernel_radius);

  var conv: f32 = 0.0;
  for (var dy = -R; dy <= R; dy++) { // loop through every cell neighbor
    for (var dx = -R; dx <= R; dx++) { 
      if (f32(dx*dx + dy*dy) > f32(R*R)) { continue; }
      // neighbor coords
      let nx = i32(x) + dx;
      let ny = i32(y) + dy;
      // clamped coords
      let cx = clamp(nx, 0, i32(W) - 1);
      let cy = clamp(ny, 0, i32(H) - 1);
      // squared distance in grid cells
      let r2 = f32(dx*dx + dy*dy);
      // sum of neighbors through mexican hat in firing rate from clamped coords 
      conv  += mexican_hat(r2) * sigmoid(u_in[u32(cy) * W + u32(cx)]);
    }
  }

  // Noise
  let seed = idx ^ (params.frame * 2654435761u);
  let eta  = rand_signed(seed) * params.noise;

  // adaptation: disabled when tau_v > 100
  let use_v      = params.tau_v < 100.0;
  let adaptation = select(0.0, v, use_v);

  // euler (time) step for u -> smaller = accurate but slower
  let du = (params.dt / params.tau) * (-u + conv - adaptation + params.tonic + eta);
  u_out[idx] = clamp(u + du, -3.0, 3.0);

  // euler step for v
  let dv = (params.dt / params.tau_v) * (u - v);
  v_out[idx] = select(0.0, v + dv, use_v);
}
