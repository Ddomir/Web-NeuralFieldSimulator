// Render shader — fullscreen quad that reads the neural field buffer
// and maps firing rate -> color using a cool-hot colormap.

struct Params {
  width : u32,
  height: u32,
  beta  : f32,
  theta : f32,
}

@group(0) @binding(0) var<uniform>       params: Params;
@group(0) @binding(1) var<storage, read> field : array<f32>;

struct VertexOut { // Vertex output
  @builtin(position) pos: vec4<f32>,
  @location(0)       uv : vec2<f32>,
}

// Two large triangles
// Positions indexed by vertex_index
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOut {
  var pos = array<vec2<f32>, 6>(
    vec2(-1.0, -1.0), vec2( 1.0, -1.0), vec2(-1.0,  1.0),
    vec2(-1.0,  1.0), vec2( 1.0, -1.0), vec2( 1.0,  1.0),
  );
  let p = pos[vi];
  var out: VertexOut;
  out.pos = vec4(p, 0.0, 1.0);
  // UV: (0,0) top-left, (1,1) bottom-right
  out.uv  = p * 0.5 + 0.5;
  return out;
}

// cool to hot: maps t ∈ [0,1] to a gradient.
// t=0 -> dark blue (inhibited), t=0.5 -> black (resting), t=1 -> white-orange (excited)
fn colormap(t: f32) -> vec3<f32> {
  // Split into two halves for inhibited / excited
  if (t < 0.5) {
    // 0 -> deep blue, 0.5 -> black
    let s = t * 2.0;
    return mix(vec3(0.0, 0.1, 0.5), vec3(0.0, 0.0, 0.0), s); // blue val
  } else {
    // 0.5 -> black, 0.75 -> red-orange, 1.0 -> white
    let s = (t - 0.5) * 2.0;
    let hot = mix(vec3(0.0, 0.0, 0.0), vec3(1.0, 0.4, 0.0), s);
    return mix(hot, vec3(1.0, 1.0, 0.9), s * s); // extra brightness near peak
  }
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
  let W = params.width;
  let H = params.height;

  // Map UV to integer cell coordinates
  let cx = u32(in.uv.x * f32(W));
  let cy = u32(in.uv.y * f32(H));
  let cxc = min(cx, W - 1u);
  let cyc = min(cy, H - 1u);

  let u = field[cyc * W + cxc];

  // Convert membrane potential to firing rate via sigmoid: f(u) ∈ [0, 1]
  let t = 1.0 / (1.0 + exp(-params.beta * (u - params.theta)));
  return vec4(colormap(t), 1.0);
}
