// Render shader — fullscreen quad that reads the neural field buffer.
// Displays firing rate f(u) = sigmoid(β·(u−θ)) rather than raw u.

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
  out.uv = vec2(p.x * 0.5 + 0.5, 1.0 - (p.y * 0.5 + 0.5));
  return out;
}

// Colormap: blue → green → yellow → orange → red
fn colormap(t: f32) -> vec3<f32> {
  // 0.00 → blue
  // 0.25 → green
  // 0.50 → yellow
  // 0.75 → orange
  // 1.00 → red
  let a = mix(vec3(0.0, 0.0, 0.6), vec3(0.0, 0.6, 0.0), clamp(t / 0.25, 0.0, 1.0));
  let b = mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0), clamp((t - 0.25) / 0.25, 0.0, 1.0));
  let c = mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 0.5, 0.0), clamp((t - 0.50) / 0.25, 0.0, 1.0));
  let d = mix(vec3(1.0, 0.5, 0.0), vec3(1.0, 0.0, 0.0), clamp((t - 0.75) / 0.25, 0.0, 1.0));
  var col = a;
  if (t > 0.25) { col = b; }
  if (t > 0.50) { col = c; }
  if (t > 0.75) { col = d; }
  return col;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
  let W = params.width;
  let H = params.height;

  let cx = min(u32(in.uv.x * f32(W)), W - 1u);
  let cy = min(u32(in.uv.y * f32(H)), H - 1u);

  let u = field[cy * W + cx];

  // Render firing rate
  let f = 1.0 / (1.0 + exp(-params.beta * (u - params.theta)));
  return vec4(colormap(f), 1.0);
}
