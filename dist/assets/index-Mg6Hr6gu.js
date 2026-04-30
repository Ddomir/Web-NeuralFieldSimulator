(function(){let e=document.createElement(`link`).relList;if(e&&e.supports&&e.supports(`modulepreload`))return;for(let e of document.querySelectorAll(`link[rel="modulepreload"]`))n(e);new MutationObserver(e=>{for(let t of e)if(t.type===`childList`)for(let e of t.addedNodes)e.tagName===`LINK`&&e.rel===`modulepreload`&&n(e)}).observe(document,{childList:!0,subtree:!0});function t(e){let t={};return e.integrity&&(t.integrity=e.integrity),e.referrerPolicy&&(t.referrerPolicy=e.referrerPolicy),e.crossOrigin===`use-credentials`?t.credentials=`include`:e.crossOrigin===`anonymous`?t.credentials=`omit`:t.credentials=`same-origin`,t}function n(e){if(e.ep)return;e.ep=!0;let n=t(e);fetch(e.href,n)}})();async function e(e){if(!navigator.gpu)return null;let t=await navigator.gpu.requestAdapter();if(!t)return null;let n=await t.requestDevice(),r=e.getContext(`webgpu`),i=navigator.gpu.getPreferredCanvasFormat();return r.configure({device:n,format:i}),{adapter:t,device:n,format:i,context:r}}var t=256,n=256,r=[{name:`Evoked Wave`,params:{dt:.02,tau:1,tau_v:5,beta:20,theta:.2,A_e:1.4/(2*Math.PI*4*4),A_i:.9/(2*Math.PI*8*8),sigma_e:4,sigma_i:8,kernelRadius:16,noise:.001}},{name:`Alpha Rest`,params:{dt:.02,tau:1,tau_v:10,beta:8,theta:.2,A_e:1/(2*Math.PI*4*4),A_i:.4/(2*Math.PI*8*8),sigma_e:4,sigma_i:8,kernelRadius:16,noise:.05,tonic:.1}},{name:`Seizure`,params:{dt:.02,tau:1,tau_v:40,beta:12,theta:.2,A_e:1.2/(2*Math.PI*4*4),A_i:.4/(2*Math.PI*8*8),sigma_e:4,sigma_i:8,kernelRadius:16,noise:.001,tonic:0}},{name:`Cell March`,params:{dt:.02,tau:1,tau_v:40,beta:15,theta:.05,A_e:.017,A_i:.011,sigma_e:5,sigma_i:11,kernelRadius:16,noise:.01,tonic:0}}],i={width:t,height:n,dt:.02,tau:1,tau_v:5,beta:20,theta:.2,A_e:1.4/(2*Math.PI*4*4),A_i:.9/(2*Math.PI*8*8),sigma_e:4,sigma_i:8,kernelRadius:16,noise:.001,frame:0,tonic:0};function a(e){let t=new ArrayBuffer(60),n=new DataView(t);return n.setUint32(0,e.width,!0),n.setUint32(4,e.height,!0),n.setFloat32(8,e.dt,!0),n.setFloat32(12,e.tau,!0),n.setFloat32(16,e.tau_v,!0),n.setFloat32(20,e.beta,!0),n.setFloat32(24,e.theta,!0),n.setFloat32(28,e.A_e,!0),n.setFloat32(32,e.A_i,!0),n.setFloat32(36,e.sigma_e,!0),n.setFloat32(40,e.sigma_i,!0),n.setUint32(44,e.kernelRadius,!0),n.setFloat32(48,e.noise,!0),n.setUint32(52,e.frame,!0),n.setFloat32(56,e.tonic,!0),t}function o(e,t){let n=t.width*t.height,r=n*4,i=new Float32Array(n);for(let e=0;e<n;e++)i[e]=(Math.random()-.5)*.2;let o=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST,s=e.createBuffer({size:r,usage:o,mappedAtCreation:!0});new Float32Array(s.getMappedRange()).set(i),s.unmap();let c=e.createBuffer({size:r,usage:o}),l=e.createBuffer({size:r,usage:o}),u=e.createBuffer({size:r,usage:o}),d=a(t),f=e.createBuffer({size:d.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,mappedAtCreation:!0});return new Uint8Array(f.getMappedRange()).set(new Uint8Array(d)),f.unmap(),{ping:s,pong:c,vPing:l,vPong:u,params:f}}function s(e,t,n){let r=n.width*n.height,i=new Float32Array(r),a=new Float32Array(r);for(let e=0;e<r;e++)i[e]=(Math.random()-.5)*.2;e.queue.writeBuffer(t.ping,0,i),e.queue.writeBuffer(t.pong,0,i),e.queue.writeBuffer(t.vPing,0,a),e.queue.writeBuffer(t.vPong,0,a)}async function c(e,t,n,r,i,a=18,o=3){let s=(a*.6)**2,c=Math.max(0,r-a),l=Math.max(0,i-a),u=Math.min(n.width-1,r+a),d=Math.min(n.height-1,i+a),f=u-c+1,p=d-l+1,m=f*4,h=e.createBuffer({size:p*m,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),g=e.createCommandEncoder();for(let e=0;e<p;e++)g.copyBufferToBuffer(t,((l+e)*n.width+c)*4,h,e*m,m);e.queue.submit([g.finish()]),await h.mapAsync(GPUMapMode.READ);let _=new Float32Array(h.getMappedRange().slice(0));h.unmap(),h.destroy();for(let e=0;e<p;e++)for(let t=0;t<f;t++){let n=c+t-r,u=l+e-i;n*n+u*u>a*a||(_[e*f+t]+=o*Math.exp(-(n*n+u*u)/s))}for(let r=0;r<p;r++)e.queue.writeBuffer(t,((l+r)*n.width+c)*4,_.buffer,r*m,m)}function l(e,t,n){e.queue.writeBuffer(t.params,0,a(n))}var u=`// Neural field update — Amari model with optional spike-frequency adaptation + noise
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
`,d=class e{pipeline;bindGroupA;bindGroupB;step=0;dispatchX;dispatchY;constructor(t,n,r){this.pipeline=t.createComputePipeline({layout:`auto`,compute:{module:t.createShaderModule({code:u}),entryPoint:`main`}});let i=this.pipeline.getBindGroupLayout(0);this.bindGroupA=e.makeBindGroup(t,i,n.params,n.ping,n.pong,n.vPing,n.vPong),this.bindGroupB=e.makeBindGroup(t,i,n.params,n.pong,n.ping,n.vPong,n.vPing),this.dispatchX=Math.ceil(r.width/16),this.dispatchY=Math.ceil(r.height/16)}static makeBindGroup(e,t,n,r,i,a,o){return e.createBindGroup({layout:t,entries:[{binding:0,resource:{buffer:n}},{binding:1,resource:{buffer:r}},{binding:2,resource:{buffer:i}},{binding:3,resource:{buffer:a}},{binding:4,resource:{buffer:o}}]})}get currentBuffer(){return this._currentBuffer===null?null:this._currentBuffer}_currentBuffer=null;dispatch(e,t){let n=this.step%2==0?this.bindGroupA:this.bindGroupB,r=this.step%2==0?t.pong:t.ping;this.step++;let i=e.beginComputePass();return i.setPipeline(this.pipeline),i.setBindGroup(0,n),i.dispatchWorkgroups(this.dispatchX,this.dispatchY),i.end(),this._currentBuffer=r,r}},f=`// Render shader — fullscreen quad that reads the neural field buffer.
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
`,p=class{pipeline;paramsBuffer;bindGroupCache=new Map;device;layout;constructor(e,t,n){this.device=e;let r=e.createShaderModule({code:f});this.pipeline=e.createRenderPipeline({layout:`auto`,vertex:{module:r,entryPoint:`vs_main`},fragment:{module:r,entryPoint:`fs_main`,targets:[{format:t}]},primitive:{topology:`triangle-list`}}),this.layout=this.pipeline.getBindGroupLayout(0),this.paramsBuffer=e.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,mappedAtCreation:!0}),this.writeParams(n.width,n.height,n.beta,n.theta,!0),this.paramsBuffer.unmap()}writeParams(e,t,n,r,i=!1){let a=new ArrayBuffer(16),o=new DataView(a);o.setUint32(0,e,!0),o.setUint32(4,t,!0),o.setFloat32(8,n,!0),o.setFloat32(12,r,!0),i?new Uint8Array(this.paramsBuffer.getMappedRange()).set(new Uint8Array(a)):this.device.queue.writeBuffer(this.paramsBuffer,0,a)}updateSigmoidParams(e,t){let n=new ArrayBuffer(8),r=new DataView(n);r.setFloat32(0,e,!0),r.setFloat32(4,t,!0),this.device.queue.writeBuffer(this.paramsBuffer,8,n)}draw(e,t,n){this.bindGroupCache.has(t)||this.bindGroupCache.set(t,this.device.createBindGroup({layout:this.layout,entries:[{binding:0,resource:{buffer:this.paramsBuffer}},{binding:1,resource:{buffer:t}}]}));let r=e.beginRenderPass({colorAttachments:[{view:n,clearValue:{r:.02,g:.02,b:.1,a:1},loadOp:`clear`,storeOp:`store`}]});r.setPipeline(this.pipeline),r.setBindGroup(0,this.bindGroupCache.get(t)),r.draw(6),r.end()}},m=[{text:`u_new`,tip:`Next membrane potential — the field state after this timestep.`,highlight:`decay`},` = `,{text:`u`,tip:`Current membrane potential at this cell.`,highlight:`decay`},` + `,{text:`(dt/τ)`,tip:`Euler step size — dt is the timestep, τ is the membrane time constant controlling how fast u responds. Smaller τ → faster dynamics.`,highlight:`tau`},` · ( `,{text:`−u`,tip:`Passive decay — without any input the field relaxes back to zero.`,highlight:`decay`},` + `,{text:`∑ w(r)·f(u)`,tip:`Lateral input: sums over all neighbors within kernel_radius. w(r) is the Mexican hat kernel (excitatory near, inhibitory far); f(u) is the sigmoid firing rate gated by β and θ.`,highlight:`conv`},` − `,{text:`v`,tip:`Spike-frequency adaptation — slow negative feedback that prevents sustained firing. Enable and tune with the Adaptation toggle.`,highlight:`adapt`},` + `,{text:`tonic`,tip:`Constant background drive added every step. Raises resting excitability across the whole field.`,highlight:`tonic`},` + `,{text:`η`,tip:`Per-cell noise drawn each step. Amplitude set by the Noise slider.`,highlight:`noise`},` )`],h=[{text:`v_new`,tip:`Next adaptation state.`,highlight:`adapt`},` = `,{text:`v`,tip:`Current adaptation value.`,highlight:`adapt`},` + `,{text:`(dt/τᵥ)`,tip:`Adaptation step size — τᵥ controls how slowly v chases u. Long τᵥ → adaptation can't keep up → runaway excitation (seizure mode).`,highlight:`tau`},` · ( `,{text:`u`,tip:`Membrane potential drives v upward — the more excited the cell, the faster adaptation builds.`,highlight:`decay`},` − `,{text:`v`,tip:`Passive decay of the adaptation variable back to zero.`,highlight:`adapt`},` )`],g=[{text:`w(r)`,tip:`Mexican hat kernel — the connectivity weight between two cells at distance r.`,highlight:`conv`},` = `,{text:`Aₑ`,tip:`Excitatory amplitude — scales the peak of the central excitatory Gaussian. Controlled by 'Excit. amp' slider.`,highlight:`excit_amp`},`·exp(−r²/`,{text:`2σₑ²`,tip:`Excitatory spread — width of the central excitatory lobe. Controlled by 'Excit. spread' slider.`,highlight:`excit_sig`},`) − `,{text:`Aᵢ`,tip:`Inhibitory amplitude — scales the surround inhibitory Gaussian. Controlled by 'Inhib. amp' slider.`,highlight:`inhib_amp`},`·exp(−r²/`,{text:`2σᵢ²`,tip:`Inhibitory spread — width of the surround inhibitory lobe. Controlled by 'Inhib. spread' slider. Must be wider than σₑ for a stable Mexican hat shape.`,highlight:`inhib_sig`},`)`],_={tau:`#7eb8f7`,decay:`#aaa`,conv:`#f7c948`,adapt:`#e07030`,tonic:`#88d8a0`,noise:`#c792ea`,excit_amp:`#6dd6a0`,excit_sig:`#3ecfcf`,inhib_amp:`#f47c7c`,inhib_sig:`#c97bf7`},v=class{kernelCanvas;kernelCtx;overlay;constructor(e,t,n){this.overlay=n;let r=document.createElement(`div`);r.className=`eq-panel`;let i=document.createElement(`div`);i.className=`eq-title`,i.textContent=`Amari Neural Field`,r.appendChild(i),r.appendChild(this.buildEquation(m,`eq-line`)),r.appendChild(this.buildEquation(h,`eq-line eq-line--sub`));let a=document.createElement(`div`);a.className=`eq-kernel-header`;let o=document.createElement(`div`);o.className=`eq-title`,o.textContent=`Mexican Hat Kernel`;let s=document.createElement(`button`);s.className=`kernel-toggle-btn`,s.textContent=`show on cursor`;let c=!1;s.addEventListener(`click`,()=>{c=!c,c?(this.overlay.enable(),s.classList.add(`active`)):(this.overlay.disable(),s.classList.remove(`active`))}),a.appendChild(o),a.appendChild(s),r.appendChild(a),r.appendChild(this.buildEquation(g,`eq-line`)),r.appendChild(this.buildKernelCanvas(t));let l=document.createElement(`div`);l.className=`eq-hint`,l.textContent=`hover terms for details`,r.appendChild(l),e.appendChild(r)}update(e){this.drawKernel(e),this.overlay.update(e)}buildKernelCanvas(e){let t=document.createElement(`div`);t.className=`kernel-canvas-wrap`;let n=document.createElement(`canvas`);return n.width=240,n.height=80,n.className=`kernel-canvas`,this.kernelCanvas=n,this.kernelCtx=n.getContext(`2d`),t.appendChild(n),this.drawKernel(e),t}drawKernel(e){let t=this.kernelCtx,n=this.kernelCanvas.width,r=this.kernelCanvas.height,{A_e:i,A_i:a,sigma_e:o,sigma_i:s,kernelRadius:c}=e;t.clearRect(0,0,n,r);let l=n,u=[],d=[],f=[];for(let e=0;e<l;e++){let t=e/(l-1)*c,n=t*t,r=i*Math.exp(-n/(2*o*o)),p=a*Math.exp(-n/(2*s*s));d.push(r),f.push(-p),u.push(r-p)}let p=Math.max(...u.map(Math.abs),1e-6),m=r/2,h=(r/2-6)/p,g=(e,r,i)=>{t.beginPath();for(let r=0;r<l;r++){let i=r/(l-1)*n,a=m-e[r]*h;r===0?t.moveTo(i,a):t.lineTo(i,a)}t.lineTo(n,m),t.lineTo(0,m),t.closePath(),t.fillStyle=i,t.fill(),t.beginPath();for(let r=0;r<l;r++){let i=r/(l-1)*n,a=m-e[r]*h;r===0?t.moveTo(i,a):t.lineTo(i,a)}t.strokeStyle=r,t.lineWidth=1,t.setLineDash([3,3]),t.stroke(),t.setLineDash([])};t.strokeStyle=`#2a2a2a`,t.lineWidth=1,t.beginPath(),t.moveTo(0,m),t.lineTo(n,m),t.stroke(),g(d,`#3ecfcf`,`rgba(109,214,160,0.10)`),g(f,`#c97bf7`,`rgba(244,124,124,0.10)`),t.beginPath();for(let e=0;e<l;e++){let r=e/(l-1)*n,i=m-u[e]*h;e===0?t.moveTo(r,i):t.lineTo(r,i)}t.strokeStyle=`#ddd`,t.lineWidth=1.5,t.stroke(),t.fillStyle=`#444`,t.font=`9px monospace`,t.fillText(`0`,2,m-2),t.fillText(`r = ${c}`,n-36,m-2)}buildEquation(e,t){let n=document.createElement(`div`);n.className=t;let r=document.createElement(`div`);r.className=`eq-expr`;let i=document.createElement(`div`);i.className=`eq-tooltip`;for(let t of e)if(typeof t==`string`){let e=document.createElement(`span`);e.className=`eq-op`,e.textContent=t,r.appendChild(e)}else{let e=document.createElement(`span`);e.className=`eq-term`,e.textContent=t.text,t.highlight&&(e.style.color=_[t.highlight]??`#ccc`),e.addEventListener(`mouseenter`,()=>{i.textContent=t.tip,i.classList.add(`eq-tooltip--visible`)}),e.addEventListener(`mouseleave`,()=>{i.classList.remove(`eq-tooltip--visible`)}),r.appendChild(e)}return n.appendChild(r),n.appendChild(i),n}},y=class{simCanvas;canvas;ctx;params;enabled=!1;mouseX=-1;mouseY=-1;ringCache=null;cacheKey=``;constructor(e,t){this.simCanvas=e,this.params={...t},this.canvas=document.createElement(`canvas`),this.canvas.style.cssText=`
      position: absolute;
      top: 0; left: 0;
      width: 100%; height: 100%;
      pointer-events: none;
      display: none;
    `,this.ctx=this.canvas.getContext(`2d`),e.parentElement.style.position=`relative`,e.insertAdjacentElement(`afterend`,this.canvas),this.syncSize(),new ResizeObserver(()=>this.syncSize()).observe(e),e.addEventListener(`mousemove`,t=>{if(!this.enabled)return;let n=e.getBoundingClientRect();this.mouseX=t.clientX-n.left,this.mouseY=t.clientY-n.top,this.draw()}),e.addEventListener(`mouseleave`,()=>{this.mouseX=-1,this.clear()})}enable(){this.enabled=!0,this.canvas.style.display=`block`}disable(){this.enabled=!1,this.canvas.style.display=`none`,this.clear()}update(e){this.params={...e},this.ringCache=null,this.enabled&&this.mouseX>=0&&this.draw()}syncSize(){let e=this.simCanvas.getBoundingClientRect();this.canvas.width=e.width,this.canvas.height=e.height,this.ringCache=null,this.enabled&&this.mouseX>=0&&this.draw()}buildRingCache(){let{A_e:e,A_i:t,sigma_e:n,sigma_i:r,kernelRadius:i}=this.params,a=this.simCanvas.getBoundingClientRect().width/this.params.width,o=i*a,s=Math.ceil(o*2)+2,c=new ImageData(s,s),l=s/2,u=s/2;for(let i=0;i<s;i++)for(let d=0;d<s;d++){let f=d-l,p=i-u,m=Math.sqrt(f*f+p*p);if(m>o)continue;let h=m/a,g=h*h,_=e*Math.exp(-g/(2*n*n))-t*Math.exp(-g/(2*r*r)),v=Math.min(Math.abs(_)/e,1)*180,y=(i*s+d)*4;_>0?(c.data[y]=109,c.data[y+1]=214,c.data[y+2]=160):(c.data[y]=244,c.data[y+1]=124,c.data[y+2]=124),c.data[y+3]=v}return c}draw(){let e=this.canvas.width,t=this.canvas.height;if(this.ctx.clearRect(0,0,e,t),this.mouseX<0)return;let n=`${this.params.A_e},${this.params.A_i},${this.params.sigma_e},${this.params.sigma_i},${this.params.kernelRadius},${e}`;(this.cacheKey!==n||!this.ringCache)&&(this.ringCache=this.buildRingCache(),this.cacheKey=n);let r=this.ringCache.width,i=new OffscreenCanvas(r,r);i.getContext(`2d`).putImageData(this.ringCache,0,0);let a=this.mouseX-r/2,o=this.mouseY-r/2;this.ctx.drawImage(i,a,o);let{kernelRadius:s}=this.params,c=e/this.params.width,l=s*c;this.ctx.beginPath(),this.ctx.arc(this.mouseX,this.mouseY,l,0,Math.PI*2),this.ctx.strokeStyle=`rgba(255,255,255,0.15)`,this.ctx.lineWidth=1,this.ctx.stroke();let u=this.findZeroCrossing();u>0&&(this.ctx.beginPath(),this.ctx.arc(this.mouseX,this.mouseY,u*c,0,Math.PI*2),this.ctx.strokeStyle=`rgba(255,255,255,0.3)`,this.ctx.lineWidth=1,this.ctx.setLineDash([3,3]),this.ctx.stroke(),this.ctx.setLineDash([]))}findZeroCrossing(){let{A_e:e,A_i:t,sigma_e:n,sigma_i:r,kernelRadius:i}=this.params,a=n,o=i;for(let i=0;i<32;i++){let i=(a+o)/2,s=i*i;e*Math.exp(-s/(2*n*n))-t*Math.exp(-s/(2*r*r))>0?a=i:o=i}return(a+o)/2}clear(){this.ctx.clearRect(0,0,this.canvas.width,this.canvas.height)}},b=[{key:`noise`,label:`Noise`,min:0,max:.5,step:.001,color:`#c792ea`},{key:`dt`,label:`Timestep (dt)`,min:.005,max:.1,step:.005},{key:`tau`,label:`Membrane τ`,min:.1,max:5,step:.1,color:`#7eb8f7`},{key:`beta`,label:`Sigmoid gain (β)`,min:1,max:30,step:.5,color:`#f7c948`},{key:`theta`,label:`Threshold (θ)`,min:-1,max:1,step:.05,color:`#f7c948`},{key:`A_e`,label:`Excit. amp (Aₑ)`,min:5e-4,max:.025,step:5e-4,color:`#6dd6a0`},{key:`A_i`,label:`Inhib. amp (Aᵢ)`,min:5e-4,max:.025,step:5e-4,color:`#f47c7c`},{key:`sigma_e`,label:`Excit. spread (σₑ)`,min:2,max:20,step:1,color:`#3ecfcf`},{key:`sigma_i`,label:`Inhib. spread (σᵢ)`,min:4,max:30,step:1,color:`#c97bf7`}],x=class{params;onChange;onPreset;equationPanel;sliderEls=new Map;tauVSlider;tauVDisplay;adaptCheckbox;constructor(e,t,n,r,i){this.params={...n},this.onChange=r,this.onPreset=i,this.build(e,t)}build(e,t){let n=document.createElement(`div`);n.id=`controls-panel`;let r=new y(t,this.params);this.equationPanel=new v(n,this.params,r),n.appendChild(this.buildPresetRow());for(let e of b)n.appendChild(this.buildSliderRow(e));n.appendChild(this.buildAdaptationRow()),e.appendChild(n)}buildPresetRow(){let e=document.createElement(`div`);e.className=`preset-row`;for(let t of r){let n=document.createElement(`button`);n.textContent=t.name,n.className=`preset-btn`,n.addEventListener(`click`,()=>{Object.assign(this.params,t.params),this.syncSliders(),this.onPreset({...this.params}),this.equationPanel.update(this.params)}),e.appendChild(n)}return e}buildSliderRow(e){let t=document.createElement(`div`);t.className=`slider-row`,e.color&&t.style.setProperty(`--slider-color`,e.color);let n=document.createElement(`label`);n.textContent=e.label;let r=document.createElement(`span`);r.className=`slider-value`,r.textContent=this.formatValue(this.params[e.key]);let i=document.createElement(`input`);return i.type=`range`,i.min=String(e.min),i.max=String(e.max),i.step=String(e.step),i.value=String(this.params[e.key]),i.addEventListener(`input`,()=>{let t=parseFloat(i.value);this.params[e.key]=t,r.textContent=this.formatValue(t),this.onChange({...this.params}),this.equationPanel.update(this.params)}),this.sliderEls.set(e.key,{slider:i,display:r}),t.appendChild(n),t.appendChild(i),t.appendChild(r),t}syncSliders(){for(let[e,{slider:t,display:n}]of this.sliderEls){let r=this.params[e];t.value=String(r),n.textContent=this.formatValue(r)}let e=this.params.tau_v<100;this.adaptCheckbox.checked=e,this.tauVSlider.disabled=!e;let t=e?this.params.tau_v:10;this.tauVSlider.value=String(t),this.tauVDisplay.textContent=this.formatValue(t)}buildAdaptationRow(){let e=document.createElement(`div`);e.className=`adapt-section`,e.style.setProperty(`--slider-color`,`#e07030`);let t=document.createElement(`div`);t.className=`slider-row adapt-header`;let n=document.createElement(`label`);n.textContent=`Adaptation (τᵥ)`,this.adaptCheckbox=document.createElement(`input`),this.adaptCheckbox.type=`checkbox`,this.adaptCheckbox.checked=this.params.tau_v<100,this.tauVDisplay=document.createElement(`span`),this.tauVDisplay.className=`slider-value`,this.tauVSlider=document.createElement(`input`),this.tauVSlider.type=`range`,this.tauVSlider.min=`0.5`,this.tauVSlider.max=`30`,this.tauVSlider.step=`0.5`;let r=this.params.tau_v<100?this.params.tau_v:10;return this.tauVSlider.value=String(r),this.tauVDisplay.textContent=this.formatValue(r),this.tauVSlider.disabled=!this.adaptCheckbox.checked,this.adaptCheckbox.addEventListener(`change`,()=>{let e=this.adaptCheckbox.checked;this.tauVSlider.disabled=!e,this.params.tau_v=e?parseFloat(this.tauVSlider.value):999,this.onChange({...this.params})}),this.tauVSlider.addEventListener(`input`,()=>{let e=parseFloat(this.tauVSlider.value);this.params.tau_v=e,this.tauVDisplay.textContent=this.formatValue(e),this.onChange({...this.params})}),t.appendChild(n),t.appendChild(this.adaptCheckbox),t.appendChild(this.tauVDisplay),e.appendChild(t),e.appendChild(this.tauVSlider),e}formatValue(e){return e<.01&&e>0?e.toExponential(2):Number.isInteger(e)?e.toFixed(0):e.toFixed(3)}},S=class{el;visible=!1;frameCount=0;lastFpsTime=performance.now();fps=0;staging=null;stagingSize=0;sampling=!1;stats={min:0,mean:0,max:0,variance:0,fresh:!1};device;getFieldBuffer;fieldCells;getParams;constructor(e,t,n,r){this.device=e,this.getFieldBuffer=t,this.fieldCells=n,this.getParams=r,this.el=document.createElement(`div`),this.el.id=`debug-overlay`,this.el.style.display=`none`,document.body.appendChild(this.el),window.addEventListener(`keydown`,e=>{e.key==="`"&&(this.visible=!this.visible,this.el.style.display=this.visible?`block`:`none`,this.visible&&this.sample())}),setInterval(()=>{this.visible&&this.sample()},1e3)}tick(){this.frameCount++;let e=performance.now(),t=e-this.lastFpsTime;t>=500&&(this.fps=Math.round(this.frameCount/(t/1e3)),this.frameCount=0,this.lastFpsTime=e,this.visible&&this.render())}render(){let e=this.stats,t=this.getParams(),n=t.A_e*2*Math.PI*t.sigma_e*t.sigma_e,r=n-t.A_i*2*Math.PI*t.sigma_i*t.sigma_i,i=1/(1+Math.exp(-t.beta*(0-t.theta))),a=t.beta*i*(1-i),o=t.tau_v/t.tau,s=4/(r*t.beta*t.beta),c=t.tau_v<100&&o>s,l=t.beta*t.beta*n/4>1,u=t.tau_v<100,d=`?`;d=u?l&&r>.6?o<3?`Tonic-clonic / bursting`:`Spatial Turing patterns`:c?o>15?`Slow delta waves`:`Excitable traveling waves`:`Sub-wave`:r<.05?`Balanced rest`:`Noise-driven rest`;let f=(e,t=2)=>e.toFixed(t);this.el.innerHTML=`<div class="dbg-section"><div>${this.fps} fps</div>`+(e.fresh?`<div>u  [${f(e.min)}, ${f(e.mean)}, ${f(e.max)}]</div>`:`<div>sampling…</div>`)+`<div>W_net: ${f(r,3)}  &nbsp; u*≈${f(r*i/(1-r*a),3)}</div><div>Regime: ${d}</div></div><div class="dbg-hint">[ \` ] toggle</div>`}async sample(){if(this.sampling)return;let e=this.getFieldBuffer();if(!e)return;this.sampling=!0;let t=this.fieldCells*4;(!this.staging||this.stagingSize!==t)&&(this.staging?.destroy(),this.staging=this.device.createBuffer({size:t,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),this.stagingSize=t);let n=this.device.createCommandEncoder();n.copyBufferToBuffer(e,0,this.staging,0,t),this.device.queue.submit([n.finish()]),await this.staging.mapAsync(GPUMapMode.READ);let r=new Float32Array(this.staging.getMappedRange().slice(0));this.staging.unmap();let i=1/0,a=-1/0,o=0;for(let e=0;e<r.length;e++)r[e]<i&&(i=r[e]),r[e]>a&&(a=r[e]),o+=r[e];let s=o/r.length,c=0;for(let e=0;e<r.length;e++){let t=r[e]-s;c+=t*t}let l=c/r.length;this.stats={min:i,max:a,mean:s,variance:l,fresh:!0},this.sampling=!1,this.render()}},C=document.getElementById(`status`),w=document.getElementById(`webgpu-canvas`);function T(e){C.textContent=e,C.classList.add(`error`),w.style.display=`none`}async function E(){let t=await e(w);if(!t){T(`WebGPU is not supported or unavailable.`);return}let{device:n,format:r,context:a}=t,u={...i},f=o(n,u),m=new d(n,f,u),h=new p(n,r,u);h.updateSigmoidParams(u.beta,u.theta),new x(document.body,w,u,e=>{Object.assign(u,e),l(n,f,u),h.updateSigmoidParams(u.beta,u.theta)},e=>{Object.assign(u,e),l(n,f,u),h.updateSigmoidParams(u.beta,u.theta),s(n,f,u)});function g(e){if(!(e.buttons&1))return;let t=m.currentBuffer;if(!t)return;let r=w.getBoundingClientRect();c(n,t,u,Math.floor((e.clientX-r.left)/r.width*u.width),Math.floor((e.clientY-r.top)/r.height*u.height))}w.addEventListener(`pointerdown`,g),w.addEventListener(`pointermove`,g);let _=new S(n,()=>m.currentBuffer,u.width*u.height,()=>u);C.textContent=`Running`,setTimeout(()=>{let e=m.currentBuffer;e&&c(n,e,u,u.width/2|0,u.height/2|0)},200);function v(){u.frame=u.frame+1&4294967295,l(n,f,u);let e=n.createCommandEncoder(),t;for(let n=0;n<12;n++)t=m.dispatch(e,f);h.draw(e,t,a.getCurrentTexture().createView()),n.queue.submit([e.finish()]),_.tick(),requestAnimationFrame(v)}requestAnimationFrame(v)}E();