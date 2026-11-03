'use strict';

// Browser-only reader for the numerical tables generated from SciPy solve_bvp.
// No Python, SciPy, JavaScript packages, or external CDNs are used at runtime.
const $ = (id) => document.getElementById(id);
const COLORS = ['#FF3A3A', '#F77A13', '#3E07F4', '#0DB51E'];
const REACTION_COLOR = '#B60C86';
const state = { data: {}, currentView: 'dimensional', last: {}, scheduled: false, drel: null };
const dimInputs = ['c0', 'cs', 'logL', 'logk', 'logD'];

function clamp(v, low, high) { return Math.max(low, Math.min(high, v)); }
function nice(n, digits=3) {
  if (!Number.isFinite(n)) return '—';
  if (n === 0) return '0';
  const a = Math.abs(n);
  return a >= 1e4 || a < 0.001 ? n.toExponential(2) : Number(n.toPrecision(digits)).toString();
}
function superscript(n) { return String(n).replace(/-/g,'⁻').replace(/[0-9]/g, c => '⁰¹²³⁴⁵⁶⁷⁸⁹'[Number(c)]); }
function sci(n, digits=2) {
  if (n === 0) return '0';
  const exp = Math.floor(Math.log10(Math.abs(n)));
  const mantissa = n / 10 ** exp;
  return `${mantissa.toFixed(digits)} × 10${superscript(exp)}`;
}

async function loadTable(spec) {
  // A .gz suffix matters: the compressed table is unpacked entirely in Safari.
  if (!('DecompressionStream' in window)) throw new Error('This browser cannot decompress data files; please update Safari.');
  const response = await fetch(spec.file);
  if (!response.ok) throw new Error(`${spec.file}: HTTP ${response.status}`);
  const unzipped = response.body.pipeThrough(new DecompressionStream('gzip'));
  const buffer = await new Response(unzipped).arrayBuffer();
  if (buffer.byteLength !== spec.uncompressed_bytes) throw new Error('Numerical table is incomplete.');
  const data = {};
  for (const f of spec.fields) {
    const dtype = {'<f8': Float64Array, '<f4': Float32Array, '<u4': Uint32Array}[f.type];
    if (!dtype) throw new Error('Unknown table field type: ' + f.type);
    data[f.name] = new dtype(buffer, f.offset, f.length);
  }
  data._buffer = buffer; // retain the complete buffer for typed-array views
  return data;
}

function locate(grid, value) {
  // Return the lower bracket: grid[i] <= value <= grid[i+1].
  let low = 0, high = grid.length - 1;
  if (value <= grid[0]) return 0;
  if (value >= grid[high]) return high - 1;
  while (low + 1 < high) {
    const mid = (low + high) >> 1;
    if (grid[mid] <= value) low = mid;
    else high = mid;
  }
  return low;
}

function dimensionalProfile(Da, gamma) {
  // Port of ENProfileLookup.concentrations(): log-Da/gamma bilinear blending
  // with piecewise-affine front alignment, preserving sharp reaction zones.
  const d = state.data.dimensional;
  if (!Number.isFinite(Da) || Da < 0 || !Number.isFinite(gamma) || gamma <= 0) throw new Error('Invalid Da or salt ratio.');
  if (gamma < d.gamma[0] - 1e-10 || gamma > d.gamma[d.gamma.length - 1] + 1e-10) throw new Error('Salt ratio is beyond precomputed data.');
  const q = clamp(Da === 0 ? d.log_da[0] : Math.log10(Da), d.log_da[0], d.log_da[d.log_da.length - 1]);
  if (Da > 0 && Math.log10(Da) > d.log_da[d.log_da.length - 1] + 1e-10) throw new Error('Da is beyond precomputed data.');
  const j = locate(d.log_da, q);
  const i = locate(d.gamma, gamma);
  const v = clamp((q - d.log_da[j]) / (d.log_da[j+1] - d.log_da[j]), 0, 1);
  const u = clamp((gamma - d.gamma[i]) / (d.gamma[i+1] - d.gamma[i]), 0, 1);
  const corners = [
    [i, j, (1-u)*(1-v)], [i, j+1, (1-u)*v],
    [i+1, j, u*(1-v)], [i+1, j+1, u*v]
  ].filter(item => item[2] > 1e-10);

  const nq = d.log_da.length;
  const entries = corners.map(([gi, di, weight]) => {
    const key = gi*nq+di;
    const a = d.offsets[key], b = d.offsets[key+1];
    return {x: d.x.subarray(a,b), c: [d.cH.subarray(a,b),d.cNa.subarray(a,b),d.cOH.subarray(a,b)],
      peak: d.peak[key], weight};
  });
  const targetPeak = clamp(entries.reduce((s, p)=>s+p.weight*p.peak,0), 1e-8, 1-1e-8);
  const merged = [];
  for (let z=0;z<=359;z++) merged.push(z/359); // common readability grid
  for (const e of entries) {
    const sourcePeak = clamp(e.peak,1e-8,1-1e-8);
    e.warped = new Float64Array(e.x.length);
    for (let k=0;k<e.x.length;k++) {
      const xx = e.x[k];
      e.warped[k] = xx <= sourcePeak ? xx*targetPeak/sourcePeak :
        targetPeak + (xx-sourcePeak)*(1-targetPeak)/(1-sourcePeak);
      merged.push(e.warped[k]);
    }
  }
  merged.sort((a,b)=>a-b);
  // De-duplicate exactly matching / numerical-roundoff x values.
  const xvals = [];
  for (const xx of merged) if (xvals.length===0 || xx-xvals[xvals.length-1]>1e-15) xvals.push(xx);
  if (xvals[0] !== 0) xvals[0]=0;
  if (xvals[xvals.length-1] !== 1) xvals[xvals.length-1]=1;
  const x = Float64Array.from(xvals);
  const cs = [new Float64Array(x.length), new Float64Array(x.length), new Float64Array(x.length)];
  for (const e of entries) {
    let j = 0;
    for (let k=0; k<x.length; k++) {
      const xv=x[k];
      while (j<e.warped.length-2 && e.warped[j+1]<xv) j++;
      const x0=e.warped[j], x1=e.warped[j+1];
      const w=x1>x0 ? clamp((xv-x0)/(x1-x0),0,1) : 0;
      for (let s=0; s<3; s++) {
        cs[s][k] += e.weight*(e.c[s][j]*(1-w)+e.c[s][j+1]*w);
      }
    }
  }
  const cl = new Float64Array(x.length);
  for (let k=0;k<x.length;k++) {
    const h=cs[0][k]=Math.max(cs[0][k],0);
    const na=cs[1][k]=Math.max(cs[1][k],0);
    const oh=cs[2][k]=Math.min(Math.max(cs[2][k],0),h+na);
    cl[k]=h+na-oh;
  }
  cs[0][0]=1; cs[1][0]=gamma; cs[2][0]=0; cl[0]=1+gamma;
  const end=x.length-1;
  cs[0][end]=0; cs[1][end]=1; cs[2][end]=1; cl[end]=0;
  return {x,c:[...cs,cl]};
}

function daProfile(q) {
  const data = state.data.da;
  const idx = clamp(Math.round((q - data.log_da[0]) / (data.log_da[1]-data.log_da[0])),0,data.log_da.length-1);
  const a=data.offsets[idx], b=data.offsets[idx+1];
  return {x:data.x.subarray(a,b),c:[data.cH.subarray(a,b),data.cNa.subarray(a,b),
    data.cOH.subarray(a,b),data.cCl.subarray(a,b)], Da:10**data.log_da[idx]};
}

function dimensionalValues() {
  const c0=Number($('c0').value),cs=Number($('cs').value);
  const L=10**Number($('logL').value),dstar=10**Number($('logD').value);
  const qk=Number($('logk').value),k=qk<-12 ? 0 : 10**qk;
  return {c0,cs,L,k,dstar,Da:k*c0*L*L/dstar,gamma:cs/c0};
}
function chooseUnit(L) {
  if (L<1e-6) return [1e9,'nm'];
  if (L<1e-3) return [1e6,'μm'];
  if (L<1) return [1e3,'mm'];
  return [1,'m'];
}

// Recover fluxes without differentiating concentration profiles.
// Conservation: J_H(x)=h0-I(x), J_OH(x)=o0-I(x), J_Na=n0,
// J_Cl=h0+n0-o0, where I(x)=integral_0^x Da*C_H*C_OH dx.
// The three unknown inlet fluxes are determined from integrated Nernst–Planck
// concentration balances and the electroneutral zero-current field closure.
// This reproduces the original BVP fluxes on tabulated snapshots, with a
// small quadrature error; off-grid results inherit concentration interpolation.
function solve3(A, b) {
  const m = A.map((row, i) => [...row, b[i]]);
  for (let k=0; k<3; k++) {
    let pivot=k;
    for (let i=k+1; i<3; i++) if (Math.abs(m[i][k]) > Math.abs(m[pivot][k])) pivot=i;
    if (Math.abs(m[pivot][k]) < 1e-13) throw new Error('Flux reconstruction matrix is singular.');
    [m[k],m[pivot]]=[m[pivot],m[k]];
    for (let i=k+1; i<3; i++) {
      const fac=m[i][k]/m[k][k];
      for (let j=k; j<4; j++) m[i][j]-=fac*m[k][j];
    }
  }
  const z=[0,0,0];
  for (let k=2; k>=0; k--) {
    let val=m[k][3];
    for (let j=k+1; j<3; j++) val-=m[k][j]*z[j];
    z[k]=val/m[k][k];
  }
  return z;
}

function recoverFlux(x, c, Da) {
  const [h, na, oh] = c, n=x.length;
  if (!state.drel || n<2) throw new Error('Missing diffusivity ratios or numerical grid.');
  const [dH,dNa,dOH,dCl] = state.drel;
  const I=new Float64Array(n), rate=new Float64Array(n);
  for (let k=0; k<n; k++) rate[k]=Da*Math.max(0,h[k])*Math.max(0,oh[k]);
  for (let k=1; k<n; k++) {
    const dx=x[k]-x[k-1];
    I[k]=I[k-1]+0.5*dx*(rate[k-1]+rate[k]);
  }

  let qH=0,qNa=0,qOH=0,qHI=0,qNaI=0,qOHI=0,intI=0;
  for (let k=1; k<n; k++) {
    const dx=0.5*(x[k]-x[k-1]);
    const den0=2*Math.max(h[k-1]+na[k-1],1e-16);
    const den1=2*Math.max(h[k]+na[k],1e-16);
    const a=h[k-1]/den0, b=h[k]/den1;
    const e=na[k-1]/den0, f=na[k]/den1;
    const g=oh[k-1]/den0, t=oh[k]/den1;
    qH+=dx*(a+b); qNa+=dx*(e+f); qOH+=dx*(g+t);
    qHI+=dx*(a*I[k-1]+b*I[k]);
    qNaI+=dx*(e*I[k-1]+f*I[k]);
    qOHI+=dx*(g*I[k-1]+t*I[k]);
    intI+=dx*(I[k-1]+I[k]);
  }

  // E=[a*J_H+b*J_Na+c*J_OH+e*I] / [2(C_H+C_Na)]
  // with J_Cl=J_H+J_Na-J_OH, and J_H=h0-I, J_OH=o0-I.
  const a=1/dH-1/dCl, b=1/dNa-1/dCl;
  const cc=-1/dOH+1/dCl, e=-1/dH+1/dOH;
  const A=[
    [a*qH-1/dH, b*qH, cc*qH],
    [a*qNa, b*qNa-1/dNa, cc*qNa],
    [-a*qOH, -b*qOH, -cc*qOH-1/dOH]
  ];
  const rhs=[
    h[n-1]-h[0]-e*qHI-intI/dH,
    na[n-1]-na[0]-e*qNaI,
    oh[n-1]-oh[0]+e*qOHI-intI/dOH
  ];
  const [h0,n0,o0]=solve3(A,rhs);
  const flux=[new Float64Array(n),new Float64Array(n),new Float64Array(n),new Float64Array(n)];
  const cl0=h0+n0-o0;
  for (let k=0; k<n; k++) {
    flux[0][k]=h0-I[k];flux[1][k]=n0;
    flux[2][k]=o0-I[k];flux[3][k]=cl0;
  }
  return flux;
}

function paint(canvasId,data) {
  // Signed fluxes use a y-axis that includes negative and positive values.
  const canvas=$(canvasId),rect=canvas.getBoundingClientRect();
  if (rect.width<1 || rect.height<1) return;
  const dpr=Math.min(window.devicePixelRatio||1,2);
  const W=rect.width,H=rect.height;
  if (canvas.width!==Math.round(W*dpr) || canvas.height!==Math.round(H*dpr)) {
    canvas.width=Math.round(W*dpr);canvas.height=Math.round(H*dpr);
  }
  const ctx=canvas.getContext('2d');
  ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,W,H);
  let rawMin=Infinity,rawMax=-Infinity;
  for(const curve of data.ys) for(let i=0;i<curve.length;i++) {
    if(Number.isFinite(curve[i])){
      rawMin=Math.min(rawMin,curve[i]);rawMax=Math.max(rawMax,curve[i]);
    }
  }
  if (!Number.isFinite(rawMin)) {rawMin=0;rawMax=1;}
  const magnitude=Math.max(Math.abs(rawMin),Math.abs(rawMax),1e-12);
  const yMin=data.signed ? Math.min(rawMin,0)-0.08*magnitude : 0;
  const yMax=data.signed ? Math.max(rawMax,0)+0.08*magnitude : Math.max(rawMax*1.14,1e-12);
  const ticks=Array.from({length:5},(_,i)=>yMin+(yMax-yMin)*i/4);
  const fs=W<420?10:11;
  ctx.font=`${fs}px system-ui, sans-serif`;
  const tickWidth=Math.max(...ticks.map(v=>ctx.measureText(nice(Math.abs(v)<magnitude*1e-11?0:v,3)).width));
  const left=Math.max(58,Math.min(110,Math.ceil(tickWidth)+28));
  const right=W<400?8:15,top=14,bottom=37;
  const px=left,py=top,pw=Math.max(30,W-left-right),ph=Math.max(35,H-top-bottom);
  ctx.fillStyle='#536c78';ctx.textAlign='right';ctx.textBaseline='middle';
  for(let i=0;i<5;i++){
    const val=ticks[i],yy=py+ph*(1-i/4);
    ctx.beginPath();ctx.strokeStyle=Math.abs(val)<1e-10*magnitude&&data.signed?'#a8b9be':'#e3e9eb';
    ctx.lineWidth=(Math.abs(val)<1e-10*magnitude&&data.signed)?1.4:1;
    ctx.moveTo(px,yy);ctx.lineTo(px+pw,yy);ctx.stroke();
    ctx.fillStyle='#536c78';ctx.fillText(nice(Math.abs(val)<magnitude*1e-11?0:val,3),px-7,yy);
  }
  if(data.signed && yMin<0 && yMax>0){
    const y0=py+ph*(yMax/(yMax-yMin));
    ctx.beginPath();ctx.strokeStyle='#8499a2';ctx.lineWidth=1.4;
    ctx.moveTo(px,y0);ctx.lineTo(px+pw,y0);ctx.stroke();
  }
  ctx.textAlign='center';ctx.textBaseline='top';
  for(let i=0;i<=4;i++){
    const xx=px+pw*i/4;
    ctx.beginPath();ctx.strokeStyle='#edf0f1';ctx.lineWidth=1;
    ctx.moveTo(xx,py);ctx.lineTo(xx,py+ph);ctx.stroke();
    ctx.fillStyle='#536c78';ctx.fillText(nice(data.xMax*i/4,3),xx,py+ph+7);
  }
  ctx.strokeStyle='#8fa2aa';ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(px,py);ctx.lineTo(px,py+ph);ctx.lineTo(px+pw,py+ph);ctx.stroke();
  ctx.save();ctx.translate(13,py+ph/2);ctx.rotate(-Math.PI/2);
  ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillStyle='#526a76';ctx.font=`${fs}px system-ui`;
  ctx.fillText(data.yLabel,0,0);ctx.restore();
  ctx.textAlign='center';ctx.textBaseline='bottom';ctx.fillStyle='#526a76';
  ctx.fillText(data.xLabel,px+pw/2,H-2);
  ctx.save();ctx.beginPath();ctx.rect(px,py,pw,ph);ctx.clip();
  for(let s=0;s<data.ys.length;s++){
    const curve=data.ys[s];ctx.beginPath();
    ctx.strokeStyle=data.colors[s];ctx.lineWidth=W<420?2:2.3;ctx.lineJoin='round';
    for(let k=0;k<data.x.length;k++){
      const vx=data.x[k],vy=curve[k];
      if(!Number.isFinite(vy))continue;
      const cx=px+pw*vx,cy=py+ph*(yMax-vy)/(yMax-yMin);
      if(k===0)ctx.moveTo(cx,cy);else ctx.lineTo(cx,cy);
    }
    ctx.stroke();
  }
  if(data.peak!==undefined && data.peakR>0){
    ctx.beginPath();ctx.setLineDash([4,4]);ctx.lineWidth=1;ctx.strokeStyle='#738b95';
    ctx.moveTo(px+pw*data.peak,py);ctx.lineTo(px+pw*data.peak,py+ph);
    ctx.stroke();ctx.setLineDash([]);
  }
  ctx.restore();
}

function drawDimensional() {
  const v=dimensionalValues();
  $('c0-value').textContent=v.c0.toFixed(2)+' mol/m³';
  $('cs-value').textContent=v.cs.toFixed(2)+' mol/m³';
  const [mul,unit]=chooseUnit(v.L);
  $('logL-value').textContent=nice(v.L*mul,4)+' '+unit;
  $('logk-value').textContent=sci(v.k);
  $('logD-value').textContent=sci(v.dstar)+' m²/s';
  const p=dimensionalProfile(v.Da,v.gamma);
  const J=recoverFlux(p.x,p.c,v.Da).map(curve=>Float64Array.from(curve,a=>a*v.dstar*v.c0/v.L));
  const c=p.c.map(arr=>Float64Array.from(arr,a=>a*v.c0));
  const r=new Float64Array(p.x.length);
  let peakR=0,peak=0;
  for(let j=0;j<r.length;j++){
    r[j]=v.k*c[0][j]*c[2][j];
    if(r[j]>peakR){peakR=r[j];peak=p.x[j];}
  }
  state.last.dimensional={x:p.x,c,J,r,peak,peakR,values:v,mul,unit};
  $('stat-da').textContent=nice(v.Da,4);
  $('stat-gamma').textContent=nice(v.gamma,4);
  $('stat-peak').textContent=peakR?nice(peak*v.L*mul,4)+' '+unit:'No reaction';
  renderDimensional();
}
function renderDimensional(){
  const d=state.last.dimensional;if(!d)return;
  const xMax=d.values.L*d.mul;
  paint('dim-concentration',{x:d.x,ys:d.c,colors:COLORS,xMax,xLabel:`x (${d.unit})`,yLabel:'cᵢ (mol/m³)'},'Concentration','dimensional');
  paint('dim-flux',{x:d.x,ys:d.J,colors:COLORS,xMax,xLabel:`x (${d.unit})`,yLabel:'jᵢ (mol/(m² s))',signed:true,peak:d.peak,peakR:d.peakR});
  paint('dim-reaction',{x:d.x,ys:[d.r],colors:[REACTION_COLOR],xMax,xLabel:`x (${d.unit})`,yLabel:'R (mol/(m³·s))',peak:d.peak,peakR:d.peakR},'Reaction','dimensional');
}
function drawDa(){
  const q=Number($('logDa').value);
  const p=daProfile(q);
  $('logDa-value').textContent=sci(p.Da);
  const r=new Float64Array(p.x.length);
  let peak=0,peakR=0;
  for(let j=0;j<r.length;j++){
    r[j]=p.Da*Math.max(p.c[0][j],0)*Math.max(p.c[2][j],0);
    if(r[j]>peakR){peakR=r[j];peak=p.x[j];}
  }
  const J=recoverFlux(p.x,p.c,p.Da);
  state.last.da={x:p.x,c:p.c,J,r,peak,peakR};
  $('stat-da-only').textContent=nice(p.Da,4);
  $('stat-da-peak').textContent=nice(peak,4);
  renderDa();
}
function renderDa(){
  const d=state.last.da;if(!d)return;
  paint('da-concentration',{x:d.x,ys:d.c,colors:COLORS,xMax:1,xLabel:'x̂',yLabel:'Ĉᵢ'},'Concentrations','da');
  paint('da-flux',{x:d.x,ys:d.J,colors:COLORS,xMax:1,xLabel:'x̂',yLabel:'Ĵᵢ',signed:true,peak:d.peak,peakR:d.peakR});
  paint('da-reaction',{x:d.x,ys:[d.r],colors:[REACTION_COLOR],xMax:1,xLabel:'x̂',yLabel:'R̂',peak:d.peak,peakR:d.peakR},'Reaction','da');
}

function setPlotFocus(panel,focus){
  panel.classList.toggle('focus-single',focus!=='all');
  for(const button of panel.querySelectorAll('.plot-choice')){
    const selected=button.dataset.focus===focus;
    button.classList.toggle('selected',selected);
    button.setAttribute('aria-pressed',selected?'true':'false');
  }
  for(const chart of panel.querySelectorAll('.chart[data-plot]')){
    chart.hidden=focus!=='all' && chart.dataset.plot!==focus;
  }
  enqueueRender();
}

function enqueueRender(){
  if(state.scheduled)return;
  state.scheduled=true;
  requestAnimationFrame(()=>{
    state.scheduled=false;
    try { if(state.currentView==='dimensional' && state.data.dimensional) drawDimensional();
      else if(state.currentView==='da' && state.data.da) drawDa(); }
    catch(error){const s=$('load-status');s.textContent='Cannot render: '+error.message;s.className='load-status error';console.error(error);}
  });
}
function activate(view){
  state.currentView=view;
  for(const type of ['dimensional','da']){
    const active=type===view;
    $('tab-'+type).classList.toggle('active',active);
    $('tab-'+type).setAttribute('aria-selected',active?'true':'false');
    $('panel-'+type).hidden=!active;
    $('panel-'+type).classList.toggle('active',active);
  }
  enqueueRender();
}
function restoreDimensional(){
  const defaults={c0:'1',cs:'0.2',logL:'-3.30',logk:'4',logD:'-8.03'};
  for(const [key,value] of Object.entries(defaults))$(key).value=value;
  enqueueRender();
}
function restoreDa(){$('logDa').value='2';enqueueRender();}

async function boot(){
  for(const id of dimInputs.concat('logDa'))$(id).addEventListener('input',enqueueRender);
  $('tab-dimensional').addEventListener('click',()=>activate('dimensional'));
  $('tab-da').addEventListener('click',()=>activate('da'));
  $('reset-dimensional').addEventListener('click',restoreDimensional);
  $('reset-da').addEventListener('click',restoreDa);
  for(const button of document.querySelectorAll('.controls-toggle')){
    button.addEventListener('click',()=>{
      const section=button.closest('.controls');
      const collapsed=section.classList.toggle('collapsed');
      button.textContent=collapsed?'Show sliders':'Hide sliders';
      button.setAttribute('aria-expanded',collapsed?'false':'true');
      enqueueRender();
    });
  }
  for(const button of document.querySelectorAll('.plot-choice')){
    button.addEventListener('click',()=>setPlotFocus(button.closest('.panel'),button.dataset.focus));
  }
  window.addEventListener('resize',enqueueRender);
  const status=$('load-status');
  try{
    const resp=await fetch('manifest.json');
    if(!resp.ok)throw new Error('Could not read manifest.json');
    const manifest=await resp.json();
    if(manifest.version!==1)throw new Error('Incompatible data version');
    state.drel=manifest.dimensional.relative_diffusivities;
    status.textContent='Loading 321 Da profiles…';
    state.data.da=await loadTable(manifest.da);
    // The smaller table is loaded first. Both modes load from the site itself.
    status.textContent='Loading 3,926 dimensional profiles…';
    state.data.dimensional=await loadTable(manifest.dimensional);
    status.textContent='Ready · drag any slider';
    status.className='load-status ready';
    enqueueRender();
  }catch(error){status.textContent='Unable to load numerical tables: '+error.message;
    status.className='load-status error';console.error(error);}
}
boot();
