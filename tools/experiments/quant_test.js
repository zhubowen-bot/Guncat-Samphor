// Compare float32 vs per-row int8 quantization of the embedded model.
const fs = require('fs');
const file = fs.readdirSync('.').find(f => f.endsWith('.html'));
const html = fs.readFileSync(file, 'utf8');
function extractWeights(html) {
  const start = html.indexOf('const weights_b64');
  const end = html.indexOf('const config', start);
  const blk = html.slice(start, end);
  const re = /"([\w.]+)"\s*:\s*\{\s*"size"\s*:\s*\[([^\]]*)\]\s*,\s*"data"\s*:\s*"([^"]*)"/gs;
  const out = {}; let m;
  while ((m = re.exec(blk))) {
    const name = m[1]; const size = m[2].split(',').map(x => parseInt(x.trim())).filter(Number.isFinite);
    const buf = Buffer.from(m[3], 'base64');
    const f32 = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
    const is2D = size.length === 2; const rows = is2D ? size[0] : 1; const cols = is2D ? size[1] : size[0];
    if (is2D) { const arr=[]; for(let r=0;r<rows;r++) arr.push(f32.slice(r*cols,(r+1)*cols)); out[name]=arr; }
    else out[name]=f32.slice(0,cols);
  }
  return out;
}
function quantize(weights) {
  const out = {};
  for (const [name, w] of Object.entries(weights)) {
    if (Array.isArray(w)) {
      const rows = w.map(row => {
        let maxAbs = 0;
        for (const v of row) maxAbs = Math.max(maxAbs, Math.abs(v));
        const scale = maxAbs / 127 || 1e-8;
        const q = new Int8Array(row.length);
        for (let i=0;i<row.length;i++) q[i] = Math.max(-127, Math.min(127, Math.round(row[i]/scale)));
        return { q, scale };
      });
      out[name] = { rows, is2D: true };
    } else {
      let maxAbs = 0;
      for (const v of w) maxAbs = Math.max(maxAbs, Math.abs(v));
      const scale = maxAbs / 127 || 1e-8;
      const q = new Int8Array(w.length);
      for (let i=0;i<w.length;i++) q[i] = Math.max(-127, Math.min(127, Math.round(w[i]/scale)));
      out[name] = { q, scale, is2D: false };
    }
  }
  return out;
}
function dequantize(qw) {
  const out = {};
  for (const [name, w] of Object.entries(qw)) {
    if (w.is2D) out[name] = w.rows.map(r => { const a = new Float32Array(r.q.length); for(let i=0;i<a.length;i++) a[i]=r.q[i]*r.scale; return a; });
    else { const a = new Float32Array(w.q.length); for(let i=0;i<a.length;i++) a[i]=w.q[i]*w.scale; out[name]=a; }
  }
  return out;
}
const config = { vocab_size: 4096, hidden_size: 128, num_layers: 6, num_heads: 4, intermediate_size: 512, norm_eps: 1e-5, max_seq_len: 256 };
const vm = html.match(/const vocab = \[([\s\S]*?)\]/);
const vocab = vm[1].match(/"(?:[^"\\]|\\.)*"/g).map(x => JSON.parse(x));
const V = vocab.length;
class MiniTransformer {
  constructor(weights, config) { this.weights=weights; this.c=config; this.headDim=config.hidden_size/config.num_heads; this.init(); }
  init(){ this.pos=0; this.kvCache=[]; }
  silu(x){ const o=new Float64Array(x.length); for(let i=0;i<x.length;i++){const v=x[i]; o[i]=v/(1+Math.exp(-v));} return o; }
  softmax(x){ let mx=-Infinity; for(const v of x) if(v>mx) mx=v; const o=new Float64Array(x.length); let s=0; for(let i=0;i<x.length;i++){o[i]=Math.exp(x[i]-mx); s+=o[i];} for(let i=0;i<o.length;i++) o[i]/=s; return o; }
  matVecMul(mat,vec){ const out=new Float64Array(mat.length); for(let r=0;r<mat.length;r++){const row=mat[r]; let s=0; for(let i=0;i<row.length;i++) s+=row[i]*vec[i]; out[r]=s;} return out; }
  rmsNorm(x,w){ let s=0; for(const v of x) s+=v*v; s=Math.sqrt(s/x.length+this.c.norm_eps); const o=new Float64Array(x.length); for(let i=0;i<x.length;i++) o[i]=w[i]*(x[i]/s); return o; }
  applyRoPE(vec,pos){ const out=new Float64Array(vec.length); const half=vec.length/2; for(let i=0;i<half;i++){const theta=pos/Math.pow(10000,(2*i)/vec.length); const c=Math.cos(theta),s=Math.sin(theta); const x1=vec[2*i],x2=vec[2*i+1]; out[2*i]=x1*c-x2*s; out[2*i+1]=x1*s+x2*c;} return out; }
  attention(x,layerIdx,currentPos){
    const wqkv=this.weights[`layers.${layerIdx}.attn.wqkv.weight`]; const qkv=this.matVecMul(wqkv,x);
    const h=this.c.num_heads,d=this.headDim; const q=[],k=[],v=[];
    for(let i=0;i<h;i++){ q.push(this.applyRoPE(qkv.slice(i*d,(i+1)*d),currentPos)); k.push(this.applyRoPE(qkv.slice((h+i)*d,(h+i+1)*d),currentPos)); v.push(qkv.slice((2*h+i)*d,(2*h+i+1)*d)); }
    this.kvCache[this.kvCache.length-1][layerIdx]={k,v};
    const headOutputs=new Float64Array(h*d);
    for(let i=0;i<h;i++){ const scores=new Float64Array(this.kvCache.length); for(let t=0;t<this.kvCache.length;t++){const hk=this.kvCache[t][layerIdx].k[i]; let s=0; for(let j=0;j<d;j++) s+=q[i][j]*hk[j]; scores[t]=s/Math.sqrt(d);} const probs=this.softmax(scores); for(let t=0;t<this.kvCache.length;t++){const hv=this.kvCache[t][layerIdx].v[i],p=probs[t]; for(let j=0;j<d;j++) headOutputs[i*d+j]+=p*hv[j];} }
    return this.matVecMul(this.weights[`layers.${layerIdx}.attn.wo.weight`],headOutputs);
  }
  mlp(x,layerIdx){ const g=this.silu(this.matVecMul(this.weights[`layers.${layerIdx}.mlp.gate_proj.weight`],x)); const u=this.matVecMul(this.weights[`layers.${layerIdx}.mlp.up_proj.weight`],x); const o=new Float64Array(g.length); for(let i=0;i<g.length;i++) o[i]=g[i]*u[i]; return this.matVecMul(this.weights[`layers.${layerIdx}.mlp.down_proj.weight`],o); }
  forward(tokenId){
    const currentPos=this.pos++; if(this.kvCache.length>=this.c.max_seq_len) this.kvCache.shift(); this.kvCache.push([]);
    let x=Float64Array.from(this.weights['embed.weight'][tokenId]);
    for(let i=0;i<this.c.num_layers;i++){ const a=this.attention(this.rmsNorm(x,this.weights[`layers.${i}.norm1.weight`]),i,currentPos); for(let j=0;j<x.length;j++) x[j]+=a[j]; const m=this.mlp(this.rmsNorm(x,this.weights[`layers.${i}.norm2.weight`]),i); for(let j=0;j<x.length;j++) x[j]+=m[j]; }
    return this.matVecMul(this.weights['embed.weight'],this.rmsNorm(x,this.weights['norm.weight']));
  }
}
const weights = extractWeights(html);
const qweights = dequantize(quantize(weights));
const m32 = new MiniTransformer(weights, config);
const m8 = new MiniTransformer(qweights, config);
function logitsAt(m, token) { return m.forward(token); }
function nll(logits, token) {
  let mx=-Infinity; for(let i=0;i<V;i++) if(logits[i]>mx) mx=logits[i];
  let sum=0; for(let i=0;i<V;i++) sum+=Math.exp(logits[i]-mx);
  return -(logits[token]-mx-Math.log(sum))/Math.log(2);
}
const sampleText = '今天天气很好，我们一起去公园散步。公园里的花都开了，空气也很清新。孩子们在草地上奔跑，老人们坐在长椅上聊天。这样的日子让人感到非常舒服。';
const chars = Array.from(sampleText);
let tot32=0, tot8=0, agree=0;
let last32=null, last8=null;
for (const ch of chars) {
  const id = vocab.indexOf(ch);
  if (id < 0) continue;
  if (last32 !== null) { const l32=last32, l8=last8; tot32+=nll(l32,id); tot8+=nll(l8,id); let b32=0,b8=0; for(let i=0;i<V;i++){if(l32[i]>l32[b32])b32=i;if(l8[i]>l8[b8])b8=i;} if(b32===b8) agree++; }
  last32 = logitsAt(m32,id);
  last8 = logitsAt(m8,id);
}
const n = chars.length-1;
console.log('avg NLL float32', (tot32/n).toFixed(4), 'bits; int8', (tot8/n).toFixed(4), 'bits; delta', ((tot8-tot32)/n).toFixed(4));
console.log('top1 agreement', (agree/n*100).toFixed(1)+'%');
// size estimate
let params=0; for(const [k,w] of Object.entries(weights)){ if(Array.isArray(w)) params+=w.reduce((a,r)=>a+r.length,0); else params+=w.length; }
console.log('params', params, 'float32 MB', params*4/1e6, 'int8 MB', params/1e6);
