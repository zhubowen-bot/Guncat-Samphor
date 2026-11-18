// Quality experiments for the embedded 2M model: valid vocab, filters, hidden prompts.
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
const weights = extractWeights(html);
const config = { vocab_size: 4096, hidden_size: 128, num_layers: 6, num_heads: 4, intermediate_size: 512, norm_eps: 1e-5, max_seq_len: 256 };
const vm = html.match(/const vocab = \[([\s\S]*?)\]/);
const vocab = vm[1].match(/"(?:[^"\\]|\\.)*"/g).map(x => JSON.parse(x));
const V = vocab.length;
const charToId = new Map(vocab.map((c,i)=>[c,i]));
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
const model=new MiniTransformer(weights,config);
// character filter: CJK, Chinese punctuation, ASCII letters/digits, common punctuation, space
function isNaturalChar(c) {
  const code = c.codePointAt(0);
  if (code >= 0x4E00 && code <= 0x9FFF) return true;
  if ('，。！？；：、（）《》“”‘’…—·「」『』【】〔〕'.includes(c)) return true;
  if (/[A-Za-z0-9]/.test(c)) return true;
  if (' ,.!?;:\'"()-'.includes(c)) return true;
  return false;
}
const naturalIds = [];
for (let i=0;i<V;i++) if (isNaturalChar(vocab[i])) naturalIds.push(i);
console.log('natural ids', naturalIds.length, 'of', V);
function tokenize(str){ const ids=[]; for(const ch of str){ let id=charToId.get(ch); if(id===undefined) id=charToId.get(' '); ids.push(id);} return ids; }
function buildDist(logits, opts){
  const {temperature=0.85, topP=0.95, repPenalty=1.1, recent=[], useFilter=true, noRepeat3=false, history=[]} = opts;
  const ids = useFilter ? naturalIds : Array.from({length:V},(_,i)=>i);
  const arr = new Float64Array(ids.length);
  const seen = new Set(recent);
  for (let k=0;k<ids.length;k++){ const id=ids[k]; let v=logits[id]; if (seen.has(id)) v = v>0 ? v/repPenalty : v*repPenalty; arr[k]=v; }
  // no-repeat 3-gram: ban tokens that would complete an existing 3-gram
  const banned = new Set();
  if (noRepeat3 && history.length>=2) {
    const a=history[history.length-2], b=history[history.length-1];
    for (let i=0;i+2<history.length;i++) if (history[i]===a && history[i+1]===b) banned.add(history[i+2]);
  }
  let order = Array.from({length:ids.length},(_,i)=>i);
  order.sort((a,b)=>arr[b]-arr[a]);
  const max=arr[order[0]];
  let probs = order.map(k=>Math.exp((arr[k]-max)/temperature));
  let sum=probs.reduce((a,b)=>a+b,0); probs=probs.map(p=>p/sum);
  let kept=[], cum=0;
  for(let j=0;j<order.length;j++){ const k=order[j]; if(banned.has(ids[k])) continue; kept.push({k,p:probs[j]}); cum+=probs[j]; if(cum>=topP) break; }
  if(kept.length===0) kept=[{k:order[0],p:1}];
  sum=kept.reduce((a,c)=>a+c.p,0);
  return { cand: kept.map(c=>({id:ids[c.k], p:c.p/sum})), entropy: -kept.reduce((a,c)=>a+(c.p/sum)*Math.log2(c.p/sum),0) };
}
function sample(dist){ let r=Math.random(); for(const c of dist.cand){ r-=c.p; if(r<=0) return c.id; } return dist.cand[dist.cand.length-1].id; }
function generate(hiddenPrompt, visiblePreamble, n, opts, seed=1){
  model.init(); let s=seed>>>0; const rnd=()=>{s=(s*1664525+1013904223)>>>0; return s/4294967296;}; const old=Math.random; Math.random=rnd;
  let last=null, history=[];
  for(const id of tokenize(hiddenPrompt+visiblePreamble)){ model.forward(id); last=id; history.push(id); }
  const out=[]; let entSum=0, nSteps=0;
  let recent=tokenize(hiddenPrompt+visiblePreamble).slice(-32);
  for(let i=0;i<n;i++){
    const logits=model.forward(last);
    const dist=buildDist(logits,{...opts,recent,history});
    const id=sample(dist); out.push(id); recent.push(id); if(recent.length>64) recent.shift(); history.push(id); last=id;
    entSum+=dist.entropy; nSteps++;
  }
  Math.random=old;
  return {text: out.map(id=>vocab[id]??'�').join(''), entropy: entSum/nSteps, n: nSteps};
}
function show(label,res){ console.log(`\n--- ${label} ---\n${res.text}\n[entropy ${res.entropy.toFixed(2)} bits/char]`); }
const hiddenPrompts = {
  none: '',
  writer: '请用自然、通顺的中文继续写作：',
  essay: '以下是一段自然流畅的中文：',
  assistant: '你好，我是人工智能助手，',
  topic: '今天，我们来聊一个话题。',
  daily: '在日常生活中，我们经常会遇到各种问题。',
  story: '从前有座山，山里有一座庙。',
  news: '据本台最新消息，',
};
const visible = '今天天气很好，';
for (const [name, hp] of Object.entries(hiddenPrompts)) {
  const r = generate(hp, visible, 80, {temperature:0.85, topP:0.95, repPenalty:1.1}, 7);
  show(name, r);
}
console.log('\n=== temperature sweep (hidden=essay) ===');
for (const T of [0.7,0.75,0.8,0.85,0.9,0.95,1.0]) {
  const r = generate(hiddenPrompts.essay, visible, 80, {temperature:T, topP:0.95, repPenalty:1.1}, 7);
  console.log(`T=${T} entropy=${r.entropy.toFixed(2)}\n${r.text}\n`);
}
console.log('\n=== no-repeat 3gram ===');
for (const nr of [false,true]) {
  const r = generate(hiddenPrompts.essay, visible, 100, {temperature:0.85, topP:0.95, repPenalty:1.1, noRepeat3:nr}, 7);
  console.log(`noRepeat3=${nr} entropy=${r.entropy.toFixed(2)}\n${r.text}\n`);
}
