// Extract and test the embedded 2M model with valid-vocab restriction.
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
function tokenize(str){ const ids=[]; for(const ch of str){ let id=vocab.indexOf(ch); if(id<0) id=vocab.indexOf(' '); ids.push(id);} return ids; }
function buildDist(logits, opts){
  const {temperature=1.0, topK=0, topP=1.0, minP=0, repetitionPenalty=1.0, recent=[]} = opts;
  let arr=Array.from(logits).slice(0,V); // restrict to known vocab
  if(repetitionPenalty!==1) for(const id of new Set(recent)) arr[id]=arr[id]>0?arr[id]/repetitionPenalty:arr[id]*repetitionPenalty;
  let idx=arr.map((v,i)=>i); idx.sort((a,b)=>arr[b]-arr[a]);
  const max=arr[idx[0]];
  let probs=idx.map(i=>Math.exp((arr[i]-max)/temperature));
  let sum=probs.reduce((a,b)=>a+b,0); probs=probs.map(p=>p/sum);
  let cand=idx.map((i,j)=>({id:i,p:probs[j]})).filter(c=>c.p>=minP);
  if(topK>0) cand=cand.slice(0,topK);
  if(topP<1){ let cum=0; const kept=[]; for(const c of cand){kept.push(c); cum+=c.p; if(cum>=topP) break;} cand=kept; }
  sum=cand.reduce((a,c)=>a+c.p,0);
  return {cand, total:sum};
}
function sampleDist(dist){ let r=Math.random()*dist.total; for(const c of dist.cand){ r-=c.p; if(r<=0) return c.id; } return dist.cand[dist.cand.length-1].id; }
function generate(prompt,n,opts,seed=1){
  model.init(); let s=seed>>>0; const rnd=()=>{s=(s*1664525+1013904223)>>>0; return s/4294967296;}; const oldRandom=Math.random; Math.random=rnd;
  const out=[]; let last=null;
  for(const id of tokenize(prompt)){ model.forward(id); last=id; }
  let recent=tokenize(prompt).slice(-32);
  for(let i=0;i<n;i++){ const logits=model.forward(last); const dist=buildDist(logits,{...opts,recent}); const id=sampleDist(dist); out.push(id); recent.push(id); if(recent.length>64) recent.shift(); last=id; }
  Math.random=oldRandom;
  return out.map(id=>vocab[id]??'�').join('');
}
function show(label,text){ console.log(`\n--- ${label} ---\n${text}`); }
const prompts=['今天天气很好，','人工智能的发展','生命的意义是什么？','他说：','在一个遥远的','作为一个AI助手，','请介绍一下北京。'];
for(const p of prompts) show('T=1 top64 valid vocab prompt='+p, generate(p,80,{temperature:1,topK:64},7));
show('T=0.9 top64', generate('今天天气很好，',80,{temperature:0.9,topK:64},7));
show('T=0.8 top64', generate('今天天气很好，',80,{temperature:0.8,topK:64},7));
show('T=0.7 top64', generate('今天天气很好，',80,{temperature:0.7,topK:64},7));
show('T=1 top1 greedy', generate('今天天气很好，',80,{temperature:0.01,topK:1},7));
show('T=1 topP=0.9 rep=1.1', generate('今天天气很好，',80,{temperature:1,topK:200,topP:0.9,repetitionPenalty:1.1},7));
show('T=0.8 topP=0.95 rep=1.15', generate('今天天气很好，',80,{temperature:0.8,topK:200,topP:0.95,repetitionPenalty:1.15},7));
show('T=1 top64 prompt=请用自然通顺的中文继续写作：', generate('请用自然通顺的中文继续写作：',100,{temperature:1,topK:64},7));
