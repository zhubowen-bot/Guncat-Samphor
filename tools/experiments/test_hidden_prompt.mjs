import { Model2M, MODEL2M_VOCAB } from '../src/model2m.js';
import { buildDistribution, isNaturalChar } from '../src/stego.js';

const model = new Model2M();
const V = MODEL2M_VOCAB.length;
const ids = [];
for (let i=0;i<V;i++) if (isNaturalChar(MODEL2M_VOCAB[i])) ids.push(i);
const map = new Map(MODEL2M_VOCAB.map((c,i)=>[c,i]));
const profile = { temperature:0.85, topP:0.95, topK:0, repPenalty:1.1, repWindow:32, noRepeat3:true };
function gen(prompt, n, seed=7) {
  model.reset();
  let s = seed>>>0; const rnd=()=>{s=(s*1664525+1013904223)>>>0; return s/4294967296;};
  const history=[]; for(const ch of prompt) history.push(map.get(ch)??0);
  let logits=null; for(const id of history) logits=model.forward(id);
  const out=[];
  for(let i=0;i<n;i++){
    const dist=buildDistribution(logits, history, profile, ids);
    let r=rnd()*dist.total, idx=0, cum=0;
    for(;idx<dist.freqs.length;idx++){cum+=dist.freqs[idx]; if(r<cum)break;}
    if(idx>=dist.freqs.length) idx=dist.freqs.length-1;
    const token=dist.ids[idx]; out.push(MODEL2M_VOCAB[token]); history.push(token); logits=model.forward(token);
  }
  return out.join('');
}
const prompts = ['天气很好，','谢邀，','生活中有很多值得记录的小事。','人工智能的发展','在我看来，'];
for (const p of prompts) {
  console.log('\n===', p, '===');
  for (const hidden of ['', '请用自然的中文写下去：', '以下是自然流畅的中文：']) {
    console.log('--- hidden:', JSON.stringify(hidden), '---');
    console.log(gen(hidden + p, 100, 7));
  }
}
