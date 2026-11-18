import { Model2M, MODEL2M_VOCAB } from '../src/model2m.js';
import { buildDistribution, isNaturalChar } from '../src/stego.js';

const model = new Model2M();
const V = MODEL2M_VOCAB.length;
const CJK = /^[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]$/u;
const CN_PUNCT = new Set(Array.from('，。！？；：、（）《》“”‘’…—·「」『』【】〔〕'));
function isCjkOnly(ch) { return CJK.test(ch) || CN_PUNCT.has(ch) || ch === ' '; }
const idsNatural = [];
const idsCjk = [];
for (let i=0;i<V;i++) { if (isNaturalChar(MODEL2M_VOCAB[i])) idsNatural.push(i); if (isCjkOnly(MODEL2M_VOCAB[i])) idsCjk.push(i); }
console.log('natural', idsNatural.length, 'cjk', idsCjk.length);

function gen(prompt, n, profile, allowed, seed) {
  model.reset();
  let s = seed >>> 0;
  const rnd = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
  const history = [];
  const map = new Map(MODEL2M_VOCAB.map((c,i)=>[c,i]));
  for (const ch of prompt) history.push(map.get(ch) ?? 0);
  let logits = null;
  for (const id of history) logits = model.forward(id);
  const out = [];
  const recent = history.slice(-32);
  for (let i=0;i<n;i++) {
    const dist = buildDistribution(logits, history, profile, allowed);
    let r = rnd() * dist.total;
    let idx = 0, cum = 0;
    for (; idx < dist.freqs.length; idx++) { cum += dist.freqs[idx]; if (r < cum) break; }
    if (idx >= dist.freqs.length) idx = dist.freqs.length-1;
    const token = dist.ids[idx];
    out.push(MODEL2M_VOCAB[token]);
    history.push(token); recent.push(token); if (recent.length>64) recent.shift();
    logits = model.forward(token);
  }
  return out.join('');
}
const base = { temperature:0.85, topP:0.95, topK:0, repPenalty:1.1, repWindow:32, noRepeat3:true };
const prompts = ['天气很好，','生活中有很多值得记录的小事。','人工智能的发展','在我看来，'];
for (const p of prompts) {
  console.log('\n===', p, '===');
  for (const [name, allowed, opts] of [
    ['natural T.85', idsNatural, base],
    ['cjk T.85', idsCjk, base],
    ['cjk T.80', idsCjk, {...base, temperature:0.8}],
    ['cjk T.90', idsCjk, {...base, temperature:0.9}],
    ['cjk T.85 noNR', idsCjk, {...base, noRepeat3:false}],
  ]) {
    const t0 = Date.now();
    const text = gen(p, 100, opts, allowed, 7);
    console.log(`--- ${name} (${((Date.now()-t0)/1000).toFixed(2)}s) ---\n${text}`);
  }
}
