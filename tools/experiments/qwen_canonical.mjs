import { AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList } from '@huggingface/transformers';

const id = 'onnx-community/Qwen2.5-0.5B';
const tok = await AutoTokenizer.from_pretrained(id);
const model = await AutoModelForCausalLM.from_pretrained(id, { dtype: 'q4', device: 'cpu' });
const vocab = tok._tokenizer.model.vocab;
const entries = Object.entries(vocab);
const tokenStr = new Map();
const completeIds = [];
for (const [sid, _] of entries) {
  const tid = Number(sid);
  if (!Number.isFinite(tid)) continue;
  let s;
  try { s = tok.decode([tid], { skip_special_tokens: false }); } catch { continue; }
  if (!s || s.includes('\uFFFD')) continue;
  const cp = s.codePointAt(0);
  if (cp < 32 || cp === 127) continue;
  tokenStr.set(tid, s);
  completeIds.push(tid);
}
console.log('complete tokens', completeIds.length, 'of', entries.length);
const completeSet = new Set(completeIds);

function isNatural(s) {
  const cps = Array.from(s);
  if (cps.length === 0) return false;
  return cps.every(ch => {
    const cp = ch.codePointAt(0);
    if (cp >= 0x4e00 && cp <= 0x9fff) return true;
    if ('，。！？；：、（）《》“”‘’…—·「」『』【】〔〕'.includes(ch)) return true;
    if (/[A-Za-z0-9 ,.!?;:'"()\-]/.test(ch)) return true;
    return false;
  });
}

class CanonicalSampler extends LogitsProcessor {
  constructor({ topK = 32, temperature = 0.85, topP = 0.95, rep = 1.1, seed = 1 }) {
    super();
    this.topK = topK; this.temperature = temperature; this.topP = topP; this.rep = rep;
    this.seed = seed >>> 0;
    this.rnd = () => { this.seed = (this.seed * 1664525 + 1013904223) >>> 0; return this.seed / 4294967296; };
    this.text = '';
    this.ids = [];
    this.history = [];
    this.acceptedTotal = 0;
    this.steps = 0;
    this.entropy = 0;
  }
  canonical(tokenId, s) {
    const text = this.text + s;
    const ids = tok.encode(text, { add_special_tokens: false });
    if (ids.length !== this.ids.length + 1) return false;
    if (ids[ids.length - 1] !== tokenId) return false;
    for (let i = 0; i < this.ids.length; i++) if (ids[i] !== this.ids[i]) return false;
    return true;
  }
  _call(input_ids, logits) {
    const data = logits.data;
    const V = data.length;
    const recent = new Set(this.history.slice(-48));
    // top-K complete tokens by adjusted logit
    const cand = [];
    for (const tid of completeIds) {
      let v = data[tid];
      if (recent.has(tid)) v = v > 0 ? v / this.rep : v * this.rep;
      cand.push([tid, v]);
    }
    cand.sort((a, b) => b[1] - a[1]);
    const top = cand.slice(0, this.topK);
    const accepted = [];
    for (const [tid, v] of top) {
      const s = tokenStr.get(tid);
      if (!s) continue;
      if (this.canonical(tid, s)) accepted.push([tid, v, s]);
    }
    this.acceptedTotal += accepted.length;
    this.steps++;
    if (accepted.length === 0) {
      // fallback: force a punctuation token that is definitely canonical
      const fallback = 1773; // '。'
      const s = tokenStr.get(fallback) || '。';
      this.text += s; this.ids.push(fallback); this.history.push(fallback);
      for (let i=0;i<V;i++) data[i] = i===fallback?100:-100;
      return logits;
    }
    let max = -Infinity;
    for (const [_, v] of accepted) if (v > max) max = v;
    const probs = accepted.map(([_, v]) => Math.exp((v - max) / this.temperature));
    let sum = probs.reduce((a,b)=>a+b,0);
    const order = accepted.map((a,i)=>i).sort((a,b)=>probs[b]-probs[a]);
    let cum = 0, kept = [];
    for (const i of order) { kept.push(i); cum += probs[i]/sum; if (cum >= this.topP) break; }
    let ssum = 0; for (const i of kept) ssum += probs[i];
    let r = this.rnd() * ssum;
    let chosen = kept[kept.length-1];
    for (const i of kept) { r -= probs[i]; if (r <= 0) { chosen = i; break; } }
    let ent = 0; for (const i of kept) { const p = probs[i]/ssum; ent -= p*Math.log2(p); }
    this.entropy += ent;
    const [tid, _, s] = accepted[chosen];
    this.text += s; this.ids.push(tid); this.history.push(tid);
    for (let i=0;i<V;i++) data[i] = i===tid?100:-100;
    return logits;
  }
}

async function generate(prompt, n, opts) {
  const proc = new CanonicalSampler(opts);
  const lp = new LogitsProcessorList(); lp.push(proc);
  const inputs = tok(prompt, { add_special_tokens: false });
  await model.generate({ ...inputs, max_new_tokens: n, do_sample: false, logits_processor: lp, eos_token_id: null });
  return proc;
}

const prompts = [
  '傍晚的时候，我走在回家的路上。',
  '人工智能正在改变我们的生活。',
  '天气很好，',
  '以下是一段文字：',
];
for (const p of prompts) {
  for (const topK of [32, 64]) {
    const t0 = Date.now();
    const proc = await generate(p, 60, { topK, temperature: 0.85, topP: 0.95, rep: 1.1, seed: 7 });
    const verify = tok.encode(proc.text, { add_special_tokens: false });
    const exact = verify.length === proc.ids.length && verify.every((v,i)=>v===proc.ids[i]);
    console.log(`\n--- prompt=${p} topK=${topK} ---`);
    console.log(p + proc.text);
    console.log('exact', exact, 'ids', proc.ids.length, 'verify', verify.length, 'avgAccepted', (proc.acceptedTotal/proc.steps).toFixed(1), 'entropy', (proc.entropy/proc.steps).toFixed(2), 'ms', Date.now()-t0);
  }
}
