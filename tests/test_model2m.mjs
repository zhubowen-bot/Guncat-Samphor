import { Model2M, MODEL2M_VOCAB, charToToken } from '../src/model2m.js';

const model = new Model2M();
const V = MODEL2M_VOCAB.length;

function buildDist(logits, { temperature = 0.85, topP = 0.95, repPenalty = 1.1, recent = [] } = {}) {
  const seen = new Set(recent);
  const arr = new Float64Array(V);
  for (let i = 0; i < V; i++) {
    let v = logits[i];
    if (seen.has(i)) v = v > 0 ? v / repPenalty : v * repPenalty;
    arr[i] = v;
  }
  const order = Array.from({ length: V }, (_, i) => i).sort((a, b) => arr[b] - arr[a]);
  const max = arr[order[0]];
  const probs = new Float64Array(V);
  let sum = 0;
  for (let k = 0; k < V; k++) {
    const p = Math.exp((arr[order[k]] - max) / temperature);
    probs[k] = p;
    sum += p;
  }
  const kept = [];
  let cum = 0;
  for (let k = 0; k < V; k++) {
    kept.push({ id: order[k], p: probs[k] / sum });
    cum += probs[k] / sum;
    if (cum >= topP) break;
  }
  const total = kept.reduce((a, c) => a + c.p, 0);
  return kept.map(c => ({ id: c.id, p: c.p / total }));
}

function generate(prompt, n, opts, seed = 1) {
  model.reset();
  let s = seed >>> 0;
  const rnd = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
  const history = [];
  for (const ch of prompt) history.push(charToToken(ch));
  let logits = null;
  for (const id of history) logits = model.forward(id);
  const out = [];
  const recent = history.slice(-32);
  for (let i = 0; i < n; i++) {
    const dist = buildDist(logits, { ...opts, recent });
    let r = rnd();
    let chosen = dist[dist.length - 1];
    for (const c of dist) { r -= c.p; if (r <= 0) { chosen = c; break; } }
    out.push(chosen.id);
    recent.push(chosen.id); if (recent.length > 64) recent.shift();
    logits = model.forward(chosen.id);
  }
  return out.map(id => MODEL2M_VOCAB[id]).join('');
}

const t0 = Date.now();
const text = generate('天气很好，', 100, { temperature: 0.85, topP: 0.95, repPenalty: 1.1 }, 7);
console.log(text);
console.log('time', (Date.now() - t0) / 1000, 's for 100 chars');
