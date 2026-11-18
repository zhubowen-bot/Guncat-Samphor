import * as T from '@huggingface/transformers';
import { AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList } from '@huggingface/transformers';
import fs from 'fs';
const id = 'onnx-community/Qwen2.5-0.5B';
const tok = await AutoTokenizer.from_pretrained(id);
const model = await AutoModelForCausalLM.from_pretrained(id, { dtype: 'q4', device: 'cpu' });
const map = JSON.parse(fs.readFileSync('qwen_char_map.json', 'utf8'));
const charToIds = new Map(map.charToIds);
const canonical = new Map();
for (const [ch, ids] of charToIds) canonical.set(ch, Math.min(...ids));
const candidateIds = [...canonical.values()];
const idToChar = new Map([...canonical].map(([ch, i]) => [i, ch]));

class Sampler extends LogitsProcessor {
  constructor(opts) { super(); this.opts = opts; this.history = []; }
  _call(input_ids, logits) {
    const V = logits.dims.at(-1);
    const data = logits.data;
    const { temperature = 0.8, topP = 0.95, rep = 1.1 } = this.opts;
    const recent = new Set(this.history.slice(-48));
    const arr = new Float64Array(candidateIds.length);
    let best = -Infinity;
    for (let k = 0; k < candidateIds.length; k++) {
      const tid = candidateIds[k];
      let v = data[tid];
      if (recent.has(tid)) v = v > 0 ? v / rep : v * rep;
      arr[k] = v; if (v > best) best = v;
    }
    const probs = new Float64Array(candidateIds.length);
    let sum = 0;
    for (let k = 0; k < arr.length; k++) { const p = Math.exp((arr[k] - best) / temperature); probs[k] = p; sum += p; }
    for (let k = 0; k < probs.length; k++) probs[k] /= sum;
    const order = Array.from({ length: probs.length }, (_, i) => i).sort((a, b) => probs[b] - probs[a]);
    let cum = 0, kept = [];
    for (const k of order) { kept.push(k); cum += probs[k]; if (cum >= topP) break; }
    let s = 0; for (const k of kept) s += probs[k];
    let r = Math.random() * s, chosen = kept[kept.length - 1];
    for (const k of kept) { r -= probs[k]; if (r <= 0) { chosen = k; break; } }
    const tid = candidateIds[chosen];
    this.history.push(tid);
    for (let i = 0; i < V; i++) data[i] = i === tid ? 100 : -100;
    return logits;
  }
}

async function gen(prompt, n, opts, seed) {
  let s = seed >>> 0;
  const old = Math.random;
  Math.random = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
  const proc = new Sampler(opts);
  const lp = new LogitsProcessorList();
  lp.push(proc);
  const inputs = tok(prompt, { add_special_tokens: false });
  const out = await model.generate({ ...inputs, max_new_tokens: n, do_sample: false, logits_processor: lp, eos_token_id: null });
  Math.random = old;
  const ids = out.tolist()[0].map(Number).slice(inputs.input_ids.dims[1]);
  return ids.map(i => idToChar.get(i) ?? '').join('');
}

const prompts = [
  '以下是一段文字：',
  '请用自然、通顺的中文继续写作：',
  '以下是一段自然流畅的中文：',
  '最近发生了一件小事。',
  '生活中有很多值得记录的小事。',
  '今天的天气很好，',
  '我想和你聊聊最近的生活。',
  '这是一个关于人工智能的故事。',
  '在我看来，',
  '有时候，',
  '傍晚的时候，我走在回家的路上。',
  '人工智能正在改变我们的生活。',
  '你有没有想过，',
  '很多人都在讨论这个问题。',
];
for (const p of prompts) {
  const text = await gen(p, 80, { temperature: 0.8, topP: 0.95, rep: 1.1 }, 7);
  console.log('\n--- ' + (p || '(empty)') + ' ---\n' + text);
}
