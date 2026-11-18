// Small benchmark for the built-in 2M backend.
//   node tests/bench.mjs
// Set BENCH_QWEN=1 to include the optional Qwen backend (downloads/caches).

import { encode2m, decode2m } from '../src/stego2m.js';
import { Model2M, MODEL2M_VOCAB } from '../src/model2m.js';
import { buildDistribution } from '../src/stego.js';

const secret = '这是一条用于基准测试的秘密消息，包含中文、English 和数字 123。';
const model = new Model2M();

function normalSampling(n = 100) {
  model.reset();
  const ids = Array.from({ length: MODEL2M_VOCAB.length }, (_, i) => i);
  const profile = { temperature: 0.85, topP: 0.95, topK: 0, repPenalty: 1.1, repWindow: 32, noRepeat3: true };
  const history = Array.from('傍晚的时候，我走在回家的路上。').map((ch) => {
    const id = MODEL2M_VOCAB.indexOf(ch);
    return id < 0 ? 0 : id;
  });
  for (const id of history) model.forward(id);
  let out = '';
  for (let i = 0; i < n; i++) {
    const dist = buildDistribution(model.logits, history, profile, ids);
    let r = Math.random() * dist.total;
    let idx = 0;
    for (; idx < dist.freqs.length; idx++) {
      r -= dist.freqs[idx];
      if (r < 0) break;
    }
    if (idx >= dist.freqs.length) idx = dist.freqs.length - 1;
    const token = dist.ids[idx];
    out += MODEL2M_VOCAB[token];
    history.push(token);
    model.forward(token);
  }
  return out;
}

const t0 = performance.now();
const sample = normalSampling(100);
const tSample = performance.now() - t0;
console.log(`normal sampling 100 chars: ${(tSample / 1000).toFixed(3)} s`);
console.log(sample.slice(0, 80) + '…');

for (const profileId of [0, 1, 2]) {
  const t1 = performance.now();
  const enc = await encode2m(secret, { profileId, attempts: 2 });
  const t2 = performance.now();
  const dec = await decode2m(enc.text, { profileId });
  const t3 = performance.now();
  console.log(`\nprofile ${profileId} (${enc.stats.profileName})`);
  console.log(`  body chars ${enc.stats.bodyChars}, payload ${enc.stats.plainBytes} B, bits/char ${enc.stats.bitsPerChar.toFixed(2)}, entropy ${enc.stats.avgEntropy.toFixed(2)}`);
  console.log(`  encode ${((t2 - t1) / 1000).toFixed(2)} s, decode ${((t3 - t2) / 1000).toFixed(2)} s, ok ${dec.text === secret}`);
}

if (process.env.BENCH_QWEN === '1') {
  const T = await import('@huggingface/transformers');
  const { loadQwen } = await import('../src/qwen.js');
  const backend = await loadQwen(T, { device: 'cpu', dtype: 'q4' });
  const t1 = performance.now();
  const enc = await backend.encode(secret, { profileId: 1, preamble: '傍晚的时候，我走在回家的路上。' });
  const t2 = performance.now();
  const dec = await backend.decode(enc.text, { profileId: 1 });
  const t3 = performance.now();
  console.log(`\nQwen 0.5B profile 1`);
  console.log(`  body chars ${enc.stats.bodyChars}, tokens ${enc.stats.bodyTokens}, bits/char ${enc.stats.bitsPerChar.toFixed(2)}`);
  console.log(`  encode ${((t2 - t1) / 1000).toFixed(2)} s, decode ${((t3 - t2) / 1000).toFixed(2)} s, ok ${dec.text === secret}`);
}
