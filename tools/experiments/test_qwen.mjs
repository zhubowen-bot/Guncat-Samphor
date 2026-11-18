import * as T from '@huggingface/transformers';
import { loadQwen } from '../src/qwen.js';

console.log('loading Qwen (q4, cpu)...');
const t0 = Date.now();
const backend = await loadQwen(T, {
  device: 'cpu',
  dtype: 'q4',
  onProgress: (p, info) => { if (p === 1 || Math.random() < 0.02) console.log('progress', p.toFixed(3), info?.file || ''); },
});
console.log('loaded in', (Date.now()-t0)/1000, 's; chars', backend.charToId.size, 'natural', backend.naturalIds.length);

const secret = '明天下午三点，老地方见。';
console.log('encoding...');
const enc = await backend.encode(secret, { profileId: 0, preamble: '天气很好，', onProgress: (p)=>{} });
console.log('cover:', enc.text);
console.log('stats:', enc.stats);
console.log('decoding...');
const dec = await backend.decode(enc.text, { profileId: 0 });
console.log('decoded:', JSON.stringify(dec.text), 'ok', dec.text === secret);
console.log('decode stats', dec.stats);
