import * as T from '@huggingface/transformers';
import { loadQwen } from '../src/qwen.js';

console.log('loading Qwen...');
const t0 = Date.now();
const backend = await loadQwen(T, { device: 'cpu', dtype: 'q4' });
console.log('loaded', (Date.now()-t0)/1000, 's; complete', backend.completeIds.length, 'natural', backend.naturalIds.length);

const secret = '明天下午三点，老地方见。';
console.log('encoding...');
const t1 = Date.now();
const enc = await backend.encode(secret, { profileId: 0, preamble: '天气很好，', onProgress: (p) => { if (Math.random() < 0.05) console.log('progress', (p*100).toFixed(0)+'%'); } });
console.log('cover:', enc.text);
console.log('stats:', enc.stats);
console.log('encode total', (Date.now()-t1)/1000, 's');
console.log('decoding...');
const t2 = Date.now();
const dec = await backend.decode(enc.text, { profileId: 0 });
console.log('decoded:', JSON.stringify(dec.text), 'ok', dec.text === secret);
console.log('decode total', (Date.now()-t2)/1000, 's', dec.stats);
