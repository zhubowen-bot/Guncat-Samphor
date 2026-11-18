import * as T from '@huggingface/transformers';
import { loadQwen } from '../src/qwen.js';

const backend = await loadQwen(T, { device: 'cpu', dtype: 'q4' });
const secret = '明天下午三点，老地方见。';
for (const pid of [1, 2]) {
  console.log('\n=== profile', pid, '===');
  try {
    const t0 = Date.now();
    const enc = await backend.encode(secret, { profileId: pid, preamble: '天气很好，' });
    console.log('cover:', enc.text);
    console.log('stats:', enc.stats);
    const dec = await backend.decode(enc.text, { profileId: pid });
    console.log('decoded ok', dec.text === secret, 'decodeMs', dec.stats.decodeMs, 'total', Date.now()-t0);
  } catch (e) { console.log('ERROR', e.message); }
}
