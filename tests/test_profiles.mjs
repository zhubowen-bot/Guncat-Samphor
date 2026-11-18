import { encode2m, decode2m } from '../src/stego2m.js';

const secret = '保密信息：明天下午三点，老地方见。';
for (const pid of [0, 1, 2]) {
  console.log('\n=== profile', pid, '===');
  try {
    const t0 = Date.now();
    const enc = await encode2m(secret, { profileId: pid, preamble: '天气很好，', compression: true });
    console.log(enc.text);
    console.log(enc.stats);
    const dec = await decode2m(enc.text, { profileId: pid });
    console.log('roundtrip', dec.text === secret, 'decodeMs', dec.stats.decodeMs, 'total', Date.now()-t0);
  } catch (e) { console.log('ERROR', e.message); }
}
