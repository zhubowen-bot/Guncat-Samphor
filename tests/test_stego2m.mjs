import { encode2m, decode2m } from '../src/stego2m.js';

const cases = [
  '你好',
  '这是一次端到端测试。',
  'Hello, world! 123',
  '保密信息：明天下午三点，老地方见。',
  '中文😀emoji 测试',
  '長' .repeat(80),
];

for (const secret of cases) {
  console.log('\n=== secret ===', JSON.stringify(secret.slice(0, 40)), 'len', secret.length);
  try {
    const t0 = Date.now();
    const enc = await encode2m(secret, { profileId: 0, preamble: '天气很好，', compression: true });
    console.log('cover:', enc.text);
    console.log('stats:', enc.stats);
    const dec = await decode2m(enc.text, { profileId: 0 });
    console.log('decoded:', JSON.stringify(dec.text));
    console.log('roundtrip:', dec.text === secret, 'decode ms', dec.stats.decodeMs, 'total ms', Date.now() - t0);
    if (dec.text !== secret) {
      console.log('MISMATCH', Array.from(secret).map((c,i)=>[i,c,dec.text[i]]));
    }
  } catch (e) {
    console.log('ERROR', e.stack);
  }
}
