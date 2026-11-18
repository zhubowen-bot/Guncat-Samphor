import { encode2m } from '../src/stego2m.js';
import { Model2M, MODEL2M_VOCAB, charToToken } from '../src/model2m.js';
import { ArithmeticEncoder } from '../src/arith.js';
import {
  PROFILES_2M, PREAMBLE_LEN, buildDistribution, findToken,
  filterIdsByChar, PayloadStreamDecoder, crc32, normalizeShortPreamble,
} from '../src/stego.js';

const secret = '这是一次端到端测试。';
const enc = await encode2m(secret, { preamble: '天气很好，', profileId: 0, attempts: 1 });
const profile = PROFILES_2M[0];
const pre = normalizeShortPreamble('天气很好，');
const chars = Array.from(enc.text);
const preLen = Array.from(pre).length;
const body = chars.slice(preLen);
const model = new Model2M();
model.reset();
const ctx = Array.from(profile.hiddenPrompt + pre).map((ch) => {
  const id = MODEL2M_VOCAB.indexOf(ch);
  return id < 0 ? 0 : id;
});
for (const id of ctx) model.forward(id);
const allowed = filterIdsByChar(MODEL2M_VOCAB, profile.charFilter);
const history = ctx.slice();
const ae = new ArithmeticEncoder();
const stream = new PayloadStreamDecoder();
for (let i = 0; i < body.length; i++) {
  const id = charToToken(body[i]);
  const dist = buildDistribution(model.logits, history, profile, allowed);
  const idx = findToken(dist, id);
  if (idx < 0) throw new Error('token not in dist');
  ae.encode(dist.cdf[idx], dist.freqs[idx], dist.total);
  stream.pushBits(ae.drainBits());
  history.push(id);
  model.forward(id);
}
stream.pushBits(ae.finish());
try {
  const parsed = await stream.finish();
  console.log('OK', parsed.text);
} catch (e) {
  console.log('ERR', e.message);
}
console.log('header', stream.header);
console.log('text', JSON.stringify(stream.text));
const stored = Uint8Array.from(stream.stored);
const plain = new TextEncoder().encode(secret);
console.log('stored len', stored.length, 'plain len', plain.length, 'header len', stream.header?.length);
console.log('stored hex', Buffer.from(stored).toString('hex'));
console.log('plain  hex', Buffer.from(plain).toString('hex'));
console.log('stored crc', crc32(stored).toString(16), 'plain crc', crc32(plain).toString(16), 'expected', stream.header?.expectedCrc?.toString(16));
for (let i = 0; i < Math.min(stored.length, plain.length); i++) {
  if (stored[i] !== plain[i]) { console.log('first mismatch', i, stored[i], plain[i]); break; }
}
