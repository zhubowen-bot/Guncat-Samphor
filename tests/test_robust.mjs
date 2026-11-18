import { encode2m, decode2m } from '../src/stego2m.js';

const secret = '保密信息：明天下午三点，老地方见。';
const enc = await encode2m(secret, { profileId: 0, preamble: '天气很好，' });
console.log('cover len', Array.from(enc.text).length);
console.log(enc.text);
let t0 = Date.now();
const dec = await decode2m(enc.text);
console.log('decoded', dec.text === secret, 'ms', Date.now()-t0, dec.stats);

// wrong preamble (modify first char)
const bad = 'X' + enc.text.slice(1);
try { await decode2m(bad); console.log('BAD: wrong preamble accepted'); } catch (e) { console.log('wrong preamble rejected:', e.message); }

// corrupt one body char
const chars = Array.from(enc.text);
chars[30] = chars[30] === '的' ? '了' : '的';
try { await decode2m(chars.join('')); console.log('BAD: corruption accepted'); } catch (e) { console.log('corruption rejected:', e.message); }

// profile autodetect for profile 1
const enc1 = await encode2m(secret, { profileId: 1, preamble: '天气很好，' });
t0 = Date.now();
const dec1 = await decode2m(enc1.text);
console.log('autodetect profile1', dec1.text === secret, dec1.stats.profileId, 'ms', Date.now()-t0);
