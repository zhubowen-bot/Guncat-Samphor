import {
  BitReader, BitWriter, ArithmeticDecoder, ArithmeticEncoder,
  bitsToBytes, bytesToBits,
} from '../src/arith.js';

function randomFreq(n, total, seed) {
  let s = seed >>> 0;
  const rnd = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
  const w = [];
  let sum = 0;
  for (let i = 0; i < n; i++) { const v = 1 + rnd() * 100; w.push(v); sum += v; }
  const f = w.map(v => Math.max(1, Math.round(v / sum * total)));
  let diff = total - f.reduce((a, b) => a + b, 0);
  const maxIdx = f.indexOf(Math.max(...f));
  f[maxIdx] += diff;
  // fix any rounding to below 1
  for (let i = 0; i < n; i++) if (f[i] < 1) { f[i] = 1; f[maxIdx] -= 1; }
  const cdf = [0];
  for (let i = 0; i < n; i++) cdf.push(cdf[i] + f[i]);
  if (cdf[n] !== total) cdf[n] = total;
  return cdf;
}

const TOTAL = 65536;
const N = 600;
const cdf = randomFreq(N, TOTAL, 12345);
console.log('cdf total', cdf[N], 'n', N);

// 1) known symbols -> encode -> decode
let s = 99;
const rnd = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
const symbols = [];
for (let i = 0; i < 200; i++) {
  const v = Math.floor(rnd() * TOTAL);
  let lo = 0, hi = N;
  while (lo + 1 < hi) { const mid = (lo + hi) >>> 1; if (cdf[mid] <= v) lo = mid; else hi = mid; }
  symbols.push(lo);
}
const enc = new ArithmeticEncoder();
for (const sym of symbols) enc.encode(cdf[sym], cdf[sym + 1] - cdf[sym], TOTAL);
const bits = enc.finish();
const bytes = bitsToBytes(bits);
console.log('encoded bits', bits.length, 'bytes', bytes.length);

const dec = new ArithmeticDecoder(new BitReader(bytes));
const decoded = [];
for (let i = 0; i < symbols.length; i++) decoded.push(dec.decode(cdf, TOTAL));
const ok1 = symbols.every((v, i) => v === decoded[i]);
console.log('known symbols round-trip', ok1);
if (!ok1) {
  const idx = symbols.findIndex((v, i) => v !== decoded[i]);
  console.log('first mismatch', idx, symbols[idx], decoded[idx]);
}

// 2) secret bits -> decode symbols -> encode -> compare bits
const secretBits = new Uint8Array(4000);
for (let i = 0; i < secretBits.length; i++) secretBits[i] = (rnd() * 2) | 0;
const secretBytes = bitsToBytes(secretBits);
const dec2 = new ArithmeticDecoder(new BitReader(secretBytes));
const symbols2 = [];
while (dec2.bitsRead < secretBits.length + 64) symbols2.push(dec2.decode(cdf, TOTAL));
console.log('symbols2', symbols2.length, 'bitsRead', dec2.bitsRead);
const enc2 = new ArithmeticEncoder();
for (const sym of symbols2) enc2.encode(cdf[sym], cdf[sym + 1] - cdf[sym], TOTAL);
const bits2 = enc2.finish();
let ok2 = true;
for (let i = 0; i < secretBits.length; i++) if (bits2[i] !== secretBits[i]) { ok2 = false; console.log('bit mismatch', i, bits2[i], secretBits[i]); break; }
console.log('secret bits -> symbols -> bits', ok2, 'encoded bits', bits2.length, 'bitsRead', dec2.bitsRead);
