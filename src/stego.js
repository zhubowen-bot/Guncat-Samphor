// Shared steganography primitives: profiles, quality-aware distributions,
// payload framing, compression, CRC, and whitening.

import { BitReader, BitWriter, ArithmeticDecoder, ArithmeticEncoder, bitsToBytes } from './arith.js';

export const VERSION = 2;
export const MAGIC = 0xA7;
export const PREAMBLE_LEN = 24;
export const TOTAL = 1 << 20; // frequency table resolution

export const MODEL_IDS = { BUILTIN_2M: 0, QWEN: 1 };

export const PROFILES_2M = {
  0: {
    id: 0,
    key: 'natural',
    name: '自然',
    description: '优先通顺度，适合日常短消息',
    temperature: 0.85,
    topP: 0.95,
    topK: 0,
    repPenalty: 1.10,
    repWindow: 32,
    charFilter: 'natural',
    noRepeat3: true,
    hiddenPrompt: '',
    minChars: 28,
    maxChars: 2000,
    attempts: 3,
  },
  1: {
    id: 1,
    key: 'balanced',
    name: '平衡',
    description: '质量与容量兼顾',
    temperature: 0.95,
    topP: 0.98,
    topK: 0,
    repPenalty: 1.06,
    repWindow: 32,
    charFilter: 'natural',
    noRepeat3: false,
    hiddenPrompt: '',
    minChars: 24,
    maxChars: 2000,
    attempts: 3,
  },
  2: {
    id: 2,
    key: 'capacity',
    name: '容量',
    description: '最大化嵌入率，文本更粗糙',
    temperature: 1.0,
    topP: 1.0,
    topK: 0,
    repPenalty: 1.0,
    repWindow: 32,
    charFilter: 'all',
    noRepeat3: false,
    hiddenPrompt: '',
    minChars: 20,
    maxChars: 2000,
    attempts: 2,
  },
};

export const PROFILES_QWEN = {
  0: {
    id: 0,
    key: 'natural',
    name: '自然',
    description: '优先通顺度，Qwen 0.5B',
    temperature: 0.80,
    topP: 0.95,
    topK: 256,
    repPenalty: 1.10,
    repWindow: 48,
    charFilter: 'natural',
    noRepeat3: true,
    hiddenPrompt: '',
    minChars: 28,
    maxChars: 2000,
  },
  1: {
    id: 1,
    key: 'balanced',
    name: '平衡',
    description: '质量与容量兼顾，Qwen 0.5B',
    temperature: 0.90,
    topP: 0.98,
    topK: 512,
    repPenalty: 1.05,
    repWindow: 48,
    charFilter: 'natural',
    noRepeat3: true,
    hiddenPrompt: '',
    minChars: 24,
    maxChars: 2000,
  },
  2: {
    id: 2,
    key: 'capacity',
    name: '容量',
    description: '最大化嵌入率，文本更粗糙，Qwen 0.5B',
    temperature: 1.0,
    topP: 1.0,
    topK: 1024,
    repPenalty: 1.0,
    repWindow: 48,
    charFilter: 'all',
    noRepeat3: false,
    hiddenPrompt: '',
    minChars: 20,
    maxChars: 2000,
  },
};

const CJK = /^[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]$/u;
// Colons and quotation marks are deliberately excluded from the natural
// profile: the tiny model uses them to produce list/quote fragments.
const CN_PUNCT = new Set(Array.from('，。！？；、（）《》…—·「」『』【】〔〕'));

export function isNaturalChar(ch) {
  if (CJK.test(ch)) return true;
  if (CN_PUNCT.has(ch)) return true;
  // Reject spaces, Latin letters, digits, colons, quotes and hyphens: the
  // tiny model uses them to produce broken English, list markers and odd
  // spacing.  The "capacity" profile still allows the full vocabulary.
  return '.,!?;()'.includes(ch);
}

export function filterIdsByChar(vocab, mode) {
  if (mode === 'all') return Array.from({ length: vocab.length }, (_, i) => i);
  const out = [];
  for (let i = 0; i < vocab.length; i++) if (isNaturalChar(vocab[i])) out.push(i);
  return out;
}

export const DEFAULT_PREAMBLE = '傍晚的时候，我走在回家的路上。';

/**
 * Keep the user's opener as-is (up to 24 chars).  If it is short and does not
 * end with punctuation, append a comma so the receiver has a visible boundary.
 * No synthetic text is inserted, which keeps the first sentence natural.
 */
export function normalizeShortPreamble(input, fallback = DEFAULT_PREAMBLE, length = PREAMBLE_LEN) {
  let chars = Array.from((input || '').trim()).filter((ch) => {
    const c = ch.codePointAt(0);
    return c >= 32 && c !== 127;
  }).slice(0, length);
  if (chars.length === 0) chars = Array.from(fallback);
  if (chars.length < length) {
    const last = chars[chars.length - 1];
    if (!'，。！？；：、'.includes(last)) chars.push('，');
  }
  return chars.join('');
}

/**
 * Candidate boundaries between the public opener and the secret-bearing body.
 * The receiver tries punctuation positions in the first 24 chars; the CRC
 * header identifies the correct one.  This avoids transmitting a length field
 * before the model state is known.
 */
export function preambleEndCandidates(text, maxLen = PREAMBLE_LEN) {
  const chars = Array.from(text);
  const limit = Math.min(maxLen, Math.max(0, chars.length - 1));
  const puncts = [];
  const sentenceEnds = [];
  for (let i = 1; i <= limit; i++) {
    const ch = chars[i - 1];
    if ('，。！？；：、'.includes(ch)) puncts.push(i);
    if ('。！？'.includes(ch)) sentenceEnds.push(i);
  }
  const ordered = [];
  const add = (i) => { if (i > 0 && i < chars.length && !ordered.includes(i)) ordered.push(i); };
  // Sentence-final punctuation first: narrative openers like
  // "傍晚的时候，我走在回家的路上。" have an internal comma, and the
  // sentence mark is the real boundary.  Then short openers ending in a
  // comma/colon, then the last punctuation and a full 24-char opener.
  if (sentenceEnds.length) add(sentenceEnds[0]);
  if (puncts.length) add(puncts[0]);
  if (puncts.length) add(puncts[puncts.length - 1]);
  if (chars.length >= maxLen) add(maxLen);
  for (let i = 1; i < puncts.length - 1; i++) add(puncts[i]);
  if (ordered.length === 0) add(Math.min(maxLen, Math.max(1, chars.length - 1)));
  return ordered;
}

/* ------------------------------------------------------------------ */
/* CRC32                                                              */
/* ------------------------------------------------------------------ */

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  return table;
})();

export function crc32(bytes) {
  let c = 0xFFFFFFFF;
  for (let i = 0; i < bytes.length; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xFF] ^ (c >>> 8);
  return (c ^ 0xFFFFFFFF) >>> 0;
}

/* ------------------------------------------------------------------ */
/* Whitening (uniformisation, not encryption)                         */
/* ------------------------------------------------------------------ */

export function xorKeystream(bytes, seed = 0xA5C3F00D) {
  const out = new Uint8Array(bytes.length);
  let x = seed >>> 0;
  for (let i = 0; i < bytes.length; i++) {
    x ^= (x << 13) >>> 0;
    x ^= x >>> 17;
    x ^= (x << 5) >>> 0;
    out[i] = bytes[i] ^ (x & 0xFF);
  }
  return out;
}

/* ------------------------------------------------------------------ */
/* Compression (browser + Node 18+)                                   */
/* ------------------------------------------------------------------ */

export async function deflateRaw(bytes) {
  if (typeof CompressionStream === 'undefined') return null;
  try {
    const cs = new CompressionStream('deflate-raw');
    const writer = cs.writable.getWriter();
    writer.write(bytes);
    writer.close();
    const buf = await new Response(cs.readable).arrayBuffer();
    return new Uint8Array(buf);
  } catch {
    return null;
  }
}

export async function inflateRaw(bytes) {
  if (typeof DecompressionStream === 'undefined') throw new Error('浏览器不支持解压');
  const ds = new DecompressionStream('deflate-raw');
  const writer = ds.writable.getWriter();
  writer.write(bytes);
  writer.close();
  const buf = await new Response(ds.readable).arrayBuffer();
  return new Uint8Array(buf);
}

/* ------------------------------------------------------------------ */
/* Payload framing                                                    */
/* ------------------------------------------------------------------ */

export const HEADER_BITS = 8 + 4 + 4 + 4 + 4 + 16 + 16 + 32; // magic, ver, model, profile, flags, nonce, len, crc
export const WHITEN_SEED = 0xA5C3F00D;

function makePadding(nbytes = 48, seed = 0x51ED270B) {
  const out = new Uint8Array(nbytes);
  let x = seed >>> 0;
  for (let i = 0; i < nbytes; i++) {
    x ^= (x << 13) >>> 0;
    x ^= x >>> 17;
    x ^= (x << 5) >>> 0;
    out[i] = x & 0xFF;
  }
  return out;
}

/**
 * Build the hidden bitstream: 72-bit header + whitened payload + random tail.
 * The random tail makes the end of the cover text look like ordinary sampling.
 */
export async function makeBitstream(payloadText, { modelId, profileId, compression = true, nonce = 0 } = {}) {
  const encoder = new TextEncoder();
  const plain = encoder.encode(payloadText);
  let stored = plain;
  let compressed = false;
  if (compression && plain.length > 32) {
    const z = await deflateRaw(plain);
    if (z && z.length + 8 < plain.length) {
      stored = z;
      compressed = true;
    }
  }
  if (stored.length > 0xFFFF) throw new Error('秘密文本过长（压缩后超过 65535 字节）');
  const crc = crc32(stored);
  const whitened = xorKeystream(stored, WHITEN_SEED ^ (nonce & 0xFFFF));

  const w = new BitWriter();
  w.writeBits(MAGIC, 8);
  w.writeBits(VERSION, 4);
  w.writeBits(modelId & 0xF, 4);
  w.writeBits(profileId & 0xF, 4);
  // bit0 = deflate; bit1 = distribution/codec version 2 (Qwen quantized
  // distribution / streaming decoder).  Old texts are rejected explicitly.
  w.writeBits((compressed ? 1 : 0) | 2, 4);
  w.writeBits(nonce & 0xFFFF, 16);
  w.writeBits(stored.length, 16);
  w.writeBits(crc, 32);
  for (let i = 0; i < whitened.length; i++) w.writeBits(whitened[i], 8);
  const meaningful = w.finish();

  const padding = makePadding(64);
  const bits = new Uint8Array(meaningful.length + padding.length);
  bits.set(meaningful, 0);
  bits.set(padding, meaningful.length);

  return {
    bits,
    meaningfulBits: HEADER_BITS + stored.length * 8,
    meta: {
      modelId,
      profileId,
      nonce,
      compressed,
      plainBytes: plain.length,
      storedBytes: stored.length,
      crc,
      headerBits: HEADER_BITS,
    },
  };
}

/** Read only the fixed header (used for fast profile auto-detection). */
export function peekHeader(bits) {
  const reader = new BitReader(bits);
  const magic = reader.readBits(8);
  const version = reader.readBits(4);
  const modelId = reader.readBits(4);
  const profileId = reader.readBits(4);
  const flags = reader.readBits(4);
  const nonce = reader.readBits(16);
  const length = reader.readBits(16);
  const crc = reader.readBits(32);
  return { magic, version, modelId, profileId, flags, nonce, length, crc };
}

/**
 * Incremental payload decoder.  Feed it the arithmetic encoder's newly
 * emitted bits; it parses the header, unwhitens bytes, inflates if needed and
 * emits plaintext chunks before the whole body has been processed.  CRC is
 * checked at the end.
 */
export class PayloadStreamDecoder {
  constructor({ onText = null, onHeader = null } = {}) {
    this.onText = onText;
    this.onHeader = onHeader;
    this.bits = [];
    this.bitPos = 0;
    this.header = null;
    this.stored = [];
    this.text = '';
    this.emitBuffer = '';
    this.textDecoder = new TextDecoder();
    this.remaining = 0;
    this.x = 0;
    this.compressed = false;
    this.error = null;
    this.writer = null;
    this.reader = null;
    this.pump = null;
    this.compPending = [];
    this.started = false;
  }

  _emit(s) {
    if (!s) return;
    if (!this.started) {
      this.started = true;
      if (this.onText) this.onText('', true);
    }
    this.text += s;
    this.emitBuffer += s;
    // Flush aggressively so the receiver sees characters while the model is
    // still recovering the remaining payload.
    if (this.emitBuffer.length >= 1) this._flushText();
  }

  _flushText() {
    if (!this.emitBuffer) return;
    const s = this.emitBuffer;
    this.emitBuffer = '';
    if (this.onText) this.onText(s, false);
  }

  _take(n) {
    if (this.bits.length - this.bitPos < n) return null;
    let v = 0;
    for (let i = 0; i < n; i++) v = (v << 1) | this.bits[this.bitPos++];
    if (this.bitPos > 8192) {
      this.bits = this.bits.slice(this.bitPos);
      this.bitPos = 0;
    }
    // CRC and other 32-bit fields must be treated as unsigned; `<<` overflows
    // into a signed 32-bit number.
    return v >>> 0;
  }

  pushBits(bits) {
    if (this.error || !bits || bits.length === 0) return;
    for (let i = 0; i < bits.length; i++) this.bits.push(bits[i]);
    this._process();
  }

  _startInflate() {
    this.ds = new DecompressionStream('deflate-raw');
    this.writer = this.ds.writable.getWriter();
    this.reader = this.ds.readable.getReader();
    this.pump = (async () => {
      try {
        while (true) {
          const { done, value } = await this.reader.read();
          if (done) break;
          this._emit(this.textDecoder.decode(value, { stream: true }));
        }
      } catch (err) {
        this.error = err;
      }
    })();
  }

  _flushCompressed() {
    if (!this.compPending.length || !this.writer) return;
    const chunk = Uint8Array.from(this.compPending);
    this.compPending = [];
    this.writer.write(chunk).catch((err) => { this.error = err; });
  }

  _process() {
    if (this.error) return;
    if (!this.header) {
      if (this.bits.length - this.bitPos < HEADER_BITS) return;
      const magic = this._take(8);
      if (magic !== MAGIC) {
        this.error = new Error('同步头不匹配（可能是档位/模型选择错误）');
        return;
      }
      const version = this._take(4);
      if (version !== VERSION) {
        this.error = new Error(`版本不兼容: ${version}`);
        return;
      }
      const modelId = this._take(4);
      const profileId = this._take(4);
      const flags = this._take(4);
      if (!(flags & 2)) {
        this.error = new Error('旧版隐语分布不兼容，请用当前版本重新生成');
        return;
      }
      const nonce = this._take(16);
      const length = this._take(16);
      const expectedCrc = this._take(32);
      this.header = { modelId, profileId, nonce, flags, length, expectedCrc };
      if (this.onHeader) {
        try {
          this.onHeader(this.header);
        } catch (err) {
          this.error = err;
          return;
        }
      }
      this.remaining = length;
      this.compressed = (flags & 1) !== 0;
      this.x = WHITEN_SEED ^ (nonce & 0xFFFF);
      if (this.compressed) this._startInflate();
    }
    while (this.remaining > 0) {
      const b = this._take(8);
      if (b === null) break;
      this.x ^= (this.x << 13) >>> 0;
      this.x ^= this.x >>> 17;
      this.x ^= (this.x << 5) >>> 0;
      const byte = b ^ (this.x & 0xFF);
      this.stored.push(byte);
      if (this.compressed) {
        this.compPending.push(byte);
        if (this.compPending.length >= 16) this._flushCompressed();
      } else {
        this._emit(this.textDecoder.decode(Uint8Array.of(byte), { stream: true }));
      }
      this.remaining--;
    }
  }

  async finish() {
    if (this.error) throw this.error;
    if (!this.header) throw new Error('同步头不完整');
    if (this.remaining > 0) throw new Error('载荷不完整');
    if (this.compressed) {
      this._flushCompressed();
      try {
        await this.writer.close();
        await this.pump;
      } catch (err) {
        throw err;
      }
      this._emit(this.textDecoder.decode());
    } else {
      this._emit(this.textDecoder.decode());
    }
    this._flushText();
    if (this.error) throw this.error;
    const actual = crc32(Uint8Array.from(this.stored));
    if (actual !== this.header.expectedCrc) {
      throw new Error('校验失败（文本可能被修改，或开头字符不一致）');
    }
    return {
      text: this.text,
      meta: {
        modelId: this.header.modelId,
        profileId: this.header.profileId,
        nonce: this.header.nonce,
        compressed: this.compressed,
        plainBytes: this.compressed ? new TextEncoder().encode(this.text).length : this.stored.length,
        storedBytes: this.stored.length,
        crc: actual,
      },
    };
  }
}

/** Parse the bit array produced by the receiver's arithmetic encoder. */
export async function parseBitstream(bits) {
  const reader = new BitReader(bits);
  const magic = reader.readBits(8);
  if (magic !== MAGIC) throw new Error('同步头不匹配（可能是档位/模型选择错误）');
  const version = reader.readBits(4);
  if (version !== VERSION) throw new Error(`版本不兼容: ${version}`);
  const modelId = reader.readBits(4);
  const profileId = reader.readBits(4);
  const flags = reader.readBits(4);
  if (!(flags & 2)) throw new Error('旧版隐语分布不兼容，请用当前版本重新生成');
  const nonce = reader.readBits(16);
  const length = reader.readBits(16);
  const expectedCrc = reader.readBits(32);
  if (length > 0xFFFF) throw new Error('长度字段非法');

  const whitened = new Uint8Array(length);
  for (let i = 0; i < length; i++) whitened[i] = reader.readBits(8);
  const stored = xorKeystream(whitened, WHITEN_SEED ^ (nonce & 0xFFFF));
  const actualCrc = crc32(stored);
  if (actualCrc !== expectedCrc) throw new Error('校验失败（文本可能被修改，或开头字符不一致）');

  let plain = stored;
  const compressed = (flags & 1) !== 0;
  if (compressed) plain = await inflateRaw(stored);
  const text = new TextDecoder().decode(plain);
  return {
    text,
    meta: { modelId, profileId, nonce, compressed, plainBytes: plain.length, storedBytes: length, crc: actualCrc },
  };
}

/* ------------------------------------------------------------------ */
/* Distribution construction                                          */
/* ------------------------------------------------------------------ */

export function buildDistribution(logits, history, profile, allowedIds) {
  const n = allowedIds.length;
  const recent = new Set(history.slice(-profile.repWindow));
  const banned = new Set();
  if (profile.noRepeat3 && history.length >= 2) {
    const a = history[history.length - 2];
    const b = history[history.length - 1];
    for (let i = 0; i + 2 < history.length; i++) {
      if (history[i] === a && history[i + 1] === b) banned.add(history[i + 2]);
    }
  }

  const scores = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    const id = allowedIds[k];
    let v = logits[id];
    if (recent.has(id)) v = v > 0 ? v / profile.repPenalty : v * profile.repPenalty;
    scores[k] = v;
  }

  const order = Array.from({ length: n }, (_, i) => i);
  order.sort((a, b) => {
    const d = scores[b] - scores[a];
    return d !== 0 ? d : allowedIds[a] - allowedIds[b];
  });

  const maxScore = scores[order[0]];
  const probs = new Float64Array(n);
  let sum = 0;
  for (let k = 0; k < n; k++) {
    const p = Math.exp((scores[order[k]] - maxScore) / profile.temperature);
    probs[k] = p;
    sum += p;
  }
  const inv = 1 / sum;
  for (let k = 0; k < n; k++) probs[k] *= inv;

  const keptIds = [];
  const keptProbs = [];
  let cum = 0;
  const limit = profile.topK > 0 ? Math.min(profile.topK, n) : n;
  for (let k = 0; k < limit; k++) {
    const kk = order[k];
    const id = allowedIds[kk];
    if (banned.has(id)) continue;
    keptIds.push(id);
    keptProbs.push(probs[k]);
    cum += probs[k];
    if (cum >= profile.topP) break;
  }
  if (keptIds.length === 0) {
    keptIds.push(allowedIds[order[0]]);
    keptProbs.push(1);
  }

  let keptSum = 0;
  for (const p of keptProbs) keptSum += p;
  const freqs = new Uint32Array(keptIds.length);
  let freqSum = 0;
  let maxFreqIdx = 0;
  let maxFreq = -1;
  for (let i = 0; i < keptProbs.length; i++) {
    const f = Math.max(1, Math.round((keptProbs[i] / keptSum) * TOTAL));
    freqs[i] = f;
    freqSum += f;
    if (f > maxFreq) { maxFreq = f; maxFreqIdx = i; }
  }
  let diff = TOTAL - freqSum;
  if (diff > 0) {
    freqs[maxFreqIdx] += diff;
  } else if (diff < 0) {
    // Remove the rounding excess from entries that can spare a slot.
    let remaining = -diff;
    while (remaining > 0) {
      let changed = 0;
      for (let i = 0; i < freqs.length && remaining > 0; i++) {
        if (freqs[i] > 1) {
          freqs[i]--;
          remaining--;
          changed++;
        }
      }
      if (changed === 0) break;
    }
  }
  // The CDF must end exactly at TOTAL.
  let totalCheck = 0;
  for (let i = 0; i < freqs.length; i++) totalCheck += freqs[i];
  if (totalCheck !== TOTAL) {
    freqs[maxFreqIdx] += TOTAL - totalCheck;
  }

  const cdf = new Uint32Array(freqs.length + 1);
  for (let i = 0; i < freqs.length; i++) cdf[i + 1] = cdf[i] + freqs[i];

  let entropy = 0;
  for (let i = 0; i < freqs.length; i++) {
    const p = freqs[i] / TOTAL;
    entropy -= p * Math.log2(p);
  }

  return {
    ids: Uint32Array.from(keptIds),
    freqs,
    cdf,
    total: TOTAL,
    entropy,
    topProb: freqs[maxFreqIdx] / TOTAL,
  };
}

export function findToken(dist, tokenId) {
  const ids = dist.ids;
  for (let i = 0; i < ids.length; i++) if (ids[i] === tokenId) return i;
  return -1;
}

/** Encode a known token sequence into bits (receiver side). */
export function encodeTokens(tokens, distributionAt, onStep) {
  const enc = new ArithmeticEncoder();
  for (let i = 0; i < tokens.length; i++) {
    const dist = distributionAt(i);
    const idx = findToken(dist, tokens[i]);
    if (idx < 0) throw new Error(`第 ${i} 个字符不在候选分布中`);
    enc.encode(dist.cdf[idx], dist.freqs[idx], dist.total);
    if (onStep) onStep(i, dist);
  }
  return bitsToBytes(enc.finish());
}

/** Decode symbols from bits, choosing tokens with `choose(dist, index)`. */
export function decodeSymbols(bits, distributionAt, choose, onStep) {
  const reader = new BitReader(bits);
  const dec = new ArithmeticDecoder(reader);
  const tokens = [];
  for (let i = 0; ; i++) {
    const dist = distributionAt(i, dec);
    const idx = dec.decode(dist.cdf, dist.total);
    const token = dist.ids[idx];
    tokens.push(token);
    if (onStep) onStep(i, dist, token, dec);
    if (choose && choose(token, i, dec, dist) === false) break;
  }
  return { tokens, bitsRead: dec.bitsRead };
}
