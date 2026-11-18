// End-to-end steganography with the built-in 2M model.

import { Model2M, MODEL2M_VOCAB, charToToken } from './model2m.js';
import { BitReader, ArithmeticDecoder, ArithmeticEncoder, bitsToBytes } from './arith.js';
import {
  PREAMBLE_LEN, MODEL_IDS, PROFILES_2M, MAGIC, VERSION, HEADER_BITS,
  buildDistribution, findToken, filterIdsByChar,
  makeBitstream, peekHeader, PayloadStreamDecoder,
  normalizeShortPreamble, preambleEndCandidates,
} from './stego.js';

const VOCAB = MODEL2M_VOCAB;
const CHAR_TO_ID = new Map(VOCAB.map((c, i) => [c, i]));
const SPACE_ID = CHAR_TO_ID.get(' ') ?? 0;
const NATURAL_IDS = filterIdsByChar(VOCAB, 'natural');
const ALL_IDS = Array.from({ length: VOCAB.length }, (_, i) => i);

// Long cover texts exceed the 256-position training window of the tiny model.
// Every RESET_EVERY body characters we restart the model with the last
// RESET_CONTEXT visible characters as public context.  The receiver can
// reproduce this exactly, so the arithmetic stream continues seamlessly.
const RESET_EVERY = 160;
const RESET_CONTEXT = 64;

function contextToken(ch) {
  const id = CHAR_TO_ID.get(ch);
  return id === undefined ? SPACE_ID : id;
}

function allowedIdsFor(profile) {
  return profile.charFilter === 'all' ? ALL_IDS : NATURAL_IDS;
}

function contextTokens(profile, preamble) {
  const text = profile.hiddenPrompt + preamble;
  return Array.from(text).map(contextToken);
}

function naturalEnd(ch) {
  return ch === '。' || ch === '！' || ch === '？' || ch === '!' || ch === '?' || ch === '.';
}

/** Heuristic fluency score; lower is better.  Used to pick the best of N
 *  independent cover-text candidates generated from different nonces. */
function qualityScore(text, stats, preambleEnd) {
  const chars = Array.from(text).slice(preambleEnd);
  const ngrams = new Set();
  let repeats = 0;
  for (let i = 0; i + 3 <= chars.length; i++) {
    const g = chars.slice(i, i + 3).join('');
    if (ngrams.has(g)) repeats++;
    else ngrams.add(g);
  }
  const repRate = repeats / Math.max(1, chars.length - 2);
  const body = chars.join('');
  const bad = (body.match(/\d[.、)]\b|[A-Za-z]{3,}|[`$#*_=+{}|\\/<>\[\]]|，，|。。|、、/g) || []).length;
  const badRate = bad / Math.max(1, chars.length);
  return (stats.avgNLL || 0) + 6 * repRate + 12 * badRate;
}

async function generateOnce2m(secretText, profile, normalizedPreamble, {
  model,
  compression,
  nonce,
  onProgress,
  onToken,
}) {
  const t0 = Date.now();
  const preLen = Array.from(normalizedPreamble).length;
  const framed = await makeBitstream(secretText, {
    modelId: MODEL_IDS.BUILTIN_2M,
    profileId: profile.id,
    compression,
    nonce,
  });
  const tFrame = Date.now();

  model.reset();
  const ctx = contextTokens(profile, normalizedPreamble);
  for (const id of ctx) model.forward(id);
  const tModelInit = Date.now();

  const reader = new BitReader(framed.bits);
  const dec = new ArithmeticDecoder(reader);
  const allowedIds = allowedIdsFor(profile);
  const history = ctx.slice();
  const out = Array.from(normalizedPreamble);
  if (onToken) onToken(normalizedPreamble, true);
  let entropySum = 0;
  let nllSum = 0;
  let steps = 0;
  let modelMs = 0;
  let sinceReset = 0;

  const resetContext = () => {
    const tail = out.slice(Math.max(preLen, out.length - RESET_CONTEXT)).join('');
    const ctx2 = Array.from(profile.hiddenPrompt + tail).map(contextToken);
    model.reset();
    for (const id of ctx2) model.forward(id);
    history.length = 0;
    history.push(...ctx2);
    sinceReset = 0;
  };

  const decodeOne = () => {
    if (sinceReset >= RESET_EVERY) resetContext();
    const m0 = performance.now();
    const dist = buildDistribution(model.logits, history, profile, allowedIds);
    const idx = dec.decode(dist.cdf, dist.total);
    const token = dist.ids[idx];
    const ch = VOCAB[token];
    out.push(ch);
    if (onToken) onToken(ch, false);
    history.push(token);
    model.forward(token);
    modelMs += performance.now() - m0;
    entropySum += dist.entropy;
    nllSum += -Math.log2(dist.freqs[idx] / dist.total);
    steps++;
    sinceReset++;
    return dist;
  };

  try {
    while (true) {
      const bodyLen = out.length - preLen;
      if (bodyLen >= profile.maxChars) {
        throw new Error(`容量不足：当前档位最多生成 ${profile.maxChars} 字隐语，请缩短秘密文本或改用“平衡/容量”档位`);
      }
      decodeOne();
      const bodyLenAfter = out.length - preLen;
      if (dec.bitsRead >= framed.meaningfulBits + 32 && bodyLenAfter >= profile.minChars) break;
      if ((steps & 15) === 0) await new Promise((r) => setTimeout(r, 0));
      if (onProgress) {
        const expected = Math.max(
          profile.minChars,
          Math.min(profile.maxChars, Math.ceil(framed.meaningfulBits / Math.max(1.2, entropySum / steps))),
        );
        onProgress(Math.min(1, bodyLenAfter / expected));
      }
    }

    // Finish with a natural punctuation mark; these extra symbols only use the
    // random tail, never the payload.
    let extra = 0;
    while (!naturalEnd(out[out.length - 1]) && extra < 12 && out.length - preLen < profile.maxChars) {
      decodeOne();
      extra++;
    }
  } catch (err) {
    if (String(err.message).includes('容量不足')) throw err;
    throw new Error(`编码失败：${err.message}`);
  }

  const text = out.join('');
  const bodyChars = out.length - preLen;
  const avgEntropy = steps > 0 ? entropySum / steps : 0;
  const avgNLL = steps > 0 ? nllSum / steps : 0;
  const totalMs = Date.now() - t0;
  return {
    text,
    stats: {
      model: 'builtin-2m',
      profileId: profile.id,
      profileName: profile.name,
      nonce,
      bodyChars,
      plainBytes: framed.meta.plainBytes,
      storedBytes: framed.meta.storedBytes,
      compressed: framed.meta.compressed,
      payloadBits: framed.meaningfulBits,
      bitsPerChar: bodyChars > 0 ? framed.meaningfulBits / bodyChars : 0,
      avgEntropy,
      avgNLL,
      utilization: avgEntropy > 0 ? (framed.meaningfulBits / bodyChars) / avgEntropy : 0,
      frameMs: tFrame - t0,
      modelInitMs: tModelInit - tFrame,
      encodeMs: totalMs,
      modelMs,
    },
  };
}

/**
 * Encode `secretText` into cover text.  We generate several independent
 * candidates (different whitening nonces) and keep the most fluent one.
 */
export async function encode2m(secretText, {
  preamble = '',
  profileId = 0,
  compression = true,
  onProgress = null,
  onToken = null,
  model: providedModel = null,
  attempts = null,
} = {}) {
  const profile = PROFILES_2M[profileId];
  if (!profile) throw new Error('未知质量档位');
  const t0 = Date.now();
  const normalizedPreamble = normalizeShortPreamble(preamble);
  const preLen = Array.from(normalizedPreamble).length;
  const model = providedModel || new Model2M();
  const plainBytes = new TextEncoder().encode(secretText).length;
  const autoAttempts = plainBytes > 600 ? 1 : plainBytes > 300 ? 2 : (profile.attempts ?? 3);
  const nAttempts = Math.max(1, Math.min(5, attempts ?? autoAttempts));
  let best = null;
  let lastError = null;
  for (let attempt = 0; attempt < nAttempts; attempt++) {
    try {
      const res = await generateOnce2m(secretText, profile, normalizedPreamble, {
        model,
        compression,
        nonce: attempt,
        onProgress: (p) => onProgress && onProgress((attempt + p) / nAttempts),
        onToken: attempt === 0 ? onToken : null,
      });
      res.stats.attempts = nAttempts;
      res.stats.qualityScore = qualityScore(res.text, res.stats, preLen);
      if (!best || res.stats.qualityScore < best.stats.qualityScore) best = res;
    } catch (err) {
      lastError = err;
      if (String(err.message).includes('容量不足')) throw err;
    }
  }
  if (!best) throw lastError || new Error('编码失败');
  best.stats.encodeMs = Date.now() - t0;
  best.stats.preamble = normalizedPreamble;
  return best;
}

/** Arithmetic-encode visible body chars back into bits. */
function encodeBodyToBits(stegoText, profile, maxBodyChars = Infinity, preambleEnd = PREAMBLE_LEN) {
  const chars = Array.from(stegoText);
  if (chars.length <= preambleEnd + 1) throw new Error('隐语文本太短');
  const preamble = chars.slice(0, preambleEnd).join('');
  const body = chars.slice(preambleEnd, preambleEnd + maxBodyChars);
  const model = new Model2M();
  const ctx = contextTokens(profile, preamble);
  for (const id of ctx) model.forward(id);
  const allowedIds = allowedIdsFor(profile);
  const history = ctx.slice();
  const enc = new ArithmeticEncoder();
  let entropySum = 0;
  for (let i = 0; i < body.length; i++) {
    const token = charToToken(body[i]);
    const dist = buildDistribution(model.logits, history, profile, allowedIds);
    const idx = findToken(dist, token);
    if (idx < 0) throw new Error(`第 ${i + 1} 个字符不在当前档位的候选集中`);
    enc.encode(dist.cdf[idx], dist.freqs[idx], dist.total);
    history.push(token);
    model.forward(token);
    entropySum += dist.entropy;
  }
  return {
    bits: bitsToBytes(enc.finish()),
    bodyLen: body.length,
    entropySum,
  };
}

/** Fast profile detection: decode just enough chars to read the header. */
function probeProfile2m(stegoText, profile, preambleEnd = PREAMBLE_LEN) {
  try {
    const { bits } = encodeBodyToBits(stegoText, profile, 112, preambleEnd);
    const h = peekHeader(bits);
    return h.magic === MAGIC && h.version === VERSION && h.modelId === MODEL_IDS.BUILTIN_2M && h.profileId === profile.id;
  } catch {
    return false;
  }
}

async function decodeWithProfile(stegoText, profile, {
  onProgress = null,
  onText = null,
  model: providedModel = null,
  preambleEnd = PREAMBLE_LEN,
} = {}) {
  const chars = Array.from(stegoText);
  if (chars.length <= preambleEnd + 1) throw new Error('隐语文本太短');
  const bodyLen = chars.length - preambleEnd;
  const model = providedModel || new Model2M();
  model.reset();
  const preamble = chars.slice(0, preambleEnd).join('');
  const ctx = contextTokens(profile, preamble);
  for (const id of ctx) model.forward(id);
  const allowedIds = allowedIdsFor(profile);
  const history = ctx.slice();
  const enc = new ArithmeticEncoder();
  const stream = new PayloadStreamDecoder({ onText });
  let entropySum = 0;
  for (let i = 0; i < bodyLen; i++) {
    if (i > 0 && i % RESET_EVERY === 0) {
      const tail = chars.slice(Math.max(preambleEnd, preambleEnd + i - RESET_CONTEXT), preambleEnd + i).join('');
      const ctx2 = Array.from(profile.hiddenPrompt + tail).map(contextToken);
      model.reset();
      for (const id of ctx2) model.forward(id);
      history.length = 0;
      history.push(...ctx2);
    }
    const token = charToToken(chars[preambleEnd + i]);
    const dist = buildDistribution(model.logits, history, profile, allowedIds);
    const idx = findToken(dist, token);
    if (idx < 0) throw new Error(`第 ${i + 1} 个字符不在当前档位的候选集中`);
    enc.encode(dist.cdf[idx], dist.freqs[idx], dist.total);
    stream.pushBits(enc.drainBits());
    history.push(token);
    model.forward(token);
    entropySum += dist.entropy;
    if (onProgress && (i & 7) === 0) onProgress(i / bodyLen);
    if ((i & 15) === 0) await new Promise((r) => setTimeout(r, 0));
  }
  stream.pushBits(enc.finish());
  const parsed = await stream.finish();
  return {
    ...parsed,
    stats: {
      model: 'builtin-2m',
      profileId: profile.id,
      profileName: profile.name,
      bodyChars: bodyLen,
      avgEntropy: bodyLen > 0 ? entropySum / bodyLen : 0,
      payloadBits: parsed.meta.storedBytes * 8 + HEADER_BITS,
      bitsPerChar: bodyLen > 0 ? (parsed.meta.storedBytes * 8 + HEADER_BITS) / bodyLen : 0,
    },
  };
}

/**
 * Decode a cover text.  The public opener has a variable length; candidate
 * boundaries are punctuation positions in the first 24 chars.  A short probe
 * identifies the correct (boundary, profile) pair before the full decode.
 */
export async function decode2m(stegoText, {
  profileId = null,
  preambleEnd = null,
  onProgress = null,
  onText = null,
  model = null,
} = {}) {
  const t0 = Date.now();
  const order = [];
  if (profileId != null && PROFILES_2M[profileId]) order.push(profileId);
  for (const id of Object.keys(PROFILES_2M)) {
    const n = Number(id);
    if (!order.includes(n)) order.push(n);
  }
  const chars = Array.from(stegoText);
  const ends = preambleEnd && preambleEnd > 0 && preambleEnd < chars.length
    ? [preambleEnd]
    : preambleEndCandidates(stegoText);
  for (const end of ends) {
    for (const pid of order) {
      const profile = PROFILES_2M[pid];
      if (probeProfile2m(stegoText, profile, end)) {
        const res = await decodeWithProfile(stegoText, profile, { onProgress, onText, model, preambleEnd: end });
        if (res.meta.modelId === MODEL_IDS.BUILTIN_2M && res.meta.profileId === pid) {
          res.stats.preambleEnd = end;
          res.stats.decodeMs = Date.now() - t0;
          return res;
        }
      }
    }
  }

  // Fallback for unusual texts (e.g. too short for a full 88-bit probe).
  let lastError = null;
  for (const end of ends.slice(0, 3)) {
    for (const pid of order) {
      try {
        const res = await decodeWithProfile(stegoText, PROFILES_2M[pid], { onProgress, onText, model, preambleEnd: end });
        if (res.meta.modelId !== MODEL_IDS.BUILTIN_2M) {
          lastError = new Error('模型标识不匹配');
          continue;
        }
        if (res.meta.profileId !== pid) {
          lastError = new Error('档位标识不匹配');
          continue;
        }
        res.stats.preambleEnd = end;
        res.stats.decodeMs = Date.now() - t0;
        return res;
      } catch (err) {
        lastError = err;
      }
    }
  }
  throw new Error(`提取失败：${lastError ? lastError.message : '未知错误'}。请确认文本未被修改、开头完整，且使用了同一模型。`);
}
