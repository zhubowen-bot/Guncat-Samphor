// Optional larger-model backend: Qwen2.5-0.5B (ONNX, Transformers.js).
//
// Token ambiguity is the main obstacle when hiding bits in a BPE model: the
// same visible text can be segmented in several ways.  This backend solves it
// by generating only tokens that keep the body text *canonically tokenized*.
// At every step the candidate token is accepted only if
//     tokenizer.encode(bodyText + tokenStr) === bodyIds + [tokenId]
// so the receiver can recover the exact token sequence by tokenizing the
// visible body once.  This keeps the model in its native token distribution
// (much more fluent than forcing single characters) while remaining lossless.

import {
  PREAMBLE_LEN, MODEL_IDS, PROFILES_QWEN, HEADER_BITS,
  findToken, isNaturalChar, normalizeShortPreamble, preambleEndCandidates,
  makeBitstream, PayloadStreamDecoder,
} from './stego.js';
import { BitReader, ArithmeticDecoder, ArithmeticEncoder } from './arith.js';

export const QWEN_MODEL_ID = 'onnx-community/Qwen2.5-0.5B';
const FALLBACK_CHARS = ['。', '，', '.', ' ', '！', '？'];
const WHITEN_NONCE_ATTEMPTS = 6;
// WebGPU and WASM use different ORT kernels; their raw logits differ by tiny
// floating-point amounts.  Quantize to a coarse grid and use a fixed candidate
// order (token id) so both backends produce the same arithmetic intervals.
const QWEN_LOGIT_STEP = 0.5;
const QWEN_TOTAL = 4096;
const quantizeLogit = (v) => Math.round(v / QWEN_LOGIT_STEP) * QWEN_LOGIT_STEP;

// Persistent weight cache.  A page reload cannot preserve the ONNX/WebGPU
// session, but it does not need to re-read the 750 MB model from the local
// web server every time.  Cache API keys must be absolute http(s) URLs, so we
// normalise both the local URL and the (unused) HF remote URL to one key.
const MODEL_CACHE_NAME = 'stego-qwen-weights-v1';
const OPFS_DIR = 'stego-qwen';
const OPFS_MANIFEST = 'manifest.json';

function createModelCache(baseUrl) {
  const modelsBase = new URL('models/', baseUrl).href;
  const remoteMarker = '/resolve/main/';
  const normalize = (key) => {
    try {
      const abs = new URL(String(key), baseUrl).href;
      if (abs.startsWith(modelsBase)) return abs;
      const at = abs.indexOf(remoteMarker);
      if (at >= 0) return modelsBase + abs.slice(at + remoteMarker.length);
      return abs;
    } catch {
      return String(key);
    }
  };
  let opening = null;
  const open = () => {
    if (!opening) {
      opening = caches.open(MODEL_CACHE_NAME).catch((err) => {
        opening = null;
        throw err;
      });
    }
    return opening;
  };
  return {
    async match(key) {
      if (typeof caches === 'undefined') return undefined;
      if (normalize(key).endsWith('.onnx')) return undefined; // large weights live in OPFS
      try {
        return await (await open()).match(normalize(key));
      } catch {
        return undefined;
      }
    },
    async put(key, response) {
      if (typeof caches === 'undefined') return;
      const norm = normalize(key);
      if (norm.endsWith('.onnx')) return; // large weights live in OPFS
      // Do not make the first model load wait for a 750 MB cache write.
      // Cache.put is atomic, so an interrupted write is never treated as a
      // valid cached model.
      void (async () => {
        try {
          const cache = await open();
          if (await cache.match(norm)) return;
          await cache.put(norm, response);
        } catch {
          // Quota exceeded / private mode: keep working without the cache.
        }
      })();
    },
  };
}

/**
 * The 750 MB ONNX file is too large for a single Cache API entry (Chromium
 * rejects entries above ~512 MB), so keep it in the Origin Private File
 * System.  OPFS supports large files and streams back at disk speed.
 */
async function openOpfsDir() {
  if (!navigator.storage || !navigator.storage.getDirectory) return null;
  try {
    const root = await navigator.storage.getDirectory();
    return await root.getDirectoryHandle(OPFS_DIR, { create: true });
  } catch {
    return null;
  }
}

async function readOpfsManifest(dir) {
  try {
    const fh = await dir.getFileHandle(OPFS_MANIFEST);
    const file = await fh.getFile();
    return JSON.parse(await file.text());
  } catch {
    return null;
  }
}

async function writeOpfsManifest(dir, manifest) {
  try {
    const fh = await dir.getFileHandle(OPFS_MANIFEST, { create: true });
    const ws = await fh.createWritable();
    await ws.write(JSON.stringify(manifest));
    await ws.close();
  } catch { /* best effort */ }
}

async function getOpfsModel(dir, url, filename) {
  const manifest = await readOpfsManifest(dir);
  if (!manifest || manifest.url !== url || manifest.file !== filename) return null;
  let file;
  try {
    const fh = await dir.getFileHandle(filename);
    file = await fh.getFile();
  } catch {
    return null;
  }
  if (!file.size || (manifest.size && file.size !== manifest.size)) return null;
  // The local server is cheap to ask; invalidate if the model was replaced.
  try {
    const head = await fetch(url, { method: 'HEAD', cache: 'no-store' });
    if (head.ok) {
      const size = Number(head.headers.get('content-length')) || 0;
      const modified = head.headers.get('last-modified') || '';
      if (size && size !== file.size) return null;
      if (modified && manifest.modified && modified !== manifest.modified) return null;
    }
  } catch { /* offline: trust the OPFS copy */ }
  return { file, modified: manifest.modified || '' };
}

async function downloadToOpfs(dir, url, filename, onProgress) {
  const res = await fetch(url, { cache: 'no-cache' });
  if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);
  const total = Number(res.headers.get('content-length')) || 0;
  const modified = res.headers.get('last-modified') || '';
  const fh = await dir.getFileHandle(filename, { create: true });
  const ws = await fh.createWritable();
  const reader = res.body.getReader();
  let loaded = 0;
  let reported = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      await ws.write(value);
      loaded += value.byteLength;
      if (onProgress && (loaded - reported >= 4 * 1024 * 1024 || (total && loaded >= total))) {
        reported = loaded;
        onProgress(total ? Math.min(0.999, loaded / total) : 0, { loaded, total });
      }
    }
    await ws.close();
  } catch (err) {
    try { await ws.abort(); } catch {}
    try { await dir.removeEntry(filename); } catch {}
    throw err;
  }
  await writeOpfsManifest(dir, { url, file: filename, size: loaded, modified });
  if (onProgress) onProgress(1, { loaded, total: loaded });
  return { file: await fh.getFile(), modified };
}

async function prepareOpfsModel(dir, url, filename, onProgress) {
  if (!dir) return null;
  try {
    const hit = await getOpfsModel(dir, url, filename);
    if (hit) return { ...hit, state: 'hit' };
  } catch { /* fall through to download */ }
  try {
    const stored = await downloadToOpfs(dir, url, filename, onProgress);
    return { ...stored, state: 'stored' };
  } catch {
    // Quota / private mode / interrupted write: use the normal HTTP path.
    return null;
  }
}

function toNumberArray(tensor) {
  const list = tensor.tolist();
  const row = Array.isArray(list[0]) ? list[0] : list;
  return row.map(Number);
}

function isNaturalString(s) {
  const cps = Array.from(s);
  if (cps.length === 0) return false;
  for (const ch of cps) {
    if (!isNaturalChar(ch)) return false;
  }
  return true;
}

export class QwenBackend {
  constructor(T, tokenizer, model) {
    this.T = T;
    this.tokenizer = tokenizer;
    this.model = model;
    this.eosId = tokenizer.eos_token_id;
    this.tokenStr = null;
    this.completeIds = null;
    this.naturalIds = null;
  }

  /**
   * Quick sanity check: some WebGPU/fp16 combinations load successfully but
   * produce degenerate, repetitive logits.  Generate a short normal
   * continuation and reject backends with an abnormally high 3-gram repeat
   * rate or a collapsed vocabulary.
   */
  async selfCheck() {
    const prompt = '傍晚的时候，我走在回家的路上。';
    const inputs = this.tokenizer(prompt, { add_special_tokens: false });
    const out = await this.model.generate({
      ...inputs,
      max_new_tokens: 24,
      do_sample: true,
      temperature: 0.8,
      top_p: 0.9,
      repetition_penalty: 1.05,
      eos_token_id: this.eosId,
      pad_token_id: this.eosId,
    });
    const full = out.tolist()[0].map(Number);
    const gen = full.slice(inputs.input_ids.dims[1]);
    if (gen.length < 6) return { ok: false, reason: '生成过短', text: '' };
    const grams = new Set();
    let repeats = 0;
    for (let i = 0; i + 3 <= gen.length; i++) {
      const g = gen.slice(i, i + 3).join(',');
      if (grams.has(g)) repeats++;
      else grams.add(g);
    }
    const repRate = repeats / Math.max(1, gen.length - 2);
    const unique = new Set(gen).size / gen.length;
    const text = this.tokenizer.decode(gen, { skip_special_tokens: true });
    const ok = repRate < 0.42 && unique > 0.30;
    return { ok, repRate, unique, text };
  }

  async buildTokenMap(onProgress = null) {
    const vocab = this.tokenizer._tokenizer?.model?.vocab;
    if (!vocab) throw new Error('无法读取 Qwen 词表');
    const entries = Object.entries(vocab);
    const tokenStr = new Map();
    const completeIds = [];
    const naturalIds = [];
    let n = 0;
    for (const [sid] of entries) {
      const id = Number(sid);
      if (!Number.isFinite(id)) continue;
      let s;
      try {
        s = this.tokenizer.decode([id], { skip_special_tokens: false });
      } catch {
        continue;
      }
      if (!s || s.includes('\uFFFD')) continue;
      let hasControl = false;
      for (const ch of Array.from(s)) {
        const cp = ch.codePointAt(0);
        if (cp < 32 || cp === 127) { hasControl = true; break; }
      }
      if (hasControl) continue;
      // Keep multi-character word tokens too: forcing single characters puts
      // the model off its training distribution and makes it repeat.
      tokenStr.set(id, s);
      completeIds.push(id);
      if (isNaturalString(s)) naturalIds.push(id);
      n++;
      if (onProgress && (n & 8191) === 0) onProgress(Math.min(0.99, n / entries.length));
    }
    this.tokenStr = tokenStr;
    this.completeIds = completeIds;
    this.naturalIds = naturalIds;
    if (onProgress) onProgress(1);
    return { complete: completeIds.length, natural: naturalIds.length };
  }

  allowedIdsFor(profile) {
    return profile.charFilter === 'all' ? this.completeIds : this.naturalIds;
  }

  contextIds(profile, preamble) {
    const text = profile.hiddenPrompt + preamble;
    return toNumberArray(this.tokenizer(text, { add_special_tokens: false }).input_ids);
  }

  /**
   * Select the most likely complete tokens and keep only those that preserve
   * the canonical tokenization of the visible body prefix.
   */
  canonicalDistribution(logits, history, profile, bodyText, bodyIds) {
    const allowed = this.allowedIdsFor(profile);
    const recent = new Set(history.slice(-profile.repWindow));
    const banned = new Set();
    if (profile.noRepeat3 && history.length >= 2) {
      const a = history[history.length - 2];
      const b = history[history.length - 1];
      for (let i = 0; i + 2 < history.length; i++) {
        if (history[i] === a && history[i + 1] === b) banned.add(history[i + 2]);
      }
    }

    // Find a generous set of high-logit candidates without sorting the full
    // 150k vocabulary.
    let maxScore = -Infinity;
    for (let k = 0; k < allowed.length; k++) {
      const id = allowed[k];
      let v = logits[id];
      if (recent.has(id)) v = v > 0 ? v / profile.repPenalty : v * profile.repPenalty;
      v = quantizeLogit(v);
      if (v > maxScore) maxScore = v;
    }
    const threshold = maxScore - 12;
    const pool = [];
    for (let k = 0; k < allowed.length; k++) {
      const id = allowed[k];
      if (banned.has(id)) continue;
      let v = logits[id];
      if (recent.has(id)) v = v > 0 ? v / profile.repPenalty : v * profile.repPenalty;
      v = quantizeLogit(v);
      if (v >= threshold) pool.push([id, v]);
    }
    pool.sort((a, b) => (b[1] - a[1]) || (a[0] - b[0]));
    // Canonicality checking is the expensive part; the prototype accepted
    // >98% of the top candidates, so 160 checks is ample.
    const maxCheck = Math.min(160, Math.max(96, profile.topK));
    const accepted = [];
    const checked = Math.min(pool.length, maxCheck);
    for (let i = 0; i < checked; i++) {
      const [id, v] = pool[i];
      const s = this.tokenStr.get(id);
      if (!s) continue;
      const ids = this.tokenizer.encode(bodyText + s, { add_special_tokens: false });
      if (ids.length !== bodyIds.length + 1) continue;
      if (ids[ids.length - 1] !== id) continue;
      let ok = true;
      for (let j = 0; j < bodyIds.length; j++) {
        if (ids[j] !== bodyIds[j]) { ok = false; break; }
      }
      if (!ok) continue;
      accepted.push([id, v, s]);
      if (accepted.length >= profile.topK) break;
    }
    if (accepted.length === 0) {
      // Force a punctuation boundary.  It is always canonical at a boundary.
      for (const ch of FALLBACK_CHARS) {
        const ids = this.tokenizer.encode(bodyText + ch, { add_special_tokens: false });
        if (ids.length === bodyIds.length + 1) {
          const id = ids[ids.length - 1];
          const s = this.tokenStr.get(id);
          if (s !== undefined) {
            accepted.push([id, maxScore, s]);
            break;
          }
        }
      }
    }
    if (accepted.length === 0) throw new Error('无法为当前上下文找到可规范分词的候选');

    let max = -Infinity;
    for (const [, v] of accepted) if (v > max) max = v;
    // Sort by token id, not probability: the arithmetic interval order must be
    // identical on WebGPU and WASM even if two candidates swap ranks.
    const sorted = accepted.slice().sort((a, b) => a[0] - b[0]);
    let sum = 0;
    const probs = sorted.map(([, v]) => {
      const p = Math.exp((v - max) / profile.temperature);
      sum += p;
      return p;
    });
    const freqs = new Uint32Array(sorted.length);
    const ids = new Uint32Array(sorted.length);
    let freqSum = 0;
    let maxIdx = 0;
    for (let j = 0; j < sorted.length; j++) {
      ids[j] = sorted[j][0];
      const f = Math.max(1, Math.round((probs[j] / sum) * QWEN_TOTAL));
      freqs[j] = f;
      freqSum += f;
      if (f > freqs[maxIdx]) maxIdx = j;
    }
    let diff = QWEN_TOTAL - freqSum;
    if (diff > 0) freqs[maxIdx] += diff;
    else if (diff < 0) {
      let remaining = -diff;
      while (remaining > 0) {
        let changed = 0;
        for (let j = 0; j < freqs.length && remaining > 0; j++) {
          if (freqs[j] > 1) { freqs[j]--; remaining--; changed++; }
        }
        if (!changed) break;
      }
    }
    const cdf = new Uint32Array(freqs.length + 1);
    for (let j = 0; j < freqs.length; j++) cdf[j + 1] = cdf[j] + freqs[j];
    let entropy = 0;
    for (let j = 0; j < freqs.length; j++) {
      const p = freqs[j] / QWEN_TOTAL;
      entropy -= p * Math.log2(p);
    }
    return {
      ids,
      freqs,
      cdf,
      total: QWEN_TOTAL,
      entropy,
      tokenStrings: sorted.map((row) => row[2]),
    };
  }

  async generateCanonical(profile, preamble, framed, onProgress, onToken) {
    const ctxIds = this.contextIds(profile, preamble);
    const controller = new EncodeController({
      backend: this, profile, framed, ctxIds, onProgress, onToken,
    });
    const lp = new this.T.LogitsProcessorList();
    lp.push((inputIds, logits) => controller._call(inputIds, logits));
    const inputs = this.tokenizer(profile.hiddenPrompt + preamble, { add_special_tokens: false });
    const maxNew = Math.min(4096, Math.ceil(framed.meaningfulBits / 1.0) + 300);
    await this.model.generate({
      ...inputs,
      max_new_tokens: maxNew,
      do_sample: false,
      repetition_penalty: 1.0,
      no_repeat_ngram_size: 0,
      eos_token_id: this.eosId,
      pad_token_id: this.eosId,
      logits_processor: lp,
    });
    if (controller.error) throw controller.error;
    if (!controller.finished) {
      throw new Error(`容量不足：当前档位最多生成 ${profile.maxChars} 字隐语，请缩短秘密文本或改用“平衡/容量”档位`);
    }
    const verify = toNumberArray(this.tokenizer(controller.bodyText, { add_special_tokens: false }).input_ids);
    const ok = verify.length === controller.bodyIds.length &&
      verify.every((v, i) => v === controller.bodyIds[i]);
    return {
      text: preamble + controller.bodyText,
      bodyText: controller.bodyText,
      bodyIds: controller.bodyIds,
      verified: ok,
      stats: {
        bodyChars: controller.bodyText.length,
        bodyTokens: controller.bodyIds.length,
        avgEntropy: controller.steps > 0 ? controller.entropySum / controller.steps : 0,
        bitsPerChar: controller.bodyText.length > 0 ? framed.meaningfulBits / controller.bodyText.length : 0,
        bitsPerToken: controller.bodyIds.length > 0 ? framed.meaningfulBits / controller.bodyIds.length : 0,
        generateMs: controller.generateMs,
      },
    };
  }

  async encode(secretText, {
    preamble = '',
    profileId = 0,
    compression = true,
    onProgress = null,
    onToken = null,
  } = {}) {
    const profile = PROFILES_QWEN[profileId];
    if (!profile) throw new Error('未知质量档位');
    if (!this.completeIds) await this.buildTokenMap();
    const t0 = Date.now();
    const pre = normalizeShortPreamble(preamble);
    const plainBytes = new TextEncoder().encode(secretText).length;
    const maxAttempts = plainBytes > 600 ? 2 : WHITEN_NONCE_ATTEMPTS;
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      if (onToken) onToken(pre, true);
      const framed = await makeBitstream(secretText, {
        modelId: MODEL_IDS.QWEN,
        profileId,
        compression,
        nonce: attempt,
      });
      const tGen = Date.now();
      const generated = await this.generateCanonical(profile, pre, framed, onProgress, onToken);
      generated.stats.generateMs = Date.now() - tGen;
      if (!generated.verified) {
        continue;
      }
      const avgEntropy = generated.stats.avgEntropy;
      return {
        text: generated.text,
        stats: {
          model: 'qwen-0.5b',
          device: this.device,
          dtype: this.dtype,
          profileId,
          profileName: profile.name,
          nonce: attempt,
          preamble: pre,
          bodyChars: generated.stats.bodyChars,
          bodyTokens: generated.stats.bodyTokens,
          plainBytes: framed.meta.plainBytes,
          storedBytes: framed.meta.storedBytes,
          compressed: framed.meta.compressed,
          payloadBits: framed.meaningfulBits,
          bitsPerChar: generated.stats.bitsPerChar,
          bitsPerToken: generated.stats.bitsPerToken,
          avgEntropy,
          utilization: avgEntropy > 0 ? generated.stats.bitsPerToken / avgEntropy : 0,
          encodeMs: Date.now() - t0,
          generateMs: generated.stats.generateMs,
        },
      };
    }
    throw new Error(`编码失败：尝试 ${maxAttempts} 次仍无法得到可规范分词的文本，请重试或改用内置 2M 模式`);
  }

  /** Run the model over the body and stream-decoded payload. */
  async decodeBodyToBits(stegoText, profile, preambleEnd, maxBodyChars = Infinity, onProgress = null, onText = null) {
    const chars = Array.from(stegoText);
    if (chars.length <= preambleEnd + 1) throw new Error('隐语文本太短');
    const pre = chars.slice(0, preambleEnd).join('');
    const bodyText = chars.slice(preambleEnd, maxBodyChars === Infinity ? undefined : preambleEnd + maxBodyChars).join('');
    const bodyIds = toNumberArray(this.tokenizer(bodyText, { add_special_tokens: false }).input_ids);
    if (bodyIds.length === 0) throw new Error('正文为空');
    for (let i = 0; i < bodyIds.length; i++) {
      if (!this.tokenStr.get(bodyIds[i])) throw new Error('正文包含无法解码的 token');
    }
    const ctxIds = this.contextIds(profile, pre);
    const stream = new PayloadStreamDecoder({
      onText,
      onHeader: (header) => {
        if (header.modelId !== MODEL_IDS.QWEN) throw new Error('模型标识不匹配');
        if (header.profileId !== profile.id) {
          const actual = PROFILES_QWEN[header.profileId]?.name || header.profileId;
          throw new Error(`档位标识不匹配（文本为 ${actual}）`);
        }
      },
    });
    const controller = new DecodeController({
      backend: this, profile, knownIds: bodyIds, ctxIds, onProgress,
      stream,
      onBits: (bits) => stream.pushBits(bits),
    });
    const lp = new this.T.LogitsProcessorList();
    lp.push((inputIds, logits) => controller._call(inputIds, logits));
    const inputs = this.tokenizer(profile.hiddenPrompt + pre, { add_special_tokens: false });
    await this.model.generate({
      ...inputs,
      max_new_tokens: bodyIds.length,
      do_sample: false,
      repetition_penalty: 1.0,
      no_repeat_ngram_size: 0,
      eos_token_id: this.eosId,
      pad_token_id: this.eosId,
      logits_processor: lp,
    });
    if (controller.error) throw controller.error;
    stream.pushBits(controller.encoder.finish());
    const parsed = await stream.finish();
    return {
      parsed,
      pre,
      bodyText,
      bodyIds,
      decodedTokens: controller.index,
      entropySum: controller.entropySum,
    };
  }

  async decodeWithProfile(stegoText, profile, { onProgress = null, onText = null, preambleEnd = PREAMBLE_LEN } = {}) {
    const { parsed, pre, bodyText, bodyIds, decodedTokens, entropySum } = await this.decodeBodyToBits(
      stegoText, profile, preambleEnd, Infinity, onProgress, onText,
    );
    return {
      ...parsed,
      stats: {
        model: 'qwen-0.5b',
        device: this.device,
        dtype: this.dtype,
        profileId: profile.id,
        profileName: profile.name,
        preamble: pre,
        preambleEnd,
        bodyChars: bodyText.length,
        bodyTokens: bodyIds.length,
        decodedTokens,
        avgEntropy: bodyIds.length > 0 ? entropySum / bodyIds.length : 0,
        payloadBits: parsed.meta.storedBytes * 8 + HEADER_BITS,
        bitsPerChar: bodyText.length > 0 ? (parsed.meta.storedBytes * 8 + HEADER_BITS) / bodyText.length : 0,
        bitsPerToken: bodyIds.length > 0 ? (parsed.meta.storedBytes * 8 + HEADER_BITS) / bodyIds.length : 0,
      },
    };
  }

  async decode(stegoText, { profileId = null, preambleEnd = null, onProgress = null, onText = null, onAttempt = null } = {}) {
    if (!this.completeIds) await this.buildTokenMap();
    const t0 = Date.now();
    const order = [];
    if (profileId != null && PROFILES_QWEN[profileId]) order.push(profileId);
    // Balanced (id 1) is the UI default for Qwen; try it before natural when
    // auto-detecting.
    for (const n of [1, 0, 2]) {
      if (!order.includes(n)) order.push(n);
    }
    const chars = Array.from(stegoText);
    const ends = preambleEnd && preambleEnd > 0 && preambleEnd < chars.length
      ? [preambleEnd]
      : preambleEndCandidates(stegoText);
    // On WebGPU the ONNX session can spend a long time recompiling for a new
    // probe length.  A full decode uses the final body length and is usually
    // much faster than a short "probe" plus a full decode.  Use the actual
    // decode at each candidate boundary/profile and rely on CRC to reject
    // wrong choices.
    let lastError = null;
    let attemptNo = 0;
    const totalAttempts = ends.length * order.length;
    for (const end of ends) {
      for (const pid of order) {
        attemptNo++;
        if (onAttempt) {
          onAttempt({
            attempt: attemptNo,
            total: totalAttempts,
            end,
            profileId: pid,
            profileName: PROFILES_QWEN[pid]?.name || String(pid),
          });
        }
        try {
          const res = await this.decodeWithProfile(stegoText, PROFILES_QWEN[pid], { onProgress, onText, preambleEnd: end });
          if (res.meta.modelId !== MODEL_IDS.QWEN) {
            lastError = new Error('模型标识不匹配');
            continue;
          }
          if (res.meta.profileId !== pid) {
            lastError = new Error('档位标识不匹配');
            continue;
          }
          res.stats.decodeMs = Date.now() - t0;
          return res;
        } catch (err) {
          lastError = err;
        }
      }
    }
    throw new Error(
      `提取失败：${lastError ? lastError.message : '未知错误'}。` +
      `已尝试 ${order.length} 个档位 × ${ends.length} 个开头位置；` +
      '请确认文本未被修改、开头完整、档位一致，并尽量使用与编码时相同的后端（WebGPU/WASM）。',
    );
  }
}

/* ------------------------------------------------------------------ */
/* Generation controllers                                             */
/* ------------------------------------------------------------------ */

class BaseController {
  constructor({ backend, profile, ctxIds, onProgress }) {
    this.backend = backend;
    this.profile = profile;
    this.history = ctxIds.slice();
    this.onProgress = onProgress;
    this.entropySum = 0;
    this.steps = 0;
    this.finished = false;
    this.error = null;
    this.generateMs = 0;
  }
  force(logits, token) {
    const data = logits.data;
    for (let i = 0; i < data.length; i++) data[i] = -100;
    data[token] = 100;
    return logits;
  }
  forceEos(logits) {
    return this.force(logits, this.backend.eosId);
  }
}

class EncodeController extends BaseController {
  constructor({ backend, profile, framed, ctxIds, onProgress, onToken }) {
    super({ backend, profile, ctxIds, onProgress });
    this.decoder = new ArithmeticDecoder(new BitReader(framed.bits));
    this.meaningfulBits = framed.meaningfulBits;
    this.bodyText = '';
    this.bodyIds = [];
    this.extra = 0;
    this.onToken = onToken;
  }
  _call(inputIds, logits) {
    if (this.error || this.finished) return this.forceEos(logits);
    try {
      const t0 = performance.now();
      const enough = this.decoder.bitsRead >= this.meaningfulBits + 32 && this.bodyText.length >= this.profile.minChars;
      if (enough) {
        const last = this.bodyText[this.bodyText.length - 1] || '。';
        const end = last === '。' || last === '！' || last === '？' || last === '!' || last === '?' || last === '.';
        if (end || this.extra >= 16) {
          this.finished = true;
          this.generateMs += performance.now() - t0;
          return this.forceEos(logits);
        }
        this.extra++;
      }
      if (this.bodyText.length >= this.profile.maxChars) {
        this.finished = true;
        this.error = new Error(`容量不足：当前档位最多生成 ${this.profile.maxChars} 字隐语，请缩短秘密文本或改用“平衡/容量”档位`);
        return this.forceEos(logits);
      }
      const dist = this.backend.canonicalDistribution(logits.data, this.history, this.profile, this.bodyText, this.bodyIds);
      const idx = this.decoder.decode(dist.cdf, dist.total);
      const token = dist.ids[idx];
      const s = this.backend.tokenStr.get(token);
      if (!s) throw new Error('生成了无法映射的 token');
      this.bodyText += s;
      this.bodyIds.push(token);
      if (this.onToken) this.onToken(s, false);
      this.history.push(token);
      this.entropySum += dist.entropy;
      this.steps++;
      this.generateMs += performance.now() - t0;
      if (this.onProgress) {
        const expected = Math.max(
          this.profile.minChars,
          Math.min(this.profile.maxChars, Math.ceil(this.meaningfulBits / Math.max(1.0, this.entropySum / this.steps))),
        );
        this.onProgress(Math.min(1, this.bodyIds.length / Math.max(1, expected)));
      }
      return this.force(logits, token);
    } catch (err) {
      this.error = err;
      this.finished = true;
      return this.forceEos(logits);
    }
  }
}

class DecodeController extends BaseController {
  constructor({ backend, profile, knownIds, ctxIds, onProgress, onBits, stream = null }) {
    super({ backend, profile, ctxIds, onProgress });
    this.knownIds = knownIds;
    this.index = 0;
    this.encoder = new ArithmeticEncoder();
    this.bodyText = '';
    this.bodyIds = [];
    this.onBits = onBits;
    this.stream = stream;
    this.done = false;
  }
  _call(inputIds, logits) {
    if (this.error || this.done || this.index >= this.knownIds.length) return this.forceEos(logits);
    try {
      const token = this.knownIds[this.index];
      const dist = this.backend.canonicalDistribution(logits.data, this.history, this.profile, this.bodyText, this.bodyIds);
      const idx = findToken(dist, token);
      if (idx < 0) throw new Error(`第 ${this.index + 1} 个 token 不在当前档位的候选集中`);
      this.encoder.encode(dist.cdf[idx], dist.freqs[idx], dist.total);
      if (this.onBits) this.onBits(this.encoder.drainBits());
      if (this.stream && this.stream.error) throw this.stream.error;
      const s = this.backend.tokenStr.get(token);
      if (s === undefined) throw new Error('无法映射正文 token');
      this.bodyText += s;
      this.bodyIds.push(token);
      this.history.push(token);
      this.entropySum += dist.entropy;
      this.index++;
      this.steps++;
      if (this.onProgress && (this.index & 3) === 0) this.onProgress(this.index / this.knownIds.length);
      // Once the payload is fully recovered, the remaining body tokens are
      // only cosmetic cover text.  Do not decode them: a tiny cross-backend
      // difference there would fail after the entire secret was already shown.
      if (this.stream && this.stream.header && this.stream.remaining <= 0) {
        this.done = true;
        if (this.onProgress) this.onProgress(1);
        return this.forceEos(logits);
      }
      return this.force(logits, token);
    } catch (err) {
      this.error = err;
      return this.forceEos(logits);
    }
  }
}

/**
 * Load the optional larger model.  `T` is the dynamically imported
 * Transformers.js module.
 */
export async function loadQwen(T, {
  device = 'webgpu',
  dtype = 'q4',
  modelId = 'Qwen2.5-0.5B',
  baseUrl = null,
  remote = false,
  onProgress = null,
  onCacheProgress = null,
} = {}) {
  const modelRel = dtype === 'q4' ? 'onnx/model_q4.onnx' : `onnx/model_${dtype}.onnx`;
  let modelUrl = null;
  if (baseUrl && T.env) {
    if (remote) {
      T.env.allowLocalModels = false;
      T.env.allowRemoteModels = true;
      const remoteHost = T.env.remoteHost || 'https://huggingface.co/';
      const remoteTemplate = T.env.remotePathTemplate || '{model}/resolve/{revision}/';
      const remotePath = remoteTemplate
        .replaceAll('{model}', modelId)
        .replaceAll('{revision}', 'main');
      modelUrl = new URL(remotePath + modelRel, remoteHost).href;
    } else {
      T.env.allowLocalModels = true;
      T.env.allowRemoteModels = false;
      modelUrl = new URL(`models/${modelId}/${modelRel}`, baseUrl).href;
    }
    // Cache the weights ourselves.  Transformers.js's default browser cache
    // key is the un-resolved local path ("/models/..."), which the Cache API
    // rejects.  A custom cache lets us use absolute local URLs and keeps the
    // 750 MB model out of the HTTP fetch path on every reload.
    const hasCacheApi = !remote && typeof caches !== 'undefined';
    T.env.useBrowserCache = false;
    T.env.useCustomCache = hasCacheApi;
    T.env.customCache = hasCacheApi ? createModelCache(baseUrl) : null;
    // Keep an origin-relative path: Transformers.js uses it for the
    // tokenizer's existence check, which does not accept absolute local URLs.
    // env.fetch resolves it to a full URL for the actual request.
    const localPrefix = remote ? '' : new URL('models/', baseUrl).pathname;
    if (!remote) T.env.localModelPath = localPrefix;
    T.env.__stegoLocalPrefix = localPrefix;
    T.env.__stegoBaseUrl = baseUrl;
    if (!T.env.__stegoLocalFetch) {
      const originalFetch = T.env.fetch.bind(T.env);
      T.env.fetch = (input, init) => {
        let target = input;
        const prefix = T.env.__stegoLocalPrefix || '';
        if (prefix && typeof input === 'string' && input.startsWith(prefix)) {
          target = new URL(input, T.env.__stegoBaseUrl).href;
        }
        const holder = T.env.__stegoOpfs;
        if (holder && holder.file && target === holder.url) {
          const headers = new Headers();
          headers.set('content-length', String(holder.file.size));
          headers.set('content-type', 'application/octet-stream');
          if (holder.modified) headers.set('last-modified', holder.modified);
          return Promise.resolve(new Response(holder.file, { status: 200, headers }));
        }
        return originalFetch(target, init);
      };
      T.env.__stegoLocalFetch = true;
    }
    T.env.__stegoOpfs = { url: modelUrl, file: null, modified: '' };
    if (T.env.backends && T.env.backends.onnx && T.env.backends.onnx.wasm) {
      T.env.backends.onnx.wasm.wasmPaths = new URL('vendor/ort/', baseUrl).href;
    }
  }
  const progress = (info) => {
    if (!onProgress) return;
    if (info && typeof info === 'object') {
      if (typeof info.progress === 'number') {
        // Transformers.js reports 0..100 for file downloads.
        const p = info.progress > 1 ? info.progress / 100 : info.progress;
        onProgress(Math.max(0, Math.min(1, p)), info);
      } else if (typeof info.loaded === 'number' && typeof info.total === 'number' && info.total > 0) {
        onProgress(info.loaded / info.total, info);
      }
    }
  };
  const localOnly = !remote && !!baseUrl;
  let cacheState = 'disabled';
  if (modelUrl) {
    const dir = await openOpfsDir();
    const prepared = await prepareOpfsModel(
      dir,
      modelUrl,
      modelRel.replace(/\//g, '_'),
      onCacheProgress,
    );
    if (prepared) {
      T.env.__stegoOpfs = { url: modelUrl, file: prepared.file, modified: prepared.modified };
      cacheState = `opfs-${prepared.state}`;
    } else {
      cacheState = 'http';
    }
  }
  const tokenizer = await T.AutoTokenizer.from_pretrained(modelId, {
    progress_callback: progress,
    local_files_only: localOnly,
  });
  const model = await T.AutoModelForCausalLM.from_pretrained(modelId, {
    device,
    dtype,
    progress_callback: progress,
    local_files_only: localOnly,
  });
  const backend = new QwenBackend(T, tokenizer, model);
  backend.device = device;
  backend.dtype = dtype;
  backend.cacheState = cacheState;
  await backend.buildTokenMap((p) => onProgress && onProgress(0.99 + p * 0.01));
  return backend;
}
