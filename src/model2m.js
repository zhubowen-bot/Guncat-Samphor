// Optimized inference for the embedded 2M char-level model.
// Weights are int8 per-row quantized; activations stay in float64 for
// deterministic, browser-independent decoding.

import { MODEL_CONFIG, MODEL_WEIGHTS } from './weights.js';
import { VOCAB } from './vocab.js';

export const MODEL2M_ID = 'builtin-2m-int8-v2';
export const MODEL2M_VOCAB = VOCAB;
export const MODEL2M_CONFIG = MODEL_CONFIG;

const V = VOCAB.length;
const HIDDEN = MODEL_CONFIG.hidden_size;
const HEADS = MODEL_CONFIG.num_heads;
const HEAD_DIM = HIDDEN / HEADS;
const LAYERS = MODEL_CONFIG.num_layers;
const INTER = MODEL_CONFIG.intermediate_size;
const MAX_SEQ = MODEL_CONFIG.max_seq_len;
const ROPE_MAX = 1024;
const ROPE_HALF = HEAD_DIM / 2;

const CHAR_TO_ID = new Map();
for (let i = 0; i < V; i++) CHAR_TO_ID.set(VOCAB[i], i);

export function charToToken(ch) {
  const id = CHAR_TO_ID.get(ch);
  if (id === undefined) throw new Error(`字符不在词表中: ${JSON.stringify(ch)}`);
  return id;
}

export function tokenToChar(id) {
  return VOCAB[id];
}

function b64ToBytes(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

/**
 * Dequantize the compact int8 representation into float32 rows.
 * Per-row symmetric scales keep the error around 0.02 bit/token.
 */
export function dequantizeWeights(raw = MODEL_WEIGHTS) {
  const out = {};
  for (const [name, w] of Object.entries(raw)) {
    const bytes = b64ToBytes(w.data);
    if (w.dtype === 'int8') {
      const [rows, cols] = w.size;
      const result = new Array(rows);
      for (let r = 0; r < rows; r++) {
        const scale = w.scales[r];
        const row = new Float32Array(cols);
        const base = r * cols;
        for (let c = 0; c < cols; c++) {
          // sign-extend int8
          row[c] = ((bytes[base + c] << 24) >> 24) * scale;
        }
        result[r] = row;
      }
      out[name] = result;
    } else {
      const f32 = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
      out[name] = f32.slice();
    }
  }
  return out;
}

export class Model2M {
  constructor(weights = dequantizeWeights()) {
    this.w = weights;
    this.config = MODEL_CONFIG;
    this.V = V;

    this.x = new Float64Array(HIDDEN);
    this.xNorm = new Float64Array(HIDDEN);
    this.qkv = new Float64Array(3 * HIDDEN);
    this.qHead = new Float64Array(HEAD_DIM);
    this.kHead = new Float64Array(HEAD_DIM);
    this.vHead = new Float64Array(HEAD_DIM);
    this.headOut = new Float64Array(HEAD_DIM);
    this.attnHeads = new Float64Array(HIDDEN);
    this.attnOut = new Float64Array(HIDDEN);
    this.mlpGate = new Float64Array(INTER);
    this.mlpUp = new Float64Array(INTER);
    this.mlpAct = new Float64Array(INTER);
    this.mlpOut = new Float64Array(HIDDEN);
    this.logits = new Float64Array(V);
    this.scores = new Float64Array(MAX_SEQ);
    this.probs = new Float64Array(MAX_SEQ);

    // Precompute RoPE cos/sin tables.  The original implementation computes
    // these inside every attention call; precomputation is a significant win.
    this.ropeCos = new Array(ROPE_MAX);
    this.ropeSin = new Array(ROPE_MAX);
    for (let p = 0; p < ROPE_MAX; p++) {
      const c = new Float64Array(ROPE_HALF);
      const s = new Float64Array(ROPE_HALF);
      for (let i = 0; i < ROPE_HALF; i++) {
        const theta = p / Math.pow(10000, (2 * i) / HEAD_DIM);
        c[i] = Math.cos(theta);
        s[i] = Math.sin(theta);
      }
      this.ropeCos[p] = c;
      this.ropeSin[p] = s;
    }

    this.reset();
  }

  reset() {
    this.pos = 0;
    this.kv = [];
    for (let l = 0; l < LAYERS; l++) this.kv.push({ k: [], v: [] });
  }

  get contextLength() {
    return this.kv[0].k.length;
  }

  rmsNorm(x, weight) {
    let sumSq = 0;
    for (let i = 0; i < x.length; i++) sumSq += x[i] * x[i];
    const inv = 1 / Math.sqrt(sumSq / x.length + this.config.norm_eps);
    const out = this.xNorm;
    for (let i = 0; i < x.length; i++) out[i] = weight[i] * x[i] * inv;
    return out;
  }

  matVec(mat, vec, out) {
    const n = mat.length;
    for (let r = 0; r < n; r++) {
      const row = mat[r];
      let s = 0;
      for (let i = 0; i < row.length; i++) s += row[i] * vec[i];
      out[r] = s;
    }
    return out;
  }

  applyRope(vec, pos) {
    const c = this.ropeCos[pos];
    const s = this.ropeSin[pos];
    for (let i = 0; i < ROPE_HALF; i++) {
      const x1 = vec[2 * i];
      const x2 = vec[2 * i + 1];
      vec[2 * i] = x1 * c[i] - x2 * s[i];
      vec[2 * i + 1] = x1 * s[i] + x2 * c[i];
    }
  }

  attention(layer, pos) {
    const w = this.w;
    const qkv = this.matVec(w[`layers.${layer}.attn.wqkv.weight`], this.xNorm, this.qkv);
    const cache = this.kv[layer];
    if (cache.k.length >= MAX_SEQ) {
      cache.k.shift();
      cache.v.shift();
    }
    const kVec = new Float64Array(HIDDEN);
    const vVec = new Float64Array(HIDDEN);
    cache.k.push(kVec);
    cache.v.push(vVec);

    const len = cache.k.length;
    const scale = 1 / Math.sqrt(HEAD_DIM);
    const attnHeads = this.attnHeads;

    for (let h = 0; h < HEADS; h++) {
      const qOff = h * HEAD_DIM;
      const kOff = (HEADS + h) * HEAD_DIM;
      const vOff = (2 * HEADS + h) * HEAD_DIM;

      for (let j = 0; j < HEAD_DIM; j++) {
        this.qHead[j] = qkv[qOff + j];
        this.kHead[j] = qkv[kOff + j];
        this.vHead[j] = qkv[vOff + j];
      }
      this.applyRope(this.qHead, pos);
      this.applyRope(this.kHead, pos);

      for (let j = 0; j < HEAD_DIM; j++) {
        kVec[qOff + j] = this.kHead[j];
        vVec[qOff + j] = this.vHead[j];
      }

      // scores against the cached keys
      let maxScore = -Infinity;
      for (let t = 0; t < len; t++) {
        const kt = cache.k[t];
        let s = 0;
        for (let j = 0; j < HEAD_DIM; j++) s += this.qHead[j] * kt[qOff + j];
        s *= scale;
        this.scores[t] = s;
        if (s > maxScore) maxScore = s;
      }
      let sum = 0;
      for (let t = 0; t < len; t++) {
        const e = Math.exp(this.scores[t] - maxScore);
        this.probs[t] = e;
        sum += e;
      }
      const invSum = 1 / sum;
      const out = this.headOut;
      out.fill(0);
      for (let t = 0; t < len; t++) {
        const p = this.probs[t] * invSum;
        const vt = cache.v[t];
        for (let j = 0; j < HEAD_DIM; j++) out[j] += p * vt[qOff + j];
      }
      for (let j = 0; j < HEAD_DIM; j++) attnHeads[qOff + j] = out[j];
    }

    this.matVec(w[`layers.${layer}.attn.wo.weight`], attnHeads, this.attnOut);
    for (let i = 0; i < HIDDEN; i++) this.x[i] += this.attnOut[i];
  }

  mlp(layer) {
    const w = this.w;
    this.matVec(w[`layers.${layer}.mlp.gate_proj.weight`], this.xNorm, this.mlpGate);
    this.matVec(w[`layers.${layer}.mlp.up_proj.weight`], this.xNorm, this.mlpUp);
    for (let i = 0; i < INTER; i++) {
      const g = this.mlpGate[i];
      this.mlpAct[i] = (g / (1 + Math.exp(-g))) * this.mlpUp[i];
    }
    this.matVec(w[`layers.${layer}.mlp.down_proj.weight`], this.mlpAct, this.mlpOut);
    for (let i = 0; i < HIDDEN; i++) this.x[i] += this.mlpOut[i];
  }

  /** Feed one token; return logits for the *next* token (reused buffer). */
  forward(tokenId) {
    if (tokenId < 0 || tokenId >= V) throw new Error(`invalid token id ${tokenId}`);
    const pos = this.pos++;
    if (pos >= ROPE_MAX) throw new Error('position exceeds RoPE table');

    const emb = this.w['embed.weight'][tokenId];
    for (let i = 0; i < HIDDEN; i++) this.x[i] = emb[i];

    for (let l = 0; l < LAYERS; l++) {
      this.rmsNorm(this.x, this.w[`layers.${l}.norm1.weight`]);
      this.attention(l, pos);
      this.rmsNorm(this.x, this.w[`layers.${l}.norm2.weight`]);
      this.mlp(l);
    }

    const finalNorm = this.rmsNorm(this.x, this.w['norm.weight']);
    const embed = this.w['embed.weight'];
    for (let r = 0; r < V; r++) {
      const row = embed[r];
      let s = 0;
      for (let i = 0; i < HIDDEN; i++) s += row[i] * finalNorm[i];
      this.logits[r] = s;
    }
    return this.logits;
  }
}
