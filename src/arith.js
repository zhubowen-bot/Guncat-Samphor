// Bit-level arithmetic coding (32-bit interval, 53-bit-safe integer math).
// The encoder is the inverse of the decoder; both are deterministic across
// JavaScript engines because every state update is integer arithmetic.

const HALF = 0x80000000;
const QUARTER = 0x40000000;
const THREE_QUARTER = 0xC0000000;

export class BitWriter {
  constructor() {
    this.bytes = [];
    this.cur = 0;
    this.nbits = 0;
  }
  writeBit(bit) {
    this.cur = (this.cur << 1) | (bit & 1);
    this.nbits++;
    if (this.nbits === 8) {
      this.bytes.push(this.cur);
      this.cur = 0;
      this.nbits = 0;
    }
  }
  writeBits(value, n) {
    for (let i = n - 1; i >= 0; i--) this.writeBit((value >>> i) & 1);
  }
  finish() {
    if (this.nbits > 0) {
      this.bytes.push(this.cur << (8 - this.nbits));
      this.cur = 0;
      this.nbits = 0;
    }
    return Uint8Array.from(this.bytes);
  }
}

export class BitReader {
  constructor(bytes) {
    this.bytes = bytes;
    this.pos = 0;
    this.cur = 0;
    this.nbits = 0;
    this.count = 0;
  }
  readBit() {
    if (this.nbits === 0) {
      this.cur = this.pos < this.bytes.length ? this.bytes[this.pos++] : 0;
      this.nbits = 8;
    }
    const bit = (this.cur >> 7) & 1;
    this.cur = (this.cur << 1) & 0xFF;
    this.nbits--;
    this.count++;
    return bit;
  }
  readBits(n) {
    let v = 0;
    for (let i = 0; i < n; i++) v = (v << 1) | this.readBit();
    return v >>> 0;
  }
}

export class ArithmeticDecoder {
  /** Decode symbols from a bitstream (the sender's side). */
  constructor(reader) {
    this.reader = reader;
    this.low = 0;
    this.high = 0xFFFFFFFF;
    this.code = reader.readBits(32);
    this.bitsRead = 32;
  }

  /**
   * @param {number[]|Uint32Array} cdf cumulative frequencies, length n+1,
   *        cdf[0]=0, cdf[n]=total.
   * @param {number} total
   * @returns {number} symbol index
   */
  decode(cdf, total) {
    const range = this.high - this.low + 1;
    let value = Math.floor(((this.code - this.low + 1) * total - 1) / range);
    if (value < 0) value = 0;
    if (value >= total) value = total - 1;

    // largest i with cdf[i] <= value
    let lo = 0;
    let hi = cdf.length - 1;
    while (lo + 1 < hi) {
      const mid = (lo + hi) >>> 1;
      if (cdf[mid] <= value) lo = mid;
      else hi = mid;
    }
    const i = lo;
    const cumLow = cdf[i];
    const cumHigh = cdf[i + 1];

    this.high = this.low + Math.floor(range * cumHigh / total) - 1;
    this.low = this.low + Math.floor(range * cumLow / total);
    if (this.high < this.low) this.high = this.low;

    this.normalize();
    return i;
  }

  normalize() {
    while (true) {
      if (this.high < HALF) {
        // no offset
      } else if (this.low >= HALF) {
        this.code = (this.code - HALF) >>> 0;
        this.low = (this.low - HALF) >>> 0;
        this.high = (this.high - HALF) >>> 0;
      } else if (this.low >= QUARTER && this.high < THREE_QUARTER) {
        this.code = (this.code - QUARTER) >>> 0;
        this.low = (this.low - QUARTER) >>> 0;
        this.high = (this.high - QUARTER) >>> 0;
      } else {
        break;
      }
      this.low = (this.low << 1) >>> 0;
      this.high = ((this.high << 1) | 1) >>> 0;
      this.code = ((this.code << 1) | this.reader.readBit()) >>> 0;
      this.bitsRead++;
    }
  }
}

export class ArithmeticEncoder {
  /** Encode known symbols into a bitstream (the receiver's side). */
  constructor() {
    this.low = 0;
    this.high = 0xFFFFFFFF;
    this.pending = 0;
    this.bits = [];
  }

  encode(cumLow, freq, total) {
    const range = this.high - this.low + 1;
    this.high = this.low + Math.floor(range * (cumLow + freq) / total) - 1;
    this.low = this.low + Math.floor(range * cumLow / total);
    if (this.high < this.low) this.high = this.low;
    this.normalize();
  }

  outputBit(bit) {
    this.bits.push(bit & 1);
    while (this.pending > 0) {
      this.bits.push((bit & 1) ^ 1);
      this.pending--;
    }
  }

  normalize() {
    while (true) {
      if (this.high < HALF) {
        this.outputBit(0);
      } else if (this.low >= HALF) {
        this.outputBit(1);
        this.low = (this.low - HALF) >>> 0;
        this.high = (this.high - HALF) >>> 0;
      } else if (this.low >= QUARTER && this.high < THREE_QUARTER) {
        this.pending++;
        this.low = (this.low - QUARTER) >>> 0;
        this.high = (this.high - QUARTER) >>> 0;
      } else {
        break;
      }
      this.low = (this.low << 1) >>> 0;
      this.high = ((this.high << 1) | 1) >>> 0;
    }
  }

  /** Return bits emitted so far and clear the buffer (for streaming). */
  drainBits() {
    const out = this.bits;
    this.bits = [];
    return out;
  }

  finish() {
    this.pending++;
    this.outputBit(this.low < QUARTER ? 0 : 1);
    return this.bits;
  }
}

/** Pack a 0/1 bit array into bytes (MSB first). */
export function bitsToBytes(bits) {
  const w = new BitWriter();
  for (const b of bits) w.writeBit(b);
  return w.finish();
}

/** Unpack bytes into a 0/1 bit array. */
export function bytesToBits(bytes) {
  const out = new Uint8Array(bytes.length * 8);
  for (let i = 0; i < bytes.length; i++) {
    const b = bytes[i];
    for (let j = 0; j < 8; j++) out[i * 8 + j] = (b >> (7 - j)) & 1;
  }
  return out;
}
