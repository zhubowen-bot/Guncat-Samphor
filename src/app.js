const $ = (id) => document.getElementById(id);

const secret = $('secret');
const stego = $('stego');
const stegoInput = $('stegoInput');
const secretOutput = $('secretOutput');
const encodeBtn = $('encodeBtn');
const decodeBtn = $('decodeBtn');
const encodeProgress = $('encodeProgress');
const decodeProgress = $('decodeProgress');
const encodeStats = $('encodeStats');
const decodeStats = $('decodeStats');
const engineBadge = $('engineBadge');

// Directory that contains vendor/ and models/.  The build script injects
// "./" for dist/index.html and "./dist/" for the convenience copy at the
// project root.
const MODEL_BASE = new URL(window.STEGO_MODEL_BASE || './', location.href).href;
const REMOTE_MODEL = !!window.STEGO_REMOTE_MODEL;
const toastEl = $('toast');

// Fixed user-facing configuration: no model/profile/backend selectors.
const FIXED_PROFILE = 1;       // balanced
const FIXED_BACKEND = 'auto';  // WebGPU first, WASM fallback
const FIXED_COMPRESSION = false;

const presets = [
  '清晨的阳光透过窗帘照进房间。',
  '生活中有很多值得记录的小事。',
  '多年以后回想起来，',
  '天气很好，', '谢邀，', '刚下飞机，人在火星，', '家人们谁懂啊，',
  '专家提醒：', '最新数据显示，', '话说天下大势，',
  '作为一个AI助手，', '别问了，别问了，', '这是可以说的吗？',
  '生活就像一盒巧克力，', '凌晨三点，睡不着。', '据本台最新消息，',
  '起猛了，看见', '虽然但是，', '不是哥们，', '温馨提示：',
];

let worker = null;
let seq = 0;
let pending = new Map();
let busy = false;

/* ------------------------------------------------------------------ */
/* Worker plumbing                                                    */
/* ------------------------------------------------------------------ */

function onWorkerMessage(event) {
  const msg = event.data || {};
  if (msg.type === 'boot') return;
  const entry = pending.get(msg.id);
  if (!entry) return;
  if (msg.type === 'progress') {
    entry.onProgress && entry.onProgress(msg);
    return;
  }
  if (msg.type === 'stream') {
    entry.onStream && entry.onStream(msg);
    return;
  }
  if (msg.type === 'ready') {
    pending.delete(msg.id);
    entry.resolve(msg);
    return;
  }
  if (msg.type === 'result') {
    pending.delete(msg.id);
    entry.resolve(msg.result);
    return;
  }
  if (msg.type === 'error') {
    pending.delete(msg.id);
    const err = new Error(msg.error || '未知错误');
    err.stack = msg.stack;
    entry.reject(err);
  }
}

function createWorker() {
  const code = $('worker').textContent;
  const blob = new Blob([code], { type: 'text/javascript' });
  const url = URL.createObjectURL(blob);
  const w = new Worker(url, { type: 'module' });
  URL.revokeObjectURL(url);
  w.addEventListener('message', onWorkerMessage);
  w.addEventListener('error', (e) => {
    toast(`工作线程错误：${e.message || '无法启动'}`, true);
    for (const p of pending.values()) p.reject(new Error('工作线程错误'));
    pending.clear();
    restartWorker();
  });
  return w;
}

function restartWorker() {
  try { worker && worker.terminate(); } catch {}
  worker = createWorker();
}

function call(type, payload = {}, onProgress = null, onStream = null) {
  const id = ++seq;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject, onProgress, onStream });
    worker.postMessage({ id, type, ...payload });
  });
}

/* ------------------------------------------------------------------ */
/* UI helpers                                                         */
/* ------------------------------------------------------------------ */

let toastTimer = null;
function toast(message, isError = false) {
  toastEl.textContent = message;
  toastEl.className = 'toast show' + (isError ? ' err' : '');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { toastEl.className = 'toast'; }, isError ? 6000 : 2600);
}

function showProgress(el, stage, progress, detail = '') {
  el.classList.remove('hidden');
  const p = Math.max(0, Math.min(1, progress || 0));
  el.querySelector('.bar').style.width = `${(p * 100).toFixed(1)}%`;
  el.querySelector('.stage').textContent = stage + (detail ? ` · ${detail}` : '');
  el.querySelector('.pct').textContent = `${Math.round(p * 100)}%`;
}

function hideProgress(el) { el.classList.add('hidden'); }

function setBusy(value) {
  busy = value;
  encodeBtn.disabled = value;
  decodeBtn.disabled = value;
  encodeBtn.textContent = value ? '处理中…' : '生成隐语文本';
  decodeBtn.textContent = value ? '处理中…' : '提取秘密';
}

function updateSecretCount() {
  const text = secret.value;
  const chars = Array.from(text).length;
  const bytes = new TextEncoder().encode(text).length;
  $('secretCount').textContent = `${chars} 字`;
  $('secretBytes').textContent = `${bytes} B`;
}

function updateStegoCount() {
  const text = stego.value;
  const chars = Array.from(text).length;
  $('stegoMeta').textContent = `${chars} 字`;
}

function updateStegoInputCount() {
  const chars = Array.from(stegoInput.value).length;
  $('stegoInputMeta').textContent = `${chars} 字`;
}

function renderStats(el, stats) {
  const rows = [];
  const num = (v, digits = 2) => (typeof v === 'number' && Number.isFinite(v) ? v.toFixed(digits) : '—');
  rows.push(['模型', 'Qwen2.5 0.5B']);
  if (stats.device) rows.push(['运行后端', `${stats.device} / ${stats.dtype || 'q4'}`]);
  rows.push(['质量档位', stats.profileName || '—']);
  if (stats.bodyChars != null) rows.push(['隐语正文', `${stats.bodyChars} 字`]);
  if (stats.bodyTokens != null) rows.push(['正文 token', `${stats.bodyTokens} 个`]);
  if (stats.decodedTokens != null) rows.push(['解码 token', `${stats.decodedTokens} 个`]);
  if (stats.preambleEnd != null) rows.push(['开头边界', `${stats.preambleEnd} 字`]);
  if (stats.attempts != null && stats.attempts > 1) rows.push(['质量择优', `${stats.attempts} 个候选`]);
  if (stats.plainBytes != null) rows.push(['明文字节', `${stats.plainBytes} B`]);
  if (stats.storedBytes != null) rows.push(['载荷字节', `${stats.storedBytes} B${stats.compressed ? ' · 已压缩' : ''}`]);
  if (stats.payloadBits != null) rows.push(['载荷比特', `${stats.payloadBits} bit`]);
  if (stats.bitsPerChar != null) rows.push(['比特 / 字', num(stats.bitsPerChar)]);
  if (stats.bitsPerToken != null) rows.push(['比特 / token', num(stats.bitsPerToken)]);
  if (stats.avgEntropy != null) rows.push(['平均熵', `${num(stats.avgEntropy)} bit`]);
  if (stats.utilization != null) rows.push(['熵利用率', `${num(Math.min(1, stats.utilization) * 100, 1)}%`]);
  if (stats.encodeMs != null) rows.push(['编码耗时', `${(stats.encodeMs / 1000).toFixed(2)} s`]);
  if (stats.decodeMs != null) rows.push(['解码耗时', `${(stats.decodeMs / 1000).toFixed(2)} s`]);
  el.textContent = '';
  for (const [k, v] of rows) {
    const box = document.createElement('div');
    box.className = 'stat';
    const kk = document.createElement('div');
    kk.className = 'k';
    kk.textContent = k;
    const vv = document.createElement('div');
    vv.className = 'v';
    vv.textContent = v;
    box.append(kk, vv);
    el.append(box);
  }
  el.classList.remove('hidden');
}

let lastPreamble = '';
function pickRandomPreamble() {
  let chosen = presets[Math.floor(Math.random() * presets.length)];
  if (presets.length > 1 && chosen === lastPreamble) {
    chosen = presets[(presets.indexOf(chosen) + 1) % presets.length];
  }
  lastPreamble = chosen;
  return chosen;
}

/* ------------------------------------------------------------------ */
/* Actions                                                            */
/* ------------------------------------------------------------------ */

let modelReadyNotified = false;

async function ensureModel(onProgress) {
  const ready = await call('init', {
    model: 'qwen',
    options: {
      baseUrl: MODEL_BASE,
      backend: FIXED_BACKEND,
      remote: REMOTE_MODEL,
    },
  }, onProgress);
  engineBadge.textContent = 'Qwen2.5 0.5B · 本地';
  if (!modelReadyNotified) {
    modelReadyNotified = true;
    if (ready && ready.cache === 'opfs-hit') {
      toast('模型已从浏览器缓存加载');
    } else if (ready && ready.cache === 'opfs-stored') {
      toast('模型已缓存到浏览器，下次刷新会快很多');
    }
  }
}

async function doEncode() {
  if (busy) return;
  const payload = secret.value.trim();
  if (!payload) {
    toast('请先输入秘密文本', true);
    secret.focus();
    return;
  }
  const chosenPreamble = pickRandomPreamble();
  setBusy(true);
  encodeStats.classList.add('hidden');
  stego.value = '';
  updateStegoCount();
  showProgress(encodeProgress, '准备模型', 0.01);
  let streamed = '';
  try {
    const onProgress = (m) => showProgress(encodeProgress, m.stage || '生成隐语', m.progress, m.detail);
    const onStream = (m) => {
      if (m.reset) streamed = m.text || '';
      else streamed += m.text || '';
      stego.value = streamed;
      updateStegoCount();
      stego.scrollTop = stego.scrollHeight;
    };
    await ensureModel(onProgress);
    const result = await call('encode', {
      model: 'qwen',
      payload,
      preamble: chosenPreamble,
      profileId: FIXED_PROFILE,
      compression: FIXED_COMPRESSION,
      options: {},
    }, onProgress, onStream);
    const replaced = streamed !== result.text;
    stego.value = result.text;
    updateStegoCount();
    renderStats(encodeStats, result.stats);
    toast(replaced ? '生成完成（已用更优候选更新）' : '生成完成');
  } catch (err) {
    stego.value = '';
    updateStegoCount();
    toast(err.message || String(err), true);
  } finally {
    setBusy(false);
    hideProgress(encodeProgress);
  }
}

async function doDecode() {
  if (busy) return;
  const text = stegoInput.value.trim();
  if (!text) {
    toast('请先粘贴隐语文本', true);
    stegoInput.focus();
    return;
  }
  setBusy(true);
  decodeStats.classList.add('hidden');
  secretOutput.value = '';
  showProgress(decodeProgress, '准备模型', 0.01);
  let streamed = '';
  try {
    const onProgress = (m) => showProgress(decodeProgress, m.stage || '提取信息', m.progress, m.detail);
    const onStream = (m) => {
      if (m.reset) streamed = m.text || '';
      else streamed += m.text || '';
      secretOutput.value = streamed;
      secretOutput.scrollTop = secretOutput.scrollHeight;
    };
    await ensureModel(onProgress);
    const result = await call('decode', {
      model: 'qwen',
      stegoText: text,
      profileId: FIXED_PROFILE,
      preambleEnd: null,
      options: {},
    }, onProgress, onStream);
    secretOutput.value = result.text;
    secretOutput.scrollTop = secretOutput.scrollHeight;
    renderStats(decodeStats, result.stats);
    toast('提取成功');
  } catch (err) {
    secretOutput.value = '';
    toast(err.message || String(err), true);
  } finally {
    setBusy(false);
    hideProgress(decodeProgress);
  }
}

async function copyText(text, label) {
  if (!text) { toast('没有可复制的内容', true); return; }
  try {
    await navigator.clipboard.writeText(text);
    toast(`${label}已复制`);
  } catch {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.append(ta);
    ta.select();
    try { document.execCommand('copy'); toast(`${label}已复制`); }
    catch { toast('复制失败，请手动选择文本', true); }
    ta.remove();
  }
}

function downloadText(text, filename) {
  if (!text) { toast('没有可下载的内容', true); return; }
  const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1500);
  toast('已开始下载');
}

async function clearModelCache() {
  let removed = false;
  try {
    const root = await navigator.storage.getDirectory();
    await root.removeEntry('stego-qwen', { recursive: true });
    removed = true;
  } catch { /* not present */ }
  try {
    const names = await caches.keys();
    for (const name of names) {
      if (name.startsWith('stego-qwen')) {
        await caches.delete(name);
        removed = true;
      }
    }
  } catch { /* cache storage unavailable */ }
  toast(removed ? '模型缓存已清除，下次使用会重新加载本地模型' : '没有找到可清除的模型缓存');
}

async function runSelfTest() {
  const btn = $('selfTestBtn');
  const out = $('selfTestResult');
  btn.disabled = true;
  out.className = 'selftest-result';
  out.textContent = '运行中…';
  const payload = '这是一次自检：隐语文本应当可以无损还原。';
  const testPreamble = pickRandomPreamble();
  try {
    await ensureModel(null);
    const t0 = performance.now();
    const enc = await call('encode', {
      model: 'qwen', payload, preamble: testPreamble, profileId: FIXED_PROFILE, compression: FIXED_COMPRESSION, options: {},
    });
    const t1 = performance.now();
    const dec = await call('decode', {
      model: 'qwen',
      stegoText: enc.text,
      profileId: FIXED_PROFILE,
      preambleEnd: null,
      options: {},
    });
    const t2 = performance.now();
    const ok = dec.text === payload;
    out.className = 'selftest-result ' + (ok ? 'ok' : 'bad');
    out.textContent = ok
      ? `通过 ✓ 隐语 ${enc.stats.bodyChars} 字 · 编码 ${((t1 - t0) / 1000).toFixed(2)} s · 解码 ${((t2 - t1) / 1000).toFixed(2)} s`
      : `失败：还原结果不一致（${dec.text}）`;
  } catch (err) {
    out.className = 'selftest-result bad';
    out.textContent = `失败：${err.message}`;
  } finally {
    btn.disabled = false;
  }
}

/* ------------------------------------------------------------------ */
/* Wiring                                                             */
/* ------------------------------------------------------------------ */

function setupTabs() {
  const buttons = Array.from(document.querySelectorAll('nav.tabs button'));
  for (const btn of buttons) {
    btn.addEventListener('click', () => {
      buttons.forEach((b) => b.classList.toggle('active', b === btn));
      document.querySelectorAll('.panel').forEach((p) => p.classList.toggle('active', p.id === `panel-${btn.dataset.tab}`));
    });
  }
}

function init() {
  // Ask the browser to keep the (large) Qwen model files in cache.  This is a
  // hint; private windows or strict storage policies may ignore it.
  if (navigator.storage && navigator.storage.persist) {
    navigator.storage.persist().catch(() => {});
  }
  const sourceNote = $('modelSource');
  if (sourceNote) {
    sourceNote.innerHTML = REMOTE_MODEL
      ? '当前为在线部署：模型从 Hugging Face 下载，<strong>首次约 750 MB</strong>，之后缓存到浏览器 OPFS，不再重复下载。'
      : '当前为本地部署：模型与运行时都在 <code>dist/</code> 内，运行时<strong>不访问外网</strong>，也不会现场下载。';
  }
  updateSecretCount();
  updateStegoCount();
  updateStegoInputCount();

  secret.addEventListener('input', updateSecretCount);
  stegoInput.addEventListener('input', updateStegoInputCount);
  encodeBtn.addEventListener('click', doEncode);
  decodeBtn.addEventListener('click', doDecode);
  $('copyStego').addEventListener('click', () => copyText(stego.value, '隐语文本'));
  $('downloadStego').addEventListener('click', () => downloadText(stego.value, '隐语文本.txt'));
  $('copySecret').addEventListener('click', () => copyText(secretOutput.value, '秘密文本'));
  $('selfTestBtn').addEventListener('click', runSelfTest);
  $('clearModelCache').addEventListener('click', clearModelCache);

  document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      const active = document.querySelector('nav.tabs button.active')?.dataset.tab;
      if (active === 'encode') doEncode();
      else if (active === 'decode') doDecode();
    }
  });

  setupTabs();

  try {
    worker = createWorker();
    engineBadge.textContent = 'Qwen2.5 0.5B · 本地';
  } catch (err) {
    engineBadge.textContent = '浏览器不支持 Worker';
    toast(`无法创建 Worker：${err.message}`, true);
    encodeBtn.disabled = true;
    decodeBtn.disabled = true;
  }
}

init();
