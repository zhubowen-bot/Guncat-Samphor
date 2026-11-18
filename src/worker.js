// Web Worker: runs the local Qwen2.5-0.5B backend on WebGPU.

import { loadQwen } from './qwen.js';

let qwen = null;
let qwenLoading = null;

function post(id, type, payload = {}) {
  self.postMessage({ id, type, ...payload });
}

async function getQwen(id, options = {}) {
  if (qwen) return qwen;
  if (!qwenLoading) {
    qwenLoading = (async () => {
      const baseUrl = options.baseUrl || self.location.href;
      const remote = !!options.remote;
      const modelId = remote ? 'onnx-community/Qwen2.5-0.5B' : 'Qwen2.5-0.5B';
      if (String(baseUrl).startsWith('file:')) {
        throw new Error('请通过 http 服务打开本页（file:// 无法运行 Worker / 读取模型）。本地部署可在 dist 目录运行 python -m http.server 8123。');
      }
      const vendorUrl = new URL('vendor/transformers.js', baseUrl).href;
      post(id, 'progress', { stage: '加载本地运行时', progress: 0.01, detail: vendorUrl });
      let T;
      try {
        T = await import(/* webpackIgnore: true */ vendorUrl);
      } catch (err) {
        throw new Error(`本地 Transformers.js 加载失败：${err.message}。请确认 dist/vendor/transformers.js 存在。`);
      }
      const hasGpu = typeof navigator !== 'undefined' && !!navigator.gpu;
      const backend = options.backend || 'auto';
      let device;
      if (backend === 'wasm-q4') device = 'wasm';
      else if (backend === 'webgpu-q4') device = 'webgpu';
      else device = hasGpu ? 'webgpu' : 'wasm';
      if (device === 'webgpu' && !hasGpu) {
        throw new Error('当前环境没有 WebGPU，无法使用 WebGPU 后端；请选择 WASM 后端，或使用支持 WebGPU 的浏览器。');
      }

      const progress = (dev) => (p, info) => {
        const file = info && info.file ? info.file : '';
        post(id, 'progress', {
          stage: '加载本地模型',
          progress: Math.max(0.02, Math.min(0.99, p || 0.02)),
          detail: `${dev}/q4${file ? ' · ' + file : ''}`,
        });
      };
      const loadAndCheck = async (dev) => {
        const backendInstance = await loadQwen(T, {
          device: dev,
          dtype: 'q4',
          modelId,
          baseUrl,
          remote,
          onProgress: progress(dev),
          onCacheProgress: (p, info) => {
            const mb = (n) => `${(n / 1048576).toFixed(0)} MB`;
            const detail = info && info.total
              ? `${mb(info.loaded || 0)} / ${mb(info.total)}`
              : `${dev}/q4`;
            post(id, 'progress', {
              stage: '首次缓存模型',
              progress: Math.max(0.01, Math.min(0.98, p || 0.01)),
              detail,
            });
          },
        });
        post(id, 'progress', { stage: '本地模型自检', progress: 0.995, detail: `${dev}/q4` });
        const check = await backendInstance.selfCheck();
        if (!check.ok) {
          throw new Error(`后端输出异常（重复率 ${(check.repRate * 100).toFixed(0)}%）`);
        }
        return backendInstance;
      };
      const loadHint = remote
        ? '请检查网络能否访问 huggingface.co，以及浏览器是否允许 OPFS/缓存。'
        : '请确认 dist/models/Qwen2.5-0.5B 与 dist/vendor 完整，并通过 http 打开。';
      try {
        return await loadAndCheck(device);
      } catch (err) {
        const networkError = remote && /fetch|network|failed to fetch|cors/i.test(err.message || '');
        if (device !== 'webgpu' || networkError) {
          throw new Error(`Qwen 模型加载失败：${err.message}。${loadHint}`);
        }
        post(id, 'progress', { stage: '回退 WASM', progress: 0.02, detail: err.message || '' });
        try {
          return await loadAndCheck('wasm');
        } catch (err2) {
          throw new Error(`Qwen 模型加载失败：${err2.message}。${loadHint}`);
        }
      }
    })().catch((err) => {
      // Allow a later retry (e.g. after switching backend).
      qwenLoading = null;
      throw err;
    });
  }
  qwen = await qwenLoading;
  return qwen;
}

self.onmessage = async (event) => {
  const msg = event.data || {};
  const id = msg.id;
  try {
    if (msg.type === 'init') {
      const backend = await getQwen(id, msg.options);
      post(id, 'ready', { model: 'qwen', cache: backend.cacheState || 'unknown' });
      return;
    }

    if (msg.type === 'encode') {
      const onProgress = (p, extra = {}) => {
        post(id, 'progress', {
          stage: extra.stage || '生成隐语',
          progress: Math.max(0, Math.min(1, p)),
          detail: extra.detail || '',
        });
      };
      const onToken = (text, reset = false) => {
        post(id, 'stream', { text: text || '', reset: !!reset });
      };
      const backend = await getQwen(id, msg.options);
      const result = await backend.encode(msg.payload, {
        preamble: msg.preamble || '',
        profileId: msg.profileId ?? 1,
        compression: !!msg.compression,
        onProgress: (p) => onProgress(p, { stage: '生成隐语' }),
        onToken,
      });
      post(id, 'result', { result });
      return;
    }

    if (msg.type === 'decode') {
      const onProgress = (p) => post(id, 'progress', {
        stage: '提取信息',
        progress: Math.max(0, Math.min(1, p)),
      });
      const onText = (text, reset = false) => post(id, 'stream', { text: text || '', reset: !!reset });
      const onAttempt = (info) => post(id, 'progress', {
        stage: `尝试档位 · ${info.profileName}`,
        progress: 0.01,
        detail: `开头 ${info.end} 字 · ${info.attempt}/${info.total}`,
      });
      const backend = await getQwen(id, msg.options);
      const result = await backend.decode(msg.stegoText, {
        profileId: msg.profileId ?? null,
        preambleEnd: msg.preambleEnd ?? null,
        onProgress,
        onText,
        onAttempt,
      });
      post(id, 'result', { result });
      return;
    }

    throw new Error('未知请求');
  } catch (err) {
    post(id, 'error', {
      error: err && err.message ? err.message : String(err),
      stack: err && err.stack ? err.stack : '',
    });
  }
};

post(0, 'boot', {});
