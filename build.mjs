// Build the single-file UI and the local Qwen package.
//
//   node build.mjs
//
// Outputs:
//   dist/index.html                 complete package (vendor/ + models/ beside it)
//   dist/与AI论道-增强版.html
//   与AI论道-增强版.html            convenience copy; references ./dist/
//
// The Qwen model and runtime are local; no network access is needed at
// runtime.  The first build copies ~830 MB of model/ORT files into dist/.

import { build } from 'esbuild';
import fs from 'node:fs';
import path from 'node:path';

const root = process.cwd();
const distDir = path.join(root, 'dist');
fs.mkdirSync(distDir, { recursive: true });

async function bundle(entry, format, outfile = null) {
  const result = await build({
    entryPoints: [entry],
    bundle: true,
    format,
    platform: 'browser',
    target: ['es2022'],
    write: false,
    minify: true,
    legalComments: 'none',
    logLevel: 'warning',
    define: { 'process.env.NODE_ENV': '"production"' },
  });
  const text = result.outputFiles[0].text;
  if (outfile) fs.writeFileSync(outfile, text);
  return text;
}

function escapeInlineScript(code) {
  return code
    .replace(/<\/script/gi, '<\\/script')
    .replace(/<!--/g, '<\\!--');
}

function copyIfMissing(src, dst) {
  if (fs.existsSync(dst)) return false;
  fs.mkdirSync(path.dirname(dst), { recursive: true });
  fs.copyFileSync(src, dst);
  return true;
}

// fs.cpSync can terminate Node silently on some Windows/OneDrive setups when
// copying a 750 MB file.  A per-file copy is slower but reliable.
function copyTree(src, dst) {
  const stat = fs.statSync(src);
  if (stat.isDirectory()) {
    fs.mkdirSync(dst, { recursive: true });
    for (const name of fs.readdirSync(src)) {
      copyTree(path.join(src, name), path.join(dst, name));
    }
  } else {
    fs.mkdirSync(path.dirname(dst), { recursive: true });
    fs.copyFileSync(src, dst);
  }
}

console.log('bundling local Transformers.js runtime…');
const vendorDir = path.join(distDir, 'vendor');
fs.mkdirSync(vendorDir, { recursive: true });
await bundle('vendor-src.js', 'esm', path.join(vendorDir, 'transformers.js'));

console.log('copying ONNX Runtime WebAssembly files…');
const ortSrc = path.join(root, 'node_modules', 'onnxruntime-web', 'dist');
const ortDst = path.join(vendorDir, 'ort');
fs.mkdirSync(ortDst, { recursive: true });
for (const name of fs.readdirSync(ortSrc)) {
  if (!name.startsWith('ort-wasm')) continue;
  copyIfMissing(path.join(ortSrc, name), path.join(ortDst, name));
}

console.log('checking local Qwen model…');
// Netlify builds cannot carry the 750 MB ONNX file: the repo does not contain
// it and Netlify's static-file limit is far below 786 MB.  In that environment
// the UI loads the model from Hugging Face and caches it in the browser OPFS.
const remoteModel = process.env.STEGO_REMOTE_MODEL === '1' || !!process.env.NETLIFY;
const modelSrc = path.join(
  root, 'node_modules', '@huggingface', 'transformers', '.cache',
  'onnx-community', 'Qwen2.5-0.5B',
);
const modelDst = path.join(distDir, 'models', 'Qwen2.5-0.5B');
const modelFile = path.join(modelDst, 'onnx', 'model_q4.onnx');
if (remoteModel) {
  console.log('remote model mode: skipping local Qwen model copy');
  // Never publish a stale 750 MB model directory from an earlier local build.
  fs.rmSync(path.join(distDir, 'models'), { recursive: true, force: true });
} else if (!fs.existsSync(modelFile)) {
  if (!fs.existsSync(modelSrc)) {
    throw new Error('找不到本地 Qwen 模型缓存；请先安装并运行一次 Node 端 Qwen 测试，或手动放置 models/Qwen2.5-0.5B。');
  }
  console.log('copying Qwen model (~750 MB, one-time)…');
  copyTree(modelSrc, modelDst);
}

console.log('bundling worker…');
const workerCode = await bundle('src/worker.js', 'esm');
console.log('bundling app…');
const appCode = await bundle('src/app.js', 'iife');

const template = fs.readFileSync('src/index.template.html', 'utf8');

function render(modelBase, remote) {
  return template
    .replace('__WORKER_CODE__', escapeInlineScript(workerCode))
    .replace('__APP_CODE__', escapeInlineScript(appCode))
    .replace('__MODEL_BASE__', modelBase)
    .replace('__REMOTE_MODEL__', remote ? 'true' : 'false');
}

const distHtml = render('./', remoteModel);
const rootHtml = render('./dist/', remoteModel);

const outDist1 = path.join(distDir, '大模型隐语.html');
const outDist2 = path.join(distDir, 'index.html');
const outRoot = path.join(root, '大模型隐语.html');
fs.writeFileSync(outDist1, distHtml);
fs.writeFileSync(outDist2, distHtml);
fs.writeFileSync(outRoot, rootHtml);

// Remove the previous name so the package does not contain two stale apps.
for (const old of [
  path.join(root, '与AI论道-增强版.html'),
  path.join(distDir, '与AI论道-增强版.html'),
]) {
  try { fs.unlinkSync(old); } catch {}
}

console.log('worker KB', (workerCode.length / 1024).toFixed(0));
console.log('app KB   ', (appCode.length / 1024).toFixed(0));
console.log('html MB  ', (distHtml.length / 1e6).toFixed(2));
console.log('wrote', outDist1);
console.log('wrote', outDist2);
console.log('wrote', outRoot);
console.log('local model:', fs.existsSync(modelFile) ? 'present' : 'MISSING');
console.log('model mode:', remoteModel ? 'remote (Hugging Face + OPFS cache)' : 'local (dist/models)');
