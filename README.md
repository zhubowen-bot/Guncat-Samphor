# 大模型隐语（Qwen2.5 本地集成）

基于语言模型采样过程的中文文本隐写工具。当前版本**只使用 Qwen2.5-0.5B**，模型与运行时全部放在本地；生成和提取都是逐字流式显示。

界面固定使用 **Qwen2.5-0.5B + 平衡档位**，不提供模型、档位、后端、压缩等选项；公开开头每次随机生成，自动附加在隐语文本开头。接收端不需要填写开头，解码器会自动识别边界。

- 本地 UI：`dist/index.html` / `dist/大模型隐语.html`
- Qwen 模型：`dist/models/Qwen2.5-0.5B/`（q4 ONNX，约 750 MB）
- 运行时：`dist/vendor/`（Transformers.js + ONNX Runtime，约 83 MB）
- 页面本身只有约 60 KB；模型不再打进 HTML

> 原版 2M 模型仅保留在源码/测试中作历史对照，**界面不再提供 2M**，也不会自动切换模型。

## 快速开始

```bash
# 依赖（构建/测试用）
npm install

# 构建（复制本地模型、打包 vendor/UI）
node build.mjs

# 通过本地 http 服务打开（file:// 无法 fetch 本地模型）
cd dist
python -m http.server 8123
```

浏览器打开 `http://127.0.0.1:8123/`。

## Netlify 部署

Netlify 不能托管 750 MB 的单文件模型（静态资源单文件上限远低于 786 MB），所以构建脚本在 Netlify 上会自动切换为**远程模型模式**：

- HTML、Transformers.js、ONNX Runtime 仍由 Netlify 托管；
- Qwen 权重从 Hugging Face（`onnx-community/Qwen2.5-0.5B`）下载，首次约 750 MB；
- 浏览器把权重缓存到 OPFS，之后刷新不再重复下载；
- 本地 `node build.mjs` 仍使用 `dist/models`，保持完全离线。

Netlify 构建会自动识别 `NETLIFY=true`，也可以显式设置 `STEGO_REMOTE_MODEL=1`。仓库里的 `netlify.toml` 已配置好：

```toml
[build]
  command = "npm run build"
  publish = "dist"

[build.environment]
  STEGO_REMOTE_MODEL = "1"
```

首次访问需要能访问 `huggingface.co`；网络不通时页面会提示模型加载失败。`.gitignore` 已排除 `dist/models`，避免把 750 MB 权重提交进仓库。

如果 Netlify 仍然报“找不到本地 Qwen 模型缓存”，确认部署的是最新 `build.mjs`；它会在 `NETLIFY=true` 时自动跳过模型。也可以在 Netlify 的环境变量里显式添加 `STEGO_REMOTE_MODEL=1`。

## 运行要求

- **推荐 WebGPU**：Chrome / Edge，开启硬件加速。WebGPU 下 Qwen 编码/解码通常几秒完成。
- **WASM 兜底**：无 WebGPU 或 WebGPU 加载失败时，自动使用本地 WASM q4。可以运行，但明显更慢（示例 89 字隐语解码约 51 s）。
- 不支持 `file://` 读取本地模型；必须用本地 http 服务。
- Qwen 需要 WebGPU 或 WASM 与足够内存（模型约 750 MB）。

## 模型加载：能避免什么，不能避免什么

**不能完全避免。** 纯静态页面刷新会销毁 Web Worker、ONNX Runtime session 和 WebGPU 设备，浏览器不允许把这些状态序列化后恢复。所以每次刷新后都必须重新创建一次 ONNX session（本机约 7 秒）。

**可以避免重新读取 750 MB。** 首次使用时模型会被缓存到浏览器 OPFS（`stego-qwen/`）；之后刷新只从 OPFS 读回权重（本机约 0.8 秒），不再走 http server，也不会重复下载。较小的 tokenizer/config 文件走 Cache API；ONNX 权重单独放 OPFS，因为 Chromium 对单条 Cache API 条目有约 512 MB 的限制。

实测（同一台机器，87 字隐语）：

| 场景 | 耗时 |
| --- | --- |
| 首次加载（下载并写入 OPFS） | 约 52 s（其中下载 + 写入约 39 s） |
| 刷新后加载 | 约 8 s（OPFS 读取约 0.8 s + ONNX/WebGPU 初始化约 7 s） |
| 旧版每次刷新 | 约 50 s |
| 同页面后续编码/解码 | 不再加载模型 |

如果确实需要 0 加载，只能让模型常驻在一个不随页面刷新的进程/标签页里（例如本地推理服务或常开 host 标签页）。本项目目前没有默认启用这种模式。

界面里的 **“清除模型缓存”** 会删除 OPFS 权重和本应用的 Cache API 条目；浏览器“清除站点数据”也会达到同样效果。模型文件被替换时，页面会用 `HEAD` 检查大小/修改时间并自动重新缓存。

## 为什么 WebGPU 和 WASM 现在可以互相解码

两种后端使用不同的 ONNX Runtime 执行内核，同一份 q4 权重算出的原始 logits 会有微小浮点差异。算术编码对分布非常敏感：如果直接把 logits softmax 后量化到 1/65536，再按概率排序候选，几十个 token 后区间就会失步。

本版本做了三层后端无关化：

1. **粗粒度 logits 量化**：`round(logit / 0.5) * 0.5`，把微小数值差异吸收到同一格；
2. **固定 CDF 顺序**：候选按 **token id** 排序，而不是按概率排序，避免近邻候选交换顺序；
3. **粗粒度频率表**：Qwen 使用 4096 的频率总表，降低边界舍入敏感度。

实测：

- WebGPU 编码的文本 → **WASM 解码成功**（`跨后端`，89 字隐语，51 s，CRC 通过）；
- 同一文本 → WebGPU 解码 2.15 s，CRC 通过。

因此 WASM 兜底被保留，不再需要删掉它；同一条消息仍建议尽量在同一后端编解码，但跨后端已经可用。

## 流式生成与流式提取

- **生成**：每生成一个 token/字符，立即追加到“隐语文本”框；公开开头先出现，正文持续追加。
- **提取**：算术编码器每产出一段比特，立即解析头部、解白化、解压缩（如有）并逐字追加到“恢复的秘密”框；CRC32 最后校验，失败会清空并提示。
- **载荷完成即停止**：秘密字节全部恢复后，解码器立即停止，不再继续解码编码器补在末尾的装饰性 token。这些 token 不承载信息，跨后端时反而可能先失配；现在统计里的“解码 token”可能小于“正文 token”，这是正常的。
- **先校验头部再输出**：解码器解析出头部后会立即核对模型与档位标识；错误档位会在头部处放弃，不会先流出一大段内容再清空重来。
- 压缩默认关闭（界面不提供开关）。压缩会提高容量，但明文需要等到大部分压缩数据恢复后才能解出，流式显示会明显延后。
- 解码端不需要填写公开开头；随机开头由解码器通过标点边界自动识别。程序会用头部里的模型/档位标识校验候选，错误档位在头部处直接放弃。

## 算法概要

1. 明文 → UTF-8（可选 deflate）→ CRC32 → xorshift32 白化。
2. 88 位头部：magic、版本、模型、档位、flags、nonce、长度、CRC32。
3. 算术编码：把秘密比特当作区间码流，按模型条件分布选择 token；接收端对 token 做算术编码恢复比特。Qwen 频率总表 4096。
4. Qwen 分词歧义处理：只接受保持正文“规范分词”的 token：

   ```text
   tokenizer.encode(bodyText + candidateStr) === bodyIds + [candidateId]
   ```

   接收端对可见正文重新分词即可恢复相同 token 序列，不需要零宽字符。
5. 质量档位：温度、重复惩罚、无重复 3-gram、自然字符过滤。界面固定“平衡”。

## 质量档位（界面固定平衡）

界面固定使用“平衡”档位（温度 0.90、top-p 0.95），不向用户暴露选择。源码中仍保留其他档位用于实验/测试：

| 档位 | 温度 | 特点 |
| --- | --- | --- |
| 自然 | 0.80 | 最通顺，隐语更长 |
| 平衡（界面固定） | 0.90 | 质量与容量折中 |
| 容量 | 1.00 | 熵更高，文本更粗糙 |

## 实测

| 项目 | 结果 |
| --- | --- |
| UI HTML | 约 60 KB |
| Qwen 模型 + 运行时 | 约 833 MB（本地） |
| 首次模型加载（缓存到 OPFS） | 约 52 s |
| 刷新后模型加载（OPFS 命中） | 约 8 s |
| WebGPU 编码（149 字隐语/36 B） | 约 4.2 s |
| WebGPU 解码（指定开头/档位） | 约 2.2–6 s |
| WASM 解码（89 字隐语，WebGPU 编码） | 约 51–68 s，CRC 通过 |
| 流式提取观测 | 解码进度 40% 时已显示“明天” |
| 流式生成观测 | 进度 16% 时已输出 55 字 |

## 源码结构

```text
src/
  index.template.html   # UI 模板
  app.js                # UI、worker 通信、流式渲染
  worker.js             # Qwen worker（WebGPU / WASM）
  qwen.js               # Qwen 后端、规范分词、量化分布、编解码
  stego.js              # 档位、分布、CRC、压缩、白化、流式载荷解码
  arith.js              # 算术编码/解码
  model2m.js            # 原版 2M（仅测试/历史，不打包进 UI）
  stego2m.js            # 原版 2M 编解码（仅测试/历史）
  weights.js            # 原版 2M 量化权重（不打包进 UI）
vendor-src.js           # 本地 Transformers.js 打包入口
build.mjs               # esbuild + 复制模型/ORT
tests/                  # 算术编码、2M 历史测试、Qwen 测试
tools/extract_model.py  # 从原 HTML 提取 2M 权重
dist/
  index.html
  大模型隐语.html
  vendor/transformers.js
  vendor/ort/
  models/Qwen2.5-0.5B/
```

## 构建与测试

```bash
npm test        # 算术编码 + 2M 历史端到端（UI 不使用 2M）
node build.mjs  # 构建 UI 并检查/复制本地 Qwen 模型与 ORT
```

## 安全边界

- **隐写 ≠ 加密**：默认只做白化和 CRC32，不做加密/认证。高敏感内容请先加密。
- **后端一致性**：虽然已做跨后端量化兼容，仍建议编码/解码使用同一后端；不同 GPU/驱动仍有极小概率出现数值差异导致 CRC 失败。
- **档位公开**：界面固定“平衡”档位；源码中的“容量”档更接近原始分布，但文本更粗糙，需要重新生成文本。
- **CRC32** 只检测偶然损坏，不能抵抗恶意篡改。
- Qwen 本地模型为 Apache-2.0；模型文件只需放在 `dist/models`，运行时不会访问外网。

## 参考

- Cachin, 2004. *An information-theoretic model for steganography.*
- Hopper, Langford & von Ahn, 2002. *Provably secure steganography.*
- Kaptchuk et al., 2021. *Meteor.* CCS.
- Ding et al., 2023. *Discop.* IEEE S&P.
- Bai et al., 2025. *Shifting-Merging.* arXiv:2501.00786.
- Wang et al., 2025. *SparSamp.* arXiv:2503.19499.
- 2025. *Provable Secure Steganography Based on Adaptive Dynamic Sampling.* arXiv:2504.12579.
- 2026. *Efficient Provably Secure Linguistic Steganography via Range Coding.* ACL 2026.
- Qwen Team, 2024. *Qwen2.5 Technical Report.* arXiv:2412.15115.
