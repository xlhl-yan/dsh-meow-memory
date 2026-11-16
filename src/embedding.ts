/**
 * v0.41.0 语义向量（混合召回）。
 *
 * 设计要点：
 *  - **懒加载**：第一次真正用到才 `import('@xenova/transformers')`。依赖缺失、模型文件不在、
 *    加载失败……一律 `failed=true` 并返回 null，调用方退回纯词面——绝不能因为向量把插件拖垮。
 *  - **离线优先**：allowRemoteModels=false，只读本地 models/ 目录，不在运行时悄悄下载。
 *  - **模型可换**：模型 id 变化会重置管线（同时应触发一次全量重算，见 migration）。
 *  - 向量按 L2 归一化输出，余弦相似度退化成点积。
 */
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

/** 默认模型：23MB 量化版，单条 32ms、常驻 ~150MB，实测性价比拐点。 */
export const DEFAULT_EMBEDDING_MODEL = 'Xenova/bge-small-zh-v1.5'

/** 模型目录：包根下的 models/（与 lib/ 同级），不进 git、不打包。 */
export function modelsDir(): string {
  // lib/index.js → ../models
  const here = dirname(fileURLToPath(import.meta.url))
  return resolve(here, '..', 'models')
}

type Extractor = (text: string, opts: { pooling: 'mean'; normalize: boolean }) => Promise<{ data: ArrayLike<number> }>

let modelId = DEFAULT_EMBEDDING_MODEL
let pipe: Extractor | null = null
let pending: Promise<Extractor | null> | null = null
let failed = false
let loadedDim = 0

export function embeddingModelId(): string {
  return modelId
}

/** 切换模型（配置变化时调用）：重置管线并清掉失败标记。 */
export function setEmbeddingModel(id: string): void {
  const next = id.trim() === '' ? DEFAULT_EMBEDDING_MODEL : id.trim()
  if (next === modelId) return
  modelId = next
  pipe = null
  pending = null
  failed = false
  loadedDim = 0
}

/** 是否已确认不可用（调用方据此跳过向量、只走词面，省掉每次的 getPipe 开销）。 */
export function embeddingUnavailable(): boolean {
  return failed
}

export function embeddingDim(): number {
  return loadedDim
}

/** 管线就绪（幂等，失败后不再重试）。 */
export function warmupEmbedding(): Promise<Extractor | null> {
  if (pipe !== null) return Promise.resolve(pipe)
  if (failed) return Promise.resolve(null)
  if (pending !== null) return pending
  pending = (async () => {
    try {
      // 动态 import：包不在（没装依赖）也不会让插件加载失败
      const mod = (await import('@xenova/transformers')) as {
        env: { allowRemoteModels: boolean; localModelPath: string }
        pipeline: (task: string, model: string, opts: Record<string, unknown>) => Promise<Extractor>
      }
      mod.env.allowRemoteModels = false
      mod.env.localModelPath = modelsDir()
      const p = await mod.pipeline('feature-extraction', modelId, { quantized: true })
      pipe = p
      return p
    } catch {
      failed = true
      return null
    } finally {
      pending = null
    }
  })()
  return pending
}

/**
 * 文本 → 归一化向量。失败返回 null（调用方降级纯词面）。
 * @param maxChars 截断长度：向量模型 max token 512，中文 ~500 字足够，超长截断控成本。
 */
export async function embed(text: string, maxChars = 500): Promise<Float32Array | null> {
  if (failed) return null
  const p = await warmupEmbedding()
  if (p === null) return null
  const input = text.replace(/\s+/g, ' ').trim().slice(0, maxChars)
  if (input.length === 0) return null
  try {
    const out = await p(input, { pooling: 'mean', normalize: true })
    const v = Float32Array.from(out.data)
    if (v.length === 0) return null
    loadedDim = v.length
    return v
  } catch {
    return null
  }
}

/** 批量算向量（迁移/补算用）：串行执行，避免一次吃满 CPU。 */
export async function embedMany(items: Array<{ id: string; text: string }>, onOne?: (done: number, total: number) => void): Promise<Array<{ id: string; vec: Float32Array }>> {
  const out: Array<{ id: string; vec: Float32Array }> = []
  for (let i = 0; i < items.length; i++) {
    const v = await embed(items[i].text)
    if (v !== null) out.push({ id: items[i].id, vec: v })
    onOne?.(i + 1, items.length)
  }
  return out
}

/** 余弦相似度；两边都归一化时等价于点积。 */
export function cosine(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) return 0
  let s = 0
  for (let i = 0; i < a.length; i++) s += a[i] * b[i]
  return s
}

/**
 * 一条记忆「喂给 embedding 的文本」（v0.41.2）。
 *
 * 这条链路试过三种取法，实测数据决定最终写法（252 条真实库 + 5 句纯语义查询）：
 *
 * | 取法 | 纯语义相关率 |
 * |---|---|
 * | 正文前 300 字（v0.41.0） | ~60-70% |
 * | 头 200 + 尾 200（方案 A） | **~50%**（更差：中间根因丢了） |
 * | **加粗句 + 开头 150（本版）** | 见 CHANGELOG |
 *
 * 关键发现：**记忆的结论都在 `**...**` 里**。这套记忆的正文写法高度一致——现象/踩坑用
 * 普通文字，结论用加粗（「**通用教训：…**」「**画风最后定在…**」）。加粗等于人工已经
 * 标好了重点，直接喂给向量比按位置截断准得多。
 *
 * 退化：老条目可能没用加粗，此时加粗集为空，自动退回「开头 300 字」。
 * ⚠️ migrate-embeddings.mjs 必须调用同一函数，否则存量向量与增量向量不同源。
 */
export function embeddingText(row: { keywords?: unknown; content?: string | null; title?: string | null }): string {
  const kw = Array.isArray(row.keywords) ? (row.keywords as string[]).join('，') : ''
  const body = (row.content ?? '').replace(/\s+/g, ' ').trim()
  // 加粗句 = 结论。过滤掉纯符号/过短的噪声加粗。
  const bold = [...body.matchAll(/\*\*(.+?)\*\*/g)]
    .map((m) => m[1].trim())
    .filter((s) => s.length >= 4 && /[\p{L}\p{N}]/u.test(s))
  const head = body.slice(0, 150)
  // v0.41.4 最终版：加粗结论（≤300 字）+ 开头 150 字。四种取法的实测（5 句纯语义 × top5，
  // 人工判定相关率）：
  //   正文前 300 字（v0.41.0）      ~60-70%
  //   头 200 + 尾 200（方案 A）      ~50%   ← 更差：中间根因丢失
  //   加粗 + 前 400（方案 C）        ~65%
  //   **加粗 + 开头 150（本版）**    ~69%   ← 最好，且命中更准（宁缺毋滥）
  // 加粗句是这套记忆的天然摘要（结论都写成 `**通用教训：…**`），比按位置截断可靠。
  // 降级：老条目没用加粗时 bold 为空，退回正文前 300 字。
  const core = bold.length > 0 ? bold.join('；').slice(0, 300) : body.slice(0, 300)
  return `${kw}。${core}。${head}`.replace(/\s+/g, ' ').trim().slice(0, 500)
}
