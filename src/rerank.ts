/**
 * v0.43 cross-encoder 精排（rerank）。
 *
 * 为什么需要：bi-encoder（bge-small）把「查询」和「文档」分别编码再比余弦，**看不到两者的
 * 交互**，所以跨不过「口语 ↔ 术语」的鸿沟（「那个烤显卡的家伙」vs 记忆里的「AutoDL 实例」）。
 * cross-encoder 把 query+doc 拼一起过一遍模型，能判断「这条记忆回答了这个问题吗」。
 *
 * 实测（5 句难句 × top5，人工判定）：混合召回相关 4.5/13 ≈ 35%，rerank 后 13.5/25 ≈ 54%。
 *
 * 设计要点：
 *  - **懒加载 + 失败降级**：模型缺失/加载失败一律返回 null，调用方保持原顺序。
 *  - **必须拿原始 logits**：transformers.js 的 `text-classification` pipeline 会把单 logit
 *    softmax 成 1，所有文档分数一样、排序完全失效（实测踩过）。用 AutoModel 取 logits。
 *  - 分数是负数 logit，**越大（越接近 0）越相关**，不是概率。
 */
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

/** 默认 reranker。量化版优先（fp32 单条 231ms，量化约 60-80ms）。 */
export const DEFAULT_RERANK_MODEL = 'Xenova/bge-reranker-base'
/** 单条文档截断长度：cross-encoder 的 token 上限 512，中文 400 字足够。 */
const DOC_MAX_CHARS = 400

export function modelsDir(): string {
  const here = dirname(fileURLToPath(import.meta.url))
  return resolve(here, '..', 'models')
}

type RerankFn = (query: string, doc: string) => Promise<number>

let fn: RerankFn | null = null
let pending: Promise<RerankFn | null> | null = null
let failed = false

export function rerankUnavailable(): boolean {
  return failed
}

/** 预热（幂等，失败后不再重试）。 */
export function warmupRerank(): Promise<RerankFn | null> {
  if (fn !== null) return Promise.resolve(fn)
  if (failed) return Promise.resolve(null)
  if (pending !== null) return pending
  pending = (async () => {
    try {
      const mod = (await import('@xenova/transformers')) as {
        env: { allowRemoteModels: boolean; localModelPath: string }
        AutoTokenizer: { from_pretrained: (m: string) => Promise<unknown> }
        AutoModelForSequenceClassification: { from_pretrained: (m: string, o?: Record<string, unknown>) => Promise<unknown> }
      }
      mod.env.allowRemoteModels = false
      mod.env.localModelPath = modelsDir()
      // 优先量化版（model_quantized.onnx），没有则用 fp32（model.onnx）
      const tok = (await mod.AutoTokenizer.from_pretrained(DEFAULT_RERANK_MODEL)) as {
        (text: string[], o: Record<string, unknown>): unknown
      }
      let model: unknown
      try {
        model = await mod.AutoModelForSequenceClassification.from_pretrained(DEFAULT_RERANK_MODEL, { quantized: true })
      } catch {
        model = await mod.AutoModelForSequenceClassification.from_pretrained(DEFAULT_RERANK_MODEL, { quantized: false })
      }
      const run = model as (inputs: unknown) => Promise<{ logits: { data: ArrayLike<number> } }>
      fn = async (query: string, doc: string): Promise<number> => {
        const inputs = tok([query], { text_pair: [doc.slice(0, DOC_MAX_CHARS)], padding: true, truncation: true })
        const { logits } = await run(inputs)
        return logits.data[0] ?? -99
      }
      return fn
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
 * 对一批文档打分（保持输入顺序）。模型不可用时返回 null。
 * @returns 与 docs 等长的分数数组；分数越大越相关（负数 logit，不是概率）。
 */
export async function rerankScores(query: string, docs: string[]): Promise<number[] | null> {
  if (failed || docs.length === 0) return null
  const f = await warmupRerank()
  if (f === null) return null
  const out: number[] = []
  for (const d of docs) {
    try {
      out.push(await f(query, d))
    } catch {
      out.push(-99)
    }
  }
  return out
}

/** 是否启用 rerank（配置 / 环境开关）。 */
export function rerankEnabled(flag?: boolean): boolean {
  if (flag === false) return false
  return !failed
}
