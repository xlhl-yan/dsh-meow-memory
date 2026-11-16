/**
 * v0.44 伪相关反馈（Pseudo Relevance Feedback / Rocchio 式查询扩展）。
 *
 * ## 解决什么
 * 口语化查询会让**两层**都失效：
 *   - bi-encoder：「那个烤显卡的家伙」和「AutoDL 实例」没有共享词，余弦够不着
 *   - cross-encoder：实测「换场那一下最费劲」给正确答案 −2.18、给噪声 −2.24，**只差 0.06**，
 *     排序等于随机；而同一句话换成「转场」两个字，答案立刻跳到 +2.27（拉开 6 分）
 *
 * 所以瓶颈是「查询里没有模型认识的技术词」。本模块从**第一轮召回的记忆条目关键词**里
 * 抽这些词补回查询——词来自用户自己的记忆库，天然贴合他自己的用词，且 **0 LLM、0 网络**。
 *
 * ## 为什么不用 LLM 改写
 * 实测插件里拿不到 LLM 调用能力（`@deepseek-ai/dsh-llm` 只有 `createUserMessage` 这类
 * 消息构造器；`dream.ts` 的做法是注入消息让主 agent 处理，不能同步拿结果）。
 *
 * ## 已知覆盖率（合作方实测 5 句难句）
 * 约 2/5 能救：池子里碰巧有技术词种子的能救（烤显卡 → 抽出「AutoDL」「租GPU」），
 * 第一轮就跑偏的救不了（换场 → 池子里一个「转场」都没有）。
 * 这是本方案的固有限制：**它只能放大池子里已有的信息，不能凭空创造**。
 */

/** 扩展词最多取几个（太多会稀释向量）。 */
export const PRF_MAX_TERMS = 5

/**
 * ⚠️ **2026-11-16 三次尝试全部失败，默认关闭。**
 *
 * 原因是 keywords 是「一句话总结」不是「概念标签」：
 *   `要重跑只能租GPU`（而不是 `租GPU`）、`19800积分总量不刷新`（而不是 `积分`）。
 *
 * **等 keywords 规范化（写入侧改成 2-6 个概念词）做完后再开**，
 * 别在 keywords 还是长句的时候浪费时间调算法——三次尝试的详细结果见文件头注释。
 */
export const PRF_ENABLED = false

/** 与检索无关的高频虚词/元词：出现在关键词里的概率高但没区分度。 */
const PRF_STOP = new Set([
  '记忆', '插件', '注入', '命中', '召回', '机制', '问题', '原因', '办法', '方法',
  '结论', '教训', '判断', '实测', '测试', '验证', '记录', '说明', '注意', '要点',
  '用户', '他说', '要求', '决策', '设计', '实现', '功能', '参数', '配置',
  '第一次', '第二次', '最后', '之前', '之后', '现在', '当时', '一下', '一个',
  '这个', '那个', '就是', '不是', '没有', '已经', '还是', '但是', '因为',
]);

/**
 * 从候选池的关键词里抽「技术词种子」。
 *
 * ## 三次尝试的教训
 * 1. **按跨候选频次抽（`>=2` 条重复）→ 一个字都抽不出来**：这套记忆的 keywords 是
 *    「一条一个说法」，池子 2 条里零重复。
 * 2. **整条 keyword 当扩展词 → 效果变负面**：keywords 是**完整短语**不是概念词
 *    （`要重跑只能租GPU`、`19800积分总量不刷新`、`70段汇总被退回看不清`），
 *    拼进查询等于塞一堆具体细节，把原语义稀释掉（实测「别那么肉」的 top5 反而更差）。
 * 3. **本版：只从短语里拆「拉丁词/数字片段」**（`AutoDL`/`TDR`/`GPU`/`736x416`）——
 *    这类才是口语查询最缺、且指向性最强的部分（cross-encoder 对「转场」两个字就能拉开
 *    6 分，对一串细节描述却没有区分度）。
 *
 * ## 已知局限（合作方实测确认）
 * 池子第一轮就跑偏时（「换场」池子里一个「转场」都没有），**纯中文概念词抽不出来**——
 * 因为源头就没有。本方案只能放大池子里已有的信息。
 */
export function extractPrfTerms(
  candidates: Array<{ keywords?: string[] }>,
  query: string,
): string[] {
  if (candidates.length === 0) return []
  const q = query.toLowerCase()
  const score = new Map<string, number>()
  const add = (t: string, w: number): void => {
    const term = t.trim()
    if (term.length < 2 || term.length > 20) return
    if (PRF_STOP.has(term)) return
    if (q.includes(term.toLowerCase())) return // 原查询已有
    score.set(term, (score.get(term) ?? 0) + w)
  }
  for (const c of candidates) {
    for (const raw of c.keywords ?? []) {
      const term = String(raw).trim()
      // 只取拉丁字母开头的技术片段：AutoDL / TDR / GPU / nvlddmkm / 736x416 / Wan2.1
      const latin = term.match(/[A-Za-z][A-Za-z0-9._-]{1,19}|[0-9]{2,4}x[0-9]{2,4}/g) ?? []
      for (const t of latin) add(t, 2)
      // 整条短语仅在「短且不含中文标点」时兜底（「ComfyUI闪退」这类）
      if (term.length <= 8 && !/[，。；、]/.test(term)) add(term, 1)
    }
  }
  return [...score.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, PRF_MAX_TERMS)
    .map(([t]) => t)
}

/** 把扩展词拼回查询。原查询放最前，保证主语义不被稀释。 */
export function buildExpandedQuery(query: string, terms: string[]): string {
  if (terms.length === 0) return query
  return `${query} ${terms.join(' ')}`
}
