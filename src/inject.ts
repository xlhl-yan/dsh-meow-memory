/**
 * meow-memory v2 — 会话开头注入。
 *
 * 结构（用户拍板）：soul/user 全量 + 记忆导引（project/topic 标题列表，
 * 正文模型自取）+ 第一条用户消息关键词命中 fact/lesson 短条目自动注入。
 * 无每轮注入——模型自己用 memory_search 深挖。
 *
 * 去重：.dsh-meow/sessions/<sessionId>.json 记录本会话注入过的 memory id，
 * 已注入不重复。
 */

import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import type { Doc } from './bm25.js'
import { keywordHitScore, search, tokenize } from './bm25.js'
import { cosine, embeddingModelId, embeddingUnavailable } from './embedding.js'
import { isGlobalProject, isGlobalScope, memoryDbPath, projectCovers, projectLabel, projectList, relativeTime, PROJECT_SUBCATEGORIES, type MemoryDb, type MemoryRow, type ProjectSubcategory } from './db.js'
import { fillTemplate, keyedValue } from './prompt-loader.js'

export interface InjectOptions {
  /** 关键词命中条数上限（fact/lesson 短条目）。 */
  hitTopK: number
  /** 导引里 project/topic 标题超长截断长度。 */
  titleMax: number
  /** 命中最低覆盖率门槛（0..1）：本轮消息被记忆关键词覆盖的比例低于此值就不注入。
   *  分母取 min(查询 token 数, 记忆 token 数)（见 bm25.keywordHitScore）。
   *  0 = 关闭门槛。 */
  hitMinCoverage: number
  /** 命中诊断日志（默认 false）：每次命中判定（含 0 条）都打一行，排查「怎么从来不命中」用。
   *  不进设置页表单——它是排查工具不是日常开关（需求方拍板 2026-11-13）。 */
  hitDebug: boolean
  /** 混合召回开关（v0.41.0，默认 true）：关掉退回纯词面（v0.40 行为），用于排查
   *  「是向量在捣乱还是词面本来就不行」。 */
  hybrid?: boolean
}

const DEFAULT_OPTS: InjectOptions = { hitTopK: 3, titleMax: 40, hitMinCoverage: 0.15, hitDebug: false, hybrid: true }

/** 命中查询的文本上限：从用户消息里取这么多字符做关键词匹配。
 *  旧值是 500——桥接/预设会在真实内容前拼大段说明模板，500 字常常只覆盖到模板，
 *  真实诉求永远匹配不到（长会话实测命中率 0%）。4000 覆盖绝大多数消息全文。 */
const MAX_QUERY_CHARS = 4000

/** 「最近注入窗口」长度：命中去重只看最近这么多条注入记录，而不是整会话的注入全集。
 *  旧语义（整会话 excluded）在长会话里会把候选池清零——本会话写的记忆先被
 *  source_session 排除，剩下的又全在 injected 里，最终没有一条能再被命中。 */
export const MAX_RECENT_INJECTED = 40

/** 命中链路的最少命中 token 数（v0.37.0）。只撞上 1 个词就注入是误注入的主要来源：
 *  实测「等视频渲染完再说」撞一个「渲染」就注入 4 条全无关的记忆。 */
export const MIN_HIT_COUNT = 2

/** 单命中放行所需的 IDF 权重和（v0.39.0）。短查询的有效词常只剩 1 个，hc>=2 在数学上
 *  永不满足（实测「甩镜怎么做」过滤后只剩「甩镜」、「AutoDL 那台什么配置」只剩
 *  「那台/配置」而没有任何一条记忆同时含这两个词）——这两条就这样全灭。改看「命中的词
 *  够不够有信息量」：「甩镜」idfSum 3.77 放行。 */
export const MIN_IDF_SUM = 2.0

// ── v0.41.0 混合召回权重与门槛 ──────────────────────────────────────────────
// 词面 0.45 + 向量 0.55：词面保住专有名词/数字的精确匹配（`3081`、`48GB` 这类词向量会糊），
// 向量补上「字面不重合但语义相关」。实测（252 条真实库 + 15 条测试集）：
//   纯词面 68% / 纯向量 74% / 混合 80%，且混合保住 top1 命中 10/10、负样本误命中 2/5→1/5。
export const HYBRID_W_LEX = 0.45
export const HYBRID_W_VEC = 0.55
/** 混合总分门槛：0.45×词面 + 0.55×向量归一化。
 *  0.40 而不是 0.30：BGE 对「任意两句中文」都给出 ~0.5 的基线余弦，闲聊句（「我有点困了」
 *  top1=0.577）和技术问句（「甩镜怎么做」top1=0.616）几乎重叠——单靠 sim 分不开。
 *  但负样本有一个共同点：**词面命中 0，是纯向量命中**。0.30 时闲聊的
 *  0.55×(0.577-0.30)/0.45 = 0.339 刚好过线；0.40 挡住它，而有词面支撑的正样本
 *  （cov 0.5 + sim 0.6 → 0.594）绰绰有余。 */
export const HYBRID_MIN_SCORE = 0.40
/** 纯向量候选（词面完全没撞上）的最低余弦：低于它认为只是泛泛相关。 */
export const HYBRID_VEC_MIN_SIM = 0.45
/** 查询向量是否参与召回（关掉即退回 v0.40 的纯词面行为）。 */
export const HYBRID_ENABLED_DEFAULT = true

/** 单命中的「强覆盖」例外阈值（v0.37.0）：只命中 1 个 token，但它覆盖了查询一半以上
 *  的信息量时仍然放行（「画风定成什么了」→ 只中「画风」，cov 0.55）。 */
export const STRONG_COVERAGE = 0.5

/** 命中诊断日志出口（apply 期由宿主 logger 注入；null = 不输出）。
 *  与 setMemoryToolGate 同款：模块级注入，避免把 ctx 一路透传到打分链路。 */
let hitDebugLogger: ((msg: string) => void) | null = null

export function setHitDebugLogger(fn: ((msg: string) => void) | null): void {
  hitDebugLogger = fn
}

/** 查询侧停用 bigram（虚词 / 疑问词 / 跨词碎片）：命中了也不代表语义相关。
 *
 *  实测（207 条真实记忆）：「今天天气怎么样」唯一命中的是「怎么」，「1 加 1 等于几」只命中
 *  「等于」，「帮我看看这个文件」命中碎片「个文」——都靠虚词把覆盖率抬到 0.17-0.50，
 *  把无关记忆注入了进来。更糟的是「甩镜怎么做」真正该命中的「甩镜」反而因为 IDF 低于
 *  「怎么」排在后面。
 *
 *  这里只做「命中侧」过滤（不参与分母统计），代价是真实但不相关的虚词命中不会被算作相关。 */
const QUERY_STOP_TOKENS = new Set([
  '怎么', '么做', '么样', '么配', '什么', '为什', '如何', '哪些', '是不', '不是',
  '这个', '那个', '个文', '一个', '一下', '可以', '应该', '帮我', '我看', '看看',
  '等于', '现在', '今天', '明天', '昨天', '想要', '需要', '希望', '这样', '那样',
  '就是', '还是', '但是', '因为', '所以', '如果', '然后', '已经', '没有', '一些',
  '这里', '那里', '你的', '我的', '他的', '自己',
  // v0.37.0 补：问句碎片。「双卡32G是什么意思」曾靠「是什/么意/意思」三个碎片命中一条
  // 毫不相关的记忆（cov=74%）——这类词是「X 是什么意思」的固定尾巴，永远不承载语义。
  '是什', '么意', '意思', '几点',
  // v0.39.0：日常应答/闲聊词。它们偶尔混进某些记忆的 keywords 里，IDF 反而很高
  //（「吃饭」在 238 条里只出现 1-2 次 → idf 5.07，比正样本「甩镜」的 3.77 还高），
  // 单命中放行后就会误命中。已核实「知道/收到」等只出现在正文（13/5 条）、不在匹配面里，
  // 加进表不会误伤；真正在 keywords 里的只有「好的」(4 条) 和「吃饭」(1 条)。
  '好的', '吃饭', '睡觉', '知道', '收到', '谢谢', '再见', '拜拜', '晚安', '早安',
  '累了', '休息', '忙吗', '干嘛', '去哪',
])

/** 短条目阈值（token 数）与它的宽松覆盖率门槛：只有寥寥几个关键词的条目（早期无
 *  keywords、回退正文前 100 字的旧数据），覆盖率分母小，按长条目门槛会被一律误伤。
 *  实测正常记忆 token 数最小 14、中位 33，所以这一档只对旧数据/极短条目生效。 */
const SHORT_DOC_TOKENS = 10
const SHORT_DOC_MIN_COVERAGE = 0.15

export function sessionsFile(workspace: string, sessionId: string, dir = '.dsh-meow'): string {
  // 会话 id 可能含 Windows 文件名非法字符（#56：飞书桥 lark-link:dm:…:0 的冒号）——
  // 读写全部 ENOENT。非安全字符做百分号编码：确定性（跨重启同 id 同文件）、无碰撞
  // （% 编码与原字符不重叠）、安全字符原样（既有会话文件不失效）。
  const safe = sessionId.replace(/[^A-Za-z0-9._-]/g, (ch) => `%${ch.charCodeAt(0).toString(16).toUpperCase().padStart(2, '0')}`)
  return join(workspace, dir, 'sessions', `${safe}.json`)
}

/** 会话记忆可见集：injected=注入过的，searched=search/find_similar 返回过的，
 *  accessed=memory_read 读过的（v0.17.0，dream 第一轮清单的"查阅"源）。
 *  前两者是"本会话上下文里已经出现过的记忆"，检索时应排除（省 token、扩大检索面）；
 *  accessed 只服务 dream 扫尾范围（读过的条目该被复查），不参与命中去重。 */
export interface SessionSeen {
  injected: string[]
  /** 命中去重窗口（滑动）：最近 MAX_RECENT_INJECTED 条注入 id。命中链路只看它，
   *  不看 injected 全集——否则长会话里注入过的记忆永不再现。首轮快照也会记进来
   *  （同一批内容不该在紧接着的几轮里重复出现）。 */
  recentInjected: string[]
  searched: string[]
  accessed: string[]
  /** 本会话写工具落库过的记忆 id（v0.23.0）：memory_remember（新建/合并）与
   *  memory_update（实际写入）成功后记录，子代理代写归父窗口文件（source_session 同源）。
   *  压缩重注入第三块数据源；releaseSeen 不清（同 projectsQueried，正是重注入数据源）。 */
  written: string[]
  /** 本会话 AI 调 memory_project 查阅过的项目名（v0.21.0，按查询顺序、去重、
   *  最多保留 MAX_REINJECT_PROJECTS 个最近项）：会话压缩后重注入项目全景的清单。
   *  '全局' 不记（全局 soul/user/rules 已在快照层，非项目）；多项目参数按逗号拆开记。 */
  projectsQueried: string[]
  /** 压缩成功落地后置 true（v0.21.0）：下一个含真实用户消息的 pre-step 注入
   *  压缩重注入块（长期记忆快照 + 项目全景），随后清回 false。 */
  reinjectPending: boolean
  /** 当前 project 锚定（最近一次带 project 参数的 memory 工具调用）：命中检索限定"全局+当前项目"。 */
  currentProject: string | null
}

/** projectsQueried 保留上限：压缩重注入的项目全景个数上限（防极端会话注入膨胀）。 */
export const MAX_REINJECT_PROJECTS = 8

/** written 保留上限：压缩重注入"本会话写过的记忆"条数上限（防极端会话注入膨胀）。 */
export const MAX_REINJECT_WRITTEN = 20

function readSeenFile(workspace: string, sessionId: string, dir: string): SessionSeen {
  try {
    const text = readFileSync(sessionsFile(workspace, sessionId, dir), 'utf8')
    const parsed = JSON.parse(text) as Record<string, unknown>
    return {
      injected: Array.isArray(parsed.injected) ? parsed.injected.filter((x): x is string => typeof x === 'string') : [],
      recentInjected: Array.isArray(parsed.recentInjected) ? parsed.recentInjected.filter((x): x is string => typeof x === 'string') : [],
      searched: Array.isArray(parsed.searched) ? parsed.searched.filter((x): x is string => typeof x === 'string') : [],
      accessed: Array.isArray(parsed.accessed) ? parsed.accessed.filter((x): x is string => typeof x === 'string') : [],
      written: Array.isArray(parsed.written) ? parsed.written.filter((x): x is string => typeof x === 'string') : [],
      projectsQueried: Array.isArray(parsed.projectsQueried) ? parsed.projectsQueried.filter((x): x is string => typeof x === 'string') : [],
      reinjectPending: parsed.reinjectPending === true,
      currentProject: typeof parsed.currentProject === 'string' && parsed.currentProject.length > 0 ? parsed.currentProject : null,
    }
  } catch {
    return { injected: [], recentInjected: [], searched: [], accessed: [], written: [], projectsQueried: [], reinjectPending: false, currentProject: null }
  }
}

/** 统一写 sessions/<id>.json（全部字段一次写全——新增字段只改这里，防散落漏写）。 */
function writeSeenFile(workspace: string, sessionId: string, seen: SessionSeen, dir: string): void {
  const file = sessionsFile(workspace, sessionId, dir)
  mkdirSync(dirname(file), { recursive: true })
  writeFileSync(file, JSON.stringify({
    injected: seen.injected,
    recentInjected: seen.recentInjected,
    searched: seen.searched,
    accessed: seen.accessed,
    written: seen.written,
    projectsQueried: seen.projectsQueried,
    reinjectPending: seen.reinjectPending,
    currentProject: seen.currentProject,
  }), 'utf8')
}

/** 读本会话已注入 id 列表（文件不存在返回空数组）。 */
export function readInjected(workspace: string, sessionId: string, dir = '.dsh-meow'): string[] {
  return readSeenFile(workspace, sessionId, dir).injected
}

/** 追加写入已注入 id（同时推进「最近注入窗口」）。 */
export function markInjected(workspace: string, sessionId: string, ids: string[], dir = '.dsh-meow'): void {
  if (ids.length === 0) return
  const seen = readSeenFile(workspace, sessionId, dir)
  const set = new Set(seen.injected)
  for (const id of ids) set.add(id)
  seen.injected = [...set]
  // 滑窗：新注入的排最后，超出长度丢最旧的。
  const recent = seen.recentInjected.filter((id) => !ids.includes(id))
  recent.push(...ids)
  seen.recentInjected = recent.slice(-MAX_RECENT_INJECTED)
  writeSeenFile(workspace, sessionId, seen, dir)
}

/** 本会话全部"已见" id（注入 + 检索 + 查阅），dream 第一轮清单范围 + 检索排除用。 */
export function readSeen(workspace: string, sessionId: string, dir = '.dsh-meow'): Set<string> {
  const s = readSeenFile(workspace, sessionId, dir)
  return new Set([...s.injected, ...s.searched, ...s.accessed])
}

/** 追加已检索返回的 id（memory_search / memory_find_similar 命中后调用）。 */
export function markSearched(workspace: string, sessionId: string, ids: string[], dir = '.dsh-meow'): void {
  if (ids.length === 0) return
  const seen = readSeenFile(workspace, sessionId, dir)
  const set = new Set(seen.searched)
  for (const id of ids) set.add(id)
  seen.searched = [...set]
  writeSeenFile(workspace, sessionId, seen, dir)
}

/** 追加 memory_read 读过的 id（v0.17.0）：dream 第一轮"查阅过"源。
 *  只由 memory_read 单条读取触发；memory_project 全景不标记（第三轮项目总结专门复查）。 */
export function markAccessed(workspace: string, sessionId: string, ids: string[], dir = '.dsh-meow'): void {
  if (ids.length === 0) return
  const seen = readSeenFile(workspace, sessionId, dir)
  const set = new Set(seen.accessed)
  for (const id of ids) set.add(id)
  seen.accessed = [...set]
  writeSeenFile(workspace, sessionId, seen, dir)
}

/** 追加本会话写工具落库过的 id（v0.23.0）：memory_remember（新建/合并）与
 *  memory_update（实际写入）成功后调用。重复写入移到末尾（最近优先），
 *  超过 MAX_REINJECT_WRITTEN 淘汰最旧的。 */
export function markWritten(workspace: string, sessionId: string, ids: string[], dir = '.dsh-meow'): void {
  if (ids.length === 0) return
  const seen = readSeenFile(workspace, sessionId, dir)
  const ordered = seen.written.filter((id) => !ids.includes(id))
  ordered.push(...ids)
  seen.written = ordered.slice(-MAX_REINJECT_WRITTEN)
  writeSeenFile(workspace, sessionId, seen, dir)
}

/** 读本会话写过的记忆 id 清单（压缩重注入第三块 + dream 第一轮清单用）。 */
export function readWritten(workspace: string, sessionId: string, dir = '.dsh-meow'): string[] {
  return readSeenFile(workspace, sessionId, dir).written
}

/** 记录 memory_project 查阅过的项目（v0.21.0）：压缩重注入清单。
 *  全局标记（isGlobalProject，语言感知）/空串跳过；多项目参数按逗号拆开逐个记；
 *  重复查询移到末尾（最近优先）；超过 MAX_REINJECT_PROJECTS 淘汰最旧的。 */
export function markProjectQueried(workspace: string, sessionId: string, project: string, dir = '.dsh-meow'): void {
  const names = project.split(',').map((p) => p.trim()).filter((p) => p.length > 0 && !isGlobalProject(p))
  if (names.length === 0) return
  const seen = readSeenFile(workspace, sessionId, dir)
  const ordered = seen.projectsQueried.filter((p) => !names.includes(p))
  ordered.push(...names)
  seen.projectsQueried = ordered.slice(-MAX_REINJECT_PROJECTS)
  writeSeenFile(workspace, sessionId, seen, dir)
}

/** 读本会话查阅过的项目清单（压缩重注入用；文件不存在返回空数组）。 */
export function readProjectQueried(workspace: string, sessionId: string, dir = '.dsh-meow'): string[] {
  return readSeenFile(workspace, sessionId, dir).projectsQueried
}

/** 压缩成功落地后置位（compaction/end 无 error 时调用）：下一个含真实用户消息的
 *  pre-step 注入压缩重注入块。 */
export function markReinjectPending(workspace: string, sessionId: string, dir = '.dsh-meow'): void {
  const seen = readSeenFile(workspace, sessionId, dir)
  if (seen.reinjectPending) return
  seen.reinjectPending = true
  writeSeenFile(workspace, sessionId, seen, dir)
}

/** 清除重注入待办（重注入块注入完成后调用；幂等）。 */
export function clearReinjectPending(workspace: string, sessionId: string, dir = '.dsh-meow'): void {
  const seen = readSeenFile(workspace, sessionId, dir)
  if (!seen.reinjectPending) return
  seen.reinjectPending = false
  writeSeenFile(workspace, sessionId, seen, dir)
}

/** 读重注入待办标记。 */
export function isReinjectPending(workspace: string, sessionId: string, dir = '.dsh-meow'): boolean {
  return readSeenFile(workspace, sessionId, dir).reinjectPending
}

/** 释放本会话已见记录（收到会话压缩信号后调用）：清空 injected/searched，
 *  允许之前注入/检索过的记忆被再次命中提取——压缩后它们的内容已不在上下文里。
 *  accessed 不清（用户拍板 2026-08-25）：它只服务 dream 扫尾范围、没有去重功能，
 *  清掉纯丢信息——长窗口压缩前读过的条目恰恰最该被 dream 复查。
 *  当前 project 锚定保留（与可见性无关）；projectsQueried/written/reinjectPending 保留
 *  （它们正是压缩重注入的数据源，见 buildReinjection）。 */
export function releaseSeen(workspace: string, sessionId: string, dir = '.dsh-meow'): void {
  const seen = readSeenFile(workspace, sessionId, dir)
  seen.injected = []
  seen.searched = []
  writeSeenFile(workspace, sessionId, seen, dir)
}

/** 读当前 project 锚定（最近一次带 project 参数的 memory 工具调用）；未锚定返回 null。 */
export function getCurrentProject(workspace: string, sessionId: string, dir = '.dsh-meow'): string | null {
  return readSeenFile(workspace, sessionId, dir).currentProject
}

/** 锚定当前 project：memory 工具调用带 project 参数时更新会话状态（命中检索用它，免扫历史）。 */
export function setCurrentProject(workspace: string, sessionId: string, project: string, dir = '.dsh-meow'): void {
  const seen = readSeenFile(workspace, sessionId, dir)
  seen.currentProject = project
  writeSeenFile(workspace, sessionId, seen, dir)
}

function toDocs(rows: MemoryRow[]): Doc[] {
  return rows.map((r) => ({
    id: r.id,
    level: r.level,
    title: r.title,
    content: r.content,
    keywords: r.keywords,
    importance: r.importance,
    created_at: r.created_at,
    updated_at: r.updated_at,
    last_accessed_at: r.last_accessed_at, // #91：衰减按最后使用时间
  }))
}

function shortTitle(row: MemoryRow, max: number): string {
  const t = row.title?.trim()
  if (t) return t.length > max ? t.slice(0, max) + '…' : t
  const c = row.content.replace(/\s+/g, ' ').trim()
  return c.length > max ? c.slice(0, max) + '…' : c
}
/* v8 ignore next -- 保留工具函数（历史/调试用），当前导引不再截断标题 */
void shortTitle

/**
 * 构造长期记忆快照正文（buildInjection / buildReinjection 共用）：
 *   顶格「===== 长期记忆 =====」→ 【关于你】(soul) / 【关于user】/ 【设计原则】/ 【记忆导引】(两行)。
 * 不含结束标记与「本轮用户prompt：」尾巴（两链路各自拼装），也不做已见记账（调用方负责）。
 * @returns null = 只有标题头，无任何可注入内容。
 */
function buildInjectionBody(
  db: MemoryDb,
  o: InjectOptions,
): { body: string; injectedIds: string[] } | null {
  // 框架词外置（v0.19.0）：labels.md 的 inject.* 键；记忆条目正文本身是数据不是文案，不外置。
  const lbl = (key: string, params?: Record<string, string>): string => fillTemplate(keyedValue('labels', key), params)
  const soul = db.list('soul', { status: 'active' })
  const user = db.list('user', { status: 'active' })

  const lines: string[] = [lbl('inject.title'), '']
  const injected: string[] = []

  const pushEntries = (label: string, rows: MemoryRow[]) => {
    if (rows.length === 0) return
    lines.push(lbl('inject.sectionFormat', { label }))
    for (const r of rows) {
      lines.push(`- ${r.content}`)
      injected.push(r.id) // 正文已注入 → 记入已见（命中链路不再重复注入）
    }
    lines.push('')
  }

  pushEntries(lbl('inject.aboutYou'), soul)
  pushEntries(lbl('inject.aboutUser'), user)

  // 设计原则（rules）：注入「全局适用（未标记 null / 空串 / 全局标记「全局」「global」）
  // 且 importance≥2」的——少而精的命令式准则。
  // 2026-09-19 口径修复（用户拍板）：此前只认 project === null，而文档（tools.md /
  // system-guide.md / dream-atomic.md）教模型写"全局"——按文档写的准则反而永不注入。
  const globalRules = db.list('rules', { status: 'active' }).filter((r) => isGlobalScope(r.project) && r.importance >= 2)
  pushEntries(lbl('inject.rules'), globalRules)

  // 记忆导引：说明 + 项目列表（正文/标题一律自取，不列）。
  // titleMax 落地（2026-09-10）：schema/设置页一直承诺「导引里项目名的截断长度」，
  // 此前是死配置（shortTitle 被 void）。这里按承诺截断项目名——只影响导引一行展示，
  // 记忆条目正文/标题依旧不截断（v0.19.0 拍板不变）。
  const projectNames = db.listProjectNames()
  if (projectNames.length > 0) {
    const max = o.titleMax
    const shown = projectNames.map((n) => (n.length > max ? n.slice(0, max) + '…' : n))
    lines.push(lbl('inject.sectionFormat', { label: lbl('inject.guide') }))
    lines.push(lbl('inject.guideSearchLine'))
    lines.push(lbl('inject.guideProjectLine'))
    lines.push(lbl('inject.guideProjects', { list: shown.join(' / ') }))
    lines.push('')
  }

  if (lines.length <= 2) return null // 只有标题头，无任何内容
  return { body: lines.join('\n').trimEnd(), injectedIds: injected }
}

/**
 * 构造首轮长期记忆注入块（用户拍板格式，PR #10 起为独立 plugin snapshot 正文）：
 *   顶格「===== 长期记忆 =====」→ 【关于你】(soul) / 【关于user】/ 【设计原则】/ 【记忆导引】(两行)。
 *   不再拼接结束标记与「本轮用户prompt：」尾巴——正文作为独立消息插在真实用户消息前，
 *   用户 prompt 保持原样（防会话标题污染；旧格式历史会话由前端折叠兼容）。
 * 首轮只注入长期记忆，不做关键词命中（命中链路从第二轮起，见 buildHitInjection）；
 * 正文注入的 id 记入已见（命中链路不再重复注入它们）。
 * @returns { text, injectedIds }；无任何可注入内容返回 null。
 */
export function buildInjection(
  db: MemoryDb,
  workspace: string,
  sessionId: string,
  _firstUserText: string = '',
  opts: Partial<InjectOptions> = {},
  dir = '.dsh-meow',
): { text: string; injectedIds: string[] } | null {
  const o = { ...DEFAULT_OPTS, ...opts }
  const built = buildInjectionBody(db, o)
  if (built === null) return null
  const text = built.body
  if (built.injectedIds.length > 0) markInjected(workspace, sessionId, built.injectedIds, dir)
  return { text, injectedIds: built.injectedIds }
}

/** 框架词（labels.md）：项目全景段落里的短词随语言包走（PR #6 外置，zh 值逐字不变）。 */
const lbl = (key: string, params?: Record<string, string>): string => fillTemplate(keyedValue('labels', key), params)
/** 子标签 → 注入段落标题（文案外置：labels.md 的 project.section.*）。 */
const sectionTitle = (sub: ProjectSubcategory): string => lbl(`project.section.${sub}`)

/** 组内排序：记忆时间戳（updated_at）旧→新，相同按创建时间；null 视为最旧。 */
function sortByUpdatedAt(list: MemoryRow[]): MemoryRow[] {
  return [...list].sort((a, b) => (a.updated_at ?? 0) - (b.updated_at ?? 0) || a.created_at - b.created_at)
}

/** 项目全景条目行（原文视图）：归属 + 完整 id + 绝对/相对时间戳，第二行完整内容。
 *  归属显示：'全局'=真全局；null=未标记（可能是数据 bug）；多值 join '/'。 */
function fmtProjectRow(r: MemoryRow): string {
  const abs = new Date(r.updated_at).toISOString().slice(0, 16).replace('T', ' ')
  return `[${projectLabel(r.project)} : ${r.level}] [${r.id}] ${abs} [${relativeTime(r.updated_at)}]\n${r.content}`
}

/**
 * 构造项目全景注入段落（memory_project 工具与压缩重注入共用，v0.21.0 从 tools.ts 迁入）：
 * 【项目：X】+ 项目设计原则(rules，放最前——规则优先于事实) + 各子标签分组
 * （组内按记忆时间戳旧→新；todo 含已完成最近 5 条）+ 检索说明尾注。
 * 非 todo 子标签只取 active；条目正文本身是数据，不做文案外置。
 * @returns null = 项目无任何可注入段落（active 条目与已完成 todo 皆空）。
 */
/** 内部构造：段落文本 + 段内出现过的全部条目 id（压缩重注入第三块按此去重，防同块重复展示）。 */
function buildProjectSection(db: MemoryDb, workspace: string, project: string, dir: string): { text: string; ids: string[] } | null {
  // 项目视图按名单语义匹配（#61）：project 字段写多项目名（'dsh,femo'）的条目，
  // 在任一名单项目的视图都该可见——SQL 精确等值会把两边都漏掉。全局标记/未标记
  // projectList 为空、不命中任何具体项目名，保持「全局内容走快照层」的原语义。
  const rows = db.list('project').filter((r) => projectList(r.project).includes(project))
  const ids: string[] = []
  const render = (list: MemoryRow[]): string => {
    for (const r of list) ids.push(r.id)
    return list.map(fmtProjectRow).join('\n')
  }
  const active = rows.filter((r) => r.status === 'active')
  // todo 已完成：stale 且 updated_at 非空，按 updated_at 取最近 5 条（展示仍按旧→新）。
  const done = sortByUpdatedAt(
    rows
      .filter((r) => r.subcategory === 'todo' && r.status === 'stale' && r.updated_at !== null)
      .sort((a, b) => (b.updated_at ?? 0) - (a.updated_at ?? 0))
      .slice(0, 5),
  )
  const bySub = new Map<ProjectSubcategory, MemoryRow[]>()
  for (const r of active) {
    const sub = r.subcategory ?? 'overview' // 早期无子类条目归概述组
    const list = bySub.get(sub) ?? []
    list.push(r)
    bySub.set(sub, list)
  }
  const sections: string[] = []
  // 项目设计原则（rules，project 特定）：放最前——规则优先于事实。
  const projectRules = sortByUpdatedAt(
    db.list('rules').filter((r) => projectList(r.project).includes(project) && r.status === 'active'),
  )
  if (projectRules.length > 0) {
    sections.push(`${lbl('inject.rules')}\n${render(projectRules)}`)
  }
  for (const sub of PROJECT_SUBCATEGORIES) {
    if (sub === 'todo') {
      const todos = sortByUpdatedAt(bySub.get('todo') ?? [])
      if (todos.length === 0 && done.length === 0) continue
      const lines = [sectionTitle('todo')]
      if (done.length > 0) {
        lines.push(lbl('project.todoDone'))
        for (const r of done) {
          ids.push(r.id)
          lines.push(fmtProjectRow(r))
        }
      }
      if (todos.length > 0) {
        lines.push(lbl('project.todoOpen'))
        for (const r of todos) {
          ids.push(r.id)
          lines.push(fmtProjectRow(r))
        }
      }
      sections.push(lines.join('\n'))
    } else {
      const list = sortByUpdatedAt(bySub.get(sub) ?? [])
      if (list.length === 0) continue
      sections.push(`${sectionTitle(sub)}\n${render(list)}`)
    }
  }
  if (sections.length === 0) return null
  const dbPath = memoryDbPath(workspace, dir)
  const text = [
    lbl('project.header', { name: project }),
    '',
    sections.join('\n\n'),
    '',
    '——',
    '说明：此处只提供 active 的记忆。',
    `如果你想看非 active 条目（archived=删除 / stale=完结），或某条记忆的具体时间戳（记忆时间戳=该窗口 dream 封存时刻）、记忆来源（source_session）、重要性、关键词等元数据，可以直接去搜记忆库 SQLite：${dbPath}`,
    '（库内结构：七层表 soul/user/project/fact/lesson/topic/rules，字段含 id/title/content/importance/keywords/status/corrected/project/subcategory/goal/source_session/created_at/updated_at（记忆时间戳=最后更新时间）/last_accessed_at；另有 dream_log 整理留痕表、windows 窗口时间表；也可按 id 用 memory_read 看单条完整元数据）',
    `如果你想了解未被记录的更多细节，可以直接去搜会话历史目录（dsh 的 session 日志，位置由 DSH_HOME 决定，默认 ~/.dsh/sessions，喵版为 dsh-home/sessions），按会话 id 查原始记录。`,
  ].join('\n')
  return { text, ids }
}

/** 段落文本视图（memory_project 工具用）：只要文本。 */
export function buildProjectSectionText(db: MemoryDb, workspace: string, project: string, dir = '.dsh-meow'): string | null {
  return buildProjectSection(db, workspace, project, dir)?.text ?? null
}

/**
 * 压缩重注入第三块（v0.23.0）：本会话写过的记忆原文回放。
 * 数据源 = sessions/<id>.json 的 written（memory_remember/memory_update 落库痕迹，
 * 子代理代写归父窗口）∪ db 层 source_session=本会话 的新建条目（双保险，覆盖插件
 * 热更新前旧代码写入的条目）。按当前库最新数据解析原文（不缓存旧文本）；只回放
 * status=active（archived/stale 内容已失效，回放会误导）；排除快照正文与项目全景
 * 已展示的 id（防同块重复）；超出 MAX_REINJECT_WRITTEN 保留最近写入的。
 * @returns null = 本会话没有可回放的写入条目。
 */
function buildWrittenSection(
  db: MemoryDb,
  workspace: string,
  sessionId: string,
  dir: string,
  exclude: ReadonlySet<string>,
): { text: string; ids: string[] } | null {
  const ordered: string[] = [...readWritten(workspace, sessionId, dir)]
  for (const { id } of db.idsBySession(sessionId)) {
    if (!ordered.includes(id)) ordered.push(id)
  }
  const rows: MemoryRow[] = []
  for (const id of ordered) {
    if (exclude.has(id)) continue
    const found = db.findById(id)
    if (!found || found.row.status !== 'active') continue
    rows.push(found.row)
  }
  const kept = rows.slice(-MAX_REINJECT_WRITTEN)
  if (kept.length === 0) return null
  return {
    text: [
      lbl('inject.sectionFormat', { label: lbl('inject.writtenSection') }),
      lbl('inject.writtenIntro'),
      ...kept.map(fmtProjectRow),
    ].join('\n'),
    ids: kept.map((r) => r.id),
  }
}

/**
 * 构造压缩重注入块（v0.21.0 三块制，v0.23.0 增第三块）：
 * ① 长期记忆快照正文（与首轮 buildInjection 同款）；
 * ② 【会话已压缩】说明 + 本会话此前查阅过的项目全景（按当前库最新数据重新构造，
 *    空项目跳过；项目全景条目按现行原则不标记已见）；
 * ③ 【本会话写过的记忆】+ 本会话写工具落库过的条目原文（见 buildWrittenSection）。
 * 不拼尾注（PR #10 起）：正文作为独立 plugin snapshot 消息插在真实用户消息前，
 * 用户 prompt 保持原样。
 * 快照条目 id 与回放的 written id 重新记入 injected——压缩后内容重新进入上下文，
 * 去重语义随之恢复（合并更新过的他窗条目不再被关键词命中重复注入）。
 * @returns null = 无任何可注入内容（库无快照正文、项目全空且无写入条目）；调用方仍应
 *          清除 reinjectPending，避免每个用户消息轮空转重查。
 */
export function buildReinjection(
  db: MemoryDb,
  workspace: string,
  sessionId: string,
  projects: readonly string[],
  opts: Partial<InjectOptions> = {},
  dir = '.dsh-meow',
): { text: string; injectedIds: string[] } | null {
  const o = { ...DEFAULT_OPTS, ...opts }
  const snapshot = buildInjectionBody(db, o)
  const projectTexts: string[] = []
  const projectIds = new Set<string>()
  for (const project of projects) {
    const built = buildProjectSection(db, workspace, project, dir)
    if (built !== null) {
      projectTexts.push(built.text)
      for (const id of built.ids) projectIds.add(id)
    }
  }
  // 第三块：本会话写过的记忆（排除快照与项目全景已展示的 id，防同块重复）。
  const written = buildWrittenSection(db, workspace, sessionId, dir, new Set([...(snapshot?.injectedIds ?? []), ...projectIds]))
  if (snapshot === null && projectTexts.length === 0 && written === null) return null
  const lines: string[] = []
  if (snapshot !== null) {
    lines.push(snapshot.body)
    lines.push('')
  }
  if (projectTexts.length > 0 || written !== null) {
    lines.push(lbl('inject.sectionFormat', { label: lbl('inject.reinjectSection') }))
    lines.push(lbl('inject.reinjectIntro'))
    lines.push('')
    if (projectTexts.length > 0) {
      lines.push(projectTexts.join('\n\n'))
      lines.push('')
    }
  }
  if (written !== null) {
    lines.push(written.text)
    lines.push('')
  }
  const text = lines.join('\n').trimEnd()
  const replayedIds = [...(snapshot?.injectedIds ?? []), ...(written?.ids ?? [])]
  if (replayedIds.length > 0) markInjected(workspace, sessionId, replayedIds, dir)
  return { text, injectedIds: replayedIds }
}

/** 关键词命中查询（首轮与每条消息链路共用）：active 的 fact/lesson/rules/topic。
 *
 *  候选池 = 四层全量（只排除「最近注入窗口」内的），跨项目与本会话建立的记忆**降权而非剔除**：
 *  - 旧实现直接 `source_session !== sessionId` 排除 + 只认当前 project，两者叠加在长会话里
 *    会把候选池清零（本会话建立的记忆先被排除，剩下的又全在 injected 全集里），实测某
 *    600 轮会话 808 次判定 0 命中。降权（×0.7）同样能防「复读刚说过的话」，但不会掐断来源。
 *  - 范围：全局/当前项目为满权，跨项目 ×0.7（长会话里话题会漂移，硬切会丢相关来源）。
 *
 *  命中打分基于条目关键词（keywords，+title）而非全文——全文匹配噪音大
 *  （常见 bigram 如"的时"每条消息都有，长条目一碰词就整条注入）。但仅有词面还不够，
 *  必须再叠「覆盖率绝对门槛」（hitMinCoverage）：重叠一个词就入榜时，长文档靠通用词
 *  就能占满 top-K。
 *
 *  查询取用户消息全文（上限 MAX_QUERY_CHARS）而不是前 500 字：桥接/别名预设会在真实
 *  内容前拼大段模板，截断后真实诉求永远进不了匹配面。 */
function hitQuery(
  db: MemoryDb,
  workspace: string,
  sessionId: string,
  userText: string,
  o: InjectOptions,
  dir: string,
  /** v0.41.0：查询向量（调用方预先算好，32ms）。null = 走纯词面（降级路径）。 */
  queryVec: Float32Array | null = null,
): Array<{ id: string; level: string; content: string; updated_at: number | null }> {
  const seenState = readSeenFile(workspace, sessionId, dir)
  const currentProject = seenState.currentProject
  const recent = new Set(seenState.recentInjected)
  const hitRows = [
    ...db.list('fact', { status: 'active' }),
    ...db.list('lesson', { status: 'active' }),
    ...db.list('rules', { status: 'active' }),
    ...db.list('topic', { status: 'active' }),
  ].filter((r) => !recent.has(r.id))
  if (hitRows.length === 0) return []
  const inScope = new Map(
    hitRows.map((r) => [
      r.id,
      isGlobalScope(r.project) || (currentProject !== null && projectCovers(r.project, currentProject)),
    ]),
  )
  const isSelf = new Map(hitRows.map((r) => [r.id, r.source_session === sessionId]))
  // 查询侧先滤掉虚词/碎片 bigram（见 QUERY_STOP_TOKENS）：剩下的才是可能有语义权重的词。
  // 用空格重新拼接再喂给 keywordHitScore——顺带消掉跨词的 bigram 碎片（「这个文件」的「个文」）。
  const query = tokenize(userText.slice(0, MAX_QUERY_CHARS))
    .filter((t) => t.length >= 2 && !QUERY_STOP_TOKENS.has(t))
    .join(' ')
  const docs = toDocs(hitRows).map((d) => ({
    ...d,
    // 匹配面 = 条目关键词（LLM 提取或自动 bigram；无关键词的旧条目回退内容前 100 字）。
    content: d.keywords.length > 0 ? d.keywords.join(' ') : d.content.slice(0, 100),
  }))
  // k 传候选池全量：门槛过滤与降权都在本层做，先截断会丢掉本该进榜的条目。
  // maxDfRatio 0.08（2026-11-16 实测回滚）：0.05 在 250 条 active 上阈值是 12，
  // 「AutoDL」出现在 13 条记忆的关键词里 → 13 > 12 被切掉 → 「AutoDL 什么配置」一条都中不了。
  // 0.08 阈值是 20：AutoDL 13 < 20 放行，「视频」23 > 20 切掉，正好卡在「项目核心词」和「纯万金油」之间。
  const hits = keywordHitScore(query, docs, { k: hitRows.length, maxDfRatio: 0.08 })
  // 覆盖率门槛按条目长度分档：长条目要求命中比例高（防"通用词一碰就整条注入"），
  // 短条目放宽（分母太小，比例天然偏高/偏低）。hitMinCoverage=0 时整档关闭。
  // 长度用去重 token 数，与 bm25 里 coverage 的分母保持同一口径。
  const docLen = new Map(docs.map((d) => [d.id, new Set(tokenize([d.content, d.title ?? ''].join(' ')).filter((t) => t.length >= 2)).size]))
  const thresholdOf = (id: string): number => {
    if (o.hitMinCoverage <= 0) return 0
    const len = docLen.get(id) ?? 0
    return len > 0 && len <= SHORT_DOC_TOKENS ? Math.min(o.hitMinCoverage, SHORT_DOC_MIN_COVERAGE) : o.hitMinCoverage
  }
  // v0.37.0：除覆盖率门槛外再加「最少命中数」下限。measured：真实使用里 8 次错 6 次，
  // 其中所有误注入的命中数都是 1（「等视频渲染完再说」撞一个「渲染」就注入 4 条）。
  // 只撞上 1 个词就注入，是误注入的主要来源；宁可不注入，也别注入错的。
  //
  // v0.37.0：除覆盖率门槛外再加「最少命中数」下限。实测真实使用里 8 次错 6 次，其中所有
  // 误注入的命中数都是 1（「等视频渲染完再说」撞一个「渲染」就注入 4 条全无关的记忆）。
  // 只撞上 1 个词就注入，是误注入的主要来源；宁可不注入，也别注入错的。
  //
  // 例外：单命中但命中的是**专有名词级 token**（含数字或拉丁字母）时放行。
  // 「3081 又怎么了？」过滤后只剩「3081」，「AutoDL 那台什么配置」的目标记忆只含「autodl」
  // ——这类词的指向性极强，命中一个就是强信号；而「渲染」「猜谜」这类普通中文词撞上不能算。
  // （试过「查询有效词 <=2 就放宽」，负样本掉到 5/10，因为大多数短查询都符合；按命中词性质
  //  判断才准。）
  const minHc = MIN_HIT_COUNT
  // ── v0.41.0 混合召回 ──
  // 词面命中 ∪ 向量命中，按 0.45×词面 + 0.55×向量 融合。词面命中要求 hc/专有名词/IDF 的
  // 那套门槛只保留在**降级路径**（没有向量时），混合模式下由总分门槛统一裁决——
  // 否则「字面不重合但语义很近」的条目会被 hc>=2 一票否决，向量就白算了。
  let vecSim: Map<string, number> | null = null
  if (queryVec !== null && o.hybrid !== false && !embeddingUnavailable()) {
    const stored = db.loadEmbeddings(embeddingModelId())
    if (stored.size > 0) {
      vecSim = new Map()
      for (const [id, v] of stored) vecSim.set(id, cosine(queryVec, v))
    }
  }
  const hybrid = vecSim !== null
  /** 向量余弦映射到 0-1：BGE 中文实测相关区间约 0.45-0.85，0.30 起步、0.75 封顶。 */
  const vecNorm = (sim: number): number => Math.max(0, Math.min(1, (sim - 0.30) / 0.45))
  const scoreOf = (h: { id: string; coverage?: number }): number => {
    const lex = h.coverage ?? 0
    if (!hybrid) return lex
    return HYBRID_W_LEX * lex + HYBRID_W_VEC * vecNorm(vecSim.get(h.id) ?? 0)
  }
  const rowById = new Map(hitRows.map((r) => [r.id, r]))
  const pool = new Map<string, (typeof hits)[number]>(hits.map((h) => [h.id, h]))
  if (hybrid) {
    // 词面完全没撞上、但语义够近的条目也要进候选池。
    for (const [id, sim] of vecSim) {
      if (!pool.has(id) && rowById.has(id) && sim >= HYBRID_VEC_MIN_SIM) {
        const r = rowById.get(id)!
        pool.set(id, { id, level: r.level, content: '', score: 0, coverage: 0, hitCount: 0 } as (typeof hits)[number])
      }
    }
  }
  const kept = [...pool.values()]
    .map((h) => ({ ...h, hybridScore: scoreOf(h) }))
    .filter((h) => {
      if (hybrid) return h.hybridScore >= HYBRID_MIN_SCORE
      if ((h.coverage ?? 0) < thresholdOf(h.id)) return false
      return (h.hitCount ?? 0) >= minHc || h.properHit === true || (h.idfSum ?? 0) >= MIN_IDF_SUM
    })
  // 命中诊断（需求方要求：没命中也要打，带最高 coverage + 被拒原因，否则只能瞎猜）。
  if (o.hitDebug && hitDebugLogger !== null) {
    const qTokens = [...new Set(tokenize(query).filter((t) => t.length >= 2))]
    const top = hits.length > 0 ? Math.max(...hits.map((h) => h.coverage ?? 0)) : 0
    const rejected = hits.filter((h) => !kept.includes(h)).slice(0, 3)
    const rejStr = rejected.map((h) => `${h.id.slice(0, 9)}(cov=${(h.coverage ?? 0).toFixed(2)},hc=${h.hitCount ?? 0})`).join(' ')
    if (kept.length === 0) {
      hitDebugLogger(`meow-memory: hit: 0 条（query tokens=${qTokens.length}，候选=${hitRows.length}，最高 coverage=${top.toFixed(2)} < ${o.hitMinCoverage}）`)
    } else {
      hitDebugLogger(`meow-memory: hit: ${kept.length} 条（query tokens=${qTokens.length}，候选=${hitRows.length}） → ${kept.map((h) => `${h.id.slice(0, 9)}(cov=${(h.coverage ?? 0).toFixed(2)},hc=${h.hitCount ?? 0})`).join(', ')}`)
    }
    hitDebugLogger(`meow-memory:   query: ${qTokens.join(' ')}`)
    if (rejStr.length > 0) hitDebugLogger(`meow-memory:   被拒(门槛 ${o.hitMinCoverage}/hc>=${minHc}): ${rejStr}`)
  }
  const byId = new Map(hitRows.map((r) => [r.id, r]))
  // 展示/注入用原文 content（docs 的 content 只是 keywords 匹配面，不能当正文）。
  return kept
    .map((h) => ({ h, weight: (inScope.get(h.id) ? 1 : 0.7) * (isSelf.get(h.id) ? 0.7 : 1) }))
    .sort((a, b) => (b.h.hybridScore ?? b.h.score) * b.weight - (a.h.hybridScore ?? a.h.score) * a.weight)
    .slice(0, Math.max(0, o.hitTopK))
    .map(({ h }) => {
      const row = byId.get(h.id)
      return { id: h.id, level: h.level, content: row?.content ?? h.content, project: row?.project ?? null, updated_at: row?.updated_at ?? null }
    })
}

/**
 * 每条用户消息的关键词命中注入（独立于首轮注入的链路）：
 * 检索 active 的 fact/lesson/rules/topic（全局+当前锚定项目），top-K 命中注入，
 * 命中 id 记入本会话已见（之后不再命中；压缩释放 seen 后恢复）。
 * 格式：顶格「可能相关的记忆，仅供参考：」→ 条目（- [id] 换行接内容、
 * 条目间空行）。返回文本作为独立 plugin snapshot 消息的正文。
 * @returns 命中注入块；无命中返回 null。
 */
export function buildHitInjection(
  db: MemoryDb,
  workspace: string,
  sessionId: string,
  userText: string,
  opts: Partial<InjectOptions> = {},
  dir = '.dsh-meow',
  /** v0.41.0：调用方（async）预算好的查询向量；null/省略 = 纯词面。 */
  queryVec: Float32Array | null = null,
): { text: string; injectedIds: string[] } | null {
  const o = { ...DEFAULT_OPTS, ...opts }
  const fresh = hitQuery(db, workspace, sessionId, userText, o, dir, queryVec)
  if (fresh.length === 0) return null
  const lines = [keyedValue('labels', 'inject.hitHeader')]
  for (const h of fresh) {
    // 原文视图：归属 + 完整 id + 绝对/相对时间戳（记忆时间戳=updated_at 最后更新时间；无时间戳不显示），第二行完整内容。
    const proj = projectLabel(h.project)
    const time = h.updated_at
      ? ` ${new Date(h.updated_at).toISOString().slice(0, 16).replace('T', ' ')} [${relativeTime(h.updated_at)}]`
      : ''
    lines.push(`[${proj} : ${h.level}] [${h.id}]${time}`)
    lines.push(h.content)
    lines.push('')
  }
  const text = lines.join('\n').trimEnd()
  const ids = fresh.map((h) => h.id)
  markInjected(workspace, sessionId, ids, dir)
  // #91：注入=使用。自动注入是记忆进入上下文的主路径，此前只有 memory_search
  // 计数——hit_count/last_accessed_at 的冷热信号接在错误事件上（被搜过≠被用过，
  // 编辑≠使用）。首轮快照注入刻意不计：全量一过没有区分度，只会淹没信号。
  for (const h of fresh) db.bumpHit(h.level as Parameters<MemoryDb['bumpHit']>[0], h.id)
  return { text, injectedIds: ids }
}

// 导出 tokenize 供索引/测试复用
export { tokenize }
