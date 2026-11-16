/**
 * base 模型的混合召回参数扫描（v0.42 校准）。
 * 复现 inject.ts 的打分逻辑，扫 vecNorm 区间 + MIN_SCORE，
 * 看 15 正 / 15 负 / 全量自命中 三条曲线。
 */
import { resolve } from 'node:path';
import { DatabaseSync } from 'node:sqlite';
import { env, pipeline } from '@xenova/transformers';
import { embeddingText, embeddingModelId } from '../lib/index.js';

const DB = 'D:/AgentTools/qq-bridge/state/agents/.dsh-meow/memory.db';
const MODEL = embeddingModelId();

// ── 打分参数（与 inject.ts 保持一致）──
const W_LEX = 0.45, W_VEC = 0.55;
const MAX_DF_RATIO = 0.08;
const MIN_HIT_COUNT = 2, MIN_IDF_SUM = 2.0, MIN_COVERAGE = 0.12;
const SHORT_DOC_TOKENS = 10, SHORT_DOC_MIN_COVERAGE = 0.15;
const STOP = new Set(['怎么', '么做', '么样', '么配', '什么', '为什', '如何', '哪些', '是不', '不是',
  '这个', '那个', '个文', '一个', '一下', '可以', '应该', '帮我', '我看', '看看',
  '等于', '现在', '今天', '明天', '昨天', '想要', '需要', '希望', '这样', '那样',
  '就是', '还是', '但是', '因为', '所以', '如果', '然后', '已经', '没有', '一些',
  '这里', '那里', '你的', '我的', '他的', '自己', '是什', '么意', '意思', '几点',
  '好的', '吃饭', '睡觉', '知道', '收到', '谢谢', '再见', '拜拜', '晚安', '早安',
  '累了', '休息', '忙吗', '干嘛', '去哪']);

function tokenize(text) {
  const out = [];
  const re = /([\u4e00-\u9fff\u3040-\u30ff]+)|((?:[^\u4e00-\u9fff\u3040-\u30ff\s])+)/g;
  let m;
  while ((m = re.exec(text)) !== null) {
    if (m[1]) {
      const s = m[1];
      for (let i = 0; i + 1 < s.length; i++) out.push(s.slice(i, i + 2));
      if (s.length === 1) out.push(s);
    } else if (m[2]) {
      const w = m[2].toLowerCase().replace(/[^\p{L}\p{N}]/gu, '');
      if (w.length >= 2) out.push(w);
    }
  }
  return out;
}

// ── 读库 ──
const dbRaw = new DatabaseSync(DB);
const rows = [];
for (const t of ['fact', 'lesson', 'rules', 'topic']) {
  try {
    rows.push(...dbRaw.prepare(`SELECT id, title, content, keywords FROM ${t} WHERE status='active'`).all().map((x) => ({ ...x, level: t })));
  } catch { /* skip */ }
}
const embRows = dbRaw.prepare('SELECT id, vec FROM embeddings WHERE model = ?').all(MODEL);
dbRaw.close();
const vecs = new Map();
for (const e of embRows) {
  const buf = e.vec;
  const copy = new Uint8Array(buf.byteLength);
  copy.set(buf);
  vecs.set(e.id, new Float32Array(copy.buffer, 0, Math.floor(copy.byteLength / 4)));
}
console.log(`库 ${rows.length} 条，base 向量 ${vecs.size} 条\n`);

// ── 词面索引（匹配面 = keywords，无则正文前 100）──
const docs = rows.map((r) => {
  let kw = [];
  try { kw = JSON.parse(r.keywords ?? '[]'); } catch { /* */ }
  if (!Array.isArray(kw)) kw = [];
  const surface = kw.length > 0 ? kw.join(' ') : (r.content ?? '').slice(0, 100);
  return { ...r, kw, toks: new Set(tokenize([surface, r.title ?? ''].join(' ')).filter((t) => t.length >= 2)) };
});
const N = docs.length;
const df = new Map();
for (const d of docs) for (const t of d.toks) df.set(t, (df.get(t) || 0) + 1);
const idf = (t) => Math.log(1 + (N - (df.get(t) ?? 0) + 0.5) / ((df.get(t) ?? 0) + 0.5));
const maxDf = Math.max(8, Math.floor(N * MAX_DF_RATIO));
const docLen = new Map(docs.map((d) => [d.id, new Set(tokenize([d.content ?? '', d.title ?? ''].join(' ')).filter((t) => t.length >= 2)).size]));

function lexScores(q) {
  const raw = [...new Set(tokenize(q).filter((t) => t.length >= 2 && !STOP.has(t)))];
  const qEff = raw.filter((t) => { const d = df.get(t) ?? 0; return d > 0 && d <= maxDf; });
  const denomW = qEff.reduce((s, t) => s + idf(t), 0) || 1;
  const out = new Map();
  for (const d of docs) {
    const hit = qEff.filter((t) => d.toks.has(t));
    if (hit.length === 0) continue;
    out.set(d.id, {
      cov: hit.reduce((s, t) => s + idf(t), 0) / denomW,
      hc: hit.length,
      proper: hit.some((t) => /[0-9]/.test(t) || /[a-z]/i.test(t)),
      idfSum: hit.reduce((s, t) => s + idf(t), 0),
    });
  }
  return out;
}

// ── 向量 ──
env.allowRemoteModels = false;
env.localModelPath = resolve('models/');
const ex = await pipeline('feature-extraction', MODEL, { quantized: true });
const embedRaw = async (t) => Float32Array.from((await ex(String(t).slice(0, 500), { pooling: 'mean', normalize: true })).data);
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };

const POS = ['甩镜怎么做', 'Qwen 用英文还是中文写 prompt', '三张参考图能不能在本地跑', 'AutoDL 那台什么配置', '画风最后定成什么了', '姓名条被角色挡住了怎么办', '群像换人怎么分配角色', '本地 6G 卡能跑多高分辨率', '豆包换人为什么脸崩', '首尾帧字幕为什么会自己变', '怎么让生成的角色动起来', '挖矿木马怎么清的', '转场为什么要拆成两段', '双卡 32G 显存能合并吗', '贴文字生成图效果怎么样'];
const NEG = ['今天天气不错', '我有点困了', '晚饭吃什么', '你在干嘛', '哈哈', '早点睡', '我出去一趟', '嗯嗯', '好的', '几点了', '帮我倒杯水', '今天好累', '周末有安排吗', '你吃饭了吗', '我在忙'];
const HARD = ['那个烤显卡的家伙还能租吗', '两个角色谁站前面谁站后面', '怎么让画面别那么肉', '换场那一下最费劲', '画出来的人不像'];

const qvCache = new Map();
for (const q of [...POS, ...NEG, ...HARD]) qvCache.set(q, await embedRaw(q));

function makeQuery(c) {
  let s = c.replace(/^\d{4}-\d{2}-\d{2}\s*/, '').replace(/^[\s*#>\-—·]+/, '').replace(/\*\*/g, '').replace(/`[^`]*`/g, '').replace(/[（(][^）)]*[）)]/g, '').trim();
  const m = s.match(/^[^。？！\n]{6,40}/);
  return m ? m[0].trim() : s.slice(0, 30);
}
const SELF = docs.map((d) => ({ id: d.id, q: makeQuery(d.content ?? '') })).filter((x) => x.q.length >= 4);
// 自命中也要查询向量：一次性算完（base 慢，约 160s，只算一次）
for (const s of SELF) if (!qvCache.has(s.q)) qvCache.set(s.q, await embedRaw(s.q));
console.log(`预计算查询向量 ${qvCache.size} 条`);

function evaluate(vecLo, vecHi, minScore) {
  const vecNorm = (s) => Math.max(0, Math.min(1, (s - vecLo) / (vecHi - vecLo)));
  const run = (q, topK = 5) => {
    const lex = lexScores(q);
    const qv = qvCache.get(q);
    const pool = new Map();
    for (const [id, v] of lex) pool.set(id, { cov: v.cov, vec: vecNorm(dot(qv, vecs.get(id) ?? new Float32Array(1))), hc: v.hc, proper: v.proper, idfSum: v.idfSum });
    for (const d of docs) {
      if (pool.has(d.id)) continue;
      const sim = vecs.has(d.id) ? dot(qv, vecs.get(d.id)) : 0;
      if (sim >= Math.max(vecHi, 0.45)) pool.set(d.id, { cov: 0, vec: vecNorm(sim), hc: 0, proper: false, idfSum: 0 });
    }
    const kept = [];
    for (const [id, p] of pool) {
      const score = W_LEX * p.cov + W_VEC * p.vec;
      if (score < minScore) continue;
      const thr = (docLen.get(id) ?? 0) <= SHORT_DOC_TOKENS ? Math.min(MIN_COVERAGE, SHORT_DOC_MIN_COVERAGE) : MIN_COVERAGE;
      if (p.cov > 0 && p.cov < thr && p.hc < MIN_HIT_COUNT && !p.proper && p.idfSum < MIN_IDF_SUM) continue;
      kept.push({ id, score });
    }
    kept.sort((a, b) => b.score - a.score);
    return kept.slice(0, topK).map((k) => k.id);
  };
  const posHit = POS.filter((q) => run(q).length > 0).length;
  const negBad = NEG.filter((q) => run(q).length > 0).length;
  const hardHits = HARD.map((q) => run(q).length).join('/');
  let selfHit = 0;
  for (const s of SELF) if (run(s.q, 5).includes(s.id)) selfHit++;
  return { posHit, negBad, hardHits, selfHit, selfTotal: SELF.length };
}

console.log('vecLo/vecHi/minScore → 正样本 | 负样本误命中 | 难句命中数 | 自命中');
for (const [lo, hi] of [[0.30, 0.75], [0.25, 0.65], [0.25, 0.60], [0.30, 0.60], [0.20, 0.60], [0.25, 0.55]]) {
  for (const ms of [0.30, 0.35, 0.40, 0.45, 0.50]) {
    const r = evaluate(lo, hi, ms);
    console.log(`${lo}/${hi}/${ms}  →  ${r.posHit}/15 | ${r.negBad}/15 | ${r.hardHits} | ${r.selfHit}/${r.selfTotal} = ${(r.selfHit / r.selfTotal * 100).toFixed(1)}%`);
  }
}
