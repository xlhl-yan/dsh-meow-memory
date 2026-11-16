/**
 * v0.44 伪相关反馈验证：对比 无扩展 vs 有扩展 的池子和最终结果。
 * 用法：node scripts/verify-prf.mjs
 */
import { resolve } from 'node:path';
import { env, pipeline } from '@xenova/transformers';
import {
  getDb, buildHitInjection, embed, embeddingModelId,
  rerankScores, rerankUnavailable, warmupRerank,
} from '../lib/index.js';

const WS = 'D:/AgentTools/qq-bridge/state/agents';
const DIR = '.dsh-meow';
const RERANK_POOL = 10;
const HARD = [
  ['那个烤显卡的家伙还能租吗', 'AutoDL / 租 GPU'],
  ['两个角色谁站前面谁站后面', '群像 / 构图'],
  ['怎么让画面别那么肉', '分辨率 / 清晰度'],
  ['换场那一下最费劲', '转场'],
  ['画出来的人不像', '人设 / 参考图'],
];

// ── 复刻 src/prf.ts 的抽取逻辑（保持同步）──
const PRF_STOP = new Set(['记忆', '插件', '注入', '命中', '召回', '机制', '问题', '原因', '办法', '方法',
  '结论', '教训', '判断', '实测', '测试', '验证', '记录', '说明', '注意', '要点',
  '用户', '他说', '要求', '决策', '设计', '实现', '功能', '参数', '配置',
  '第一次', '第二次', '最后', '之前', '之后', '现在', '当时', '一下', '一个',
  '这个', '那个', '就是', '不是', '没有', '已经', '还是', '但是', '因为']);

function extractTerms(candidates, query) {
  if (candidates.length === 0) return [];
  const q = query.toLowerCase();
  const freq = new Map();
  for (const c of candidates) {
    const seen = new Set();
    for (const raw of c.keywords ?? []) { const term = String(raw).trim(); const latin = term.match(/[A-Za-z][A-Za-z0-9._-]{1,19}|[0-9]{2,4}x[0-9]{2,4}/g) ?? []; for (const t of latin) { if (t.length<2||t.length>20||PRF_STOP.has(t)||q.includes(t.toLowerCase())) continue; freq.set(t, (freq.get(t) ?? 0) + 2); } if (term.length <= 8 && !/[，。；、]/.test(term) && !PRF_STOP.has(term) && !q.includes(term.toLowerCase())) freq.set(term, (freq.get(term) ?? 0) + 1);
    }
  }
  return [...freq.entries()].sort((a, b) => b[1] - a[1]).slice(0, 5).map(([t]) => t);
}

env.allowRemoteModels = false;
env.localModelPath = resolve('models/');
await pipeline('feature-extraction', embeddingModelId(), { quantized: true });
await warmupRerank();
const db = getDb(WS, DIR);

async function run(q, usePrf) {
  const qv = await embed(q);
  const sid = `prf-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
  const poolOpts = { hitTopK: RERANK_POOL, hitMinCoverage: 0.12, hybridMinScore: 0.15 };

  let wide = buildHitInjection(db, WS, sid, q, poolOpts, DIR, qv, true);
  if (wide === null) return { ids: [], terms: [], poolBefore: 0, poolAfter: 0 };
  const poolBefore = wide.candidates.length;
  let terms = [];
  if (usePrf) {
    terms = extractTerms(wide.candidates, q);
    if (terms.length > 0) {
      const exp = `${q} ${terms.join(' ')}`;
      const ev = await embed(exp).catch(() => null);
      const more = buildHitInjection(db, WS, sid, exp, poolOpts, DIR, ev, true);
      if (more !== null) {
        const m = new Map(wide.candidates.map((c) => [c.id, c]));
        for (const c of more.candidates) if (!m.has(c.id)) m.set(c.id, c);
        wide = { ...wide, candidates: [...m.values()] };
      }
    }
  }
  const poolAfter = wide.candidates.length;
  const scores = await rerankScores(q, wide.candidates.map((c) => c.content));
  const ranked = wide.candidates.map((c, i) => ({ c, s: scores?.[i] ?? -99 }))
    .sort((a, b) => b.s - a.s).slice(0, 5).map((x) => x.c);
  return { ids: ranked.map((c) => c.id), rows: ranked, terms, poolBefore, poolAfter };
}

for (const [q, expect] of HARD) {
  const off = await run(q, false);
  const on = await run(q, true);
  console.log(`\n════ 「${q}」  期望≈${expect}`);
  console.log(`  无扩展: 池 ${off.poolBefore} 条`);
  console.log(`  有扩展: 池 ${off.poolBefore} → ${on.poolAfter} 条   扩展词=[${on.terms.join(', ') || '无'}]`);
  console.log(`  ── 有扩展后的 top5 ──`);
  for (const c of on.rows ?? []) {
    console.log(`     ${(c.content ?? '').replace(/\n/g, ' ').slice(0, 52)}`);
  }
}
db.close();
