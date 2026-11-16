/**
 * v0.43 完整链路验证：混合召回 → （低置信时）rerank 精排。
 * 复刻 index.ts 的 buildHitWithRerank，用来在插件之外量效果和延迟。
 * 用法：node scripts/verify-rerank.mjs
 */
import { resolve } from 'node:path';
import { env, pipeline } from '@xenova/transformers';
import {
  getDb, buildHitInjection, embed, embeddingModelId,
  rerankScores, rerankUnavailable, warmupRerank,
} from '../lib/index.js';

const RERANK_POOL = 10;

const WS = 'D:/AgentTools/qq-bridge/state/agents';
const DIR = '.dsh-meow';
const RERANK_TRIGGER = 0.75;

const HARD = [
  ['那个烤显卡的家伙还能租吗', 'AutoDL / 租 GPU'],
  ['两个角色谁站前面谁站后面', '群像 / 构图'],
  ['怎么让画面别那么肉', '分辨率 / 清晰度'],
  ['换场那一下最费劲', '转场'],
  ['画出来的人不像', '人设 / 参考图'],
];
const POS = ['甩镜怎么做', 'Qwen 用英文还是中文写 prompt', '三张参考图能不能在本地跑', 'AutoDL 那台什么配置', '画风最后定成什么了', '姓名条被角色挡住了怎么办', '群像换人怎么分配角色', '本地 6G 卡能跑多高分辨率', '豆包换人为什么脸崩', '首尾帧字幕为什么会自己变', '怎么让生成的角色动起来', '挖矿木马怎么清的', '转场为什么要拆成两段', '双卡 32G 显存能合并吗', '贴文字生成图效果怎么样'];
const NEG = ['今天天气不错', '我有点困了', '晚饭吃什么', '你在干嘛', '哈哈', '早点睡', '我出去一趟', '嗯嗯', '好的', '几点了', '帮我倒杯水', '今天好累', '周末有安排吗', '你吃饭了吗', '我在忙'];

env.allowRemoteModels = false;
env.localModelPath = resolve('models/');
await pipeline('feature-extraction', embeddingModelId(), { quantized: true });
// 预热 rerank：首次加载要 1.3 秒（量化）/ 2.6 秒（fp32），不预热的话第一个查询的
// 延迟里混了加载时间，量出来的数字是假的。
const tWarm = Date.now();
await warmupRerank();
console.log(`rerank 模型预热 ${Date.now() - tWarm}ms\n`);
const db = getDb(WS, DIR);

async function hitWithRerank(q, topK = 5) {
  const qv = await embed(q);
  const sid = `vr-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  // 与 index.ts 的 buildHitWithRerank 同一套逻辑：先用正常门槛探置信度，
  // 低置信时用低门槛重召回一个大池子再精排。
  const probe = buildHitInjection(db, WS, sid, q, { hitTopK: topK, hitMinCoverage: 0.12 }, DIR, qv);
  if (probe === null) return { ids: [], reranked: false, ms: 0, topScore: 0, pool: 0 };
  if (probe.topScore >= RERANK_TRIGGER || rerankUnavailable()) {
    return { ids: probe.injectedIds, reranked: false, ms: 0, topScore: probe.topScore, pool: probe.candidates.length };
  }
  const wide = buildHitInjection(db, WS, sid, q,
    { hitTopK: RERANK_POOL, hitMinCoverage: 0.12, hybridMinScore: 0.15 }, DIR, qv);
  if (wide === null || wide.candidates.length <= 1) {
    return { ids: probe.injectedIds, reranked: false, ms: 0, topScore: probe.topScore, pool: probe.candidates.length };
  }
  const t0 = Date.now();
  const scores = await rerankScores(q, wide.candidates.map((c) => c.content));
  const ms = Date.now() - t0;
  if (scores === null) return { ids: probe.injectedIds, reranked: false, ms, topScore: probe.topScore, pool: wide.candidates.length };
  const reranked = wide.candidates.map((c, i) => ({ c, s: scores[i] ?? -99 }))
    .sort((a, b) => b.s - a.s).slice(0, topK).map((x) => x.c);
  return { ids: reranked.map((c) => c.id), reranked: true, ms, topScore: probe.topScore, pool: wide.candidates.length };
}

console.log('=== 难句（完整链路）===');
let hardMs = 0, hardCount = 0;
for (const [q, expect] of HARD) {
  const r = await hitWithRerank(q);
  hardMs += r.ms; if (r.reranked) hardCount++;
  console.log(`\n════ 「${q}」  期望≈${expect}`);
  console.log(`  topScore=${r.topScore.toFixed(2)}  ${r.reranked ? `已精排 ${r.ms}ms` : '未触发精排'}  命中 ${r.ids.length} 条`);
  for (const id of r.ids) {
    const row = db.findById(id)?.row;
    console.log(`    ${(row?.content ?? '').replace(/\n/g, ' ').slice(0, 56)}`);
  }
}
console.log(`\n难句触发精排 ${hardCount}/${HARD.length} 句，累计 ${hardMs}ms`);

let posHit = 0, posRerank = 0;
for (const q of POS) { const r = await hitWithRerank(q); if (r.ids.length > 0) posHit++; if (r.reranked) posRerank++; }
let negBad = 0, negRerank = 0;
for (const q of NEG) { const r = await hitWithRerank(q); if (r.ids.length > 0) negBad++; if (r.reranked) negRerank++; }
console.log(`正样本 ${posHit}/15（其中触发精排 ${posRerank} 次）`);
console.log(`负样本误命中 ${negBad}/15（其中触发精排 ${negRerank} 次）`);
db.close();
