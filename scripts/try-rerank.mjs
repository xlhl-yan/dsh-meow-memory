/**
 * B 方案验证：混合召回大池子 → bge-reranker 精排 → top-K。
 * 用法：node scripts/try-rerank.mjs [--topn 20]
 */
import { resolve } from 'node:path';
import { env, pipeline, AutoTokenizer, AutoModelForSequenceClassification } from '@xenova/transformers';
import { getDb, buildHitInjection, embed, embeddingModelId } from '../lib/index.js';

const WS = 'D:/AgentTools/qq-bridge/state/agents';
const argN = process.argv.indexOf('--topn');
const TOPN = argN > 0 ? Number(process.argv[argN + 1]) : 20;

const HARD = [
  ['那个烤显卡的家伙还能租吗', 'AutoDL / 租 GPU'],
  ['两个角色谁站前面谁站后面', '群像 / 构图'],
  ['怎么让画面别那么肉', '分辨率 / 清晰度'],
  ['换场那一下最费劲', '转场'],
  ['画出来的人不像', '人设 / 参考图'],
];

env.allowRemoteModels = false;
env.localModelPath = resolve('models/');
const embModel = embeddingModelId();
await pipeline('feature-extraction', embModel, { quantized: true });

// cross-encoder：必须直接拿原始 logits（text-classification pipeline 会把单 logit
// softmax 成 1，分数全一样，排序失效）。分数越大（越接近 0）越相关。
const tok = await AutoTokenizer.from_pretrained('Xenova/bge-reranker-base');
const rrModel = await AutoModelForSequenceClassification.from_pretrained('Xenova/bge-reranker-base', { quantized: false });

const db = getDb(WS, '.dsh-meow');

async function rerankScore(q, d) {
  const inputs = tok([q], { text_pair: [d], padding: true, truncation: true });
  const { logits } = await rrModel(inputs);
  return logits.data[0];
}

console.log(`embedding: ${embModel} | rerank: bge-reranker-base(fp32) | 召回池 top${TOPN}\n`);

const tAll = Date.now();
for (const [q, expect] of HARD) {
  const qv = await embed(q);
  const sid = `rr-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  const r = buildHitInjection(db, WS, sid, q, { hitTopK: TOPN, hitMinCoverage: 0.12 }, '.dsh-meow', qv);
  const ids = r === null ? [] : r.injectedIds;
  if (ids.length === 0) { console.log(`════ 「${q}」  召回池为空\n`); continue; }

  const t0 = Date.now();
  const scored = [];
  for (const id of ids) {
    const row = db.findById(id)?.row;
    if (row === undefined) continue;
    const docText = (row.content ?? '').replace(/\s+/g, ' ').slice(0, 400);
    scored.push({ id, row, score: await rerankScore(q, docText) });
  }
  const ms = Date.now() - t0;
  scored.sort((a, b) => b.score - a.score);

  console.log(`════ 「${q}」  期望≈${expect}  池 ${ids.length} 条 / rerank ${ms}ms (${Math.round(ms / ids.length)}ms/条)`);
  for (const s of scored.slice(0, 5)) {
    console.log(`  ${s.score.toFixed(2).padStart(7)} [${s.row.level}] ${(s.row.content ?? '').replace(/\n/g, ' ').slice(0, 52)}`);
  }
  console.log('');
}
console.log(`总耗时 ${((Date.now() - tAll) / 1000).toFixed(1)}s（5 句）`);
db.close();
