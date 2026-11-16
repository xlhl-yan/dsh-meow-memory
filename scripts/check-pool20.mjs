/** 池子 20 能不能捞回「转场中间的片段」（纯向量排第 11）？ */
import { resolve } from 'node:path';
import { env, pipeline } from '@xenova/transformers';
import { getDb, buildHitInjection, embed, embeddingModelId, rerankScores, warmupRerank } from '../lib/index.js';

const WS = 'D:/AgentTools/qq-bridge/state/agents';
env.allowRemoteModels = false;
env.localModelPath = resolve('models/');
await pipeline('feature-extraction', embeddingModelId(), { quantized: true });
await warmupRerank();
const db = getDb(WS, '.dsh-meow');

const q = '换场那一下最费劲';
const qv = await embed(q);
const sid = `p20-${Date.now()}`;
const poolOpts = { hitTopK: 20, hitMinCoverage: 0.12, hybridMinScore: 0.15 };
const wide = buildHitInjection(db, WS, sid, q, poolOpts, '.dsh-meow', qv, true);
const c = wide?.candidates ?? [];
console.log(`池子 ${c.length} 条，精排中...\n`);
const t0 = Date.now();
const scores = await rerankScores(q, c.map((x) => x.content));
console.log(`精排耗时 ${Date.now() - t0}ms（${Math.round((Date.now() - t0) / c.length)}ms/条）\n`);
const ranked = c.map((x, i) => ({ x, s: scores?.[i] ?? -99 })).sort((a, b) => b.s - a.s);
ranked.forEach((r, i) => {
  const t = (r.x.content ?? '').replace(/\n/g, ' ');
  const mark = t.includes('转场中间') ? ' ⭐ 就是它' : '';
  console.log(`  ${String(i + 1).padStart(2)}. ${r.s.toFixed(2).padStart(7)}  ${t.slice(0, 46)}${mark}`);
});
const target = ranked.findIndex((r) => (r.x.content ?? '').includes('转场中间'));
console.log(`\n目标记忆「转场中间的片段」排在精排第 ${target + 1} 名（共 ${ranked.length} 条）`);
db.close();
