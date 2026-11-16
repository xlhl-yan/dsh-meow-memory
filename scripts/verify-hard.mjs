/**
 * 难句测试集（完全避开词面）：口语化转述 vs 技术术语。
 * 用法：node scripts/verify-hard.mjs [--vector-only]
 *   --vector-only  只看纯向量 top5（跳过门槛，衡量模型真实能力）
 */
import { resolve } from 'node:path';
import { env, pipeline } from '@xenova/transformers';
import { getDb, buildHitInjection, embed, embeddingModelId } from '../lib/index.js';

const WS = 'D:/AgentTools/qq-bridge/state/agents';
const vectorOnly = process.argv.includes('--vector-only');

const HARD = [
  ['那个烤显卡的家伙还能租吗', 'AutoDL / 租 GPU'],
  ['两个角色谁站前面谁站后面', '群像 / 构图分配'],
  ['怎么让画面别那么肉', '分辨率 / 清晰度'],
  ['换场那一下最费劲', '转场'],
  ['画出来的人不像', '人设 / 参考图'],
];

env.allowRemoteModels = false;
env.localModelPath = resolve('models/');
const model = embeddingModelId();
const ex = await pipeline('feature-extraction', model, { quantized: true });
const embedRaw = async (t) => Float32Array.from((await ex(t.slice(0, 500), { pooling: 'mean', normalize: true })).data);
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };

const db = getDb(WS, '.dsh-meow');
console.log(`模型: ${model} ｜ 模式: ${vectorOnly ? '纯向量 top5（无门槛）' : '完整链路'}\n`);

if (vectorOnly) {
  const rows = ['fact', 'lesson', 'rules', 'topic'].flatMap((l) => db.list(l, { status: 'active' }));
  const vecs = db.loadEmbeddings(model);
  for (const [q, expect] of HARD) {
    const qv = await embedRaw(q);
    const top = rows.filter((r) => vecs.has(r.id))
      .map((r) => ({ r, sim: dot(qv, vecs.get(r.id)) }))
      .sort((a, b) => b.sim - a.sim).slice(0, 5);
    console.log(`════ 「${q}」  期望≈${expect}`);
    for (const { r, sim } of top) {
      console.log(`  ${sim.toFixed(3)} [${r.level}] ${(r.content || '').replace(/\n/g, ' ').slice(0, 56)}`);
    }
    console.log('');
  }
} else {
  for (const [q, expect] of HARD) {
    const qv = await embed(q);
    const sid = `hard-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    const r = buildHitInjection(db, WS, sid, q, { hitTopK: 5, hitMinCoverage: 0.12 }, '.dsh-meow', qv);
    console.log(`════ 「${q}」  期望≈${expect}`);
    if (r === null) { console.log('  （无命中）\n'); continue; }
    for (const id of r.injectedIds) {
      const f = db.findById(id);
      console.log(`  [${f?.row.level ?? '?'}] ${(f?.row.content ?? '').replace(/\n/g, ' ').slice(0, 56)}`);
    }
    console.log('');
  }
}
db.close();
