/** 核实宽池子到底多大：hybridMinScore 0.15 + hitTopK 20。 */
import { resolve } from 'node:path';
import { env, pipeline } from '@xenova/transformers';
import { getDb, buildHitInjection, embed, embeddingModelId } from '../lib/index.js';

const WS = 'D:/AgentTools/qq-bridge/state/agents';
const HARD = ['那个烤显卡的家伙还能租吗', '两个角色谁站前面谁站后面', '怎么让画面别那么肉', '换场那一下最费劲', '画出来的人不像'];

env.allowRemoteModels = false;
env.localModelPath = resolve('models/');
await pipeline('feature-extraction', embeddingModelId(), { quantized: true });
const db = getDb(WS, '.dsh-meow');

console.log('查询 → 宽池子(0.15/20条) | 正常池子(0.40)');
for (const q of HARD) {
  const qv = await embed(q);
  const sid = `p-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
  const wide = buildHitInjection(db, WS, sid, q, { hitTopK: 20, hitMinCoverage: 0.12, hybridMinScore: 0.15 }, '.dsh-meow', qv);
  const normal = buildHitInjection(db, WS, sid + 'b', q, { hitTopK: 5, hitMinCoverage: 0.12 }, '.dsh-meow', qv);
  console.log(`「${q}」  ${wide?.candidates.length ?? 0} 条  |  ${normal?.candidates.length ?? 0} 条  (topScore=${(normal?.topScore ?? 0).toFixed(2)})`);
}
db.close();
