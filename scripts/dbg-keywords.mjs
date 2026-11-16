/** 调试：候选的 keywords 到底有没有传出来。 */
import { resolve } from 'node:path';
import { env, pipeline } from '@xenova/transformers';
import { getDb, buildHitInjection, embed, embeddingModelId } from '../lib/index.js';

const WS = 'D:/AgentTools/qq-bridge/state/agents';
env.allowRemoteModels = false;
env.localModelPath = resolve('models/');
await pipeline('feature-extraction', embeddingModelId(), { quantized: true });
const db = getDb(WS, '.dsh-meow');

const q = '那个烤显卡的家伙还能租吗';
const qv = await embed(q);
const sid = `dbg-${Date.now()}`;
const hit = buildHitInjection(db, WS, sid, q, { hitTopK: 10, hitMinCoverage: 0.12, hybridMinScore: 0.15 }, '.dsh-meow', qv);

console.log(`池子 ${hit?.candidates.length ?? 0} 条\n`);
for (const c of hit?.candidates ?? []) {
  const kw = c.keywords;
  console.log(`  ${c.id.slice(0, 9)}  keywords 类型=${typeof kw}  是数组=${Array.isArray(kw)}  长度=${Array.isArray(kw) ? kw.length : 'N/A'}`);
  if (Array.isArray(kw) && kw.length > 0) console.log(`     → ${kw.slice(0, 6).join(' / ')}`);
}
db.close();
