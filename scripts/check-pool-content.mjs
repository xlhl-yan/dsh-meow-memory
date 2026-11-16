/** 宽池子 20 条里到底有没有相关货？决定查询扩展有没有意义。 */
import { resolve } from 'node:path';
import { env, pipeline } from '@xenova/transformers';
import { getDb, buildHitInjection, embed, embeddingModelId } from '../lib/index.js';

const WS = 'D:/AgentTools/qq-bridge/state/agents';
const HARD = [
  ['换场那一下最费劲', ['转场', '过渡', '场景']],
  ['怎么让画面别那么肉', ['分辨率', '清晰', '锐', '糊']],
];

env.allowRemoteModels = false;
env.localModelPath = resolve('models/');
await pipeline('feature-extraction', embeddingModelId(), { quantized: true });
const db = getDb(WS, '.dsh-meow');

for (const [q, kw] of HARD) {
  const qv = await embed(q);
  const sid = `w-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
  const wide = buildHitInjection(db, WS, sid, q, { hitTopK: 20, hitMinCoverage: 0.12, hybridMinScore: 0.15 }, '.dsh-meow', qv);
  const c = wide?.candidates ?? [];
  console.log(`\n════ 「${q}」  宽池子 ${c.length} 条（期望关键词: ${kw.join('/')}）`);
  let hit = 0;
  c.forEach((x, i) => {
    const t = (x.content ?? '').replace(/\n/g, ' ');
    const ok = kw.some((k) => t.includes(k));
    if (ok) hit++;
    console.log(`  ${String(i + 1).padStart(2)}. ${ok ? '✅' : '  '} ${t.slice(0, 50)}`);
  });
  console.log(`  → 池内命中期望关键词 ${hit}/${c.length} 条`);
}
db.close();
