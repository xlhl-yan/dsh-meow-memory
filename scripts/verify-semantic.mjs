import { resolve } from 'node:path';
import { env, pipeline } from '@xenova/transformers';
import { getDb, buildHitInjection, embed, embeddingModelId } from '../lib/index.js';

const WS = 'D:/AgentTools/qq-bridge/state/agents';
env.allowRemoteModels = false;
env.localModelPath = resolve('models/');
await pipeline('feature-extraction', embeddingModelId(), { quantized: true });

const QUERIES = [
  '机器学不动了，算力不够用',
  '两个人一起出镜该怎么分',
  '画面糊成一片看不清',
  '那个视频的普通过渡',
  '用什么办法把真人换成二次元',
];

const db = getDb(WS, '.dsh-meow');
for (const q of QUERIES) {
  const qv = await embed(q);
  const sid = `sem-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  const r = buildHitInjection(db, WS, sid, q, { hitTopK: 5, hitMinCoverage: 0.12 }, '.dsh-meow', qv);
  console.log(`\n════ 「${q}」 ════`);
  if (r === null) { console.log('  （无命中）'); continue; }
  for (const id of r.injectedIds) {
    const found = db.findById(id);
    const c = (found?.row.content ?? '').replace(/\n/g, ' ').slice(0, 62);
    console.log(`  [${found?.row.level ?? '?'}] ${c}`);
  }
}
db.close();
