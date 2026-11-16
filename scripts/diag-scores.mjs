import { resolve } from 'node:path';
import { env, pipeline } from '@xenova/transformers';
import { getDb } from '../lib/index.js';
import { tokenize } from '../lib/index.js';

const MODEL = 'Xenova/bge-small-zh-v1.5';
const WS = 'D:/AgentTools/qq-bridge/state/agents';
env.allowRemoteModels = false;
env.localModelPath = resolve('models/');
const ex = await pipeline('feature-extraction', MODEL, { quantized: true });
const embed = async (t) => Float32Array.from((await ex(t.slice(0, 500), { pooling: 'mean', normalize: true })).data);
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };

const db = getDb(WS, '.dsh-meow');
const rows = ['fact', 'lesson', 'rules', 'topic'].flatMap((l) => db.list(l, { status: 'active' }));
const vecs = db.loadEmbeddings(MODEL);
console.log(`库 ${rows.length} 条，有向量 ${vecs.size} 条\n`);

const QS = ['我有点困了', '你在干嘛', '我在忙', '今天天气不错', '好的', '甩镜怎么做', 'get_unread 队列逻辑'];
for (const q of QS) {
  const qv = await embed(q);
  const scored = rows.map((r) => ({ r, sim: vecs.has(r.id) ? dot(qv, vecs.get(r.id)) : 0 })).sort((a, b) => b.sim - a.sim);
  console.log(`「${q}」 top5:`);
  for (const s of scored.slice(0, 5)) {
    console.log(`   sim=${s.sim.toFixed(3)}  ${s.r.content.replace(/\n/g, ' ').slice(0, 40)}`);
  }
}
db.close();
