/** 诊断：为什么 rerank 没把明显相关的记忆排上来。 */
import { resolve } from 'node:path';
import { env, AutoTokenizer, AutoModelForSequenceClassification } from '@xenova/transformers';

const WS = 'D:/AgentTools/qq-bridge/state/agents';
env.allowRemoteModels = false;
env.localModelPath = resolve('models/');

const tok = await AutoTokenizer.from_pretrained('Xenova/bge-reranker-base');
const model = await AutoModelForSequenceClassification.from_pretrained('Xenova/bge-reranker-base', { quantized: true });

const { DatabaseSync } = await import('node:sqlite');
const db = new DatabaseSync(`${WS}/.dsh-meow/memory.db`);
const rows = db.prepare("SELECT id, content FROM lesson WHERE id LIKE '0mw4%' OR content LIKE '%转场中间%' LIMIT 5").all();
db.close();

async function score(q, d) {
  const inputs = tok([q], { text_pair: [d], padding: true, truncation: true });
  const { logits } = await model(inputs);
  return logits.data[0];
}

const Q = '换场那一下最费劲';
const docs = [
  ['转场中间的片段是全新内容', '应该是高分（就是讲转场的）'],
  ['扫雷（及同类推理）三条硬规矩', '应该是低分（无关）'],
  ['耗时任务别自己干等', '应该是低分（无关）'],
];

console.log(`查询：「${Q}」\n`);
for (const [d, tag] of docs) {
  console.log(`  ${(await score(Q, d)).toFixed(2).padStart(7)}  ${tag}  ← ${d}`);
}

console.log('\n=== 换个说法再试 ===');
for (const q2 of ['换场', '转场', '场景切换', '转场之间怎么衔接']) {
  console.log(`\n查询：「${q2}」`);
  for (const [d] of docs) console.log(`  ${(await score(q2, d)).toFixed(2).padStart(7)}  ${d}`);
}

// 库里实际那几条的分数
console.log('\n=== 库里真实的转场记忆 ===');
const { DatabaseSync: D2 } = await import('node:sqlite');
const d2 = new D2(`${WS}/.dsh-meow/memory.db`);
const real = d2.prepare("SELECT id, content FROM lesson WHERE content LIKE '%转场%' AND status='active' LIMIT 6").all();
d2.close();
for (const r of real) {
  const c = (r.content ?? '').replace(/\s+/g, ' ').slice(0, 300);
  console.log(`  ${(await score('换场那一下最费劲', c)).toFixed(2).padStart(7)}  ${c.slice(0, 50)}`);
}
