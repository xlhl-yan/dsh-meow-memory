/**
 * v0.41.0 迁移：给存量记忆算语义向量，写进 embeddings 表。
 * 用法：node scripts/migrate-embeddings.mjs <工作区路径>
 */
import { resolve } from 'node:path';
import { env, pipeline } from '@xenova/transformers';
import { MemoryDb, memoryDbPath } from '../lib/index.js';

const MODEL = 'Xenova/bge-small-zh-v1.5';
const ws = process.argv[2] ?? 'D:/AgentTools/qq-bridge/state/agents';
const dir = process.argv[3] ?? '.dsh-meow';

const dbPath = memoryDbPath(ws, dir);
console.log(`库: ${dbPath}`);

env.allowRemoteModels = false;
env.localModelPath = resolve('models/');
const t0 = Date.now();
const ex = await pipeline('feature-extraction', MODEL, { quantized: true });
console.log(`模型加载 ${Date.now() - t0}ms`);

const db = new MemoryDb(dbPath);
const rows = [];
for (const level of ['soul', 'user', 'project', 'fact', 'lesson', 'topic', 'rules']) {
  try {
    rows.push(...db.list(level, { status: 'active' }));
  } catch { /* 表不存在 */ }
}
console.log(`active 记忆 ${rows.length} 条`);

const missing = new Set(db.missingEmbeddingIds(MODEL));
const todo = rows.filter((r) => missing.has(r.id));
console.log(`需要补算 ${todo.length} 条（已有向量 ${rows.length - todo.length} 条）`);

const t1 = Date.now();
let done = 0, failed = 0;
for (const r of todo) {
  const kw = Array.isArray(r.keywords) ? r.keywords.join('，') : '';
  const text = `${kw}。${(r.content ?? '').slice(0, 300)}`;
  try {
    const out = await ex(text.replace(/\s+/g, ' ').trim().slice(0, 500), { pooling: 'mean', normalize: true });
    const vec = Float32Array.from(out.data);
    db.setEmbedding(r.id, MODEL, vec);
    done++;
  } catch {
    failed++;
  }
  if ((done + failed) % 50 === 0) process.stdout.write(`  ${done + failed}/${todo.length}\r`);
}
const ms = Date.now() - t1;
const stats = db.embeddingStats(MODEL);
console.log(`\n完成：${done} 条成功${failed > 0 ? `，${failed} 条失败` : ''}，用时 ${(ms / 1000).toFixed(1)}s（${Math.round(ms / Math.max(1, done))}ms/条）`);
console.log(`embeddings 表：${stats.total} 条，维度 ${stats.dim}`);
db.close();
