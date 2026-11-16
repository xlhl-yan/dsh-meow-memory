/**
 * v0.41.0 迁移：给存量记忆算语义向量，写进 embeddings 表。
 *
 * 用法：
 *   node scripts/migrate-embeddings.mjs <工作区路径>           # 只补缺的
 *   node scripts/migrate-embeddings.mjs <工作区路径> --force    # 全量重算（喂料口径变了必须用）
 */
import { resolve } from 'node:path';
import { env, pipeline } from '@xenova/transformers';
import { MemoryDb, memoryDbPath, embeddingText, embeddingModelId } from '../lib/index.js';

const MODEL = embeddingModelId();
const args = process.argv.slice(2);
const force = args.includes('--force');
const ws = args.find((a) => !a.startsWith('--')) ?? 'D:/AgentTools/qq-bridge/state/agents';
const dir = '.dsh-meow';

const dbPath = memoryDbPath(ws, dir);
console.log(`库: ${dbPath}`);
console.log(`模型: ${MODEL}${force ? '  [--force 全量重算]' : ''}`);

env.allowRemoteModels = false;
env.localModelPath = resolve('models/');
const t0 = Date.now();
const ex = await pipeline('feature-extraction', MODEL, { quantized: true });
console.log(`模型加载 ${Date.now() - t0}ms`);

const db = new MemoryDb(dbPath);
if (force) {
  db.deleteAllEmbeddings(MODEL);
  console.log('已清空旧向量');
}

const rows = [];
for (const level of ['soul', 'user', 'project', 'fact', 'lesson', 'topic', 'rules']) {
  try {
    rows.push(...db.list(level, { status: 'active' }));
  } catch { /* 表不存在 */ }
}

const missing = new Set(db.missingEmbeddingIds(MODEL));
const todo = rows.filter((r) => missing.has(r.id));
console.log(`active 记忆 ${rows.length} 条，需要算 ${todo.length} 条`);

const t1 = Date.now();
let done = 0, failed = 0;
for (const r of todo) {
  // 与增量写入完全同源（embeddingText 取头尾，不截断结论）
  const text = embeddingText(r);
  try {
    const out = await ex(text.slice(0, 500), { pooling: 'mean', normalize: true });
    db.setEmbedding(r.id, MODEL, Float32Array.from(out.data));
    done++;
  } catch {
    failed++;
  }
  if ((done + failed) % 50 === 0) process.stdout.write(`  ${done + failed}/${todo.length}\r`);
}
const ms = Date.now() - t1;
const stats = db.embeddingStats(MODEL);
console.log(`\n完成：${done} 条${failed > 0 ? `，失败 ${failed}` : ''}，用时 ${(ms / 1000).toFixed(1)}s（${Math.round(ms / Math.max(1, done))}ms/条）`);
console.log(`embeddings 表：${stats.total} 条，维度 ${stats.dim}`);
db.close();
