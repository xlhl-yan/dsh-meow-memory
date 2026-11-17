/**
 * 观察期分析：从 dream-debug.log 统计真实使用中的召回表现。
 *
 * 关注三件事：
 *   ① topScore 分布 —— 多少查询会触发精排（触发率 = 延迟成本）
 *   ② 池子大小 —— RERANK_POOL=20 是否够用（如果经常顶到 20，说明还有货被切在外面）
 *   ③ 精排耗时 —— 真实延迟
 *
 * 用法：node scripts/analyze-log.mjs [日志路径] [--days 1]
 */
import { existsSync, readFileSync } from 'node:fs';

const LOG = process.argv[2] && !process.argv[2].startsWith('--')
  ? process.argv[2]
  : 'D:/AgentTools/qq-bridge/state/agents/.dsh-meow/dream-debug.log';
const daysArg = process.argv.indexOf('--days');
const DAYS = daysArg > 0 ? Number(process.argv[daysArg + 1]) : 7;

if (!existsSync(LOG)) {
  console.error(`日志不存在：${LOG}`);
  process.exit(1);
}

const since = Date.now() - DAYS * 86400_000;
const lines = readFileSync(LOG, 'utf8').split('\n');

const hits = [];
const reranks = [];
for (const line of lines) {
  const m = line.match(/^\[(.+?)\] (hit-chain|rerank) (.*)$/);
  if (!m) continue;
  const t = Date.parse(m[1]);
  if (!Number.isFinite(t) || t < since) continue;
  const rest = m[3];
  const f = (k) => {
    const r = rest.match(new RegExp(`${k}=([^\\s]+)`));
    return r ? r[1] : undefined;
  };
  if (m[2] === 'hit-chain') {
    hits.push({ t, mode: f('hybrid'), topScore: f('topScore'), hit: f('hit'), via: f('via') });
  } else {
    reranks.push({ t, pool: Number(f('pool')), kept: Number(f('kept')), ms: Number((rest.match(/(\d+)ms/) ?? [])[1]) });
  }
}

const pct = (n, d) => (d === 0 ? '0%' : `${((n / d) * 100).toFixed(0)}%`);

console.log(`日志：${LOG}`);
console.log(`窗口：最近 ${DAYS} 天    命中判定 ${hits.length} 次    精排 ${reranks.length} 次\n`);

if (hits.length === 0) {
  console.log('没有 hit-chain 记录。可能原因：');
  console.log('  - 还没重启 DSH 加载新代码（v0.44 之后才记 topScore 字段）');
  console.log('  - 或者这段时间没有对话');
  process.exit(0);
}

// ① topScore 分布
const scores = hits.map((h) => Number(h.topScore)).filter((n) => Number.isFinite(n));
if (scores.length > 0) {
  console.log('① topScore 分布（决定精排触发率）');
  const buckets = [[0, 0.3], [0.3, 0.5], [0.5, 0.65], [0.65, 0.75], [0.75, 1.01]];
  for (const [lo, hi] of buckets) {
    const n = scores.filter((s) => s >= lo && s < hi).length;
    const bar = '█'.repeat(Math.round((n / scores.length) * 30));
    const tag = hi <= 0.75 ? ' ← 会触发精排' : '';
    console.log(`   ${lo.toFixed(2)}~${hi > 1 ? '1.00' : hi.toFixed(2)}  ${String(n).padStart(4)}  ${pct(n, scores.length).padStart(4)}  ${bar}${tag}`);
  }
  const ranged = scores.filter((s) => s < 0.75).length;
  const blank = hits.filter((h) => h.hit === 'null' || h.topScore === '-').length;
  console.log(`   → 触发精排 ${ranged}/${scores.length} = ${pct(ranged, scores.length)}    无命中 ${blank}/${hits.length} = ${pct(blank, hits.length)}\n`);
}

// ② 池子大小
if (reranks.length > 0) {
  const pools = reranks.map((r) => r.pool).filter(Number.isFinite);
  const atCap = pools.filter((p) => p >= 20).length;
  console.log('② 精排池子大小（RERANK_POOL=20）');
  console.log(`   平均 ${(pools.reduce((a, b) => a + b, 0) / pools.length).toFixed(1)} 条    顶到 20 的 ${atCap}/${pools.length} = ${pct(atCap, pools.length)}`);
  if (atCap / pools.length > 0.5) console.log('   ⚠️ 过半顶格——可能还有答案被切在池外，考虑提到 30');
  console.log('');

  // ③ 耗时
  const ms = reranks.map((r) => r.ms).filter(Number.isFinite);
  ms.sort((a, b) => a - b);
  console.log('③ 精排耗时');
  console.log(`   p50 ${ms[Math.floor(ms.length * 0.5)]}ms   p90 ${ms[Math.floor(ms.length * 0.9)]}ms   最慢 ${ms[ms.length - 1]}ms\n`);
}

// ④ 模式自检
const lex = hits.filter((h) => h.mode === 'lex').length;
if (lex > 0) console.log(`⚠️ 有 ${lex} 次走了纯词面（hybrid=lex）——向量模型可能没加载成功\n`);
