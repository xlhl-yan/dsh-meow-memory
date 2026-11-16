/**
 * 扫 HYBRID_VEC_MIN_SIM：候选入池的向量门槛。
 * 降它 = 更多候选进池，由 HYBRID_MIN_SCORE 总门槛筛。看难句是否改善。
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { execSync } from 'node:child_process';

const FILE = 'src/inject.ts';
const RE = /export const HYBRID_VEC_MIN_SIM = [\d.]+/;

function setSim(v) {
  const s = readFileSync(FILE, 'utf8');
  if (!RE.test(s)) throw new Error('未找到 HYBRID_VEC_MIN_SIM');
  writeFileSync(FILE, s.replace(RE, `export const HYBRID_VEC_MIN_SIM = ${v}`), 'utf8');
}

for (const v of [0.45, 0.40, 0.35, 0.30]) {
  setSim(v);
  execSync('npm run build', { stdio: 'ignore' });
  const hybrid = execSync('node scripts/verify-hybrid.mjs', { encoding: 'utf8' });
  const hard = execSync('node scripts/verify-hard.mjs', { encoding: 'utf8' });
  const pos = hybrid.match(/正样本 (\d+)\/15/g) ?? [];
  const neg = hybrid.match(/负样本误命中 (\d+)\/15/g) ?? [];
  const self = hybrid.match(/混合: \d+\/\d+ = ([\d.]+)%/)?.[1] ?? '?';
  // 难句：数命中条数
  const blocks = hard.split('════').slice(1);
  const counts = blocks.map((b) => (b.match(/^\s+\[/gm) ?? []).length);
  console.log(`VEC_MIN_SIM=${v}  正/${pos[1] ?? pos[0]}  负/${neg[1] ?? neg[0]}  自命中 ${self}%  难句命中[${counts.join('/')}]`);
}
setSim(0.45);
execSync('npm run build', { stdio: 'ignore' });
console.log('已还原 0.45');
