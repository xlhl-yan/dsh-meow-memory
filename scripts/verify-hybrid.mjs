/**
 * v0.41.0 全量验证：纯词面 vs 混合召回，对比 15 正 15 负 + 全量自命中。
 * 用法：node scripts/verify-hybrid.mjs
 */
import { resolve } from 'node:path';
import { env, pipeline } from '@xenova/transformers';
import { getDb, buildHitInjection } from '../lib/index.js';

const MODEL = 'Xenova/bge-small-zh-v1.5';
const WS = 'D:/AgentTools/qq-bridge/state/agents';
const DIR = '.dsh-meow';

env.allowRemoteModels = false;
env.localModelPath = resolve('models/');
const ex = await pipeline('feature-extraction', MODEL, { quantized: true });

const embedText = async (t) => {
  const out = await ex(t.replace(/\s+/g, ' ').trim().slice(0, 500), { pooling: 'mean', normalize: true });
  return Float32Array.from(out.data);
};

const POSITIVE = [
  ['甩镜怎么做', '转场/20步'],
  ['Qwen 用英文还是中文写 prompt', 'Qwen 中文'],
  ['三张参考图能不能在本地跑', '三张装不下'],
  ['AutoDL 那台什么配置', '33009 24GB'],
  ['画风最后定成什么了', '动漫化'],
  ['姓名条被角色挡住了怎么办', '文字处理'],
  ['群像换人怎么分配角色', '8个AI'],
  ['本地 6G 卡能跑多高分辨率', '864x480'],
  ['豆包换人为什么脸崩', '两个极端'],
  ['首尾帧字幕为什么会自己变', '字幕一致'],
  ['怎么让生成的角色动起来', '动作幅度'],
  ['挖矿木马怎么清的', 'CompShare'],
  ['转场为什么要拆成两段', '一条一个动作'],
  ['双卡 32G 显存能合并吗', '显存不合并'],
  ['贴文字回生成图效果怎么样', '后期拼被否'],
];
const NEGATIVE = [
  ['今天天气不错', ''], ['我有点困了', ''], ['晚饭吃什么', ''], ['你在干嘛', ''], ['哈哈', ''],
  ['早点睡', ''], ['我出去一趟', ''], ['嗯嗯', ''], ['好的', ''], ['几点了', ''],
  ['帮我倒杯水', ''], ['今天好累', ''], ['周末有安排吗', ''], ['你吃饭了吗', ''], ['我在忙', ''],
];

const db = getDb(WS, DIR);
const OPTS = { hitTopK: 5, hitMinCoverage: 0.12, hitDebug: false };

async function runOne(text, hybrid) {
  const sid = `verify-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  const qv = hybrid ? await embedText(text) : null;
  const r = buildHitInjection(db, WS, sid, text, { ...OPTS, hybrid }, DIR, qv);
  return r === null ? [] : r.injectedIds;
}

for (const hybrid of [false, true]) {
  const label = hybrid ? '混合（词面45%+向量55%）' : '纯词面（v0.40）';
  console.log(`\n════════ ${label} ════════`);
  let posHit = 0;
  for (const [text, expect] of POSITIVE) {
    const ids = await runOne(text, hybrid);
    if (ids.length > 0) posHit++;
    if (ids.length === 0) console.log(`  ❌ 「${text}」 期望≈${expect}`);
  }
  let negBad = 0;
  for (const [text] of NEGATIVE) {
    const ids = await runOne(text, hybrid);
    if (ids.length > 0) { negBad++; console.log(`  ⚠️ 负样本误命中「${text}」→ ${ids.length} 条`); }
  }
  console.log(`  正样本 ${posHit}/15 = ${(posHit / 15 * 100).toFixed(0)}%   负样本误命中 ${negBad}/15 = ${(negBad / 15 * 100).toFixed(0)}%`);
}

// 全量自命中
function makeQuery(content) {
  let s = content.replace(/^\d{4}-\d{2}-\d{2}\s*/, '').replace(/^[\s*#>\-—·]+/, '')
    .replace(/\*\*/g, '').replace(/`[^`]*`/g, '').replace(/[（(][^）)]*[）)]/g, '').trim();
  const m = s.match(/^[^。？！\n]{6,40}/);
  return m ? m[0].trim() : s.slice(0, 30);
}
console.log(`\n════════ 全量自命中 ════════`);
for (const hybrid of [false, true]) {
  const rows = ['fact', 'lesson', 'rules', 'topic'].flatMap((l) => db.list(l, { status: 'active' }));
  let total = 0, self = 0;
  for (const r of rows) {
    const q = makeQuery(r.content || '');
    if (q.length < 4) continue;
    total++;
    const ids = await runOne(q, hybrid);
    if (ids.includes(r.id)) self++;
  }
  console.log(`  ${hybrid ? '混合' : '纯词面'}: ${self}/${total} = ${(self / total * 100).toFixed(1)}%`);
}
db.close();
