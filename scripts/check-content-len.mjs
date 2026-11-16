import { DatabaseSync } from 'node:sqlite';
const db = new DatabaseSync('D:/AgentTools/qq-bridge/state/agents/.dsh-meow/memory.db');
const rows = db.prepare("SELECT id, content, keywords FROM lesson WHERE status='active'").all();
const lens = rows.map((r) => (r.content || '').length).sort((a, b) => a - b);
const over300 = rows.filter((r) => (r.content || '').length > 300).length;
console.log(`lesson active: ${rows.length} 条`);
console.log(`长度：最短 ${lens[0]} / 中位 ${lens[Math.floor(lens.length / 2)]} / 最长 ${lens[lens.length - 1]}`);
console.log(`超过 300 字：${over300} 条 = ${(over300 / rows.length * 100).toFixed(0)}%`);
console.log(`超过 400 字：${rows.filter((r) => (r.content || '').length > 400).length} 条`);

// 看几条长记忆：前 300 字 vs 后 200 字，验证「结论在后面」的假设
const longOnes = rows.filter((r) => (r.content || '').length > 600).slice(0, 3);
for (const r of longOnes) {
  const c = r.content || '';
  console.log(`\n===== ${r.id.slice(0, 9)} (${c.length} 字) =====`);
  console.log(`【前 150】${c.slice(0, 150).replace(/\n/g, ' ')}`);
  console.log(`【后 150】${c.slice(-150).replace(/\n/g, ' ')}`);
}
db.close();
