# Fork 说明：召回质量改造

> 本文件只属于 `feat/recall-overhaul` 分支。`README.md` 保持上游原样，方便日后合并上游更新。

**上游**：[`Phant0Meow/dsh-meow-memory`](https://github.com/Phant0Meow/dsh-meow-memory)（基线 0.32.7）
**本分支**：`feat/recall-overhaul`，当前 **0.41.0**，专项改造**召回质量**

除召回层外，插件其余部分（七层记忆、dream、反思、信任门、客户端 UI、配置、i18n）与上游一致。

## 这个 fork 在解决什么

上游的命中链路是**纯词面匹配**（CJK bigram + IDF 加权覆盖率 + 覆盖率门槛）。实测在真实使用中
命中率极低（634 轮会话命中不到 50 次），且调参空间很快见底：

| 版本 | 改动 | 结果 |
|---|---|---|
| 0.33.0 | 解除 `source_session` 自锁（原候选池恒空，808 次判定 0 命中） | 从「完全不命中」拉回可用 |
| 0.33.1 | 新增 `trustByDefault`（换插件目录后旧会话集体失忆） | 修 |
| 0.34–0.35 | `memory_search` 的 render 读 `content_max`，截断提成配置 | 修 |
| 0.36.0 | 命中门槛重做（分母改 min、门槛 0.4→0.15、查询侧停用词） | 短查询能中了，负样本仍漏 |
| 0.37.0 | IDF 加权分母只算真实词、`hc>=2` + 专有名词例外、`maxDfRatio` 0.08→0.05 | 正 5/6、负 10/10 |
| 0.38.0 | `qq_wait_for_messages` 轮也能注入（消息只在 `tool/result` 里） | 时机修好 |
| 0.39.0 | 单命中改看 `idfSum`（短查询有效词只剩 1 个时 `hc>=2` 数学上不可能） | 正 15/15、负 0/15 |
| 0.40.0 | `maxDf` 绝对下限 2→8（小库上主题词被误杀，只剩跨词碎片） | 修 |
| **0.41.0** | **召回层换成词面 + 语义向量混合** | **自命中 94.4% → 96.8%** |

每次修好一个场景就冒出新场景，根因是**词面匹配无法表达语义**：

- 讨论「这个记忆插件」和问「记忆插件怎么配」，字面几乎一样但意图不同 → 元话题撞库
- 「get_unread 队列逻辑」本该命中「会话 seq 水位」那条，词面却因为「队列」命中
  「ComfyUI `/queue clear`」

## 0.41.0：词面 + 语义向量混合

```
0.45 × 词面(IDF 加权 coverage) + 0.55 × 向量(BGE 余弦)  → 总分门槛 0.40 → topK
```

| 指标（252 条真实库，15 条测试集） | 纯词面 | 混合 |
|---|---|---|
| 正样本命中 | 15/15 = 100% | **15/15 = 100%** |
| 负样本误命中 | 0/15 = 0% | **0/15 = 0%** |
| 全量自命中（252 条） | 238/252 = 94.4% | **244/252 = 96.8%** |

更细的对比（10 正 + 5 负的子集，按 top5 相关性打分）：

| 方法 | top5 相关率 | top1 命中 | 负样本误命中 |
|---|---|---|---|
| 纯词面 | 68% | 10/10 | 2/5 |
| 纯向量 small | 74% | 9/10 | 2/5 |
| **混合** | **80%** | **10/10** | **1/5** |

### 为什么是混合而不是纯向量

纯向量会丢专有名词精度：`3081`、`48GB`、`wakeIntervalSec` 这类词向量会糊，词面反而准。

### 为什么门槛是 0.40

BGE 对**任意两句中文**都给 ~0.5 的基线余弦：

| 查询 | top1 sim | 性质 |
|---|---|---|
| 「我有点困了」 | 0.577 | 负样本（闲聊） |
| 「你在干嘛」 | 0.562 | 负样本 |
| 「我在忙」 | 0.572 | 负样本 |
| 「甩镜怎么做」 | 0.616 | 正样本 |

正负样本的 sim **几乎重叠**，单靠余弦分不开。但负样本有个共同点：**词面命中恒为 0（纯向量
命中）**。0.30 时闲聊的 `0.55×(0.577-0.30)/0.45 = 0.339` 刚好过线；0.40 挡住它，而有词面
支撑的正样本（cov 0.5 + sim 0.6 → 0.594）绰绰有余。

## 安装语义模型（需手动放一次）

模型**不随仓库分发**（22.9MB ONNX，已在 `.gitignore` 排除）。放到包根目录：

```
meow-memory/
└── models/
    └── Xenova/bge-small-zh-v1.5/
        ├── config.json
        ├── tokenizer.json
        ├── tokenizer_config.json
        └── onnx/model_quantized.onnx
```

```bash
mkdir -p models/Xenova/bge-small-zh-v1.5/onnx
for f in config.json tokenizer.json tokenizer_config.json onnx/model_quantized.onnx; do
  curl -L -o "models/Xenova/bge-small-zh-v1.5/$f" \
    "https://hf-mirror.com/Xenova/bge-small-zh-v1.5/resolve/main/$f"
done
```

**给存量记忆补算向量**

```bash
node scripts/migrate-embeddings.mjs <工作区路径>   # 例：D:/AgentTools/qq-bridge/state/agents
node scripts/verify-hybrid.mjs                    # 可选：词面 vs 混合的全量对比
node scripts/diag-scores.mjs                      # 可选：看某句话的向量分分布
```

> **缺模型时插件照常工作**，只是 `hit-chain` 日志里显示 `hybrid=lex`（纯词面）。
> 补上模型后重启即切到 `hybrid=vec`。

## 性能与模型选择

| | bge-small(23MB) | bge-base(312MB) |
|---|---|---|
| 模型加载 | 162ms | 901ms |
| 单条推理 | **32ms** | 78ms |
| 常驻内存 | **150MB** | 1288MB |
| 批量 252 条 | **13.7s** | 143s |
| top5 相关率 | 80% | 82% |

默认选 small：base 只多 +2pp，却要多 1.1GB 常驻内存和 10 倍批量耗时。
想换只需改 `src/embedding.ts` 的 `DEFAULT_EMBEDDING_MODEL` 并重跑迁移。

## 排查

`<工作区>/.dsh-meow/dream-debug.log` 里每次命中都有一行：

```
hit-chain pid=... sid=... via=user textLen=42 hybrid=vec hit=yes
```

- `hybrid=vec` → 向量参与了
- `hybrid=lex` → 纯词面（模型没加载 / 库没向量 / 主动关掉 `hybrid`）
- `hit=yes|null` → 有没有注入

## 回退

想把召回换回纯词面：`InjectOptions.hybrid = false`（内部 opts，不改库、不改数据）。
向量表留着不影响任何东西。

## 已知限制

- **元话题撞库仍未根治**：用户在讨论「记忆插件本身」时，「记忆」「插件」「命中」这些词必然
  与库里所有插件相关记忆词面重合。混合召回缓解但没消除——这类工作区建议直接
  `disabledWorkspaces` 关掉注入。
- **向量按条目算，模型换了要全量重算**（维度变化）。
- **中文闲聊与技术的向量基线重叠**，所以门槛必须靠词面兜底（见上）。

## 开发

```bash
npm install          # @xenova/transformers 需要装（build 时 external，不进 bundle）
npm run build        # esbuild 打包 src → lib
npm test             # 658 + 11 + 32 + 69 + 24 + 31 + 71 + client 系列
```

> `models/` 不进 git；`lib/` 也不进（由 `prepare` 构建）。
