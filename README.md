# meow-memory 🐱📝

| [中文](README.md) | [English](README.en.md) | [Português (BR)](README.pt-br.md) | [MIT License](LICENSE) |
| :---: | :---: | :---: | :---: |

为 [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness)（DSH）打造的跨会话记忆插件。

当前版本 **0.41.0** ｜ 版本变更见 [CHANGELOG.md](CHANGELOG.md) ｜ 兼容 **dsh 0.1.5**

> **这是 [`Phant0Meow/dsh-meow-memory`](https://github.com/Phant0Meow/dsh-meow-memory) 的 fork**，
> 分支 `feat/recall-overhaul`，专项改造**召回质量**。上游基线 0.32.7；本分支从 0.33 起逐步
> 重做命中链路（解除会话自锁 → 默认信任 → 门槛重做 → wait 轮注入 → 短查询 → 混合召回），
> 当前 **0.41.0**。除召回层外，插件其余部分与上游一致。

## 🔍 召回层：词面 + 语义向量混合（0.41.0）

纯词面匹配（CJK bigram + IDF 加权覆盖率）有个先天缺陷：**字面重合 ≠ 语义相关**。
讨论「这个记忆插件」和问「记忆插件怎么配」，字面几乎一样但意图不同；反过来
「get_unread 队列逻辑」本该命中「会话 seq 水位」那条，词面却因为「队列」去命中
「ComfyUI /queue clear」。

五轮调参（分母口径、IDF 加权、最少命中数、专有名词例外、停用词、maxDf）之后确认这条路
走不通，所以**只换打分，不动其余**：

```
0.45 × 词面(IDF 加权 coverage) + 0.55 × 向量(BGE 余弦)  → 总分门槛 0.40 → topK
```

| 指标（252 条真实库） | 纯词面 | 混合 |
|---|---|---|
| 15 条正样本命中 | 15/15 = 100% | **15/15 = 100%** |
| 15 条负样本误命中 | 0/15 = 0% | **0/15 = 0%** |
| 全量自命中 | 238/252 = 94.4% | **244/252 = 96.8%** |

**为什么混合而不是纯向量**：纯向量会丢专有名词精度——`3081`、`48GB`、`wakeIntervalSec`
这类词向量会糊，词面反而准。

**为什么门槛是 0.40**：BGE 对任意两句中文都给 ~0.5 的基线余弦，闲聊（「我有点困了」top1
0.577）和技术问句（「甩镜怎么做」top1 0.616）几乎重叠。但闲聊的**词面命中恒为 0**，
0.40 正好挡住纯向量命中的闲聊，又放过有词面支撑的正样本。

**降级安全**：依赖缺失、模型文件不在、加载失败 → 自动退回纯词面，插件不会挂。
`hit-chain` 日志里的 `hybrid=vec|lex` 就是给这个用的。

### 安装语义模型（需手动放一次）

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

**方式：镜像站直接下（1~2 分钟）**

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
```

> 缺模型时插件照常工作，只是 `hybrid=lex`（纯词面）。补上模型后重启即切到混合。

### 换成更大的模型

改 `src/embedding.ts` 的 `DEFAULT_EMBEDDING_MODEL` 并重跑迁移。实测对比：

| | bge-small(23MB) | bge-base(312MB) |
|---|---|---|
| 单条推理 | **32ms** | 78ms |
| 常驻内存 | **150MB** | 1288MB |
| 批量 252 条 | **13.7s** | 143s |
| top5 相关率 | 80% | 82% |

默认选 small：base 只多 +2pp，却要多 1.1GB 常驻内存和 10 倍批量耗时。

## ✨ 功能特性

- **七层记忆**：`soul`（AI 自身）/ `user`（用户信息与偏好）/ `project`（项目信息，含 `subcategory`）/ `fact`（原子事实）/ `lesson`（教训与纠正）/ `topic`（进行中的话题，带目标句）/ `rules`（设计原则与行为准则）。每层一张 SQLite 表，UUID 带时间前缀，id 顺序即创建顺序。
- **首轮注入（长期记忆块）**：第一条用户消息前注入 soul / user / 设计原则 / 记忆导引。记忆作为独立 plugin snapshot 消息放在真实 user 消息之前，**不改写用户 prompt**；**首轮不跑关键词命中**（命中从第二轮起）。
- **每消息关键词命中**：从第二条用户消息起，每条真实用户消息检索 fact/lesson/rules/topic（全局 + 当前 project 锚定），top-2 命中注入。命中基于**条目关键词**而非全文。
- **当前 project 锚定**：带 project 参数即锚定该会话的当前项目；未锚定时命中只搜全局（用户闲聊不误伤）。
- **缓存友好设计**：静态 `meow-memory:guide` section 在 system prompt 中注册一次——文本恒定，破坏不了 provider 的 KV 缓存。
- **压缩后重注入**：会话压缩后，下一轮自动重新注入长期记忆快照 + 本会话查阅过的项目全景 + 本会话写入过的记忆原文，AI 不会因为压缩突然失忆。
- **七层之上的三件事**：写入自动去重合并、按 `updated_at` 展示记忆时间戳、全局信息用全局标记（区别于留空）。
- **按窗口 dream**：窗口空闲 ≥3 小时即允许整理，由该窗口自己的主 agent 分三轮处理（原子记忆 / topic / 项目总结），**峰时抑制**（默认北京时间 09:00–12:00、14:00–18:00，API 峰谷电价峰时）不触发。也可敲 `/dream` 立刻手动整理。
- **反思**：单次任务内连续 ≥7 个工具 step 后，插件询问模型自上次整理以来是否有值得记忆的内容。最后工具是 `memory_*` 视为已主动记忆、不重复反思。
- **外部记忆库信任门**：工作区自带的 `.dsh-meow` 若非本机创建（如 clone 的外部仓库），默认暂停注入/检索/整理——首轮提示一次，`/meow-trust` 一键信任、即刻生效；本机创建与升级前打开过的工作区自动信任，正常使用零感知。
- **客户端 UI**：注入文本折叠成横条、反思/dream 轮折叠成一条可展开的卡片、会话列表显示 dream 状态图标（🌙 已整理 / 呼吸灯 进行中 / 灰月牙斜杠 已跳过）。
- **无需另装运行时依赖**：`node:sqlite` + esbuild 产物（`lib/index.js`）；`schemastery` 和 `dsh-llm` 由 DSH 宿主解析提供。**无原生模块。**

## 📦 安装

### 一键安装（推荐）

```sh
dsh plugin --profile web add github:Phant0Meow/dsh-meow-memory
```

一条命令装完即生效：安装时自动编译（包内含 `prepare` 脚本），自动挂载，重启 `dsh web` 后新会话自动加载插件。

> pnpm ≥10 默认会阻止安装期的构建脚本：首次 `add` 可能失败并提示 `allowBuilds`，按提示把输出的键加进 profile 的 `pnpm-workspace.yaml` 后重跑即可。

### 卸载

```sh
dsh plugin --profile web remove meow-memory
```

### PT-BR Fork

想要巴西葡语打包版本的仓库：[coutogilson](https://github.com/coutogilson) 的 fork：

```sh
dsh plugin --profile web add github:coutogilson/dsh-meow-memory
```

### 手动安装（开发者，任意 DSH 安装，无需 npm）

1. 把本包复制（或软链）到 profile 的 `node_modules`：

   ```sh
   mkdir -p ~/.dsh/profiles/web/node_modules
   ln -s /path/to/meow-memory ~/.dsh/profiles/web/node_modules/meow-memory
   ```

   （Windows：`New-Item -ItemType Junction ...` —— NTFS junction，无需管理员权限。）

2. 把 `meow-memory` 加进 profile `package.json` 的 `dsh.profile.bundles`。

3. 重启 `dsh web`。新会话自动加载插件。

## 🔌 兼容性

支持 **dsh 0.1.5**（含最新的 `0.1.5-rc.1`），同时向下兼容旧版本——升级 dsh 不需要改本插件，也不需要改任何配置。

插件不写死版本号，而是运行时探测宿主能力，因此新旧版走各自正确的分支。已实测两代：`0.1.5-rc.1` 上首次注入、记忆工具调用、客户端渲染全部正常；`0.1.1-rc.2` 上行为与历史版本完全一致。

## ⚙️ 配置

所有字段均可选（profile patch 或 `cordis.patch.yml`）。**也可以不手编文件**：DSH 设置页里有本插件的「喵记忆」标签页（与「通用」「模型」平级），下面这些项全部图形化可改、字段级保存、可单项恢复默认（恢复默认 = 回到插件出厂默认，不受 patch 基线影响）；保存后热重载/重启 meow-memory 插件生效。

```yaml
- id: meow-memory
  name: 'meow-memory'
  config:
    enabled: true          # 总开关
    projectDir: '.dsh-meow' # 记忆目录（相对工作区）
    promptLang: 'zh'       # ⚠️ 首次使用建议显式配置（见下方说明）
    hitTopK: 2             # 每条用户消息关键词命中的条目数上限
    hitMinCoverage: 0.4    # 命中覆盖率门槛（0..1）：记忆关键词有多大比例被消息命中才注入；0=关闭
    trustByDefault: true   # 默认信任工作区自带的记忆库（关闭 #64 预置库防护）；false 恢复 /meow-trust 门
    searchContentMax: 400  # memory_search / find_similar 每条命中默认显示多少字正文；0=全文
    dream:
      enabled: true
      idleMinutes: 180     # 窗口空闲 ≥180 分钟（3 小时）允许 dream
```

**完整字段、类型与默认值见 [docs/配置参考.md](docs/配置参考.md)**，含 `titleMax` / `autoMigrate` / `homeDir` / `disabledWorkspaces` / `backup` / `delegate` 等全部配置项。

> ⚠️ **`promptLang` 首次使用建议显式配置。** 它决定注入/反思/dream 文案与工具描述的语言，同时影响模型写记忆条目用的语言——关键词按条目语言提取。内置 `zh` / `en` / `pt-br` 三套语言包。配置值就是 `src/prompts/` 的子目录名，**逐字匹配**：填一个不存在的名字（例如 `pt`）不会报错，而是静默落回中文包。详见 [docs/配置参考.md](docs/配置参考.md)。

### 界面语言

插件界面文案跟随 DSH 的「设置 → 通用 → 语言」（内置中文 / English / Português (Brasil)），与 `promptLang` 是两层、互不影响。详见 [docs/配置参考.md](docs/配置参考.md)。

## 🧠 工作原理

```
第一条用户消息（首轮）         第二条起的每条消息                空闲≥3h 且非峰时
┌────────────────────┐        ┌────────────────────┐        ┌──────────────────────┐
│ ===== 长期记忆 ===== │        │ 可能相关的记忆，仅供  │        │ 按窗口 dream：        │
│ 【关于你】(soul)     │        │ 参考：keywords 命中   │        │ 三轮（含项目总结）      │
│ 【关于user】         │        │ top-2（全局+当前     │        │ 七层+提取过的，       │
│ 【设计原则】(rules)   │        │ project 锚定）      │        │ updated_at 以 T 封存  │
│ 【记忆导引】          │        └────────────────────┘        └──────────────────────┘
└────────────────────┘        已见 id 按会话记录
独立 plugin snapshot        (sessions/<id>.json)
↓ 原样 user prompt          压缩信号 → 释放已见
 每会话只注入一次，
 首轮不做命中
```

### 工具

| 工具 | 用途 |
|---|---|
| `memory_remember` | 写入一条记忆（必填 content/project/keywords/importance，自动去重合并） |
| `memory_search` | 关键词检索（BM25 × 近期权重，支持 level/project/status/days 过滤） |
| `memory_project` | 取回某项目的完整注入段落（**project 参数必填**） |
| `memory_find_similar` | 查重与冲突检测 |
| `memory_read` | 按 id 读取单条完整元数据 |
| `memory_update` | 修正 status / importance / goal / keywords |
| `memory_dream` | 手动触发本窗口的记忆整理 |
| `memory_home` | 查看 / 切换全局数据目录 |

另有 `/dream` 命令：在输入框敲 `/dream` 立即手动唤起本窗口的记忆整理，不受峰时抑制。

**参数、返回与副作用见 [docs/工具参考.md](docs/工具参考.md)**（含 `memory_project` 的项目锚定副作用与 `memory_home` 的三个预设目录标记）。

## 🛠 开发

```sh
npm install
npm run build          # esbuild 打包 → lib/index.js（宿主 peer 保持 external）
npm run test           # 228 项逻辑测试：db / bm25 / migrate / inject / reflect / dream / tools / apply
```

`@deepseek-ai/*` 包位于 dsh-meow pnpm workspace 中，不在本包的 `node_modules` 里。
在 Windows 上，`npm run link-workspace`（或 `scripts/link-workspace.ps1`）创建 workspace
包的 junction 镜像，使 esbuild 能解析它们；`build.mjs` 通过 `nodePaths` 引用。
构建不需要安装可选的 `dsh-llm` peer。直接用 Node 运行测试时，仍需从匹配的 DSH
workspace 链接 `@deepseek-ai/dsh-llm`；正常插件运行由 DSH 的 profile loader
解析宿主提供的 peer。不要把它加回 devDependencies，否则 `file:` / `link:`
安装可能再次向 profile 引入宿主核心包的另一份副本（#84）。

`node tests/build-host-peers.mjs` 单独验证无宿主 peer 的干净构建（#87）。
`node tests/host-peers.mjs /path/to/node_modules/@deepseek-ai/dsh` 使用另行安装的
官方 DSH，验证隔离 profile 的真实 Host 装载和消息契约，不调用模型。

## 📚 文档

| 想了解 | 看这里 |
| --- | --- |
| 七个工具的参数、返回与副作用 | [docs/工具参考.md](docs/工具参考.md) |
| 全部配置项、类型与默认值 | [docs/配置参考.md](docs/配置参考.md) |
| 为什么这样设计（缓存、命中、dream、踩过的坑） | [docs/设计说明.md](docs/设计说明.md) |
| 装不上、检索不准、报错怎么读 | [docs/常见问题.md](docs/常见问题.md) |
| 版本变更历史 | [CHANGELOG.md](CHANGELOG.md) |
| 加界面语言 | [src/i18n/README.md](src/i18n/README.md) |
| 加 prompt 语言包 | [src/prompts/README.md](src/prompts/README.md) |

## 🙏 致谢

感谢每一位贡献者让 meow-memory 越来越好：

- **[daveycodez](https://github.com/daveycodez)** — 英文语言包与英文分词（[PR #6](https://github.com/Phant0Meow/dsh-meow-memory/pull/6)，v0.22.0 发布）
- **[chenmzh](https://github.com/chenmzh)** — 记忆注入改为独立 plugin snapshot 消息，根治会话标题污染（[PR #10](https://github.com/Phant0Meow/dsh-meow-memory/pull/10)）
- **[cuddly-guacamole](https://github.com/cuddly-guacamole)** — dsh 0.1.2-alpha.4 双版本 Session events 兼容（[PR #11](https://github.com/Phant0Meow/dsh-meow-memory/pull/11)）
- **[coutogilson](https://github.com/coutogilson)** — 巴西葡语 prompt 语言包（`src/prompts/pt-br/`）与跟随 DSH 语言设置的界面文案层

## 📄 License

MIT —— 见 [LICENSE](LICENSE)。
