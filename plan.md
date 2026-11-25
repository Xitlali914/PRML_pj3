# PRML Project 3 开发方案（v1.1）

> 项目：Training Language Models as Text-Game Agents
> 基座：`Qwen/Qwen3-4B-Base`
> 任务：TextWorldExpress — Sorting、Map Reader
> 本文档是唯一开发依据；所有数据规模、超参数、路径与命令均在此固定。

---

## 1. 目标与约束

### 1.1 课程要求

| 项目 | 要求 |
|---|---|
| 基座 | Qwen3-4B；必须实际更新参数，LoRA 需合并后提交 |
| 任务 | Sorting、Map Reader |
| 动作预算 | 每个测试实例最多 10 次动作 |
| 指标 | 成功率、成功时平均步数、无效动作比例；两任务各占 50% |
| 评测 | 单张 RTX 4090（24GB）；参数固定；不联网；不访问其他模型、参考答案或额外环境模拟 |
| 历史 | 实例之间清空 |
| 动作 | 必须由模型产生；不得内置任务专用求解器 |
| 提交物 | `model/`、`run.py`、`code/`、`report.pdf` |

### 1.2 环境事实

1. Sorting：单房间，3–5 个物体；1000 个训练实例中 750 个带单位（质量/体积/长度各 250），
   250 个为纯计数；同一实例内不混维度。
2. Sorting 训练规模：size 3/4/5 = 390/260/350。
3. Map Reader：地图必须 `read map` 获得（1 步）；最少动作 `2d+3`。
4. 距离分布稀疏：27 组参数 × 60 seed 中 d=1/2/3 = 1002/436/182，生成 1000 条 d=3
   需要扫描全部 27 组参数 × 1000 seed。
5. 分折：train 0–999，dev 10000–10999，test 20000–20999。
6. 性能：JVM 启动 0.24 s，环境步进约 1.6k steps/s。
7. TextWorldExpress 1.1.0（commit `db0e6c0`），jar SHA256
   `23da9ab053dde2de003b4556a2734af3225dc3a0ef8af7354df53aafb5c77d99`。

### 1.3 固定决策

| 决策 | 值 |
|---|---|
| 基座 | `Qwen/Qwen3-4B-Base`，纯文本 prompt，无 chat template |
| 流程 | 零样本基线 → SFT → GRPO |
| Sorting 规模 | 训练/dev/test 均为 3–5 个物体 |
| Map Reader 距离 | 训练/dev/test 均为 d=1/2/3，不含 d=4 |
| 动作上限 | 10 |
| 评测解码 | greedy |
| 训练框架 | 自研 PyTorch + PEFT，不用 vLLM/TRL/verl |
| 运行时 | `third_party/` 内置 TextWorldExpress 与 py4j |
| KL 参考 | 关闭 LoRA adapter 的同一模型 |

### 1.4 内部分数目标

| 任务 | dev 成功率 | 无效动作比例 | 成功平均步数 |
|---|---|---|---|
| Sorting | ≥ 85% | ≤ 2% | ≤ 8 |
| Map Reader | ≥ 65% | ≤ 3% | ≈ 2d+3 |

---

## 2. 总体流程

```
阶段 0  zero-shot：Qwen3-4B-Base + harness → dev 200 → 基线指标
阶段 1  SFT：turns_train 10499 turn → LoRA 2 epoch → dev 选最佳 → 合并
阶段 2  GRPO：SFT adapter + 环境在环 rollout（G=8）→ 300 update → dev 每 25 update
阶段 3  评测：local-test 只跑一次 → 官方测试集只跑一次
```

---

## 3. 仓库结构与路径

```
PRML_pj3/
├── README.md / plan.md / requirements.txt / run.py
├── third_party/                  # 运行时（入库）
│   ├── textworld_express/        # Python 封装 + jar + 数据 + LICENSE
│   ├── py4j/                     # py4j 0.10.9.9 + LICENSE
│   └── NOTICE.md
├── code/                         # 本项目代码（入库）
│   ├── paths.py
│   ├── env/{twx.py, prompt.py}
│   ├── data/{generate_all.py, generate_instances.py, build_expert.py,
│   │         build_sft_turns.py, validate_data.py, checksums.py}
│   ├── eval/evaluate.py
│   └── train/{model.py, sft.py, reward.py, rollout.py, grpo.py}
├── data/                         # 生成数据（不入库）
│   ├── instances/ sft/
│   ├── expected_checksums.json   # 唯一入库的数据文件
│   └── .gitkeep
├── model/                        # 权重（不入库）
│   ├── Qwen3-4B-Base/
│   ├── sft/{adapter,merged}/
│   └── grpo/{adapter,merged}/
├── results/ runs/                # 不入库
```

路径由 `code/paths.py` 唯一定义：`ROOT`、`THIRD_PARTY`、`DATA`、`MODEL`、`RUNS`、`RESULTS`。
训练/评测入口均提供 `--base-model`、`--output-dir`、`--data-dir` 覆盖默认值。

---

## 4. 环境与 Harness

### 4.1 运行时

`code/env/twx.py` 在导入前把 `third_party/` 插到 `sys.path[0]`，并校验：
- `textworld_express` 必须来自 `third_party/`；
- jar SHA256 必须等于上表值。

Java 由系统提供（服务器用 OpenJDK 17）。

### 4.2 环境封装

```python
class TWXEnv:
    def reset(game, seed, fold, params) -> Step
    def step(action) -> Step
    def close() -> None
```

- 一个进程持有 1 个 `TWXEnv`，按实例 `reset` 复用；
- `invalid=True` 当且仅当观测以 `Unknown action` 开头；
- 回合结束由 harness 判定：`success`、`failure` 或 `steps == 10`；
- 不读取 `infos['done']` / `envStepLimit`；
- `inventory` 由文本块解析为 `list[str]`。

### 4.3 Prompt（纯文本，Qwen3-4B-Base）

```text
You are an agent playing a text game. At each turn, choose exactly one action.
Reply with the action inside <action></action> and nothing else.

Task: {task}

Current observation:
{observation}

Inventory: {inventory}

Valid actions:
- {action_1}
...

History:
[1] {action} -> {observation}
...

Action:
```

- 空背包写 `nothing`；空历史写 `(none)`；
- 历史最多 6 步，每条历史观测截断到 120 token；
- prompt 上限 1536 token：先丢最旧历史，再截断当前观测；
- 不提供 gold path、地图内容或物体数量。

### 4.4 解析与重试

- 目标：`<action>take 25g of oak</action>`；
- 取最后一个 `<action>...</action>`；缺失取第一行非空文本；
- 规范化后与 `valid_actions` 匹配（精确 → 大小写不敏感）；
- 失败时追加修复提示，temperature=0 重试一次；仍失败则把原文本发给环境并记无效；
- 重试不消耗动作预算，但写入轨迹日志。

### 4.5 回合执行

```python
for step in range(10):
    prompt = render_prompt(task, env.observation, env.inventory,
                           env.valid_actions, history, tokenizer)
    action, parsed = policy.act(prompt, env.valid_actions)
    result = env.step(action)
    if result.invalid or not parsed: invalid += 1
    history.append((action, result.observation))
    if result.success or result.failure: break
```

训练与评测共用同一渲染器、解析器与无效判定。
训练 rollout：temperature 1.0、top_p 1.0；评测：greedy。

---

## 5. 数据生成

### 5.1 原则

1. 只用 train 0–999 生成训练数据；dev 10000–10999；test 20000–20999；
2. 不使用 gold path、对象树等内部状态；
3. 文件无时间戳、顺序固定；同环境重复生成字节一致；
4. 数据文件全部 gitignore；只有 `data/expected_checksums.json` 入库；
5. 100% 专家轨迹必须重放成功。

### 5.2 数据清单

| 文件 | 条数 | 用途 |
|---|---|---|
| `data/instances/sorting_train.jsonl` | 1000（390/260/350） | GRPO 实例池 |
| `data/instances/mapreader_train.jsonl` | 3000（d=1/2/3 各 1000） | GRPO 实例池 |
| `data/instances/dev.jsonl` | 200（100 + 100） | 选模 |
| `data/instances/test.jsonl` | 200（100 + 100） | 本地最终测试 |
| `data/sft/sorting_expert.jsonl` | 600（size 各 200） | SFT |
| `data/sft/mapreader_expert.jsonl` | 900（d 各 300） | SFT |
| `data/sft/turns_train.jsonl` | 10499 | SFT 训练 |
| `data/sft/turns_val.jsonl` | 601 | SFT 选模 |
| `data/expected_checksums.json` | 全部数据文件 SHA256 | 复现校验 |

turn 总数 11100：sorting 4800，mapreader 6300。

### 5.3 Sorting 实例

- seed 0–999，`params=""`，`fold="train"`；
- 读 `hidden_problemsize` 作为难度；
- 记录任务、观测、背包、合法动作；
- 1000 条全部保留。

### 5.4 Map Reader 实例

参数网格（27 组，固定顺序）：

| 参数 | 取值 |
|---|---|
| `numLocations` | 6, 8, 10 |
| `maxDistanceApart` | 1, 2, 3 |
| `maxDistractorItemsPerLocation` | 0, 1, 2 |
| `includeDoors` | 0 |
| `limitInventorySize` | 0 |

生成算法：

1. 固定顺序遍历 27 组参数 × seed 0–999（27000 次 reset）；
2. 按 `actualDistanceApart` 放入 d=1/2/3 候选桶；
3. 每桶按 `(参数序号, seed)` 升序，候选数 N；
4. 按 `index = i * N // 1000`（i=0..999）等距抽取 1000 条；
5. 抽中记录按 `(参数序号, seed)` 写入。

### 5.5 dev / test

- Sorting：dev 10000–10999，test 20000–20999；按 size 升序取 34/33/33，共 100；
- Map Reader：dev 扫描 10000–10199，test 扫描 20000–20199，各 27 组参数；
  按 d 等距抽取 34/33/33，共 100；
- 两游戏合并，按 id 排序；
- test 生成后冻结，评测前用 `expected_checksums.json` 校验。

### 5.6 专家轨迹

#### Sorting

1. 从 `valid_actions` 取 `take ` 动作；
2. 解析：带单位 `(\d+(?:\.\d+)?)\s*(mg|g|kg|ml|l|mm|cm|m)\s+of\s+(.+)`，
   或纯计数 `(\d+(?:\.\d+)?)\s+([a-z].+)`；
3. 换算：质量 mg、体积 ml、长度 mm；纯计数直接比较；
4. 同实例混维度则丢弃；
5. 数值升序；平局尝试两种顺序，仍失败则丢弃；
6. 依次 `take`、`put ... in box`；
7. 校验 `score == 1.0` 且步数 `= 2 × size`。

#### Map Reader

1. `read map`，解析 `The <room> connects to <room1>, <room2> and <room3>.`；
2. 房间名去掉前缀 `the` 后归一化（地图文本会给列表首项加 `the`）；
3. 起点 `You are in the <room>.`；目标 `coin that is located in the <room>`；
4. BFS 求最短路；在当前房间观测中找方向，`move <dir>`；
5. `take coin`；BFS 返回起点；`put coin in box`；
6. 校验成功且步数 `= 2d+3 ≤ 10`。

#### SFT 实例选择

- Sorting：size 3/4/5 各按 seed 升序取前 200，共 600；
- Map Reader：每桶 1000 条中按 `i*1000//300` 等距取 300，共 900。

### 5.7 turn 展开

- 每步生成一个 turn：任务、观测、背包、合法动作、历史、目标动作、split；
- 历史为该步之前的全部 `(action, observation)`，渲染时只取最近 6 步；
- 轨迹按 id 升序，序号 `% 20 == 0` 进入 val（75 条轨迹，601 turn）；
- 同一轨迹的 turn 不跨划分；
- 目标动作必须属于该步 `valid_actions`。

### 5.8 校验门槛

`code/data/validate_data.py` 必须全部通过：

- 实例数：1000/3000/200/200；
- id 唯一；train seed 0–999，dev 10000–10999，test 20000–20999；
- Sorting size ∈ {3,4,5}；Map Reader d ∈ {1,2,3}；
- 专家轨迹 600/900，100% 重放成功，分数 1.0，步数符合公式；
- 每条轨迹的观测/背包/合法动作与重放环境逐步一致；
- 每个 turn 目标动作合法，历史与轨迹一致，split 正确；
- 文件哈希与 `expected_checksums.json` 一致。

### 5.9 命令

```bash
python -m code.data.generate_all          # 生成 + 校验 + 哈希（约 3 分钟）
python -m code.data.validate_data         # 只校验
```

生成只依赖 Java 与 `third_party/`，不读模型、不联网。

---

## 6. 模型与训练基础设施

### 6.1 模型

- `Qwen/Qwen3-4B-Base`，bf16；
- 36 层、hidden 2560、vocab 151936、8 KV head；
- tokenizer 同目录，`pad_token = eos_token`；
- 纯文本模板，无 chat token。

### 6.2 LoRA

| 项 | 值 |
|---|---|
| r / alpha / dropout | 32 / 64 / 0.05 |
| 目标模块 | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| 可训练参数 | 约 84M |
| 合并 | `merge_and_unload()` 后保存完整权重 |

### 6.3 显存预算（24GB）

| 项 | 估算 |
|---|---|
| 4B bf16 权重 | 8.1 GB |
| LoRA + AdamW 状态 | 1.0 GB |
| 激活（micro-batch 4，1.6k token，grad checkpoint） | 4–5 GB |
| logits（`logits_to_keep`，只算动作位置） | < 0.1 GB |
| rollout KV cache（8 条 × 2k） | < 0.1 GB |
| 峰值 | 约 14–15 GB |

### 6.4 turn 级样本

- 输入 = prompt token + action token，动作在末尾；
- 前向 `logits_to_keep=max_action_len`，只保留动作位置 logits；
- loss 只在动作 token 上计算；
- 同一轨迹所有 turn 共享轨迹级优势。

### 6.5 日志

`runs/<stage>/<timestamp>/` 下写 `config.json`、`train.jsonl`、`dev.jsonl`、`checkpoint/`。
日志可含时间戳，数据文件不含。

---

## 7. 阶段 0：零样本基线

```bash
python -m code.eval.evaluate \
  --model-dir model/Qwen3-4B-Base \
  --instances data/instances/dev.jsonl \
  --output results/zero_shot_dev.jsonl
```

greedy、最多 10 步、1 次修复重试；输出三项指标与逐实例轨迹。作为报告基线。

---

## 8. 阶段 1：SFT

| 项 | 值 |
|---|---|
| 数据 | `turns_train.jsonl`（10499 turn） |
| 验证 | `turns_val.jsonl`（601 turn）+ dev 200 |
| epoch | 2 |
| lr | 1e-4，cosine，warmup 5% |
| micro-batch | 4 |
| 梯度累积 | 4（有效 16） |
| max length | prompt 1536 + action 48 |
| weight decay | 0.01 |
| grad clip | 1.0 |
| 精度 | bf16 + TF32 + gradient checkpointing |
| 保存 | 每 epoch 在 dev 上评测，保留成功率最高版本 |

```bash
python -m code.train.sft --base-model model/Qwen3-4B-Base --output-dir model/sft
```

产出 `model/sft/adapter/` 与 `model/sft/merged/`。进入 GRPO 前 dev 成绩须高于零样本。

---

## 9. 阶段 2：GRPO

### 9.1 目标函数

每个 prompt 采样 G=8 条轨迹，组内标准化：

```
A_i = (R_i - mean(R)) / (std(R) + 1e-4)
ρ = exp(log π_θ(a|s) - log π_old(a|s))
L = -1/Σ|a| Σ min(ρ·A, clip(ρ, 0.8, 1.2)·A) + 0.001 · KL(π_θ || π_ref)
```

- `π_old` 为 rollout 时记录的概率；
- `π_ref` 为关闭 LoRA adapter 的同一模型；
- KL 用 k3：`exp(ref_logp - logp) - (ref_logp - logp) - 1`。

### 9.2 Rollout

| 项 | 值 |
|---|---|
| 每 update prompt 数 | 8 |
| 每个 prompt 轨迹数 | 8 |
| 温度 / top_p | 1.0 / 1.0 |
| 每步最大 token | 48，遇 `</action>` 停止 |
| 每回合动作上限 | 10 |
| 早停 | success / failure |
| 环境 | 8 个常驻 JVM，轨迹同步推进 |

- 每步把活跃轨迹拼成 batch 生成，记录每个动作 token 的 logp；
- 退化组（8 条奖励相同）丢弃；一次 update 内 8 组全退化则跳过更新。

### 9.3 奖励

```
R = 1.0 × success
  + 0.5 × max(score, 0)
  - 0.5 × taskfailure
  - 0.05 × invalid_count
  - 0.01 × steps
  - 0.05 × unparsed_count
```

### 9.4 课程

| update | 采样分布 |
|---|---|
| 1–80 | sorting size 3/4 + mapreader d=1/2，四桶等概率 |
| 81–200 | 六桶等概率 |
| 201–300 | size 5 与 d=3 合计 50%，其余四桶各 12.5% |

### 9.5 超参数

| 项 | 值 |
|---|---|
| 初始化 | `model/sft/adapter` |
| update 数 | 300 |
| lr | 1e-5，cosine，warmup 5% |
| micro-batch | 4 turn |
| 梯度累积 | 4（有效 16） |
| PPO clip | 0.2（0.8–1.2） |
| KL 系数 | 0.001 |
| 熵正则 | 0 |
| dev 评测 | 每 25 update，greedy，200 条 |
| 保存 | 两任务平均成功率最高的 adapter 与 merged |

```bash
python -m code.train.grpo --base-model model/Qwen3-4B-Base \
  --sft-adapter model/sft/adapter --output-dir model/grpo
```

### 9.6 监控

每 update 记录：平均奖励、组内标准差、成功率、无效动作率、平均步数、动作 token 数、
clip 比例、KL、loss、跳过组数。

---

## 10. 评测

```bash
python run.py --model-dir model/grpo/merged \
  --instances data/instances/test.jsonl --output results/final_test.jsonl
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--model-dir` | 必填 | 合并后完整模型 |
| `--instances` | 必填 | 实例 JSONL |
| `--output` | 必填 | 逐实例轨迹 |
| `--metrics` | 同名 `.metrics.json` | 指标 |
| `--max-actions` | 10 | 动作上限 |
| `--limit` | 0 | 调试 |
| `--device` | cuda | 设备 |

指标：成功率、成功平均步数、无效动作比例；分任务与总体；
`overall_score = 0.5 × success_sorting + 0.5 × success_mapreader`。

约束：合并权重、`eval()`、离线、每实例重建 prompt、harness 计数 10 步、greedy。
`test.jsonl` 只在主实验结束后跑一次；官方测试集只跑一次；调参只看 dev。

---

## 11. 消融实验设计（主实验跑通后实现）

共享：同一基座、同一数据、dev 200、同一 seed 流程；每次只改一项。

| 编号 | 变体 | 唯一差异 | 训练量 |
|---|---|---|---|
| A | 无 SFT | 跳过 SFT，直接 GRPO 300 update | 同主实验 GRPO |
| B | 稀疏奖励 | 去掉 `0.5 × max(score,0)` | 同上 |
| C | 无无效惩罚 | 去掉 `0.05 × invalid_count` | 同上 |
| D | 无 KL | KL 系数 0 | 同上 |
| E | 无课程 | 300 update 六桶等概率 | 同上 |
| F | 小 group | G=4，其余不变 | 同上 |
| G | 少 SFT | SFT 只 1 epoch | 仅 SFT |

报告列：两任务成功率、成功平均步数、无效动作比例、与主实验差值、GPU 小时。
每个变体只跑一次。

---

## 12. 工程与复现

依赖：`torch`、`transformers`、`peft`、`accelerate`。数据生成不需要这些；
运行时由 `third_party/` 提供，无需 pip。

复现：

```bash
python -m code.data.generate_all
python -m code.eval.evaluate --model-dir model/Qwen3-4B-Base \
  --instances data/instances/dev.jsonl --output results/zero_shot_dev.jsonl
python -m code.train.sft --base-model model/Qwen3-4B-Base --output-dir model/sft
python -m code.train.grpo --base-model model/Qwen3-4B-Base \
  --sft-adapter model/sft/adapter --output-dir model/grpo
python run.py --model-dir model/grpo/merged \
  --instances data/instances/test.jsonl --output results/final_test.jsonl
```

可复现性：数据固定种子/顺序/哈希；训练固定随机种子；评测 greedy；版本记录在
`third_party/NOTICE.md` 与 `config.json`。

Git：入库 `third_party/`、`code/`、`run.py`、`requirements.txt`、`plan.md`、`README.md`、
`data/expected_checksums.json`、`.gitkeep`；不入库 `data/instances/`、`data/sft/`、
`model/*`、`results/`、`runs/`。

---

## 13. 里程碑

| 周 | 交付 |
|---|---|
| W1 | 数据生成与校验；零样本基线；SFT 跑通 |
| W2 | SFT 超过基线；GRPO 跑通少量 update |
| W3 | GRPO 300 update；选最佳 checkpoint |
| W4 | 测试集评测；消融；报告 |

---

## 14. 风险

| 风险 | 应对 |
|---|---|
| d=3 只有 9 步、sorting 5 物体 10 步 | 专家覆盖；课程后期强化 hard 桶 |
| 24GB 显存 | micro-batch 4 + gradient checkpointing + `logits_to_keep` |
| GRPO 组内无方差 | 六桶均衡、课程、G=8；退化组跳过 |
| 训练不稳定 | SFT 初始化、KL=0.001、clip=0.2、小 lr |
| 数据不确定 | 固定顺序、无时间戳、哈希校验 |
| 测试泄漏 | 分折 + 哈希冻结 + test 只跑一次 |

---

## 15. 待确认

1. 课程评测配置是否与本文一致（Sorting ≤5 物体，Map Reader d≤3）。
2. `Qwen3-4B-Base` 是否满足课程“Qwen3-4B 为基座”的要求。
3. 官方测试集实例格式，用于适配 `run.py`。
4. “两任务各占 50%”的口径。
