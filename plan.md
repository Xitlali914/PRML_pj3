# PRML Project 3 开发方案（v0.1 草案）

> 状态：方案稿，供组内讨论；标注「待确认」的事项需要和助教或组员对齐后再定稿。
> 依据：课程 `plan.md`、TextWorldExpress 1.1.0 源码与本机实测（sorting / mapreader）、
> Qwen3-4B 的公开模型卡。
> 目标：在单张 RTX 4090（24GB）的评测预算内，用 Qwen3-4B 完成 Sorting 与 Map Reader
> 两个文本游戏任务，并给出可复现的训练与评测流程。

---

## 1. 任务与约束

### 1.1 课程要求（硬约束）

| 项目 | 要求 |
|---|---|
| 基座模型 | Qwen3-1.7B 或 Qwen3-4B；**必须实际更新参数**（LoRA 需合并后提交） |
| 任务 | TextWorldExpress：Sorting、Map Reader |
| 评测动作预算 | 每个实例**最多 10 次动作** |
| 评测指标 | 成功率、成功时的平均步数、无效动作比例；两个任务各占 50% |
| 评测环境 | 单张 RTX 4090（24GB）；模型参数固定；不联网；不访问其他模型、参考答案或额外环境模拟 |
| 历史管理 | 不同实例之间清空交互历史 |
| 动作来源 | 必须由模型产生；不得内置任务专用求解器或参考策略替代模型 |
| 提交物 | `model/`、`run.py`、`code/`、`report.pdf`（ICLR 2025 模板，英文，≤10 页） |
| 数据 | 训练数据来源与规模需说明；不得用测试集训练或调参 |

### 1.2 环境实测事实（影响设计）

以下是在本机对 TextWorldExpress 1.1.0（commit `db0e6c0`）的实测结论：

1. **Sorting**：单房间；3–5 个物体，带单位（`mg/g/kg`，也可能出现体积/长度单位）；
   需要按换算后的数量升序，逐个 `take` 后 `put ... in box`。
   - `score = 正确放入数 / 总数`（0、0.2/0.25…、1.0）；
   - **顺序错误 → `score=-1.0`、`taskfailure=True`、回合立即结束**；
   - 5 个物体时最少 10 个动作，正好用满预算，零容错。
2. **Map Reader**：初始观测不含地图，必须执行 `read map`（消耗 1 步）；
   - 最少动作数 = `read map(1) + 2d + take(1) + put(1) = 2d + 3`（`d` 为起点到硬币的距离）；
   - 拿到硬币 `+0.5`，放入箱子再 `+0.5`；
   - 默认参数 `maxDistanceApart=4`：`d=4` 时 `2d+3=11 > 10`，**在不读地图、且第一次就选对方向的
     情况下**才可能用 10 步完成（`2d+2`）。也就是说默认难度下 `d=4` 实例在 10 步预算内
     只有很小概率能满分，这一点需要和助教确认评测配置（见 §13）。
3. **随机种子分折**：train `0–999`、dev `10000–10999`、test `20000–20999`，各 1000 个；
   同一个 seed 在不同 fold 下生成不同实例。
4. **接口**：`reset(seed, gameFold, generateGoldPath)`、`step(action)` 返回
   `(obs, reward, done, infos)`；`infos['validActions']` 给出当前合法动作；
   `infos['score']` 是归一化分，`reward` 是相对上一步的增量。
5. **性能**：JVM 启动约 0.24 s；Python 桥接步进约 1.6k steps/s。环境不是瓶颈，
   模型生成才是。
6. `getObjectTree()` 在 Windows 下有文件锁 bug（不影响训练/评测）。

### 1.3 本方案的内部分数目标（非课程承诺）

| 任务 | 目标成功率（dev） | 目标无效动作比例 | 目标成功平均步数 |
|---|---|---|---|
| Sorting | ≥ 85% | ≤ 2% | ≤ 8（5 物体时接近 10） |
| Map Reader | ≥ 65%（d≤3） | ≤ 3% | ≈ 2d+3 |

这些目标用于判断训练是否收敛，不作为报告中的结论；最终以课程测试集为准。

---

## 2. 总体技术路线

### 2.1 三阶段流水线

```
阶段 0（基线）        阶段 1（冷启动）           阶段 2（在线强化）          阶段 3（评测）
zero-shot 评测  →  SFT（专家轨迹，LoRA）  →  GRPO（环境在环，LoRA）  →  合并权重，run.py
   ↓                    ↓                         ↓                        ↓
 确认任务可解性      学会格式与基本策略        优化成功率/步数/无效动作      dev → 最终 test
```

- **为什么需要 SFT 冷启动**：GRPO 的奖励是回合级的稀疏信号；4B 模型在 zero-shot 下
  很可能连“只输出一个合法动作”的格式都不稳定，直接 GRPO 会得到大量全零组
  （组内奖励无方差 → 优势全为 0）。先用少量专家轨迹把格式和基本策略教会，再上 GRPO。
- **为什么用 GRPO 而不是 PPO**：不需要价值网络，省显存；组内相对优势适合
  回合级、稀疏、可自动验证的奖励；与课程“奖励设计自行决定”的要求契合。
- **为什么用 LoRA**：单卡 24GB 下，4B 全参数 + Adam 状态 + 激活无法容纳；
  LoRA 只训练约 1–2% 参数，且评测时合并后与全参数模型无差别（满足“LoRA 需先合并”）。

### 2.2 系统组成

```
┌──────────────────────────── 训练（远程 Linux GPU） ────────────────────────────┐
│  instance pool ──► rollout workers ──► trajectory buffer ──► GRPO updater       │
│  (JSONL 清单)      (env + policy)       (turn samples)        (LoRA + AdamW)     │
│                        ▲                      │                                  │
│                        └──── TextWorldExpress (JVM, py4j) ◄── 动作/奖励 ─────────┘
└──────────────────────────────────────────────────────────────────────────────────┘
┌──────────────────────────── 评测（单卡 4090 24GB） ────────────────────────────┐
│  run.py：加载合并后的模型 → 逐实例 reset → ≤10 步交互 → 指标 + 轨迹 JSONL        │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 关键设计决策

| 决策 | 选择 | 理由 |
|---|---|---|
| 模型 | Qwen3-4B-Base（默认；官方 Qwen3-4B 后训练版为备选） | 课程指定 4B；纯基座无 chat template，SFT 冷启动必需；官方版只换 renderer（见 `docs/DATA.md` 附录 B） |
| 微调方式 | LoRA（r=32），评测前合并 | 24GB 显存；提交要求允许 |
| 训练样本 | 按“回合（turn）”切分，而非整条轨迹 | 动作只有几十个 token，避免长序列 logits 显存爆炸 |
| rollout 引擎 | 先 HF `generate`，vLLM 作为加速备选 | 轨迹短（≤10 步 × ≤48 token），HF 已够用；vLLM 与训练共卡复杂 |
| 参考策略 | 关闭 LoRA adapter 的同一模型 | 零额外显存，得到精确的参考分布 |
| 动作约束 | 默认不强制约束（自由生成 + 1 次修复重试） | 保留“无效动作比例”的真实含义；约束解码作为消融 |
| 评测解码 | greedy（temperature=0） | 可复现；避免评测随机性 |

---

## 3. 交互 Harness 设计

Harness 是“环境 ↔ 模型”的胶水，训练和评测共用同一套代码，保证分布一致。

### 3.1 环境封装

```python
class TWXEnv:
    def reset(self, game: str, seed: int, fold: str, params: str) -> tuple[str, list[str]]
    def step(self, action: str) -> StepResult   # obs, score, success, failure, done, valid_actions
    def close(self) -> None
```

- 每个进程持有少量 env（JVM 常驻），按实例复用；`reset` 清空状态。
- `done` 由 harness 自己判定：`steps >= 10` 或 `task success` 或 `task failure`；
  **不依赖** `envStepLimit` 的 off-by-one 行为。
- 动作原样传给环境（小写、去首尾空白），环境返回 `Unknown action` 记为无效动作。

### 3.2 Prompt 模板

> 下面是逻辑结构；实际渲染随基座切换（Base 用纯文本模板、官方版用 chat template），
> 具体模板与截断规则见 [`docs/DATA.md`](docs/DATA.md) 附录 B。

```
<system>
You are an agent playing a text game. At each turn, choose exactly one action.
Reply with the action inside <action></action> and nothing else.
</system>

<user>
Task: {task_description}

Current observation:
{observation}

Inventory: {inventory}

Valid actions:
- take 25g of oak
- put 25g of oak in box
...

History:
Step 1: take 25g of oak → You take the 25g of oak.
Step 2: put 25g of oak in box → You put the 25g of oak in the box.
</user>
```

- **历史**：保留最近 `k=6` 步的“动作 → 观测摘要”（观测超过 120 token 时截断）；
  初始观测与任务描述始终保留。总 prompt 上限 **1536 token**，超限从最旧的历史开始丢。
- **合法动作列表**：训练与评测都提供（与人类控制台一致）；消融时去掉，观察影响。
- **不提供 gold path、地图内容、物体数量等任何环境内部信息**；地图必须由模型
  自己 `read map` 获得（这是任务的一部分）。

### 3.3 输出格式与解析

模型输出：

```
<action>take 25g of oak</action>
```

- 解析规则：取最后一个 `<action>...</action>`；若缺失，取第一行非空文本；
- **修复重试**：解析失败或动作不在 `validActions` 中时，追加一条
  “Your previous reply could not be parsed. Reply with exactly one action from the list.”
  再生成一次（temperature=0）；仍失败则把原始文本发给环境，计为无效动作。
  - 重试**不消耗环境动作预算**（课程允许自行决定重试方式），但每次重试计入日志与报告。
- 无效动作定义（用于指标）：环境返回 `Unknown action` 或最终无法解析。

### 3.4 回合执行伪代码

```python
def run_episode(env, policy, instance, max_actions=10):
    obs, valid = env.reset(**instance)
    history = []
    invalid = 0
    for step in range(max_actions):
        prompt = build_prompt(instance.task, obs, env.inventory, valid, history)
        action, meta = policy.act(prompt, valid_actions=valid)   # 含 1 次修复重试
        result = env.step(action)
        if meta.unparsed or result.unknown_action:
            invalid += 1
        history.append((action, shorten(result.obs)))
        obs, valid = result.obs, result.valid_actions
        if result.success or result.failure or step + 1 == max_actions:
            break
    return Trajectory(steps=history, invalid=invalid, success=result.success,
                      score=result.score, steps_used=len(history))
```

### 3.5 训练与评测的一致性

- 同一份 `build_prompt`、同一个解析器、同一套无效动作判定；
- 训练 rollout 用 `temperature=1.0, top_p=0.95`；评测用 greedy；
- 训练时允许修复重试，评测也保留同样策略（并在报告中说明）。

---

## 4. 数据构造

数据分两类：**实例（prompt 池）** 和 **专家轨迹（SFT 用）**。
两类都只使用 train fold；dev/test 只用于选模和最终评测。

> 详细的数据清单、字段规范、生成命令与验收标准见 [`docs/DATA.md`](docs/DATA.md)。

### 4.1 实例生成

所有实例由 TextWorldExpress 程序化生成，无需人工标注。

#### Sorting

- 固定 `gameParams=""`；枚举 train seed `0–999`；
- 读取 `getGenerationProperties()['hidden_problemsize']`（3/4/5）做难度分桶；
- 每种规模保留约 200 个实例，共约 600 个 prompt。

#### Map Reader

- 参数网格（控制难度，同时避免 `d=4` 的预算问题）：

| 参数 | 取值 |
|---|---|
| `numLocations` | 6, 8, 10 |
| `maxDistanceApart` | 1, 2, 3 |
| `maxDistractorItemsPerLocation` | 0, 1, 2 |
| `includeDoors` | 0 |
| `limitInventorySize` | 0 |

- 每个 (params, seed) 组合生成后读取 `actualDistanceApart` 分桶（1/2/3），
  每桶保留约 300 个，共约 900 个 prompt；
- 同一 `(game, params, seed)` 只保留一个实例（去重键）；
- `d=4` 的实例单独留 50 个作为**压力测试集**，不进入训练主分布（预算原因，见 §1.2）。

#### 清单格式（JSONL）

```json
{"id":"sorting-train-000123","game":"sorting","params":"","seed":123,"fold":"train",
 "difficulty":{"size":4},"task":"...","initial_obs":"..."}
{"id":"map-train-000045-d2","game":"mapreader","params":"numLocations=8,maxDistanceApart=2,maxDistractorItemsPerLocation=1",
 "seed":45,"fold":"train","difficulty":{"distance":2},"task":"...","initial_obs":"..."}
```

### 4.2 难度分桶与课程

按 (任务, 难度) 分桶，训练时按桶采样：

| 桶 | Sorting | Map Reader |
|---|---|---|
| easy | size=3 | d=1 |
| medium | size=4 | d=2 |
| hard | size=5 | d=3 |

课程策略：先用 easy+medium 训练，成功率超过阈值后逐步加入 hard；
每个桶维护最近 50 次的成功率，低于 0.2 的桶加权过采样，高于 0.8 的桶降权。
这样 GRPO 的组内奖励始终有方差，避免“全对/全错”的无效更新。

### 4.3 专家轨迹（SFT 冷启动）

专家轨迹用**确定性构造 + 校验**生成，只在离线数据构造阶段使用，
**不进入评测代码**（满足“评测时不得内置求解器”）。

#### Sorting 专家

1. 从初始观测解析物体与数量（如 `21kg of marble`、`47g of brick`）；
2. 单位换算到统一基准（质量：mg；体积：ml；长度：mm）；
3. 按数量升序排列；数量相同则尝试两种顺序，取能成功的那种；
4. 依次执行 `take <obj>`、`put <obj> in box`；
5. 用环境校验 `score == 1.0`，失败则丢弃该条。

#### Map Reader 专家

1. `read map`，解析 `The <room> connects to <room1>, <room2> and <room3>.` 得到邻接表；
2. 从任务描述得到硬币所在房间，从观测得到起点房间；
3. 在邻接表上 BFS 求最短路径；
4. 逐步导航：在当前观测里找 `To the <Dir> you see the <next_room>.`，执行 `move <dir>`；
5. `take coin`；按最短路返回起点；`put coin in box`；
6. 校验 `tasksuccess`，且动作数 ≤10（`d=4` 的轨迹会超预算，直接丢弃）。

#### 规模与格式

- Sorting 600 条、Map Reader 900 条专家轨迹（每条含 5–10 个 turn）；
- 每个 turn 的训练样本 =（prompt, 专家动作）；prompt 中的历史使用**专家自己的历史**
  （teacher forcing），保证训练/推理一致；
- 按实例 95/5 划分 SFT 训练/验证，验证集只用于选 epoch，不参与梯度。

### 4.4 数据质量检查（写进 `code/data/validate.py`）

- [ ] 不包含 test fold seed（20xxx）；
- [ ] 实例 ID 唯一，`(game, params, seed)` 无重复；
- [ ] 每条专家轨迹都能在环境中重放成功（100% 校验）；
- [ ] prompt 长度分布：P99 ≤ 1536 token；
- [ ] 每个难度桶的实例数、平均步数、单位类型统计写入 `data/stats.json`。

### 4.5 测试集策略（防泄漏）

| 集合 | 来源 | 用途 |
|---|---|---|
| train | seed 0–999 | 生成 prompt 池与 SFT 数据 |
| dev | seed 10000–10999，固定 200 条 | 选 SFT epoch、GRPO checkpoint、消融对比 |
| local-test | seed 20000–20999，固定 200 条 | **封存**，所有决策完成后只跑一次 |
| official-test | 课程提供 | 最终提交前运行，只跑一次 |

- dev/local-test 清单在项目开始时生成并冻结（记录文件哈希）；
- 任何超参、奖励权重、prompt 模板的选择都只看 dev；
- 报告中如实说明 local-test 与 official-test 的使用次数。

---

## 5. 训练基础设施

### 5.1 模型与精度

- `Qwen/Qwen3-4B-Base`（默认；官方 `Qwen/Qwen3-4B` 为备选），bf16；Base 用纯文本模板，
  官方版用 chat template + `enable_thinking=False`（见 `docs/DATA.md` 附录 B）；
- LoRA：`r=32, alpha=64, dropout=0.05`，目标模块
  `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`；
- 训练精度 bf16 + TF32；开启 gradient checkpointing；
- 评测前 `merge_and_unload()` 得到完整权重。

### 5.2 24GB 显存预算（单卡 4090）

| 项目 | 估算 |
|---|---|
| 4B bf16 权重 | 8.0 GB |
| LoRA 参数 + AdamW 状态 | ~0.6 GB |
| 激活（micro-batch=1，1.5k prompt，grad ckpt） | 3–5 GB |
| rollout KV cache（64 条 × 2k context，GQA） | ~0.6 GB |
| logits（只保留动作 token 的 logits） | < 0.1 GB |
| **峰值** | **约 13–15 GB** |

- 关键技巧：**按 turn 训练**，每条样本以动作结尾，前向时只保留最后 `len(action)` 个位置的
  logits（`logits_to_keep`），避免 `batch × seq × vocab` 的大张量；
- 参考模型不另存：计算 ref logprob 时临时 `disable_adapter()`；
- 若仍 OOM 的降级顺序：prompt 1536→1024 → micro-batch 1 → 关掉 KL → 4-bit QLoRA。

### 5.3 训练样本表示

一条轨迹被拆成若干 turn 样本：

```python
TurnSample(
    input_ids   = [prompt tokens] + [action tokens],
    action_mask = [0]*len(prompt) + [1]*len(action),
    adv         = A_i,          # 整条轨迹共享同一个组内优势
    logp_old    = [...],        # rollout 时记录，更新前可用训练模型重算
)
```

- 同一轨迹的所有 turn 共享优势 `A_i`（轨迹级信用分配）；
- 训练时按 turn 打乱、拼 batch；loss 只在 `action_mask=1` 的 token 上计算；
- 这样既保留了多轮上下文，又把序列长度控制在 1.5k 左右。

### 5.4 工具链（建议）

```text
torch 2.4+ / transformers 4.51+ / peft / accelerate / datasets
py4j / orjson（TextWorldExpress 依赖）
可选：vllm（rollout 加速）、liger-kernel（chunked CE）
```

- **训练框架：自研轻量 GRPO loop**（约 500 行，rollout + loss + 日志），
  不用 TRL/verl 的现成多轮接口。理由：多轮环境交互 + turn 级样本 + 
  “关闭 adapter 当参考模型”这些需求在通用框架里都要改源码，自研更可控、也便于写报告。
- 若后续需要多卡：同一套代码用 `accelerate`/`FSDP` 包装即可；LoRA 下不需要 ZeRO-3。

---

## 6. GRPO 设计

### 6.1 目标函数

对每个 prompt `q`，采样 `G` 条轨迹 `{τ_i}`，得到奖励 `{R_i}`，组内标准化：

```
A_i = (R_i - mean(R)) / (std(R) + 1e-4)
```

对轨迹中的每个动作 token（turn `t`，token `j`）：

```
ρ_{i,t,j} = exp( log π_θ(a_{i,t,j} | s_{i,t,j}) - log π_old(a_{i,t,j} | s_{i,t,j}) )
L = - 1/Σ|a| Σ_{i,t,j} min( ρ·A_i, clip(ρ, 1-ε, 1+ε)·A_i )
    + β · KL(π_θ || π_ref)
```

- `ε=0.2`（单侧上限 0.28，可选）；
- `β=0.001`，KL 用 k3 估计（无偏、方差小）；`β=0` 作为消融；
- 参考分布 `π_ref` = 关闭 LoRA adapter 的同一模型；
- 不加价值网络、不加 GAE；熵正则默认关闭（短动作空间下容易抑制探索）。

### 6.2 Rollout

| 项目 | 设置 |
|---|---|
| 每个 prompt 的轨迹数 `G` | 8（消融 4/16） |
| 每个 update 的 prompt 数 | 8（共 64 条轨迹） |
| 采样温度 / top_p | 1.0 / 0.95 |
| 每条动作最大 token | 48（`</action>` 后停止） |
| 每回合最大动作数 | 10 |
| 早停 | 成功/失败立即结束 |
| 环境 | 每 worker 常驻 1 个 JVM，顺序执行 64 条轨迹 |

- 同一 prompt 的 8 条轨迹共享初始观测，但采样独立；
- 记录每条轨迹的 `score / success / failure / invalid_count / steps`；
- **退化组处理**：若 8 条全成功或全失败，该组优势全为 0，直接跳过（不产生梯度），
  并把该实例放回池子、调整采样权重。

### 6.3 奖励设计

回合级总奖励（所有 turn 共享）：

```
R = 1.0  * success
  + 0.5  * max(score, 0)          # 稠密进度：sorting 的完成比例 / mapreader 的 0.5
  - 0.5  * taskfailure            # sorting 顺序错误（score=-1）
  - 0.05 * invalid_count          # 抑制无效动作（对应评测指标）
  - 0.01 * steps                  # 鼓励更短的成功路径（对应平均步数指标）
  - 0.05 * unparsed_count         # 格式错误
```

- 取值范围约 `[-1.0, 1.5]`，组内标准化后使用；
- **为什么用 `score` 做稠密项**：sorting 每放对一个 +1/N，mapreader 拿到硬币 +0.5，
  都是环境自带的、无法伪造的信号；不用人工势函数，避免 reward hacking；
- **失败惩罚**：sorting 放错顺序会直接 `-1`，模型必须学会“先比较再动手”；
- **步数惩罚很小**（0.01/步）：主要作用是让“成功但磨蹭”与“成功且高效”区分开，
  同时不鼓励冒险；
- 所有系数在 dev 上做一次小网格（见 §9），不针对 test 调整。

### 6.4 优势与 loss 细节

- 优势按轨迹广播到该轨迹的所有动作 token；
- `logp_old` 在 rollout 时记录；更新前用训练模型（adapter 开启、no_grad）重算一遍，
  避免 HF generate 与训练前向的数值差异；
- loss 按动作 token 总数归一化，避免长轨迹主导梯度；
- 梯度裁剪 1.0；AdamW（LoRA 参数，`lr=1e-5`，weight decay 0.01，cosine + 5% warmup）；
- 每 `K=4` 个 update 同步一次“旧策略”（把当前 adapter 视为下一步的 behavior policy）；
- 每 25 个 update 在 dev 上评一次，保存 dev 最优 checkpoint（按两任务平均成功率）。

### 6.5 课程与采样

```
update 0–80    : easy + medium（sorting 3/4，mapreader d=1/2）
update 80–200  : 全分布（加入 sorting 5、d=3）
update 200–300 : 难例优先（成功率 <0.4 的实例权重 ×2）
```

- 每个桶的目标成功率区间 `[0.2, 0.8]`；低于下界过采样，高于上界降权；
- 每 20 个 update 重新统计一次桶成功率；
- prompt 池不重复使用过快：1000+ 条实例、300 个 update × 8 条 ≈ 2400 次抽取，
  平均每条被抽 2 次左右，可接受。

### 6.6 稳定性与反 reward hacking

监控指标（每个 update 记录）：

| 指标 | 健康范围 | 异常处理 |
|---|---|---|
| 组内奖励标准差 | > 0.05 | 太低 → 提高难度或温度 |
| clip 比例 | 0.05–0.3 | 过高 → 降 lr |
| KL | < 0.02 | 过高 → 增大 β 或降 lr |
| 平均动作 token 数 | 10–30 | 暴涨 → 检查是否在“念咒” |
| 无效动作率 | 下降趋势 | 上升 → 检查解析器/格式 |
| 成功率 | 上升 | 停滞 → 换课程或检查奖励 |
| 步数（成功时） | 下降或稳定 | 上升 → 加大步数惩罚 |

反 hacking 检查：每周人工看 20 条轨迹，确认没有“反复 look around 刷分”、
“输出合法动作列表原文”等退化行为。

### 6.7 超参数总表

| 超参 | 值 |
|---|---|
| LoRA r / alpha / dropout | 32 / 64 / 0.05 |
| 学习率 | 1e-5（GRPO），1e-4（SFT） |
| batch | 8 prompts × G=8 |
| micro-batch | 1 turn |
| 梯度累积 | 使每次更新约 64 个 turn |
| max prompt / action tokens | 1536 / 48 |
| PPO clip ε | 0.2 |
| KL 系数 β | 0.001 |
| 训练 update 数 | 300（早停） |
| 采样温度 | 1.0 |
| 评测温度 | 0（greedy） |

---

## 7. 训练流程（可执行步骤）

### 阶段 0：基线与数据（1–2 天）

1. 搭好 `code/env`（TWX 封装、prompt、解析器）与 `code/eval`；
2. 生成 train/dev/local-test 清单（§4.1），跑数据校验；
3. 跑 zero-shot Qwen3-4B（非思考）在 dev 上的基线，记录成功率/步数/无效动作；
4. 确认 sorting、mapreader 的可解性与预算（尤其 `d=4`）。

### 阶段 1：SFT 冷启动（2–3 天）

1. 生成并校验专家轨迹（§4.3）；
2. LoRA SFT：1–2 epoch，lr 1e-4，batch 8，prompt 1536；
3. 每 100 步在 dev 上评一次，选最佳 epoch；
4. 目标：dev 成功率 sorting ≥50%、mapreader ≥30%，无效动作 ≤5%。

### 阶段 2：GRPO（5–7 天）

1. 从 SFT checkpoint 初始化；
2. 按 §6.5 课程训练 300 update；
3. 每 25 update 在 dev 上评，保存最佳；
4. 若 100 update 内 dev 无提升：检查奖励尺度、组内方差、KL、学习率；
5. 训练日志、配置、随机种子全部落盘。

### 阶段 3：收尾（2–3 天）

1. 合并 LoRA，跑 local-test（仅一次）；
2. 若时间允许，做 §9 的消融；
3. 生成 `run.py` 的最终配置与说明；
4. 写报告、整理提交物。

**预计总时长**：4 周左右（含 buffer）。若服务器只有单卡 4090，GRPO 300 update
按每次 3–5 分钟估算约 15–25 小时，可分多晚跑完。

---

## 8. 评测方案（`run.py`）

### 8.1 接口

```bash
python run.py \
  --model-dir model/ \
  --config eval_config.json \
  --output results/test_run.jsonl
```

`eval_config.json`：

```json
{
  "games": ["sorting", "mapreader"],
  "instances": "data/test_instances.jsonl",
  "max_actions": 10,
  "temperature": 0.0,
  "max_new_tokens": 48,
  "prompt_max_tokens": 1536,
  "seed": 1234
}
```

### 8.2 指标

- `success_rate`：`成功实例数 / 实例总数`（分任务 + 总体）；
- `avg_steps_on_success`：仅统计成功实例的动作数均值；
- `invalid_action_ratio`：无效动作数 / 总动作数（分任务 + 总体）；
- `score`：按课程“两任务各 50%”的加权（默认用成功率；同时报告其他指标）；
- 逐实例输出：`id, game, seed, params, success, steps, invalid, actions[], observations[]`。

### 8.3 约束落实

| 要求 | 实现 |
|---|---|
| 参数固定 | 加载合并后的 bf16 权重，`model.eval()`，无 adapter 切换 |
| 不联网 | 本地模型目录；`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1` |
| 不访问其他模型/参考答案 | 代码中不 import 任何 API；不读取 gold path |
| 不同实例清空历史 | 每个实例新建 prompt，history 从空开始 |
| 10 次动作 | harness 计数，第 10 步后强制结束 |
| 24GB 预算 | bf16 4B 约 8GB；单进程；greedy |
| 可复现 | 固定 seed、greedy、依赖版本锁定 |

### 8.4 运行预算

- 200 个实例 × 10 步 × ≤48 token ≈ 10 万 token 生成；4090 上几分钟到十几分钟；
- 环境步进可忽略（1.6k steps/s）；
- 不需要 vLLM；若加载慢，可缓存 tokenizer/模型。

---

## 9. 实验与分析计划

### 9.1 主实验

| 配置 | 说明 |
|---|---|
| Qwen3-4B zero-shot | 非思考模式，直接评测 |
| + SFT | 专家轨迹冷启动 |
| + GRPO | 完整方案（主结果） |

### 9.2 消融

1. 奖励：去掉稠密 `score` 项 / 去掉无效动作惩罚 / 去掉步数惩罚；
2. 课程：uniform 采样 vs 课程采样；
3. 组大小：G=4 vs 8；
4. 动作约束：自由生成 + 修复 vs 合法动作前缀约束解码；
5. 思考模式：非思考 vs 思考（token 预算 256）；
6. 模型规模：Qwen3-1.7B vs 4B（若时间允许）；
7. 微调方式：LoRA vs 全参数（若服务器显存允许）。

### 9.3 错误分析（报告素材）

- Sorting：单位换算错误、顺序比较错误、拿错物体、超步数；
- Map Reader：不用 `read map`、导航绕路、忘记返回、拿硬币失败、超步数；
- 格式：未按 `<action>` 输出、输出解释性文字、非法动作。

每类抽 10 条轨迹，统计占比并给出典型例子。

### 9.4 报告结构（对应课程要求）

1. Abstract；2. Introduction；3. Related Work（TextWorld/TextWorldExpress、GRPO、
   LLM agent）；4. Method（harness、数据、SFT、GRPO、奖励）；5. Experiments
   （主结果、消融、错误分析、case study）；6. Conclusion & Reflection；References。
   图表：训练曲线、两任务成功率对比、步数分布、无效动作率、典型轨迹。

---

## 10. 工程结构与复现

```
PRML_pj3/
├── README.md
├── plan.md
├── run.py                     # 评测入口（最终提交）
├── code/
│   ├── env/                   # TWX 封装、prompt、解析、回合执行
│   ├── data/                  # 实例生成、专家轨迹、校验、清单
│   ├── train/                 # sft.py、grpo.py、rollout.py、reward.py
│   ├── eval/                  # 本地评测脚本、指标
│   └── configs/               # yaml/json 配置
├── data/                      # 清单与统计（大文件 gitignore）
├── model/                     # 合并后的权重（不入库，提交时单独打包）
├── results/                   # 评测输出与日志（gitignore）
└── report/                    # LaTeX 源文件与图
```

- 依赖锁定：`requirements.txt` + `environment.yml`；TextWorldExpress 固定 commit
  `db0e6c0` 与 jar 的 sha256；Java 版本写入文档；
- 配置驱动：所有超参在 `code/configs/*.yaml`，不在代码里写死；
- 日志：每个 run 一个目录（`config.yaml`、`train.jsonl`、`dev_eval.jsonl`、checkpoint）；
- 随机种子：Python/NumPy/Torch/采样全部固定；评测 greedy；
- 单测：prompt 构造、解析器、奖励计算、env 封装各 3–5 个用例；
- CI 可选：本地 `pytest` + 5 个实例的 smoke test。

---

## 11. 里程碑

| 周 | 交付 |
|---|---|
| W1 | harness + 数据清单 + 专家轨迹 + zero-shot 基线 |
| W2 | SFT + dev 评测 + 第一版 GRPO 跑通（少量 update） |
| W3 | GRPO 调参 + 课程 + 消融 |
| W4 | local-test、报告初稿、提交物整理 |

> 如果课程截止日期更紧，优先级：harness > SFT > GRPO 主结果 > 消融 > 额外分析。

---

## 12. 风险与应对

| 风险 | 影响 | 应对 |
|---|---|---|
| Map Reader `d=4` 在 10 步内不可解 | 上限受限 | 训练聚焦 d≤3；向助教确认评测参数；报告中分析 |
| 24GB OOM | 训练中断 | LoRA + turn 级样本 + grad ckpt；降 prompt 长度；QLoRA 兜底 |
| GRPO 组内无方差 | 无梯度 | 课程采样、难度桶、G=8、温度 1.0 |
| 多轮 GRPO 不稳定 | 崩溃/退化 | SFT 初始化、KL、clip、小 lr、每 25 update 评测 |
| Reward hacking（无效动作/刷步数） | 指标虚高 | 无效动作惩罚 + 步数惩罚 + 人工轨迹检查 |
| 测试集泄漏 | 成绩无效 | 冻结本地 test 清单与哈希；只跑一次 |
| 时间不足 | 无法完成 | 先保 4B SFT+GRPO 主结果；消融可裁剪 |
| Windows 开发 / Linux 训练差异 | 环境问题 | 依赖锁定；Docker/conda；Java 版本写文档 |

---

## 13. 待确认问题（需和助教/组员对齐）

1. **评测配置**：课程统一环境的 `gameParams` 与测试实例数量？特别是 Map Reader 的
   `maxDistanceApart` 是否包含 4（10 步预算下 `d=4` 基本不可满分）。
2. **动作约束**：评测是否允许用 `validActions` 做约束解码/动作校验？还是必须自由生成？
   （本方案默认自由生成 + 修复重试，约束解码仅作消融。）
3. **模型变体**：使用 `Qwen3-4B-Base`（纯基座）是否满足课程“Qwen3-4B 为基座”的要求？
   若必须使用官方后训练版，我们只切换 renderer，数据不变。
4. **训练算力**：课程是否对训练时长/卡数有隐性要求？（评测预算明确是 4090 24GB。）
5. **提交细节**：`model/` 的打包方式、`run.py` 的配置格式是否有统一模板？
6. **评分口径**：“两个任务各占 50%”是按成功率加权，还是综合三项指标？报告里如何呈现？

---

## 附录 A：与课程 `plan.md` 的对应关系

| 课程要求 | 本方案位置 |
|---|---|
| 基座 Qwen3-4B、实际更新参数 | §2.3、§5.1（LoRA + 合并） |
| 训练算法/奖励/数据自定 | §4、§6 |
| 交互 harness 自选 | §3 |
| 10 次动作、三项指标、50/50 | §1.1、§8.2 |
| 不联网、固定参数、清空历史 | §8.3 |
| 不得内置求解器 | §4.3（专家仅用于离线 SFT）、§3.3 |
| 数据来源与规模 | §4.4、报告 Method |
| 不用测试集训练/调参 | §4.5 |
| 提交物 | §10 |
| 报告要求 | §9.4 |
