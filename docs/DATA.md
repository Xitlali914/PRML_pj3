# 数据生成与数据清单（DATA v1.0 草案）

> 配套文档：[plan.md](../plan.md)。本文档只回答三件事：
> **需要哪些数据、每条数据长什么样、怎么生成和验收。**
> 实现代码放在 `code/data/`，生成产物放在 `data/`。
>
> 基座约定：本文默认 **`Qwen/Qwen3-4B-Base`**（纯基座、无 chat template，因此 SFT 冷启动是必需的）。
> 若最终采用官方后训练版 `Qwen/Qwen3-4B`（带 chat template、关闭 thinking），
> 数据内容不变，只需切换 renderer，见附录 B。

---

## 0. 约定

1. **只使用 train fold（seed 0–999）生成训练数据**；dev（10000+）只用于选模，
   test（20000+）封存，官方测试集由课程提供。
2. **数据与渲染解耦**：所有文本按结构化字段存储（task / observation / history / action），
   训练时再由 `renderer` 拼成 prompt。这样换基座、改模板不用重生成数据。
3. **可复现**：每个文件带 `content_hash`，生成脚本固定随机种子；同名文件重新生成应字节一致。
4. **规模是目标值**，实际数量与分布由 `code/data/stats.py` 统计并写入 `data/stats.json`。

---

## 1. 数据总览：需要哪些数据

### 1.1 训练数据

| # | 数据 | 文件 | 目标规模 | 生成方式 | 用途 |
|---|---|---|---|---|---|
| T1 | Sorting 训练实例池 | `data/manifests/sorting_train.jsonl` | 1000 条 | seed 0–999 全量，读 `hidden_problemsize` 分桶 | GRPO 的 prompt 池 |
| T2 | Map Reader 训练实例池 | `data/manifests/mapreader_train.jsonl` | 3000 条（d=1/2/3 各约 1000） | 参数网格 × seed，按 `actualDistanceApart` 分层 | GRPO 的 prompt 池 |
| T3 | Sorting 专家轨迹 | `data/sft/sorting_expert.jsonl` | 600 条轨迹（≈4800 turn） | 确定性构造 + 环境重放校验 | SFT |
| T4 | Map Reader 专家轨迹 | `data/sft/mapreader_expert.jsonl` | 900 条轨迹（≈6300 turn） | `read map` + BFS 导航 + 环境校验 | SFT |
| T5 | SFT turn 展开集 | `data/sft/turns_train.jsonl`、`turns_val.jsonl` | ≈10.5k / 550 turn | 由 T3/T4 展开，95/5 划分 | SFT 直接输入 |
| T6 | 可选：RFT 数据 | `data/rft/*.jsonl` | ≤5k turn | GRPO rollout 中成功的轨迹 | 可选第三阶段 SFT |

> **重要概念**：GRPO 不需要“标准答案动作”，它只需要实例池（T1/T2）+ 环境奖励。
> 专家轨迹（T3/T4）只服务于 SFT 冷启动；GRPO 阶段是 on-policy 自己采样。

### 1.2 验证与测试数据

| # | 数据 | 文件 | 规模 | 用途 |
|---|---|---|---|---|
| V1 | dev 集 | `data/manifests/dev.jsonl` | 200（100 sorting + 100 mapreader） | 选 SFT epoch、GRPO checkpoint、消融 |
| V2 | local-test | `data/manifests/local_test.jsonl` | 200（100 + 100） | 封存，所有决策完成后**只跑一次** |
| V3 | d=4 压力集 | `data/manifests/mapreader_stress_d4.jsonl` | 50 | 只做压力测试，**不进训练** |
| V4 | 官方测试集 | 课程提供（格式待确认） | 未知 | 最终提交前跑一次 |

### 1.3 运行与分析数据（不参与训练）

| # | 数据 | 位置 | 规模 | 说明 |
|---|---|---|---|---|
| A1 | GRPO rollout 日志 | `runs/<run_id>/rollouts/update_*.jsonl.gz` | 约 19k 条轨迹 / 50–100MB | 训练诊断、case study |
| A2 | 训练指标 | `runs/<run_id>/train.jsonl` | 每 update 一行 | 曲线、消融 |
| A3 | dev 评测记录 | `runs/<run_id>/dev_eval.jsonl` | 每 25 update | 选模 |
| A4 | 错误分析样本 | `data/analysis/error_samples.jsonl` | 每类 10 条，约 100 | 报告 §错误分析 |
| A5 | gold path（仅分析） | `data/analysis/gold_paths.jsonl` | 与实例池同规模 | **绝不进入 prompt 或训练目标** |
| A6 | 评测输出 | `results/*.jsonl` | 每次评测一份 | 指标与逐轮轨迹 |

### 1.4 元数据

| # | 数据 | 文件 | 说明 |
|---|---|---|---|
| M1 | 划分哈希 | `data/manifests/split_hashes.json` | 每个 manifest 的 sha256，冻结后不再变 |
| M2 | 统计 | `data/stats.json` | 各桶数量、长度分位数、步数分布 |
| M3 | 数据卡 | `data/DATASHEET.md` | 报告 Method 直接引用 |
| M4 | 数据配置 | `code/configs/data.yaml` | 参数网格、数量、种子 |

---

## 2. 基座与数据格式的关系

| | `Qwen3-4B-Base`（默认） | `Qwen3-4B` 官方版（备选） |
|---|---|---|
| chat template | 无 | 有 |
| prompt 渲染 | 纯文本模板（附录 B.1） | `apply_chat_template`，`enable_thinking=False` |
| zero-shot 格式能力 | 很弱 | 较好 |
| SFT 必要性 | 必需 | 建议（可减小规模） |
| 目标 token | `<action>...</action>` + EOS | 同左 + `<|im_end|>` |
| 数据是否要改 | 否 | 否，只换 renderer |

所有数据文件都只存**结构化内容**，`renderer` 版本记录在训练配置里（`renderer: base-v1` / `chat-v1`）。

---

## 3. 训练数据

### 3.1 实例池（T1 / T2）

实例池是 GRPO 的“题目集合”。每条实例 = 一个 (game, params, seed) 对应的初始局面。

#### 3.1.1 Sorting（T1）

- 生成：`seed ∈ [0, 999]`，`params=""`；`reset(game="sorting", seed, fold="train")`；
- 读取 `getGenerationProperties()["hidden_problemsize"]`（3/4/5）作为难度桶；
- 记录初始观测、背包、合法动作、任务描述；
- 共 1000 条（不筛掉任何规模；训练时按桶均衡采样）。

#### 3.1.2 Map Reader（T2）

参数网格：

| 参数 | 取值 | 说明 |
|---|---|---|
| `numLocations` | 6, 8, 10 | 房间数 |
| `maxDistanceApart` | 1, 2, 3 | 起点到硬币的最大距离 |
| `maxDistractorItemsPerLocation` | 0, 1, 2 | 干扰物 |
| `includeDoors` | 0 | 简化 |
| `limitInventorySize` | 0 | 不限制 |

生成策略（分层拒绝采样）：

```python
buckets = {1: [], 2: [], 3: []}
for params in grid:                 # 27 组参数
    for seed in range(1000):        # 只允许 train fold
        reset(game="mapreader", seed=seed, params=params, fold="train")
        d = getGenerationProperties()["actualDistanceApart"]
        if d in buckets and len(buckets[d]) < 1000:
            buckets[d].append(record(...))
        if all(len(b) >= 1000 for b in buckets.values()):
            stop
```

- 只保留 `d ∈ {1,2,3}`；`d=4` 单独进入压力集 V3；
- 每桶约 1000 条，共约 3000 条；
- 记录 `actualDistanceApart`、`numLocations` 等生成属性，供分层采样。

#### 3.1.3 实例字段规范

```json
{
  "id": "sorting-train-000123",
  "game": "sorting",
  "params": "",
  "seed": 123,
  "fold": "train",
  "difficulty": {"bucket": "medium", "size": 4},
  "task": "Your task is to sort objects by quantity. ...",
  "initial_observation": "You are in the kitchen. ...",
  "initial_inventory": [],
  "initial_valid_actions": ["take 47g of brick", "inventory", "..."],
  "generation_properties": {"seed": 123, "hidden_problemsize": 4, "gameSet": 1},
  "env": {"package": "textworld_express", "version": "1.1.0", "jar_sha256": "..."},
  "source": "code/data/generate_instances.py",
  "content_hash": "sha256:..."
}
```

Map Reader 的 `difficulty` 为 `{"bucket": "d2", "distance": 2, "num_locations": 8}`。

#### 3.1.4 采样策略

- 先按**难度桶均匀**采样（d=1/2/3 或 size=3/4/5 等概率）；
- 桶内按实例均匀采样；
- 每个 update 记录被抽中的 `instance_id` 列表（写入 rollout 日志），保证可追溯；
- 课程阶段（easy→hard）只改变桶的权重，不改变数据文件；
- 池子用完后允许重复抽取，但在同一 update 内不重复。

---

### 3.2 SFT 专家轨迹（T3 / T4）

#### 3.2.1 为什么需要、边界在哪

- Base 模型不会输出 `<action>` 格式，也不会稳定地做单位换算/地图导航，需要少量示范；
- 专家只用于**离线数据构造**，不进入 `run.py`，不参与评测决策，符合“评测时不得内置求解器”；
- 专家轨迹必须能在环境中**逐条重放成功**，否则丢弃。

#### 3.2.2 Sorting 专家

```
输入：实例的 initial_valid_actions
1. 取出所有以 "take " 开头的动作，得到物体字符串（如 "25g of oak"）
2. 用正则解析数量与单位：(\d+(?:\.\d+)?)\s*(mg|g|kg|ml|l|mm|cm|m)\s+of\s+(.+)
3. 换算到统一基准：质量→mg，体积→ml，长度→mm
4. 按数值升序排序；若两个物体数值相同，尝试两种顺序，取能成功的一种
5. 依次执行 take <obj>、put <obj> in box
6. 校验 score == 1.0 且步数 == 2 × 物体数
```

- 不插入 `look around`/`inventory` 等多余动作；
- 出现未知单位 → 记录并跳过该实例（统计里报告跳过数）。

#### 3.2.3 Map Reader 专家

```
1. 执行 read map，解析 "The <room> connects to <room1>, <room2> and <room3>."
   - 先按 ", " 分割，再处理最后一段的 " and "
   - 房间名可含空格，不能按空格切
2. 起点：初始观测中的 "You are in the <room>."
3. 目标：任务描述中的 "coin that is located in the <room>"
4. 在邻接表上 BFS 求起点→目标的最短路径
5. 导航循环：
   - 在当前观测里找 "To the <North|South|East|West> you see the <room>."
   - 取路径上的下一个房间对应的方向，move <dir>
6. 到目标房间后 take coin
7. BFS 求目标→起点的最短路，反向导航
8. put coin in box
9. 校验 tasksuccess 且步数 == 2d + 3 ≤ 10；否则丢弃
```

- 专家轨迹**必须包含 `read map`**（教会模型主动读地图）；
- `d=3` 时 9 步，`d=4` 时 11 步 → `d=4` 轨迹全部丢弃；
- 每一步都重新读取观测来确认方向，不假设方向可跨房间复用。

#### 3.2.4 轨迹展开为 turn（T5）

SFT 以 turn 为单位训练：每个 turn 的输入 = 截至当前步的上下文，输出 = 专家动作。

```python
for t in range(len(trajectory)):
    context = build_context(
        task=instance.task,
        observation=trajectory[t].observation,
        inventory=trajectory[t].inventory,
        valid_actions=trajectory[t].valid_actions,
        history=trajectory[:t],           # 最近 6 步，超长截断
    )
    assert trajectory[t].action in trajectory[t].valid_actions
    write_turn(context, target=trajectory[t].action)
```

- 上下文构造必须与训练/评测的 `build_prompt` **完全一致**（同一份代码、同一套截断规则）；
- 历史截断规则：最多 6 步，单条观测超过 120 token 截断；
- 按**实例**划分 train/val（95/5），同一实例的所有 turn 只能落在同一侧。

#### 3.2.5 规模与划分

| 数据 | 轨迹数 | 平均步数 | turn 数 |
|---|---|---|---|
| Sorting（size 3/4/5 各 200） | 600 | 8 | ≈4800 |
| Map Reader（d 1/2/3 各 300） | 900 | 7 | ≈6300 |
| 合计 | 1500 | — | ≈11100 |
| train / val | — | — | ≈10545 / 555 |

- 如果 dev 显示欠拟合（格式仍不稳），优先**增加轨迹数**（生成成本低），而不是盲目加 epoch；
- 如果出现过拟合（train 准确率高、dev 格式/成功率下降），减少 epoch 或增大 LoRA dropout。

### 3.3 可选：RFT 数据（T6）

- 从 GRPO rollout 中筛出成功轨迹（`success=true`），按实例去重；
- 只保留模型自己生成的轨迹（on-policy），最多 5k turn；
- 用途：GRPO 后期如果 dev 停滞，做 1 个 epoch 的拒绝采样微调；
- 必须记录来源 run_id 与 update 区间，报告中说明。

---

## 4. 验证与测试数据

### 4.1 dev（V1）

- seed 取 `10000–10999`（dev fold）；
- Sorting 100 条：size 3/4/5 各约 33 条；
- Map Reader 100 条：d=1/2/3 各约 33 条，参数同训练网格；
- 固定清单，生成后冻结哈希；**所有选模、消融、奖励权重调整只看 dev**。

### 4.2 local-test（V2）

- seed 取 `20000–20999`（test fold）；
- 结构同 dev（100 + 100），但**独立抽样**；
- 生成后只允许 `run.py` 读取，不允许用于任何训练/筛选/调参；
- 在 `split_hashes.json` 中记录哈希；所有决策完成后跑一次，结果写进报告。

### 4.3 压力集（V3）

- Map Reader `d=4`，50 条；
- 只用于分析 10 步预算的边界（`2d+3=11>10`），不参与训练、不作为选模依据；
- 报告中单独说明其成功率上限。

### 4.4 官方测试集（V4）

- 课程提供，格式未知；`run.py` 的实例加载器做成可插拔（JSONL 字段映射），
  本地 manifest 与官方格式用同一个 `Instance` dataclass。

### 4.5 防泄漏规则（硬性）

- 任何训练文件不得包含 seed ≥ 10000 的实例；
- 不得用 test 结果反向筛选训练数据或超参；
- 不得把 gold path、`getObjectTree()`、环境内部状态写入 prompt 或训练目标；
- 不得在 prompt 中预先给出地图内容（必须由模型 `read map` 获得）；
- dev/test manifest 生成后冻结，修改必须升版本并记录原因。

---

## 5. 训练运行时数据（GRPO rollout）

每个 update 保存一份压缩日志，用于诊断与报告：

```json
{
  "update": 12,
  "instance_id": "map-train-000045-d2",
  "group": 3,
  "sample": 5,
  "reward": {"success": 1.0, "progress": 1.0, "failure": 0.0,
             "invalid": -0.05, "steps": -0.07, "format": 0.0, "total": 1.88},
  "advantage": 0.87,
  "success": true,
  "score": 1.0,
  "steps": 7,
  "invalid_actions": 1,
  "turns": [
    {"prompt_tokens": 812, "action": "read map", "logp": -0.21, "score_after": 0.0},
    {"prompt_tokens": 980, "action": "move north", "logp": -0.34, "score_after": 0.0}
  ]
}
```

- 体积：300 update × 8 prompt × 8 sample ≈ 19,200 条轨迹，压缩后 50–100MB；
- 默认保存全部；磁盘紧张时保留最近 50 个 update + 每 10 个 update 抽样 10%；
- 观测文本不重复存储（可用 `instance_id` + 动作序列重放），需要 case study 时再重放。

---

## 6. 生成流水线与脚本

### 6.1 目录结构

```
data/
├── manifests/
│   ├── sorting_train.jsonl
│   ├── mapreader_train.jsonl
│   ├── mapreader_stress_d4.jsonl
│   ├── dev.jsonl
│   ├── local_test.jsonl
│   └── split_hashes.json
├── sft/
│   ├── sorting_expert.jsonl
│   ├── mapreader_expert.jsonl
│   ├── turns_train.jsonl
│   ├── turns_val.jsonl
│   └── stats.json
├── analysis/
│   ├── gold_paths.jsonl          # 仅分析
│   └── error_samples.jsonl
├── stats.json
└── DATASHEET.md
```

### 6.2 脚本职责

| 脚本 | 职责 | 输入 | 输出 |
|---|---|---|---|
| `code/data/generate_instances.py` | 生成实例池/dev/test，分层采样 | `configs/data.yaml` | `data/manifests/*.jsonl` |
| `code/data/build_expert.py` | 生成并校验专家轨迹 | manifests | `data/sft/*_expert.jsonl` |
| `code/data/build_sft_turns.py` | 轨迹展开为 turn，划分 train/val | expert 轨迹 | `turns_*.jsonl` |
| `code/data/validate.py` | 全部质量门槛检查 | `data/` | 通过/失败报告 |
| `code/data/stats.py` | 数量、长度、分布统计 | `data/` | `data/stats.json` |
| `code/data/make_datasheet.py` | 从 stats 生成数据卡 | stats + 配置 | `data/DATASHEET.md` |
| `code/data/freeze_splits.py` | 计算并写入哈希 | manifests | `split_hashes.json` |

### 6.3 命令（在仓库根目录执行）

```bash
# 1) 生成实例池（约几分钟）
python code/data/generate_instances.py --config code/configs/data.yaml

# 2) 生成专家轨迹并校验（约 10–20 分钟，取决于实例数）
python code/data/build_expert.py --manifests data/manifests --out data/sft

# 3) 展开 turn 并划分
python code/data/build_sft_turns.py --sft data/sft --val-ratio 0.05

# 4) 质量检查 + 统计 + 冻结哈希
python code/data/validate.py --data data --strict
python code/data/stats.py --data data --out data/stats.json
python code/data/freeze_splits.py --manifests data/manifests
python code/data/make_datasheet.py --data data
```

### 6.4 生成顺序与依赖

```
data.yaml ──► generate_instances ──► manifests ──► build_expert ──► expert 轨迹
                                              │                         │
                                              └──► stats/validate ◄─────┘
                                                                        │
                                              freeze_splits ◄───────────┘
                                                                        │
                                              build_sft_turns ──► turns_*.jsonl
```

---

## 7. 数据质量门槛（验收清单）

`validate.py --strict` 必须全部通过，否则不允许开始训练。

### 7.1 结构

- [ ] 所有 JSONL 每行可解析，字段完整，类型正确（jsonschema）；
- [ ] `id` 全局唯一；`(game, params, seed)` 无重复；
- [ ] 每条实例的 `initial_valid_actions` 非空；
- [ ] 每条 turn 的 `target_action ∈ context.valid_actions`。

### 7.2 专家轨迹

- [ ] 100% 轨迹能在环境中从头重放成功（`success=true`）；
- [ ] Map Reader 轨迹都包含 `read map`，且 `steps = 2d+3 ≤ 10`；
- [ ] Sorting 轨迹步数 `= 2 × 物体数`，且目标顺序数值升序；
- [ ] 丢弃的实例（未知单位、平局失败、d=4）数量记录在 `stats.json`。

### 7.3 划分与泄漏

- [ ] 训练文件不含 seed ≥ 10000；
- [ ] dev 全部来自 10000–10999；local-test 全部来自 20000–20999；
- [ ] dev/local-test 各 200 条，桶分布符合预期；
- [ ] manifest 哈希与 `split_hashes.json` 一致；
- [ ] prompt/目标中不含 `gold`、`objectTree`、地图文本（Map Reader 初始 prompt 不含地图）。

### 7.4 长度与预算

- [ ] 用 Qwen3 tokenizer 统计，`prompt_tokens` 的 P99 ≤ 1536；
- [ ] `target_tokens` 的 P100 ≤ 48；
- [ ] 每条 SFT 轨迹的步数 ≤ 10；
- [ ] 每类数据的平均 turn 数与预估一致（±10%）。

### 7.5 确定性

- [ ] 同一配置连续生成两次，manifest 的 sha256 一致；
- [ ] 环境版本、jar sha256、Java 版本写入 `stats.json`。

---

## 8. 统计与数据卡

### 8.1 `data/stats.json` 内容

```json
{
  "data_version": "1.0",
  "generated_at": "...",
  "env": {"version": "1.1.0", "jar_sha256": "...", "java": "..."},
  "counts": {
    "sorting_train": {"total": 1000, "size_3": 210, "size_4": 350, "size_5": 440},
    "mapreader_train": {"total": 3000, "d1": 1000, "d2": 1000, "d3": 1000},
    "dev": {"total": 200}, "local_test": {"total": 200}
  },
  "sft": {"trajectories": 1500, "turns": 11100, "train": 10545, "val": 555},
  "lengths": {"prompt_p50": 640, "prompt_p99": 1310, "target_p99": 24},
  "dropped": {"unknown_unit": 0, "tie_failed": 0, "d4": 0}
}
```

### 8.2 数据卡（`data/DATASHEET.md`）

报告 Method 直接引用，至少包含：

- 名称、版本、日期、生成脚本 commit；
- 环境版本、Java、jar sha256；
- 数据来源：TextWorldExpress 程序化生成（无人工标注）；
- 规模与划分表（T1–V4）；
- 生成过程（实例、专家、turn 展开）；
- 用途与禁止用途；
- 分布（难度桶、单位类型、房间数、距离）；
- 质量检查结果；
- 许可：TextWorldExpress Apache-2.0；Qwen3 模型许可见模型卡；
- 局限：英文、程序化分布、训练不含 d=4、专家为规则生成而非人类示范。

---

## 9. 版本、哈希与复现

- 目录版本：`data/v1.0/`（可选）；`DATA_VERSION` 写入每个文件头/统计；
- `split_hashes.json` 记录每个 manifest 的 sha256；
- 重新生成后哈希必须一致（环境与脚本版本固定）；
- 训练配置里记录：`data_version`、`renderer_version`、manifest 哈希；
- 任何数据修改（增删实例、改参数网格）都要升 `data_version` 并在报告中说明。

---

## 10. 体积与预算

| 产物 | 估算体积 | 是否入库 |
|---|---|---|
| manifests（T1+T2+V1–V3） | ≈6 MB | ✅ |
| SFT 轨迹 + turn | ≈25 MB | ❌（可重新生成） |
| rollout 日志 | 50–100 MB / 300 update | ❌ |
| 评测结果 | < 5 MB | ❌ |
| 总计入库 | < 10 MB | — |

---

## 11. 各阶段用哪些数据

| 阶段 | 数据 | 说明 |
|---|---|---|
| zero-shot 基线 | V1 dev | 不训练，只看格式与成功率 |
| SFT | T5 turns_train | 1–2 epoch，val 选模 |
| GRPO | T1/T2 实例池 + 环境奖励 | on-policy，不需要目标动作 |
| 课程 | T1/T2 的难度桶权重 | 不改数据文件 |
| 可选 RFT | T6 | 仅成功轨迹 |
| dev 选模 | V1 | 每 25 update |
| 最终评测 | V2 local-test → V4 官方 | 各只跑一次 |
| 报告 | A2/A4/M3 | 曲线、错误分析、数据卡 |

---

## 12. 负面清单（明确不收集、不使用）

- ❌ 任何 test fold（seed ≥ 20000）实例进入训练数据；
- ❌ gold path 作为 SFT 目标或 prompt 内容；
- ❌ 环境内部对象树、隐藏状态、物体总数等“上帝视角”信息进入 prompt；
- ❌ 把地图文本预先放进 Map Reader 的初始观测（必须由 `read map` 获得）；
- ❌ 人工编写特定实例的解法或提示；
- ❌ 用 dev/test 结果反向筛选训练实例或调超参；
- ❌ 为“让指标好看”而删除失败实例或只保留简单实例。

---

## 附录 A：单位换算

| 维度 | 单位 | 换算到基准 |
|---|---|---|
| 质量 | mg / g / kg | 1 / 1000 / 1,000,000（mg） |
| 体积 | ml / l | 1 / 1000（ml） |
| 长度 | mm / cm / m | 1 / 10 / 1000（mm） |

- 先枚举训练实例里实际出现的所有单位，写入 `stats.json`；
- 出现表外单位 → 该实例不进入 SFT，记录并统计；
- 数值相同（平局）→ 尝试两种顺序，仍失败则跳过。

---

## 附录 B：Prompt 渲染

### B.1 `Qwen3-4B-Base`（默认，纯文本）

```text
You are an agent playing a text game. At each turn, choose exactly one action.
Reply with the action inside <action></action> and nothing else.

Task: {task}

Current observation:
{observation}

Inventory: {inventory}

Valid actions:
{valid_actions}

History:
{history}

Action:
```

- 目标：`<action>take 25g of oak</action>`，后接 EOS；
- 不使用任何 chat 特殊 token；
- 训练与评测必须使用同一个 `renderer_version`。

### B.2 `Qwen3-4B` 官方版（备选，chat）

```python
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": user_content},
]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
```

- 目标：`<action>...</action><|im_end|>`；
- 数据文件不变，仅切换 renderer。

### B.3 截断规则（两种 renderer 共用）

- 历史最多 6 步；
- 单条观测超过 120 token 时截断并加 `...`；
- 总 prompt 超过 1536 token 时，从最旧的历史开始丢弃；
- 这些规则在数据统计（§7.4）与训练/评测中必须一致。

---

## 附录 C：完整示例（Map Reader 的一个 turn）

```json
{
  "id": "map-train-000045-d2-expert-t2",
  "instance_id": "map-train-000045-d2",
  "trajectory_id": "map-train-000045-d2-expert",
  "turn_index": 2,
  "game": "mapreader",
  "difficulty": {"bucket": "d2", "distance": 2, "num_locations": 8},
  "context": {
    "task": "Your task is to take the coin that is located in the parlor, and put it into the box found in the chamber.",
    "observation": "You are in the study. To the North you see the parlor. To the South you see the chamber.",
    "inventory": [],
    "valid_actions": ["move north", "move south", "look around", "inventory", "read map"],
    "history": [
      {"action": "read map", "observation": "The map reads: ..."},
      {"action": "move north", "observation": "You are in the study. ..."}
    ]
  },
  "target_action": "move north",
  "target_in_valid_actions": true,
  "prompt_tokens": 934,
  "target_tokens": 9,
  "split": "train"
}
```
