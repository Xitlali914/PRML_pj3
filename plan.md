# PRML Project 3 开发方案（v1.2 封存版）

> 项目：Training Language Models as Text-Game Agents
> 基座：`Qwen/Qwen3-4B`（官方后训练版，`enable_thinking=False`）
> 任务：TextWorldExpress — Sorting、Map Reader
> 本版本完成 **零样本 → SFT → GRPO**；消融实验与官方测试集评测留待后续。

---

## 1. 目标与约束

| 项 | 要求/选择 |
|---|---|
| 基座 | Qwen3-4B；必须实际更新参数，LoRA 合并后提交 |
| 任务 | Sorting（3–5 个物体，含单位/纯计数）、Map Reader（d≤3） |
| 动作预算 | 每个测试实例最多 10 次环境动作 |
| 指标 | 成功率、成功时平均步数、无效动作比例；两任务各 50% |
| 评测环境 | 单张 RTX 4090（24GB），参数固定、不联网、无参考答案 |
| 历史 | 不同实例之间清空 |
| 动作来源 | 由模型生成，不内置任务求解器 |
| 提交物 | `model/`、`run.py`、`code/`、`report.pdf` |

---

## 2. 实验环境

| 项 | 值 |
|---|---|
| 登录 | `ssh -p 20018 g19@180.163.52.129` |
| 开发容器 | Ubuntu 22.04，4 核 / 16GB，无 GPU |
| GPU | Slurm `RTX 4090 D 24GB`；每组 ≤2 卡，单任务 ≤24h，总额 350 卡时 |
| Python | `/home/g19/.venv`：3.10.12、torch 2.14.1+cu130、transformers 5.20.2、peft 0.21.2 |
| Java | 无系统 Java；`jdk4py 25.0.5.1` 提供用户级 JRE |
| 基座 | `/shared/models/Qwen3-4B`，复制到 `model/Qwen3-4B` |
| 运行时 | `third_party/` 内置 TextWorldExpress 1.1.0 + py4j（jar SHA256 `23da9ab0…c77d99`） |

服务器上的 GPU 任务必须通过 `srun` / `sbatch` 申请；`scripts/` 已处理
`srun` 工作目录和 `jdk4py` 的 PATH 注入。

---

## 3. 架构与代码结构

```text
PRML_pj3/
├── third_party/                 # TextWorldExpress + py4j（离线运行时）
├── code/
│   ├── paths.py                 # 统一路径
│   ├── env/twx.py               # TWXEnv 封装、jar 校验、jdk4py PATH
│   ├── env/prompt.py            # Qwen3 chat prompt、截断、解析、修复重试
│   ├── data/                    # 实例/专家/turn 生成、校验、哈希
│   ├── eval/evaluate.py         # 通用 greedy 评测器
│   └── train/
│       ├── sft.py               # SFT
│       ├── grpo.py              # GRPO 主循环、续训
│       ├── rollout.py           # 8 条轨迹批量 rollout
│       ├── reward.py            # 奖励
│       └── model.py             # 加载、LoRA、合并
├── scripts/                     # 服务器复现脚本
├── report/EXPERIMENTS.md        # 实验报告
└── data/ model/ results/ logs/  # 运行时产物，不入库
```

数据流：

```text
TextWorldExpress ──► 实例池/dev/test
        │                  │
        └──► 专家轨迹 ──► SFT turn ──► SFT adapter
                                              │
环境 rollout（G=8）◄── GRPO ◄─────────────────┘
        │                │
        └──► reward ─────┘──► 每 25 update dev 评测 ──► merged
```

---

## 4. 数据管线

### 4.1 分折

- train：seed 0–999；
- dev：seed 10000–10999，固定 200 条；
- test：seed 20000–20999，固定 200 条，封存只跑一次；
- 官方测试集：课程提供，尚未拿到。

### 4.2 规模

| 数据 | 数量 |
|---|---|
| Sorting train | 1000（size 3/4/5 = 390/260/350） |
| Map Reader train | 3000（d=1/2/3 各 1000） |
| dev / test | 各 200（100 + 100） |
| SFT 专家轨迹 | 600 + 900 |
| SFT turn | 11100（train 10499 / val 601） |

### 4.3 生成规则

- Sorting：从初始 `valid_actions` 解析数量与单位；质量 mg、体积 ml、长度 mm；
  纯计数直接比较；按升序 `take` / `put ... in box`；
- Map Reader：`read map` 解析房间图，房间名去掉前缀 `the`；BFS 最短路，
  按当前房间观测的方向移动；取硬币、返回、放箱；要求步数 `2d+3≤10`；
- Map Reader 参数网格：`numLocations∈{6,8,10}`、`maxDistanceApart∈{1,2,3}`、
  `maxDistractorItemsPerLocation∈{0,1,2}`；按实际距离分层抽 1000/桶；
- 所有专家轨迹构建时重放校验，100% 成功才写入。

### 4.4 确定性

- 固定遍历顺序、无时间戳、canonical JSON（sort_keys、LF）；
- 生成后与 `data/expected_checksums.json` 逐文件比对；
- Windows 与服务器 Linux 生成结果哈希一致。

---

## 5. Harness 与 Prompt

### 5.1 环境接口

`TWXEnv` 提供 `reset(game, seed, fold, params)` 与 `step(action)`；
`done` 由 harness 按 `success` / `failure` / `steps==10` 判定，不依赖
`envStepLimit`；无效动作＝观测以 `Unknown action` 开头。

### 5.2 Qwen3 chat prompt

系统消息固定为：

```text
You are an agent playing a text game. At each turn, choose exactly one action.
Reply with the action inside <action></action> and nothing else.
```

用户消息包含 Task、Current observation、Inventory、Valid actions、History；
渲染调用：

```python
tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
```

- 历史最多 6 步，每条历史观测截断到 120 token；
- prompt ≤1536 token：先丢历史，再截断当前观测；
- 目标格式 `<action>...</action><|im_end|>`；
- 解析失败时追加一次修复提示（temperature=0），仍失败则原文交给环境并记无效；
- 重试不消耗环境动作预算。

### 5.3 动作与评测

- 训练 rollout：temperature 1.0、top_p 1.0；
- 评测：greedy；
- 每实例最多 10 次环境动作，实例间 history 清空。

---

## 6. 阶段 0：零样本

```bash
python -m code.eval.evaluate \
  --model-dir model/Qwen3-4B \
  --instances data/instances/dev.jsonl \
  --output results/zero_shot_dev.jsonl --device cuda
```

dev 200 上总体 38%（Sorting 18%、Map Reader 58%），作为基线。

---

## 7. 阶段 1：SFT

| 项 | 值 |
|---|---|
| 数据 | `data/sft/turns_train.jsonl`（10499 turn） |
| 验证 | `turns_val.jsonl`（601 turn）+ dev 200 |
| epoch | 2 |
| lr | 1e-4，cosine，warmup 5% |
| micro-batch / 梯度累积 | 4 / 4 |
| max length | prompt 1536 + action 48 |
| LoRA | r=32，alpha=64，dropout=0.05，全部线性层 |
| 精度 | bf16 + TF32 + gradient checkpointing |
| 选择 | 每 epoch 在 dev 上评测，保留总体成功率最高者 |

```bash
python -m code.train.sft --base-model model/Qwen3-4B --output-dir model/sft
```

产出 `model/sft/adapter`、`model/sft/merged`；dev 92%。

---

## 8. 阶段 2：GRPO

### 8.1 目标函数

```
A_i = (R_i - mean(R)) / (std(R) + 1e-4)
ρ = exp(log π_θ(a|s) - log π_old(a|s))
L = -1/Σ|a| Σ min(ρ·A, clip(ρ, 0.8, 1.2)·A) + 0.001 · KL(π_θ || π_ref)
```

`π_old` 为 rollout 时记录的概率；`π_ref` 为关闭 LoRA adapter 的同一模型。

### 8.2 奖励

```
R = 1.0×success + 0.5×max(score,0) - 0.5×taskfailure
    - 0.05×invalid_count - 0.01×steps - 0.05×unparsed_count
```

### 8.3 Rollout

| 项 | 值 |
|---|---|
| 每 update prompt 数 | 8 |
| G | 8 |
| 温度 / top_p | 1.0 / 1.0 |
| 每步最大 token | 48，遇 `</action>` 停止 |
| 每回合动作上限 | 10 |
| 环境 | 8 个常驻 JVM，轨迹同步推进 |
| 退化组 | 组内奖励相同则丢弃；一次 update 全退化则跳过 |

### 8.4 课程与超参

| update | 采样 |
|---|---|
| 1–80 | sorting size 3/4、mapreader d=1/2 四桶等概率 |
| 81–200 | 六桶等概率 |
| 201–300 | size 5 与 d=3 合计 50% |

lr 1e-5 cosine；clip 0.2；KL 0.001；micro-batch 4 / 梯度累积 4；
每 25 update dev 评测；保存 adapter + optimizer/scheduler/RNG state，
`latest` 指向最新 checkpoint，支持 24h 任务续训。

### 8.5 复现

```bash
python -m code.train.grpo \
  --base-model model/Qwen3-4B \
  --sft-adapter model/sft/adapter \
  --output-dir model/grpo
```

300 update；dev 最佳 94.5%（update 250），实际耗时 8h38m。

---

## 9. 评测

- 本地 dev/test 用 `code.eval.evaluate`；
- `scripts/eval_test.sh` 调用本地 test；
- **官方测试文件尚未提供，`run.py` 仅为占位入口**；待格式确定后实现
  实例加载/字段映射；
- 指标：成功率、成功平均步数、无效动作比例；两任务各 50%。

---

## 10. 复现步骤

```bash
cd ~/PRML_pj3

# 0) 环境/权重
bash scripts/setup_server.sh
cp -a /shared/models/Qwen3-4B model/Qwen3-4B

# 1) 数据
bash scripts/gen_data.sh

# 2) 零样本
bash scripts/zero_shot.sh

# 3) SFT
sbatch scripts/sft.sh

# 4) GRPO（自动续训）
sbatch scripts/grpo.sh

# 5) 本地 test
bash scripts/eval_test.sh
```

实验数据、指标与 job 记录见 `report/EXPERIMENTS.md`。

---

## 11. 消融实验（待补充）

主实验与复现验证完成后执行，本节先留空：

- 无 SFT（base → GRPO）
- 去掉稠密进度奖励
- 去掉无效动作惩罚
- 去掉 KL
- 无课程
- G=4
- SFT 1 epoch

每个变体的命令、结果表、数据位置在补充时填写。

---

## 12. 风险与注意

| 风险 | 处理 |
|---|---|
| 24h Slurm 限制 | 每 25 update 保存 state，`latest` 续训 |
| 4090 24GB 显存 | LoRA + turn 级样本 + `logits_to_keep` |
| Map Reader 10 步预算 | 训练/评测都限制 d≤3；d=3 为 9 步 |
| 测试泄漏 | train/dev/test 分折 + 哈希冻结；test 只跑一次 |
| 格式重试 | 只允许一次修复重试，记录 retries/unparsed |
| 官方测试格式未知 | `run.py` 占位，拿到文件后再适配 |
