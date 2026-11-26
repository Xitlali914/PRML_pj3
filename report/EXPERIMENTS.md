# 实验报告（封存版本 v1.0）

> 本报告记录封存版本 `v1.0` 的服务器环境、零样本、SFT 与 GRPO 结果。
> 消融实验与最终官方测试集评测暂未执行，对应章节留白。

## 0. 结论摘要

| 阶段 | Sorting 成功率 | Map Reader 成功率 | 总体 | 无效动作率 |
|---|---|---|---|---|
| Zero-shot | 18% | 58% | 38% | 0.39% |
| SFT（epoch 1） | 94% | 90% | 92% | 0.13% |
| GRPO（update 250） | 98% | 91% | 94.5% | 0.00% |

在 dev 200 条上，GRPO 相比零样本总体成功率提升 **56.5 个百分点**，
无效动作率从 0.39% 降到 0。Sorting 提升最明显（18% → 98%）。

---

## 1. 服务器与实验环境

| 项 | 值 |
|---|---|
| 登录 | `ssh -p 20018 g19@180.163.52.129` |
| 开发容器 | Ubuntu 22.04，4 核 CPU / 16GB 内存（cgroup），无 GPU |
| GPU | Slurm 分配，NVIDIA RTX 4090 D 24GB |
| Slurm 限制 | 每组最多 2 卡，单任务 ≤24h，总额 350 卡时；`train` / `debug` 分区 |
| Python | `/home/g19/.venv`，Python 3.10.12 |
| 关键依赖 | torch 2.14.1+cu130、transformers 5.20.2、peft 0.21.2、py4j 0.10.9.9 |
| Java | 无系统 Java；使用 `jdk4py 25.0.5.1`（用户级，无 root） |
| 基座 | `/shared/models/Qwen3-4B`，官方后训练版（chat template，`enable_thinking=False`） |
| 运行时 | `third_party/` 内置 TextWorldExpress 1.1.0（jar SHA256 `23da9ab0…c77d99`）与 py4j |

---

## 2. 数据生成

### 复现命令

```bash
cd ~/PRML_pj3
bash scripts/gen_data.sh
```

### 数据汇总

| 数据 | 数量 | 说明 |
|---|---|---|
| Sorting train | 1000 | size 3/4/5 = 390/260/350 |
| Map Reader train | 3000 | d=1/2/3 各 1000 |
| dev | 200 | 100 + 100 |
| test | 200 | 100 + 100（本地，尚未跑最终评测） |
| SFT 专家轨迹 | 600 + 900 | 100% 环境重放成功 |
| SFT turn | 11100 | train 10499 / val 601 |

### 存放位置

```text
data/instances/{sorting_train,mapreader_train,dev,test}.jsonl
data/sft/{sorting_expert,mapreader_expert,turns_train,turns_val}.jsonl
data/expected_checksums.json        # 生成后与仓库内哈希逐文件比对
```

生成过程无时间戳、顺序固定；本机 Windows 与服务器 Linux 生成的文件哈希完全一致。

---

## 3. 零样本测试

### 复现命令

```bash
bash scripts/zero_shot.sh
```

等价命令：

```bash
python -m code.eval.evaluate \
  --model-dir model/Qwen3-4B \
  --instances data/instances/dev.jsonl \
  --output results/zero_shot_dev.jsonl \
  --device cuda
```

### 结果汇总（dev 200）

| 任务 | n | 成功率 | 成功平均步数 | 无效动作率 | retries | unparsed |
|---|---|---|---|---|---|---|
| Sorting | 100 | 18% | 8.78 | 0.00% | 1 | 0 |
| Map Reader | 100 | 58% | 6.67 | 0.74% | 14 | 6 |
| **总体** | **200** | **38%** | **7.17** | **0.39%** | — | — |

### 存放位置

```text
results/zero_shot_dev.jsonl
results/zero_shot_dev.metrics.json
```

---

## 4. SFT

### 复现命令

```bash
sbatch scripts/sft.sh
```

等价命令：

```bash
python -m code.train.sft \
  --base-model model/Qwen3-4B \
  --output-dir model/sft
```

### 训练配置

| 项 | 值 |
|---|---|
| 数据 | `turns_train.jsonl`（10499 turn） |
| epoch | 2 |
| lr | 1e-4，cosine，warmup 5% |
| micro-batch / 梯度累积 | 4 / 4（有效 16） |
| LoRA | r=32, alpha=64, dropout=0.05，全部线性层 |
| 精度 | bf16 + TF32 + gradient checkpointing |
| 选择规则 | 每 epoch 在 dev 200 上评测，保留总体成功率最高者 |

### 结果汇总

| epoch | Sorting | Map Reader | 总体 | 成功平均步数 | 无效动作率 |
|---|---|---|---|---|---|
| 1（选中） | 94% | 90% | **92%** | 7.65 | 0.13% |
| 2 | 97% | 87% | 92% | 7.57 | 0.26% |

epoch 1 与 epoch 2 总体同为 92%，按“严格大于”规则保留 epoch 1。

### 存放位置

```text
model/sft/epoch_1/{adapter,...}      # 选中
model/sft/epoch_2/{adapter,...}
model/sft/adapter/                   # 选中的 LoRA
model/sft/merged/                    # 合并权重
model/sft/sft_log.jsonl
model/sft/result.json
results/sft_epoch1.jsonl / .metrics.json
results/sft_epoch2.jsonl / .metrics.json
logs/sft-<jobid>.log
```

---

## 5. GRPO

### 复现命令

```bash
sbatch scripts/grpo.sh
```

脚本在存在 `model/grpo/latest` 时自动续训，否则从 `model/sft/adapter` 开始。

### 训练配置

| 项 | 值 |
|---|---|
| 初始化 | `model/sft/adapter` |
| update | 300 |
| 每个 prompt 轨迹数 G | 8 |
| 每 update prompt 数 | 8 |
| 采样 | temperature 1.0，top_p 1.0 |
| 每回合动作上限 | 10 |
| lr | 1e-5，cosine |
| clip / KL | 0.2 / 0.001 |
| 课程 | 1–80 easy/medium；81–200 全分布；201–300 hard 优先 |
| 保存 | 每 25 update 保存 adapter + optimizer state + RNG，`latest` 指向最新 |
| 评测 | 每 25 update，dev 200，greedy |

### 结果汇总（dev 200）

| update | Sorting | Map Reader | 总体 | 成功平均步数 | 无效动作率 |
|---|---|---|---|---|---|
| 25 | 94% | 87% | 90.5% | — | 3.69% |
| 50 | 97% | 87% | 92.0% | — | 5.40% |
| 75 | 96% | 76% | 86.0% | — | 0.00% |
| 100 | 97% | 90% | 93.5% | — | 0.00% |
| 175 | 97% | 90% | 93.5% | — | 0.00% |
| 200 | 98% | 89% | 93.5% | — | 0.86% |
| **250（选中）** | **98%** | **91%** | **94.5%** | **7.32** | **0.00%** |
| 300 | 98% | 91% | 94.5% | 7.31 | 0.00% |

- Slurm 作业 `1323`，状态 `COMPLETED`，实际耗时 **08:37:55**；
- 最佳 update 250（与 300 同分，按严格大于保留 250）；
- Map Reader 在部分 update 出现较多格式重试（如 update 250 有 78 次 retries），
  但最终动作全部合法，`invalid_action_ratio=0`。

### 存放位置

```text
model/grpo/merged/                    # 最终合并权重（update 250）
model/grpo/adapter/                   # 最佳 LoRA
model/grpo/update_{25..300}/{adapter,state.pt}
model/grpo/latest -> update_300       # 续训入口
model/grpo/result.json                # best_update=250
model/grpo/grpo_log.jsonl             # 每个 update 与每次 dev 评测
results/grpo_update<N>.jsonl / .metrics.json
logs/grpo-1323.log
```

---

## 6. 阶段对比

| 指标 | Zero-shot | SFT | GRPO |
|---|---|---|---|
| Sorting 成功率 | 18% | 94% | 98% |
| Map Reader 成功率 | 58% | 90% | 91% |
| 总体成功率 | 38% | 92% | 94.5% |
| 成功平均步数 | 7.17 | 7.65 | 7.32 |
| 无效动作率 | 0.39% | 0.13% | 0.00% |

观察：

1. SFT 是主要增益来源，尤其是 Sorting 的单位换算/排序；
2. GRPO 进一步把 Sorting 从 94% 提到 98%，Map Reader 从 90% 提到 91%，
   并把无效动作率压到 0；
3. Map Reader 在 GRPO 中期有波动（update 75 掉到 76%），课程后段恢复并稳定；
4. 成功平均步数从零样本的 7.17 到 GRPO 的 7.32，基本持平，没有用“磨蹭”换成功率。

---

## 7. 消融实验（待补充）

> 主实验封存后再执行；本节留空。

| 变体 | Sorting | Map Reader | 总体 | 无效动作率 | 说明 |
|---|---|---|---|---|---|
| 无 SFT |  |  |  |  |  |
| 去掉稠密进度奖励 |  |  |  |  |  |
| 去掉无效动作惩罚 |  |  |  |  |  |
| 去掉 KL |  |  |  |  |  |
| 无课程 |  |  |  |  |  |
| G=4 |  |  |  |  |  |
| SFT 1 epoch |  |  |  |  |  |

复现命令与数据位置在消融完成后补充。

---

## 8. 最终测试（待补充）

课程官方测试文件尚未提供；`run.py` 目前仅为占位入口。
本地 `data/instances/test.jsonl` 的评测命令：

```bash
bash scripts/eval_test.sh
```

结果位置：

```text
results/final_test.jsonl
results/final_test.metrics.json
```

---

## 9. 复现命令速查

```bash
cd ~/PRML_pj3

# 0) 环境与权重（一次性）
bash scripts/setup_server.sh

# 1) 数据
bash scripts/gen_data.sh

# 2) 零样本
bash scripts/zero_shot.sh

# 3) SFT
sbatch scripts/sft.sh

# 4) GRPO（可重复提交，自动续训）
sbatch scripts/grpo.sh

# 5) 本地 test
bash scripts/eval_test.sh

# 查看状态
squeue --me
sacct -j <jobid> -X -o JobID,JobName,State,Elapsed,AllocTRES
tail -f logs/grpo-<jobid>.log
grep dev_score model/grpo/grpo_log.jsonl | tail -5
```
