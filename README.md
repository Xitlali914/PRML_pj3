# PRML Project 3: Training Language Models as Text-Game Agents

本仓库是 PRML 课程项目 3 的完整实现：以 **Qwen3-4B** 为基座，在
TextWorldExpress 的 **Sorting** 与 **Map Reader** 两个任务上完成
**零样本基线 → SFT → GRPO** 的训练与评测。实验记录见
[`report/EXPERIMENTS.md`](report/EXPERIMENTS.md)，设计与环境见
[`plan.md`](plan.md)。

## 目录结构

```text
PRML_pj3/
├── plan.md / README.md
├── requirements.txt
├── run.py                      # 官方测试入口（占位，待测试文件格式确定）
├── third_party/                # 内置 TextWorldExpress 1.1.0 + py4j（无需联网）
├── code/
│   ├── paths.py
│   ├── env/{twx.py, prompt.py} # 环境封装、Qwen3 chat prompt、动作解析
│   ├── data/                   # 实例、专家轨迹、SFT turn、校验
│   ├── eval/evaluate.py        # 零样本/SFT/GRPO 共用评测器
│   └── train/{sft.py, grpo.py, rollout.py, reward.py, model.py}
├── scripts/                    # 服务器 Slurm 与复现脚本
├── report/EXPERIMENTS.md       # 实验报告（已入库）
├── data/                       # 生成数据（gitignore，脚本重建）
├── model/                      # 模型权重（gitignore）
└── results/ runs/ logs/        # 输出（gitignore）
```

## 实验环境

代码在课程服务器上验证：

- Ubuntu 22.04，开发容器 4 核 / 16GB，无 GPU；
- Slurm 分配 **RTX 4090 D 24GB**，每组最多 2 卡，单任务 ≤24h；
- `/home/g19/.venv`：Python 3.10、torch 2.14.1+cu130、transformers 5.20.2、peft；
- 无系统 Java，安装 `jdk4py` 提供用户级 JRE；
- 基座：`/shared/models/Qwen3-4B`（官方后训练版，`enable_thinking=False`）。

## 复现全流程

以下命令均在**服务器**上、仓库根目录执行。

### 0. 准备环境与权重（一次性）

```bash
cd ~/PRML_pj3

# 安装用户级 JDK（无 root）；已安装可跳过
bash scripts/setup_server.sh

# 把课程提供的模型复制到项目 model/（约 7.6GB）
mkdir -p model
cp -a /shared/models/Qwen3-4B model/Qwen3-4B

# Python 依赖已预装；若新机器则：
# uv pip install --python ~/.venv/bin/python -r requirements.txt
```

确认：

```bash
ls model/Qwen3-4B/config.json
~/.venv/bin/python -c "import torch, transformers, peft; print(torch.__version__)"
```

### 1. 生成数据（CPU，约 3 分钟）

```bash
bash scripts/gen_data.sh
```

等价命令：

```bash
srun -p debug -c 4 -t 10 bash -lc "cd ~/PRML_pj3 && ~/.venv/bin/python -m code.data.generate_all"
```

输出：

```text
data/instances/{sorting_train,mapreader_train,dev,test}.jsonl
data/sft/{sorting_expert,mapreader_expert,turns_train,turns_val}.jsonl
```

生成脚本会与仓库内 `data/expected_checksums.json` 逐文件比对；哈希不一致会直接失败。
数据规模：Sorting 1000、Map Reader 3000、dev/test 各 200、SFT 11100 turn。

### 2. 零样本基线

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

输出：`results/zero_shot_dev.jsonl`、`results/zero_shot_dev.metrics.json`。

### 3. SFT

```bash
sbatch scripts/sft.sh
```

等价命令：

```bash
python -m code.train.sft --base-model model/Qwen3-4B --output-dir model/sft
```

- 2 epoch，LoRA r=32，lr 1e-4，有效 batch 16；
- 每个 epoch 在 dev 200 上评测，保留最佳；
- 产出 `model/sft/{adapter,merged}`、`model/sft/result.json`；
- 耗时约 1.5h（4090 D）。

### 4. GRPO

```bash
sbatch scripts/grpo.sh
```

等价命令：

```bash
python -m code.train.grpo \
  --base-model model/Qwen3-4B \
  --sft-adapter model/sft/adapter \
  --output-dir model/grpo
```

- 300 update，G=8，8 prompt/update，KL=0.001，clip=0.2；
- 每 25 update 在 dev 上评测并存 `model/grpo/update_N/{adapter,state.pt}`；
- 24h 任务被打断后，再次 `sbatch scripts/grpo.sh` 会自动从 `model/grpo/latest` 续训；
- 产出 `model/grpo/merged`（最佳 update 的合并权重）。

### 5. 评测

**本地 test（`data/instances/test.jsonl`）**：

```bash
bash scripts/eval_test.sh
```

输出 `results/final_test.jsonl` 与 `.metrics.json`。

**官方测试集**：测试文件尚未提供，`run.py` 目前是占位入口：

```bash
python run.py --help
```

待官方格式确定后再实现实例加载与字段映射。

## 结果

| 阶段 | Sorting | Map Reader | 总体 | 无效动作率 |
|---|---|---|---|---|
| Zero-shot | 18% | 58% | 38% | 0.39% |
| SFT | 94% | 90% | 92% | 0.13% |
| GRPO | 98% | 91% | 94.5% | 0.00% |

详细表格、复现命令与数据位置见 [`report/EXPERIMENTS.md`](report/EXPERIMENTS.md)。

## 监控训练

```bash
squeue --me
sacct -j <jobid> -X -o JobID,JobName,State,Elapsed,AllocTRES
tail -f logs/grpo-<jobid>.log
grep dev_score model/grpo/grpo_log.jsonl | tail -5
cat model/grpo/result.json
```

## 关键约定

- 所有 prompt 走 Qwen3 chat template，`enable_thinking=False`；
- 每实例最多 10 次环境动作，greedy 评测；
- 数据只使用 train fold（seed 0–999），dev 10000+ 选模，test 20000+ 只跑一次；
- 训练/评测/数据生成共用同一套 harness 和动作解析；
- 运行时由 `third_party/` 提供，不依赖联网安装；
- `data/`、`model/`、`results/`、`logs/` 不入库，按上面命令重建。
