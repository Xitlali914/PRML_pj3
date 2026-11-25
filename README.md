# PRML Project 3: Training Language Models as Text-Game Agents

本仓库是 PRML 课程项目 3 的代码与文档仓库。项目要求以 Qwen3-1.7B 或 Qwen3-4B 为基座模型，
训练一个能够通过多轮文本交互完成任务的智能体；训练算法、奖励设计与数据构造由各组自行决定，
但必须实际更新模型参数。本组选择 **Qwen3-4B** 作为基座模型。

## 任务环境：TextWorldExpress

[TextWorldExpress](https://github.com/cognitiveailab/TextWorldExpress) 是文本游戏基准的高速
重实现（Jansen & Côté, 2022），用 Scala 编写并通过 py4j 提供 Python API。课程使用其中两个任务：

- **Sorting（"sorting"）**：智能体需要按数量从小到大的顺序，把物体逐个放入答案箱。
  物体数量可能带计量单位（如 `mg`、`g`、`kg`，也涉及体积、长度），需要跨单位比较。
  得分为正确排序物体数占总数的比例，全部放对得 1.0；顺序错误会导致任务失败并结束。
- **Map Reader（"mapreader"）**：智能体需要根据提供的地图，前往某个房间找到硬币，
  再把它带回起点房间放入箱子。拿到硬币得 0.5 分，放入箱子得 1.0 分。
  地图需要通过 `read map` 动作读取，初始观测中不包含地图内容。

## 交互与评测协议

课程对交互实现与评测的要求（摘要）：

- 每个测试实例最多执行 **10 次动作**；
- 分别统计两个任务的**成功率**、**成功时的平均步数**、**无效动作比例**，
  两个任务分数各占 50%；
- 评测预算为单张 **RTX 4090（24GB）**；评测时模型参数固定，不访问网络、其他模型、
  参考答案或额外环境模拟；不同实例之间清空交互历史；
- 动作决策必须由模型产生，不允许内置任务专用求解器或参考策略替代模型。

## 交付物

- `model/`：HuggingFace 格式的完整权重（LoRA 需先合并）；
- `run.py`：可直接执行的交互评测入口，支持 Sorting 和 Map Reader，接收模型目录与评测配置，
  输出评测结果和逐轮轨迹；
- `code/`：数据处理与训练代码，附复现说明；
- `report.pdf`：实验报告（ICLR 2025 模板，英文，不超过 10 页）。

## 参考资料

- TextWorldExpress: <https://github.com/cognitiveailab/TextWorldExpress>
- Jansen, P. A., & Côté, M.-A. (2022). *TextWorldExpress: Simulating Text Games at One Million
  Steps Per Second.* arXiv:2208.01174. <https://arxiv.org/abs/2208.01174>
- Qwen3 模型: <https://github.com/QwenLM/Qwen3>
