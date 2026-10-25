> 所属：[TinyLLM Docs](../README.md) › [模型推断](index.md)

# 推断时的推理增强（Reasoning Scaling）

> LLM Reasoning

## Inference Compute Scaling

> 推断计算增强

推断时计算扩展（也叫推断计算增强、测试时增强等）包含一系列在推断阶段（即用户输入提示词时）提升模型推理能力的方法，
  
这些方法无需对底层模型权重进行训练或修改。其核心思想是通过增加计算资源来换取性能提升，
  
借助思维链推理（chain-of-thought reasoning）及多种采样程序等技术，使固定参数的模型展现出更强的推理能力。

## Reinforcement Learning

> 强化学习

强化学习是一类通过最大化奖励信号来提升模型推理能力的训练方法。其奖励机制可分为两类：

- 广义奖励：如任务完成度或启发式评分
- 精准可验证奖励：如数学问题正确答案或编程任务通过率

与推断时计算扩展（inference-time compute scaling）不同，
  
RL 通过动态调整模型参数（weights updating）实现能力提升。
  
该机制使模型能够基于环境反馈，通过试错学习不断优化其推理策略。

> 注： 在开发推理模型时，
  
需明确区分此处的纯强化学习（RL）方法与常规大语言模型开发中用于偏好微调的基于人类反馈的强化学习（RLHF）（如图 2 所示）。
  
二者的核心差异在于奖励信号的来源：RLHF 通过人类对模型输出的显式评分或排序生成奖励信号，
  
直接引导模型符合人类偏好行为；纯 RL 则依赖自动化或环境驱动的奖励信号（如数学证明的正确性），
  
其优势在于客观性，但可能降低与人类主观偏好的对齐度。典型场景对比：纯 RL 训练：以数学证明任务为例，
  
系统仅根据证明步骤的正确性提供奖励；RLHF 训练：需人类评估员对不同输出进行偏好排序，
  
以优化符合人类标准（如表述清晰度、逻辑流畅性）的响应。

## Knowledge Distillation

> 知识蒸馏

模型蒸馏是指将高性能大模型习得的复杂推理模式迁移至更轻量化模型的技术。在 LLM 领域，该技术通常表现为：
  
使用高性能大模型生成的高质量标注指令数据集进行监督微调（Supervised Fine-Tuning, SFT）。
  
这种技术在 LLM 文献中常统称为知识蒸馏（Knowledge Distillation）或蒸馏（Distillation）。

与传统深度学习的区别：经典知识蒸馏中，「学生模型」需同时学习「教师模型」的输出结果和 logits，
  
而 LLM 的蒸馏通常仅基于输出结果进行迁移学习。

> 注：本场景采用的监督微调（SFT）技术与常规大语言模型开发中的 SFT 类似，
  
其核心差异体现在训练样本由专为推理任务开发的模型生成（而非通用 LLM）。
  
也因此，其训练样本更集中于推理任务，通常包括中间推理步骤。

## 本仓库实现

- **CoT（思维链提示）**：`render_prompt(prompt, cot=True)` 在 MATH-500 模板上追加 "Think step by step ..." 指令（对应书 ch04 的 prompt_suffix 机制），见 [评测](evaluation.md)。
- **Self-consistency / 加权 self-consistency**：`inference/strategies/voting_and_verification.py` 的 `self_consistency`（多数投票）与 `weighted_self_consistency`（按 `scorer_fn` 分数加权投票，全 0 分退化为多数票，对应书 ch05 的 heuristic/logprob 打分）。
- **Best-of-N / Self-refinement**：同一模块的 `best_of_n`（外部 reward 选优）与 `inference/strategies/advance_text_generation.py` 的 `generate_with_self_refinement`（反馈→修订循环）。
- 端到端把上述策略跑在 MATH-500 上：`python -m inference.harness.math500 --strategy ...`，详见 [评测 harness](evaluation.md)。
