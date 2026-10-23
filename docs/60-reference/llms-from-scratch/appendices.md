> 所属：[TinyLLM Docs](../../README.md) › [LLMs-from-scratch 分析](index.md)

### 3.8 附录 A–E

| 附录 | 内容 | 本项目 |
|---|---|---|
| A | PyTorch 基础 + DDP | ✅ 远超：`tools/distributed/` 覆盖 DP/DDP/ZeRO/FSDP2/TP/PP/组合并行（TP 为手写教学实现），另有 accelerate/DeepSpeed 对照示例 |
| B | 参考文献 | ➖ |
| C | 习题解答索引 | ➖ |
| D | 训练循环增强：warmup、cosine decay、梯度裁剪 | 🟡 `test/train_loop_template.py` 与 smolGPT 有实现，主 exp 未接入 |
| E | LoRA：低秩适配、rank/alpha、应用与合并 | 🟡 MiniMind 有 `model_lora.py`，但没有通用 GPT-2 LoRA 微调脚本 |

### 3.9 `pkg/llms_from_scratch`：从脚本到包

- 把 ch02–ch07、附录 A/D/E、Llama 3、Qwen3 的代码组织成统一 API（如 `from llms_from_scratch.ch04 import GPTModel`）。
- 有单元测试与 CI，可直接 `pip install llms-from-scratch`。
- **对本项目**：项目已有 `pyproject.toml` 与包结构，但没有稳定的公共 API、测试套件和 CI；`layers/`、`models/` 更像“源码目录”而不是“库”。

### 3.10 `reasoning-from-scratch` 子模块：第二本书

- **Ch2**：用预训练 LLM 生成文本、多轮对话、`torch.compile` 优化。
- **Ch3**：推理模型评测（MATH-500、答案解析、验证器）。
- **Ch4**：推断时扩展（CoT、self-consistency、batched 版本）。
- **Ch5**：self-refinement、best-of-n、更多推断时策略。
- **Ch6**：RLVR + GRPO（原始版、batched、FSDP）。
- **Ch7**：GRPO 改进（tracking、clip ratio、KL、format reward）。
- **Ch8**：推理蒸馏（Ollama/OpenRouter 生成数据 + 蒸馏训练）。
- **附录 C–G**：Qwen3 源码、更大 LLM、batching/throughput、评测方法（MMLU、Elo/Bradley-Terry、LLM judge）、聊天界面。

**对本项目**：`qwen3_reasoning/` 的目录命名与这本书高度对应（评测、推断时、RL、蒸馏），但只有数学验证器、选择题评测、leaderboard 骨架与 Qwen3 推断 demo 是真实代码，RL/蒸馏/推断时扩展/LLM judge 等主线仍是空壳。如果项目想继续“推理模型”方向，这是一条清晰的补全路线。

---
