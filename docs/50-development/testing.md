> 所属：[TinyLLM Docs](../README.md) › [Development](index.md)

# 测试

本目录只放**可重复运行的单元测试/集成测试**，不再放教学 demo。

## 运行

```bash
# 全量
python -m unittest discover -s tests -v

# 只跑单元测试
python -m unittest discover -s tests/unit -v
```

## 目录约定

| 目录 | 内容 |
|---|---|
| `unit/` | 单模块、无网络、秒级完成：attention、FFN、norm、tokenizer、GPT 前向/反向、BERT MLM/NSP、BERT 训练流水线（run_bert）、BERT 情感分类（bert_sentiment）、tools（profiling/checkpoint/data）、**预训练**（pretraining：LR 调度、梯度累积、词表 padding/扩容、特殊 token 扩展、滑动窗口数据集、checkpoint 恢复、legacy 入口导入）、**推断**（inference：sample_next_token 解码策略、generate 贪心确定性、temperature/top-k/top-p、Qwen3 KV cache 一致性、EOS 早停）、**nano_vllm**（safetensors 权重加载与 qkv/gate_up 打包映射、tp=1/tp=2 生成一致性，子进程隔离）、**推断注册表**（inference_registry：MODEL_REGISTRY、load_model 错误路径、generate_text facade 离线生成、公共 API 懒加载契约）、**layers 回归**（layers_regression：2026-10-20 P0 修复的防 drift 测试——attention 缩放、Qwen3-optimized cache、GPT-2 cache 一致性、loader 属性命名、embedding/FixPE、SwiGLU、RMSNorm 数值、BPE rank 合并与互逆、Qwen3 书本 checkpoint 重映射）、**解码策略**（decoding_strategies：repetition penalty、beam search、beam1=greedy 一致性）、**ALiBi**（alibi：斜率公式、因果偏置、外推性质、SDPA 集成）、**SWA**（swa：窗口掩码、窗口外 token 不影响输出、cache 有界、KV cache 窗口截断）、**RoPE 缩放**（rope_scaling：NTK 降频、YaRN 分带、mscale）、**MoE 扩展**（moe_extensions：aux loss 均衡性、shared expert）、**推断时扩展**（inference_time_scaling：self-consistency 投票与嵌套 \boxed 解析、Best-of-N、self-refinement、stable 早停）、**GRPO**（grpo：组内优势、clip 比率、学习信号、completion mask 排除 padding、EOS 提前停止、多 prompt 批处理）、**REINFORCE 与轨迹蒸馏**（reinforce：策略梯度方向、baseline 降方差、学习信号、verifier 过滤、completion-only SFT 损失、单轨迹过拟合）、**蒸馏**（distillation：KL 蒸馏、alpha 混合、学生逼近教师）、**排行榜与评委**（leaderboards：Bradley-Terry 拟合/中心与排序、Elo 零和守恒与 400 分差、re-export 同源）、**LLM 评委**（llm_judges：prompt 结构、parse_score 全格式、Ollama 不可达报错、后端模块 import 零副作用）、**PPO**（ppo：ratio=1 基线、clip 截断、padding mask、critic 收敛、互斥校验、学习信号）、**Prompt Tuning**（prompt_tuning：基座冻结、虚拟 token 前向、玩具过拟合）、**NF4**（nf4：量化往返误差有界、零块保护）、**LoRA**（lora：零初始化、alpha/rank 缩放语义、dropout、目标层过滤、合并/不合并等价、DoRA 两简化变体不等价回归、论文版 DoRA、clf_head 四策略冻结语义、shim 兼容，共 25 项）、**MLA**（mla：增量=整段一致性、latent cache 显存优势、DeepSeekModel 生成与越界防护） |
| `integration/` | 跨模块或需要小模型生成的测试（后续补充） |

## 说明

- 使用标准库 `unittest`，不额外引入 pytest。
- 测试使用极小配置（embed_dim=32、2 层、vocab=128），CPU/MPS 均可跑。
- 原来的 `test/` 里大部分是书本 walkthrough / FLOPs 分析 / 权重加载 demo，不是单元测试；它们已迁到：
  - `pre_training/examples/`：attention、FFN、LayerNorm、embedding、residual、训练模板、GPT-2 示例；
  - `pre_training/analysis/`：FLOPs 分析；
  - `inference/loaders/gpt2/`：GPT-2 权重加载；
  - `pre_training/examples/gpt2_book/`：书本 ch05 的 standalone 训练示例。

## 待办（2026-10-18 盘点）

- [ ] `pre_training/experiments/bert/`（DAPT/curpos 等 4 个脚本）无测试覆盖，未验证可运行
- [ ] 预训练主链路的端到端集成测试未固化：tiny 模型 + tiny 语料跑通 `exp.train`（warmup_cosine + 累积 + checkpoint + resume）应入 `tests/integration/`
