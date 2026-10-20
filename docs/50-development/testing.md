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
| `unit/` | 单模块、无网络、秒级完成：attention、FFN、norm、tokenizer、GPT 前向/反向、BERT MLM/NSP、BERT 训练流水线（run_bert）、BERT 情感分类（bert_sentiment）、tools（profiling/checkpoint/data）、**预训练**（pretraining：LR 调度、梯度累积、词表 padding/扩容、特殊 token 扩展、滑动窗口数据集、checkpoint 恢复、legacy 入口导入）、**推断**（inference：sample_next_token 解码策略、generate 贪心确定性、temperature/top-k/top-p、Qwen3 KV cache 一致性、EOS 早停）、**nano_vllm**（safetensors 权重加载与 qkv/gate_up 打包映射、tp=1/tp=2 生成一致性，子进程隔离）、**推断注册表**（inference_registry：MODEL_REGISTRY、load_model 错误路径、generate_text facade 离线生成、公共 API 懒加载契约）、**layers 回归**（layers_regression：2026-10-20 P0 修复的防 drift 测试——attention 缩放、Qwen3-optimized cache、GPT-2 cache 一致性、loader 属性命名、embedding/FixPE、SwiGLU、RMSNorm 数值、BPE rank 合并与互逆） |
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
