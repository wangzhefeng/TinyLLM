> 所属：[TinyLLM Docs](../README.md) › [Pre-training](index.md)

# 预训练训练循环

## 主链路

```text
run_gpt2_124M.py / run_llama.py / run_qwen.py
  → pre_training/experiments/exp_pretrain_gpt2_124M.py
    → pre_training/data_provider/
    → models + layers
    → training.Trainer
    → training.checkpoint + tools.checkpoint
```

BERT 使用 `run_bert.py` 的独立 MLM+NSP 流程，但 checkpoint 使用同一 schema。

## 支持能力

- 单卡 / DataParallel / DDP；DDP 非 step micro-batch 使用 `no_sync`。
- CUDA AMP、梯度裁剪、梯度累积；CPU/MPS 不强制启用 CUDA AMP。
- step 级 warmup+cosine 或 epoch 级学习率调整。
- 词表对齐 padding、训练/验证 loss、tokens/sec、早停、采样。
- 周期 resume 与 best checkpoint 共用统一字段，保留历史 checkpoint 兼容读取。
- CPU Dataset + 可选 pin_memory + batch 阶段设备传输，支持 DataLoader workers。

`training.Trainer` 统一 AMP、梯度累积/裁剪、DDP 同步和 optimizer step。`torch.compile` 是实验基类内部能力，未作为公开 CLI 能力承诺。GPT-2/Llama/Qwen 实验基类（`exp_pretrain_gpt2_124M.py` 等）继承 `exp_basic.py`：模型实现经 `models.registry.MODEL_REGISTRY` 解析（`model_name` → registry 键 → family/config_key → 模型模块），不再各自硬编码注册表。

## 恢复训练

```bash
python run_gpt2_124M.py --save_every_steps 500 ...
python run_gpt2_124M.py --resume latest --save_every_steps 500 ...
```

checkpoint 包含 model、optimizer、scheduler、epoch、step、config、extra；同时保留旧 `model_state_dict`/`optimizer_state_dict` 别名。历史 `optmizer` 只在读取时兼容，不再新写。best 与周期 resume 是不同保存策略，不再使用互不兼容的 schema。

## 启动

```bash
python run_gpt2_124M.py --help
bash pre_training/scripts/pretrain_gpt2_124M.sh
bash pre_training/scripts/pretrain_gpt2_124M_DP.sh
bash pre_training/scripts/pretrain_gpt2_124M_DDP.sh
python run_llama.py --help
python run_qwen.py --help
python run_bert.py --help
```

集成测试使用合成 tiny 数据真实执行训练、checkpoint 和 latest 恢复；CPU/gloo 双进程已验证，CUDA/NCCL 需在对应硬件另测。

## 相关

- [data](data.md)
- [测试](../50-development/testing.md)
- [checkpoint 工具](../70-tools/checkpoint.md)
- [权重 IO（models/io）](../30-inference/loaders.md)
