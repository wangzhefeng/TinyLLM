> 所属：[TinyLLM Docs](../README.md) › [Pre-training](index.md)

# 预训练训练循环

> 所属：[TinyLLM Docs](../README.md) › [Pre-training](index.md)

## 主链路

```text
run_gpt2_124M.py
  → pre_training/experiments/exp_pretrain_gpt2_124M.py
    → pre_training/data/data_loader.py
    → layers/tokenizers/
    → models/gpt2/gpt2_124M.py
    → utils/{optimizer,criterion,calc_loss,learning_rate,early_stopping,plot_losses}
```

## 支持能力

- 单卡 / DataParallel / DDP
- AMP（CUDA）
- `torch.compile`（可选）
- 训练/验证 loss、tokens/sec、早停、checkpoint
- 训练中采样生成
- 超参网格搜索：`pre_training/experiments/hparam_search.py`

## 已知 TODO

- 主循环的 warmup/cosine 与梯度裁剪仍是注释状态；参考 `pre_training/examples/train_loop_template.py`。
- 生成采样使用 `layers/inference.py`；KV cache 的教材实现见 [architecture](../40-models/architecture.md)。

## 启动

```bash
python run_gpt2_124M.py --help
bash pre_training/scripts/pretrain_gpt2_124M.sh
bash pre_training/scripts/pretrain_gpt2_124M_DP.sh
bash pre_training/scripts/pretrain_gpt2_124M_DDP.sh
```

## 相关

- [examples](examples.md)
- [data](data.md)
- [测试](../50-development/testing.md)
- [LLMs-from-scratch ch05 分析](../60-reference/llms-from-scratch/ch05.md)
