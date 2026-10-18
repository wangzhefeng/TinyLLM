> 所属：[TinyLLM Docs](../README.md) › [Pre-training](index.md)

# 预训练训练循环

## 主链路

```text
run_gpt2_124M.py
  → pre_training/experiments/exp_pretrain_gpt2_124M.py
    → pre_training/data_provider/data_loader.py
    → layers/tokenizers/
    → models/gpt2/gpt2_124M.py（llama2 / llama3_8B / qwen3 亦可）
    → utils/{optimizer,criterion,calc_loss,learning_rate,early_stopping,plot_losses}
    → tools/checkpoint（周期 checkpoint 与 resume）
```

## 支持能力

- 单卡 / DataParallel / DDP（DDP 下非 step micro-batch 使用 `no_sync`）
- AMP（CUDA）、`torch.compile`（可选）
- **step 级学习率调度**：线性 warmup + 余弦衰减（附录 D），`--lr_schedule warmup_cosine`；`epoch` 回退旧的 epoch 级 `lradj`
- **梯度裁剪**：`--grad_clip 1.0`（0 关闭）
- **梯度累积**：`--grad_accum_steps N`，有效 batch = batch_size × N × world_size
- **词表对齐 padding**：`--pad_vocab_to_multiple_of 64`（50257 → 50304，ch05/10）
- **训练恢复**：`--save_every_steps N` 周期保存（含 optimizer/scaler/step），`--resume latest` 或指定文件恢复
- 训练/验证 loss、tokens/sec、早停、best checkpoint、训练中采样生成
- 超参网格搜索：`pre_training/experiments/hparam_search.py`

## 恢复训练

```bash
python run_gpt2_124M.py --save_every_steps 500 ...        # 周期保存到 <checkpoints>/<setting>/steps/
python run_gpt2_124M.py --resume latest --save_every_steps 500 ...
```

checkpoint 保存 model + optimizer + global_step + epoch/batch 位置 + AMP scaler；
resume 后在原 epoch 内跳过已完成的 batch。早停的 best checkpoint（`checkpoint.pth`）与周期
resume checkpoint（`step_*.pt`，默认保留最近 3 个）相互独立。

## 相关工具

- 内存高效权重加载（mmap）：`utils/weight_loading.py`（ch05/08）
- tokenizer 扩展 / embedding 扩容：`layers/tokenizers/extension.py`（ch05/09）
- 生成采样：`layers/inference.py`；KV cache 见 [architecture](../40-models/architecture.md)

## 启动

```bash
python run_gpt2_124M.py --help
bash pre_training/scripts/pretrain_gpt2_124M.sh
bash pre_training/scripts/pretrain_gpt2_124M_DP.sh
bash pre_training/scripts/pretrain_gpt2_124M_DDP.sh
# 其他模型（ch05/14）
python run_llama.py --help
python run_qwen.py --help
```

## 相关

- [examples](examples.md)
- [data](data.md)
- [测试](../50-development/testing.md)
- [checkpoint 工具](../70-tools/checkpoint.md)
- [LLMs-from-scratch ch05 分析](../60-reference/llms-from-scratch/ch05.md)

## 待办（2026-10-18 盘点）

- [ ] **`--init_from` 入口**：加载 OpenAI/HF 权重作为预训练起点（区别于 `--resume` 恢复训练状态）；权重加载现成：[inference/loaders/gpt2/](../../inference/loaders/gpt2/)
- [ ] **加速对照基准**： vocab padding / 梯度累积 / fused AdamW 已就绪，缺开/关加速的 tok/sec 对照脚本（可用 `tools/profiling` 的 `ThroughputMeter`）
- [ ] **perplexity 输出**：评估目前只有 loss，加 `exp(valid_loss)` 即可
- [ ] `experiments/hparam_search.py` 内联了自有 warmup 逻辑，应复用 `utils/learning_rate.warmup_cosine_lr`
- [ ] `run_llama.py` / `run_qwen.py` 默认用 tiktoken GPT-2 分词器做演示；真实 Llama/Qwen3 分词器链路（`--vocab_size 32000/151936` + 对应 eos）未验证
- [ ] **Muon 优化器**接入 `utils/optimizer.py`（2D 参数用 Muon、其余 AdamW），对应上游 ch05/18
