# 分布式训练工具

> 所属：[TinyLLM Docs](../README.md) › [Tools](index.md)

`tools/distributed/` 收集 LLM 训练相关的分布式/并行参考实现，**不属于预训练主链路**。

## 内容

| 目录 | 主题 |
|---|---|
| `dp/` | DataParallel 最小示例 |
| `ddp/` | DDP、torchrun、多机、`nn_ddp` 小网络示例 |
| `fsdp/` | FSDP 自动/手动 wrapping、checkpoint、mixed precision、prefetch |
| `tensor_parallelism/` | Llama2 风格张量并行 |
| `deepspeed_zero/` | DeepSpeed ZeRO 示例 |
| `fairscale/` | fairscale 示例 |
| `log_utils.py` | 工具库日志辅助 |

## 运行

从仓库根目录执行，保证 `tools` 可导入：

```bash
# DDP 小网络示例
bash tools/distributed/ddp/nn_ddp/ddp_script.sh

# FSDP
bash tools/distributed/fsdp/run.sh

# 张量并行
bash tools/distributed/tensor_parallelism/run_example.sh
```

部分脚本假设在对应子目录内运行（例如 FSDP 的 `cd tools/distributed/fsdp`）；运行前先看脚本头部。

## 状态

- 参考实现，未接入 `run_gpt2_124M.py` 主链路。
- 依赖 `torch.distributed`、DeepSpeed、fairscale 等可选包；缺失时只影响对应示例。
- 原位于 `pre_training/distributed/`，2026-10-18 起独立为 `tools/distributed/`。

## 相关

- [预训练训练循环](../10-pretraining/training.md)
- [LLMs-from-scratch 附录 A](../60-reference/llms-from-scratch/appendices.md)
- [Tools 索引](index.md)
