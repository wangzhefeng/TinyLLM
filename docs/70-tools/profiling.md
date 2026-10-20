# Profiling 工具

> 所属：[TinyLLM Docs](../README.md) › [Tools](index.md)

`tools/profiling/` 提供训练/推断通用的轻量 profiling 能力，不依赖具体模型。

## API

| 函数 | 用途 |
|---|---|
| `count_parameters(model, trainable_only=False)` | 参数量 |
| `parameter_report(model)` | total / trainable / frozen |
| `ThroughputMeter`、`measure_throughput()` | tokens/s |
| `memory_report(device=None)` | CUDA/MPS 显存 |
| `estimate_transformer_flops(N, T)` | 约 `6*N*T` 训练 FLOPs |
| `estimate_mfu(achieved, peak)` | MFU（0–1） |
| `human_bytes(n)` | 字节格式化 |

## 示例

```python
import torch
from tools.profiling import (
    count_parameters,
    measure_throughput,
    memory_report,
)

model = torch.nn.Linear(768, 768)
print(count_parameters(model))
print(memory_report())

with measure_throughput() as meter:
    # ... forward/backward ...
    meter.update(batch_size * seq_len)

print(meter.stats.tokens_per_second)
```

## 说明

- `estimate_transformer_flops` 是 `6*N*T` 近似，不含 attention 的序列长度项；精确值用 `torch.utils.flop_counter` 或 profiler。
- `memory_report` 在 CPU 返回 0；MPS 只保证 allocated，driver 值可能不可用。
- 更细的逐算子分析请直接用 `torch.profiler`，本工具只做常用汇总。

## 相关

- [预训练训练循环](../10-pretraining/training.md)
- [数据工具](data.md)
- [Tools 索引](index.md)
