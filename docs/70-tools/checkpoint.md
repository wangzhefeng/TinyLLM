# Checkpoint 工具

> 所属：[TinyLLM Docs](../README.md) › [Tools](index.md)

`tools/checkpoint/` 提供原子保存、恢复和轮转，适用于预训练、SFT、PEFT。

## API

| 函数/类 | 用途 |
|---|---|
| `save_checkpoint(path, model, optimizer, scheduler, step, extra)` | 原子写入单个 `.pt` |
| `load_checkpoint(path, model, optimizer, scheduler)` | 恢复状态，返回 payload |
| `latest_checkpoint(dir)` | 最近修改的 `.pt/.pth` |
| `CheckpointManager(dir, keep_last=3)` | `step_XXXXXXXX.pt` 轮转与恢复 |

## 示例

```python
import torch
from tools.checkpoint import CheckpointManager, load_checkpoint

manager = CheckpointManager("saved_results/ckpt", keep_last=3)
manager.save(
    step=100,
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    extra={"loss": 1.23},
)

payload = manager.load_latest(model=model, optimizer=optimizer)
print(payload["step"], payload["extra"])
```

## 说明

- 写入采用临时文件 + `os.replace`，避免中断留下半个 checkpoint。
- `CheckpointManager` 只保留按文件名排序的最新 `keep_last` 个。
- `torch.load(..., weights_only=False)` 用于恢复 optimizer/scheduler；只加载可信来源的 checkpoint。
- 分片/异步 checkpoint 暂未实现；需要时在 `tools/checkpoint/` 增加后端，不要改训练主循环。

## 相关

- [预训练训练循环](../10-pretraining/training.md)
- [Profiling](profiling.md)
- [Tools 索引](index.md)
