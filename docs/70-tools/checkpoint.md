# Checkpoint 工具

> 所属：[TinyLLM Docs](../README.md) › [Tools](index.md)

`tools/checkpoint/` 提供原子保存、恢复和轮转；`training/checkpoint.py` 提供训练统一 schema。

## API

| 函数/类 | 用途 |
|---|---|
| `save_checkpoint(...)` | 原子写入单个 `.pt` |
| `load_checkpoint(...)` | 恢复模型、optimizer、scheduler，兼容旧字段 |
| `latest_checkpoint(dir)` | 查找最近 checkpoint |
| `CheckpointManager(dir, keep_last=3)` | `step_XXXXXXXX.pt` 轮转与恢复 |
| `training.save_training_checkpoint(...)` | 保存统一训练状态与 config/epoch |

统一字段为 `model`、`optimizer`、`scheduler`、`epoch`、`step`、`config`、`extra`。为兼容现有调用，也写入 `model_state_dict` 等旧别名；历史 `optmizer` 仅读取兼容。best 与 resume 共用格式，只是保存策略不同。

## 示例

```python
from training.checkpoint import save_training_checkpoint, load_training_checkpoint

save_training_checkpoint(
    "results/checkpoints/tiny.pt",
    model=model,
    optimizer=optimizer,
    epoch=2,
    step=100,
    config={"name": "tiny"},
)
payload = load_training_checkpoint(
    "results/checkpoints/tiny.pt",
    model=model,
    optimizer=optimizer,
)
```

## 说明

- 写入采用临时文件 + `os.replace`，避免中断留下半个文件。
- 只加载可信来源 checkpoint；恢复 optimizer/scheduler 需要 `weights_only=False`。
- 分片/异步 checkpoint 暂未实现。

## 相关

- [预训练训练循环](../10-pretraining/training.md)
- [项目架构](../00-overview/architecture.md)
