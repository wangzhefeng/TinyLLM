# 工具库（Tools）

> 所属：[TinyLLM Docs](../README.md)

独立于三个主域的 LLM 工程工具，按主题放在 `tools/`。工具库不是主训练链路的一部分，默认作为参考实现/可复用组件。

## 目录

| 工具 | 内容 | 文档 |
|---|---|---|
| `distributed/` | 分布式并行学习动线 00→08：单卡/DP/DDP/ZeRO/FSDP2/TP/PP/组合/成熟框架 | [distributed.md](distributed.md)、[distributed-frameworks.md](distributed-frameworks.md)、[distributed-roadmap.md](distributed-roadmap.md) |
| `profiling/` | 参数量、吞吐、显存、FLOPs/MFU | [profiling.md](profiling.md) |
| `checkpoint/` | 原子保存/恢复、轮转、latest | [checkpoint.md](checkpoint.md) |
| `data/` | JSONL 读写、精确去重、packing | [data.md](data.md) |

## 约定

- 工具库代码不依赖 `pre_training/`、`post_training/`、`inference/` 的内部实现；
- 可复用的通用逻辑放共享 `utils/`，工具库只保留“分布式/性能/部署”等专项实现；
- 新增工具时先建 `docs/70-tools/<name>.md`，再在 `tools/<name>/` 放代码。

## 相关

- [预训练](../10-pretraining/index.md)
- [架构](../00-overview/architecture.md)
- [代码→文档映射](../00-overview/repo-map.md)
