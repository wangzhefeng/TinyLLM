> 所属：[TinyLLM Docs](../README.md) › [Pre-training](index.md)

# 预训练扩展（extras）

书本之外的预训练参考与失修入口，未接入主链路。分布式的独立内容已移到 [工具库](../70-tools/distributed.md)。

## 内容

| 子目录 | 内容 | 状态 |
|---|---|---|
| `references/nanotron/` | nanotron 配置与启动脚本 | 参考 |
| `legacy/` | `run_llama.py`、`run_qwen.py` | 失修：与 `Model_Pretrain` 签名不匹配 |

## nanotron

- 上游：<https://github.com/huggingface/nanotron>
- 位置：`pre_training/references/nanotron/`
- 内容：`config_tiny_llama.yaml`、`pretrain.sh`、`generate.sh`

## 已移除

smolGPT、TinyLlama、Karpathy 参考等未接入项目已删除，见 [reference-projects](../50-development/reference-projects.md)。

## 相关

- [training](training.md)
- [LLMs-from-scratch 附录 A](../60-reference/llms-from-scratch/appendices.md)
- [分布式工具库](../70-tools/distributed.md)
