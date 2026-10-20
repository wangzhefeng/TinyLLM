> 所属：[TinyLLM Docs](../README.md) › [Development](index.md)

# 已移除的外部参考项目

以下项目曾放在 `pre_training/extras/` 与 `post_training/imported/`，但：

- 没有被主代码 import；
- 没有测试或脚本依赖；
- 与当前 LLM 预训练/后训练/推断主线重复或超出范围；
- 都可以从上游重新 clone。

因此从仓库中删除，仅保留链接与用途说明。

| 项目 | 上游 | 原位置 | 原用途 | 删除理由 |
|---|---|---|---|---|
| smolGPT（已删除） | <https://github.com/Om-Alve/smolGPT> | `pre_training/extras/smolGPT/` | 小 GPT 预训练参考：warmup/cosine、梯度累积、`torch.compile`、checkpoint | 主项目已有 `pre_training/examples/train_loop_template.py` 与 `pre_training/experiments/`，需要时重新 clone |
| TinyLlama（已删除） | <https://github.com/jzhang38/TinyLlama> | `pre_training/extras/tinyllama/` | Llama 预训练参考（`llama_model.py` + `pretrain.py`） | 书本主线之外，未接入；需要 Llama 预训练时再引入 |
| MiniMind（已删除） | <https://github.com/jingyaogong/minimind> | `post_training/imported/minimind/` | MiniMind 模型、LoRA、SFT/DPO 数据、tokenizer | 后训练仓库已有 SFT/DPO/PEFT 实现，内容重复 |

## 当前参考项目

- `tools/distributed/`：FSDP/TP/DeepSpeed 参考。
- `pre_training/references/nanotron/`：nanotron 配置参考。

## 说明

- 这些项目的内容没有“丢失”，只是从工作区移除；需要时按上表 URL 重新 clone。
- 如果以后要把它们的某段实现纳入 TinyLLM，请先落到对应域（`pre_training/` 或 `post_training/`）并补测试，而不是整目录复制。
