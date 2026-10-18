> 所属：[TinyLLM Docs](../README.md) › [Development](index.md)

# 清理记录与剩余候选

> 更新时间：2026-10-18  
> 状态：已按决策执行一轮清理；**未 commit / 未 push**。  
> 注：以下为历史路径；2026-10-18 三域内部重构后，当前结构见 [architecture](../00-overview/architecture.md)。

## 1. 已删除

### 1.1 空壳 / 生成物 / 重复

- `pre_training/examples/gpt2_book/loss.pdf`
- `pre_training/extras/legacy_experiments/`（5 个空壳）
- `pre_training/extras/legacy_entrypoints/run_gemma.py`（空壳）
- `post_training/finetuning/model_peft/` 下空壳：
  - `adalora_demo.py`、`bitfit_demo.py`、`qlora_demo.py`、`lorafa_demo.py`、`increlora_demo.py`
  - `prefix_tuning/`、`spectrum/`、`finetuning_all_layers/`、`finetuning_last_layers/`
  - `knowledge_distillation/main.py`
- `post_training/finetuning/scripts/rm_dsstore.sh`（与根目录重复）
- `inference/toolkit/scripts/rm_dsstore.sh`（与根目录重复）
- `docs/inference-import-notes.md`（历史说明）
- `dataset/finetuning/THUCNews/tokenizer_data/`（空目录）
- `logs/`（空目录）

### 1.2 外部参考 / 超出范围

- `pre_training/extras/smolGPT/`
- `pre_training/extras/tinyllama/`
- `post_training/imported/minimind/`
- `todo/other/`（transformer_variants、DyT、data_tools、energy_demo、llm_libraries、training_utils）
- `pre_training/extras/karpathy_proj/`（nanoGPT/nanochat/minGPT 旧参考）
- `post_training/finetuning/model_tts/`（TTS/STT 音频）
- `post_training/imported/`（合并后删除）

### 1.3 重复实现合并后删除

- `post_training/finetuning/model_pretrain/` → 合并到 `inference/loaders/gpt2/`
- `post_training/finetuning/layers/tokenizers/` → 根 `layers/tokenizers/` 为更新版本
- `post_training/finetuning/layers/inference.py` → 根 `layers/inference.py` 为完整版本

## 2. 明确保留

| 内容 | 原因 |
|---|---|
| `pre_training/extras/distributed_training/` | 进阶预训练工程参考 |
| `pre_training/extras/tiny_llama_nanotron/` | nanotron 配置参考 |
| `pre_training/extras/legacy_entrypoints/run_llama.py`、`run_qwen.py` | 按你的选择保留，待修 |
| `post_training/finetuning/model_peft/{unsloth_exps,lit_gpt_exps,uniem_finetune}` | 按你的选择保留 |
| `models/gpt2/gpt.py`、`models/llama/llama31_8B.py`、`llama32_1B.py` | 按你的选择保留，待实现 |
| `docs/archive/tokenizers-premerge-2026-10-18.bundle` | 保留 tokenizers 原仓库历史 |
| `downloaded_models/qwen3_model/`（2.8 GB） | 按你的选择保留，避免重新下载 |
| `dataset/finetuning/THUCNews/` | 数据集保留 |

## 3. 仍可选的清理项（未执行）

| 内容 | 大小 | 说明 |
|---|---:|---|
| `dataset/finetuning/THUCNews/THUCNews.zip` | 27 MB | 与解压后的 CSV + text_data 重复 |
| `docs/assets/post-training/prompt/` | 7.6 MB | Prompt Engineering PDF |
| `post_training/finetuning/model_peft/{unsloth_exps,lit_gpt_exps,uniem_finetune}` | 64 KB | 外部工具/嵌入模型实验 |
| `pre_training/extras/legacy_entrypoints/run_llama.py`、`run_qwen.py` | 16 KB | 签名不匹配，运行即报错 |
| `model_peft` 中其余 `*_demo.py` / `main.py` | — | 部分仍是空壳或外部 demo，可逐项判断 |

## 4. 不在清理范围

- `pre_training/` 主线（data_provider、exp、examples、scripts）
- `layers/`、`config/`、`utils/`
- `models/{gpt2,llama,qwen3,gemma3,bert,deepseek}`
- `post_training/finetuning/{data_provider,exp,run,model_peft,model_rlhf}`
- `post_training/reasoning/`
- `inference/{toolkit,apps,loaders,examples}`
- `tests/`
- `docs/` 关键文档
- `dataset/` 数据

## 5. 验证

```bash
python -m unittest discover -s tests -v   # 14/14
python run_gpt2_124M.py --help            # OK
```

当前状态：暂存区为空，`HEAD` 未变化，未 commit / 未 push。
