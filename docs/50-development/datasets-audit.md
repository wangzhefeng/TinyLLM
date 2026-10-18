# 数据集审计与清理记录

> 所属：[TinyLLM Docs](../README.md) › [Development](index.md)  
> 执行日期：2026-10-18。清理后 `dataset/` 约 **5.8 GB**（清理前约 **76 GB**）。

## 1. 来源

三个目录来自旧项目的测试数据：

| 目录 | 清理前 | 内容 |
|---|---:|---|
| `dataset/tinyllm/` | 46 GB | 预训练、THUCNews、IMDB、sentiment、MATH-500 |
| `dataset/tinyllmfinetuning/` | 30 GB | SFT、DPO/GRPO、PEFT、分类、医疗/法律/数学/SQL |
| `dataset/tinyllminference/` | 159 MB | MMLU、auxiliary_train |

## 2. 已删除

### 2.1 空目录 / 重复 / 缓存

- `tinyllm/tokenizer/`、`tinyllmfinetuning/qwen2_5_7B_instruct_lora/`（空目录）
- `mobvoi...jsonl.tar.bz2`（11 GB 压缩副本）
- `THUCNews.zip`（与 `finetuning/THUCNews/THUCNews.zip` 重复）
- `IMDB Dataset.csv.zip`（与 CSV 重复）
- Orca/SQL/GSM8K/MATH-500 的 `*.arrow` HF cache
- `tiny_llm/`、`pretrain/gpt/the-verdict.txt`（重复语料）
- `tinyllminference/auxiliary_train/`（MMLU 辅助训练，152 MB）

### 2.2 按决策删除

- 通用 SFT：`sft_data_zh.jsonl`（16 GB）、`sft_data_en.jsonl`（11 GB）、`sft_data_multi.csv`（537 MB）
- 中文预训练原始语料：`mobvoi...jsonl`（31 GB），保留 `pretrain_data.csv`（4.3 GB）
- 超范围/重复：`adapter/aclImdb`（461 MB）、DISC-Law QA 子集（约 266 MB）、`cluener_public`（3.6 MB）

## 3. 已归类

| 类别 | 位置 |
|---|---|
| 预训练 | `dataset/pretraining/{the-verdict.txt,gpt/,wikitext/,general_corpus/}` |
| 分类 | `dataset/finetuning/{THUCNews,sms_spam_collection,imdb_classification,sentiment.zip}` |
| SFT | `dataset/finetuning/{medical,math,sql,legal,lora}` |
| 偏好 | `dataset/finetuning/preference/{dpo,grpo}` |
| 评测 | `dataset/evaluation/{math500/,mmlu/,possibly_contaminated_urls.txt}` |

完整布局与代码路径见 [datasets.md](datasets.md)。

## 4. 代码路径对齐

- BERT：`dataset/finetuning/THUCNews/task2.csv`、`index_2_word.txt` 已就位。
- SMS 分类：`dataset/finetuning/sms_spam_collection/` 与 `.zip` 已就位。
- DeepSeek 医疗 LoRA：路径更新为 `dataset/finetuning/medical/AI-ModelScope/...`。
- GSM8K：路径更新为 `dataset/finetuning/math/gsm8k/gsm8k-math-SFT`。
- MATH-500：移动到 `dataset/evaluation/math500/math500_test.json`，代码路径已同步。

## 5. 结果

- `dataset/` 从约 76 GB 降到约 **5.8 GB**。
- 旧 THUCNews 副本、HF math500 cache、type_keywords 已清理。
- 代码期望的数据路径已对齐。
- 数据不进入 git；未 commit / 未 push。
