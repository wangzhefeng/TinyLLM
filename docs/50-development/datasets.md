# 数据集

> 所属：[TinyLLM Docs](../README.md) › [Development](index.md)

本地数据统一放在根目录 `dataset/` 下，根 `.gitignore` 忽略整个目录，不进入 git。  
清理结果见 [cleanup-log.md](cleanup-log.md)。

## 当前布局

```text
dataset/
├── pretraining/
│   ├── the-verdict.txt
│   ├── gpt/pg145.txt
│   ├── wikitext/EleutherAI___wikitext_document_level/
│   └── general_corpus/pretrain_data.csv
├── finetuning/
│   ├── THUCNews/                    # CSV、text_data、task2.csv、index_2_word.txt
│   ├── sms_spam_collection/ + sms_spam_collection.zip
│   ├── imdb_classification/         # IMDB Dataset.csv
│   ├── sentiment/ + sentiment.zip     # 中文情感分类（text\tlabel，train/valid/test）
│   ├── medical/AI-ModelScope/       # 医疗 reasoning SFT
│   ├── math/{orca,gsm8k}/
│   ├── sql/SQL-Create-Context-SFT/
│   ├── legal/DISC-Law/
│   ├── lora/llama3_1_8B_instruct_lora/
│   └── preference/{dpo,grpo}/
├── evaluation/
│   ├── math500/math500_test.json    # MATH-500
│   ├── mmlu/{dev,val,test}/         # MMLU 57 科
│   └── possibly_contaminated_urls.txt
```

## 代码路径约定

| 代码 | 期望路径 |
|---|---|
| `pre_training/data_provider/bert/data_processor.py` | `dataset/finetuning/THUCNews/task2.csv`、`index_2_word.txt` |
| `post_training/sft/bert_sentiment/` | `dataset/finetuning/sentiment/sentiment.{train,valid,test}.data` |
| `post_training/sft/data_provider/text_clf/data_config.py` | `dataset/finetuning/sms_spam_collection[.zip]` |
| `post_training/peft/model_peft/deepseek_lora/` | `dataset/finetuning/medical/AI-ModelScope/...` |
| `post_training/preference/model_rlhf/grpo/qwen_gsm8k_grpo/` | `dataset/finetuning/math/gsm8k/...` |
| `post_training/reasoning/evaluation/.../math_verifiers.py` | `dataset/evaluation/math500/math500_test.json` |

## 说明

- `dataset/pretraining/general_corpus/pretrain_data.csv` 是中文预训练语料（仅 `text` 列，约 4.3 GB）；原始 JSONL/tar 已删除。
- 通用 SFT 的 `sft_data_zh/en/multi` 已删除；`type_keywords` 元数据也已清理。
- MMLU 的 `auxiliary_train`、IMDB 原始 `aclImdb`、DISC-Law QA 子集、CLUENER、旧 THUCNews 副本已删除。
- 数据不提交 git；迁移/备份时单独处理。

## MMLU 数据

- 位置：`dataset/evaluation/mmlu/{dev,val,test}/`，每套 57 个学科 CSV。
- `dev` 用于 few-shot priming，`test` 用于评测。
- 来源：Hendrycks et al., *Measuring Massive Multitask Language Understanding*, ICLR 2021；另包含 ETHICS 相关引用。
- 原 `auxiliary_train`（RACE/ARC/OBQA 等）已删除；它用于辅助训练，不是评测必需。

## 清理摘要

- 三套旧目录 `dataset/tinyllm*` 从约 76 GB 清理到约 5.8 GB。
- 删除：重复压缩包、HF arrow cache、MMLU auxiliary_train、IMDB 原始 `aclImdb`、DISC-Law QA 子集、CLUENER、通用 SFT 大语料、mobvoi 原始 JSONL/tar。
- 保留：`pretrain_data.csv`、THUCNews、SMS、IMDB CSV、sentiment、医疗/数学/SQL/法律 SFT、DPO/GRPO、MMLU、MATH-500。

## 相关

- [清理记录](cleanup-log.md)
- [预训练数据](../10-pretraining/data.md)
- [项目架构](../00-overview/architecture.md)
