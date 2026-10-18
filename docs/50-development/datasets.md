# 数据集

> 所属：[TinyLLM Docs](../README.md) › [Development](index.md)

本地数据统一放在根目录 `dataset/` 下，根 `.gitignore` 忽略整个目录，不进入 git。  
三套旧项目数据的审计与清理记录见 [datasets-audit.md](datasets-audit.md)。

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
│   ├── sentiment.zip
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
| `models/bert/data_processor.py` | `dataset/finetuning/THUCNews/task2.csv`、`index_2_word.txt` |
| `post_training/sft/data_provider/text_clf/data_config.py` | `dataset/finetuning/sms_spam_collection[.zip]` |
| `post_training/peft/model_peft/deepseek_lora/` | `dataset/finetuning/medical/AI-ModelScope/...` |
| `post_training/preference/model_rlhf/grpo/qwen_gsm8k_grpo/` | `dataset/finetuning/math/gsm8k/...` |
| `post_training/reasoning/evaluation/.../math_verifiers.py` | `dataset/evaluation/math500/math500_test.json` |

## 说明

- `dataset/pretraining/general_corpus/pretrain_data.csv` 是中文预训练语料（仅 `text` 列，约 4.3 GB）；原始 JSONL/tar 已删除。
- 通用 SFT 的 `sft_data_zh/en/multi` 已删除；`type_keywords` 元数据也已清理。
- MMLU 的 `auxiliary_train`、IMDB 原始 `aclImdb`、DISC-Law QA 子集、CLUENER、旧 THUCNews 副本已删除。
- 数据不提交 git；迁移/备份时单独处理。

## 相关

- [数据集审计](datasets-audit.md)
- [MMLU 来源说明](datasets-mmlu.md)
- [预训练数据](../10-pretraining/data.md)
