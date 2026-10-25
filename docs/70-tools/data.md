# Data 工具

> 所属：[TinyLLM Docs](../README.md) › [Tools](index.md)

`tools/data/` 提供与模型无关的 JSONL/文本处理工具，用于数据清洗、去重和 token packing。

## API

| 函数 | 用途 |
|---|---|
| `iter_jsonl`、`read_jsonl`、`write_jsonl` | JSONL 读写（原子写入） |
| `read_json`、`write_json` | JSON 读写 |
| `normalize_text` | 空白折叠、可选小写 |
| `text_fingerprint` | SHA-256 去重指纹 |
| `dedupe_records`、`dedupe_jsonl` | 精确去重，保持顺序 |
| `read_text_label_tsv` | `text\tlabel` TSV 读取（bert_sentiment 与 DAPT 共用） |
| `pack_token_ids` | 定长 token 分块 / padding |

## 示例

```python
from tools.data import dedupe_jsonl, pack_token_ids, read_jsonl

dedupe_jsonl(
    "raw.jsonl",
    "deduped.jsonl",
    key="text",
    normalize=True,
    lowercase=True,
)

chunks = list(pack_token_ids(range(10), max_length=4, pad_id=0))
# [[0,1,2,3], [4,5,6,7], [8,9,0,0]]
```

## CLI

```bash
# 统计
python -m tools.data stats raw.jsonl --key text

# 去重
python -m tools.data dedupe raw.jsonl clean.jsonl --key text

# 保留大小写 / 保留空文本
python -m tools.data dedupe raw.jsonl clean.jsonl --case-sensitive --keep-empty
```

## 说明

- 去重是**精确哈希**，不做 MinHash/SimHash 近似去重；需要近似去重时再引入依赖。
- `write_jsonl` 先写临时文件再替换，避免中断产生半文件。
- `pack_token_ids` 只做扁平 token 流分块；按样本 packing、attention mask 由训练侧负责。

## 相关

- [数据集说明](../50-development/datasets.md)
- [预训练数据](../10-pretraining/data.md)
- [Tools 索引](index.md)
