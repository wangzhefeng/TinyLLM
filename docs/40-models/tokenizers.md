> 所属：[TinyLLM Docs](../README.md) › [Models](index.md)

# Tokenizer 架构

`layers/tokenizers/` 同时包含生产主链路与教学实现，通过统一 protocol 和 registry 使用。

## 目录结构

```text
layers/tokenizers/
├── base.py                  # TokenizerProtocol（@runtime_checkable）
├── registry.py              # 名称归一 + 懒加载工厂 + 协议校验
├── tokenization.py          # choose_tokenizer / text_to_token_ids / token_ids_to_text 门面
├── bpe/                     # simple_bpe、tiktoken_bpe、openai_gpt2_bpe、llama3_8b_bpe、vocab_download
├── sp/                      # llama2_7b_sentencepiece
├── unigram/                 # 自研 Unigram（unigram.py、main.py）
├── simple/                  # simple_tokenizer（SimpleTokenizer；registry key 仍为 "simple_custom"）
├── qwen3/                   # qwen3_tokenizer
├── gemma3/                  # gemma3_tokenizer（原顶层 gemma.py）
├── chat.py                  # Alpaca / Qwen3 / Gemma3 chat 格式化
├── chat_format_tokenizer.py # ChatFormat（llama3 对话包裹）
├── extension.py             # ExtendedTokenizer、词表 padding、embedding resize
└── models/                  # SimpleBPE 训练/加载产物（运行时生成，已 gitignore）
```

实现层统一为「实现 = 子包」形态（`gemma3/`、`simple/` 与 `qwen3/`、`unigram/`、`sp/`、`bpe/` 一致）。

## 公共接口

`base.TokenizerProtocol` 约定：

- `encode(text, **kwargs) -> list[int]`；
- `decode(token_ids) -> str`；
- `n_vocab`、`eos_token_id`、`pad_token_id`。

`registry.py` 负责名称归一与懒加载，`tokenization.py` 保留 `choose_tokenizer`、`text_to_token_ids`、`token_ids_to_text` 门面。名称大小写不敏感，未知名称会列出可用实现。

**协议校验（2026-10-25 起）**：`create_tokenizer` 实例化后会用 `isinstance(tok, TokenizerProtocol)` 校验，不满足时抛 `TypeError` 并列出缺失成员；全部内置实现已补齐协议属性（教学实现无 eos/pad 概念的置 `None`，`ExtendedTokenizer` 委托被包装的 tokenizer）。

**实例缓存（2026-10-25 起）**：`choose_tokenizer` 对不带 kwargs 的调用按归一化名称缓存实例——重复调用返回同一对象，`text_to_token_ids`/`token_ids_to_text` 因此不再每次新建实例；带 kwargs 的调用绕过缓存、行为不变。需要新实例时直接用 `create_tokenizer`。

```python
from layers.tokenizers.tokenization import choose_tokenizer
from layers.tokenizers.registry import create_tokenizer

gpt2 = choose_tokenizer("gpt2")                      # 缓存，重复调用同一实例
qwen = choose_tokenizer("qwen3", tokenizer_file_path="/path/tokenizer.json")  # 绕过缓存
```

## 实现分级

| 类型 | 实现 |
|---|---|
| 主链路 | tiktoken GPT-2、SentencePiece Llama2、tiktoken Llama3、Qwen3、Gemma3 |
| 教学 | SimpleTokenizer、自研 BPE、自研 Unigram、OpenAI GPT-2 移植 |
| 扩展 | `ExtendedTokenizer`、词表 padding、embedding resize |

自研 Unigram（`layers/tokenizers/unigram/unigram.py`，2026-10-24）：种子词表（字符 +
高频子串）→ hard-EM 重估 → 按似然损失剪枝，Viterbi 切分 + unk 兜底，JSON save/load。
注册名 `unigram` 返回未训练实例，需调用方 `train(corpus)` 或 `load(path)`。

Qwen3 的 base/chat eos 通过 `base_model` 显式指定，不再根据文件名猜测。OpenAI 词表下载通过显式 `ensure_openai_gpt2_vocab()` 调用（默认目录已锚定 `downloaded_models/gpt2_model`），Gemma 使用 `ensure_gemma_tokenizer()`；构造器不主动联网。Llama3 默认使用仓库内 `downloaded_models/llama_model/Meta-Llama-3-8B/original/tokenizer.model`（来自 HF gated 仓库，需手动获取），可用 `tokenizer_file_path=` 覆盖。

## Chat template

`layers/tokenizers/chat.py` 统一 Alpaca、Qwen3、Gemma3 格式（原 `post_training.common` 的 instruction_format 垫片已删除，消费方直接用 chat.py）。

## 相关

- [推断生成](../30-inference/index.md)
- [模型配置](../00-overview/architecture.md)
- [测试](../50-development/testing.md)
