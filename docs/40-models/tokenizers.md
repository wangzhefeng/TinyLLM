> 所属：[TinyLLM Docs](../README.md) › [Models](index.md)

# Tokenizer 架构

`layers/tokenizers/` 同时包含生产主链路与教学实现，通过统一 protocol 和 registry 使用。

## 公共接口

`base.TokenizerProtocol` 约定：

- `encode(text, **kwargs) -> list[int]`；
- `decode(token_ids) -> str`；
- `n_vocab`、`eos_token_id`、`pad_token_id`。

`registry.py` 负责名称归一与懒加载，`tokenization.py` 保留 `choose_tokenizer`、`text_to_token_ids`、`token_ids_to_text` 门面。名称大小写不敏感，未知名称会列出可用实现。Qwen3/Gemma3 需要显式传入本地 tokenizer 文件。

```python
from layers.tokenizers.tokenization import choose_tokenizer

gpt2 = choose_tokenizer("gpt2")
qwen = choose_tokenizer("qwen3", tokenizer_file_path="/path/tokenizer.json")
```

## 实现分级

| 类型 | 实现 |
|---|---|
| 主链路 | tiktoken GPT-2、SentencePiece Llama2、tiktoken Llama3、Qwen3、Gemma3 |
| 教学 | SimpleTokenizer、自研 BPE、OpenAI GPT-2 移植、HF 对照 |
| 扩展 | `ExtendedTokenizer`、词表 padding、embedding resize |
| 未完成 | `unigram/`，不属于可用能力 |

Qwen3 的 base/chat eos 通过 `base_model` 显式指定，不再根据文件名猜测。OpenAI 词表下载通过显式 `ensure_openai_gpt2_vocab()` 调用，Gemma 使用 `ensure_gemma_tokenizer()`；构造器不主动联网。

## Chat template

`layers/tokenizers/chat.py` 统一 Alpaca、Phi-3、Qwen3、Gemma3 格式。旧 `post_training.common.data_provider.instruction_format` 仅保留兼容转发。

## 相关

- [推断生成](../30-inference/index.md)
- [模型配置](../00-overview/architecture.md)
- [测试](../50-development/testing.md)
