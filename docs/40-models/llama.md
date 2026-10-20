> 所属：[TinyLLM Docs](../README.md) › [Models](index.md)

# Llama

## 实现

- 模型：`models/llama/llama2.py`（RoPE + SwiGLU）、`llama3_8B.py`（GQA + rope_base 500k）、
  `llama31_8B.py` / `llama32_1B.py`（复用 Llama3 Model，仅 config 不同：rope_freq 频率缩放）
- 配置：`config/llama/model_cfgs.py`（`LLAMA2_CONFIG_7B` / `LLAMA3_CONFIG_8B` /
  `LLAMA31_CONFIG_8B` / `LLAMA32_CONFIG_1B`）
- 权重加载：`inference/loaders/llama/`（`tok_embeddings.weight` → `model.tok_embed`）
- 结构约定：embedding 统一命名 `tok_embed`；block 用 `nn.ModuleList`（为 KV cache 传参预留）；
  eos：Llama2=2、Llama3/3.1/3.2=128001
- 注：Llama 3.2 官方权重启用 tie_word_embeddings，教学实现未 tying（参数量略高）

## 模型下载文档

* https://www.llama.com/llama-downloads/
* https://www.llama.com/llama-downloads/

## 模型下载

1. 安装 llama-stack

```bash
$ uv pip install llama-stack
$ uv pip install llama-stack -U
```

2. 查看模型列表

```bash
$ llama model list
$ llama model list --show all
```

3. 下载模型

```bash
$ llama model download --source meta --model-id MODEL_ID
```

当脚本需要输入唯一的字定义 URL 时，粘贴 URL `.env/Llama_model_url`。

# Llama 资料

* [jzhang38/TinyLlama](https://github.com/jzhang38/TinyLlama)
