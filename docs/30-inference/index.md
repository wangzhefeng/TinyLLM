# 模型推断（Inference）

> 所属：[TinyLLM Docs](../README.md)

整合原 `TinyLLMInference` 与项目内推断代码：权重加载、生成、serving、评测、UI。

## 分页

| 页面 | 内容 |
|---|---|
| [loaders.md](loaders.md) | GPT-2 / Qwen3 / Gemma3 / Llama 权重加载 |
| [serving.md](serving.md) | 推断原理、API 客户端、服务化 |
| [nano-vllm.md](nano-vllm.md) | 迷你 vLLM 引擎：分页 KV + continuous batching |
| [evaluation.md](evaluation.md) | Ollama/OpenAI 评测、推理评测 |
| [reasoning-scaling.md](reasoning-scaling.md) | 推断时推理增强、RL、蒸馏概念 |

## 代码结构

```text
inference/
├── registry.py  # MODEL_REGISTRY：模型注册表（family/config/权重来源/tokenizer/eos）
├── loading.py   # load_model(model_key, source=...)：统一权重加载入口
├── generation.py# generate_text / generate_text_stream：统一生成 facade
├── serving/     # nano_vllm
├── api/         # Ollama/OpenAI 客户端
├── evaluation/  # judge.py 统一评测（--backend ollama|openai）+ 兼容 shim
├── apps/        # app.py 单一 Chainlit UI（GPT2_WEIGHTS=orig|own）+ 兼容 shim
├── loaders/     # gpt2 / qwen3 / gemma3 / llama 各族加载库（loading.py 的底层）
├── examples/    # GPT-2 生成示例、BERT fill-mask 示例
└── scripts/     # run_app.sh
```

## 公共 API

```python
from inference import MODEL_REGISTRY, list_models, load_model, generate_text

list_models()                                    # 注册的模型 key
model, cfg, extras = load_model("qwen3-0.6b")    # 统一下载+加载（懒加载重依赖）
text = generate_text(model, "Hello", family="gpt2", max_new_tokens=50)
```

## 入口

```bash
bash inference/scripts/run_app.sh        # Chainlit UI（orig|own）
python inference/loaders/qwen3/qwen3_06B_load.py   # USE_CACHE=True 即 KV cache 版
python inference/examples/bert_fill_mask.py --text1 '今天天气[MASK]好'
```

## 共享原语

`inference/generate_core.py` 与 `layers/kv_cache.py` 是 LLMs-from-scratch ch04/ch05 的教材实现（2026-10-20 起生成逻辑已从 `layers/` 归位到 `inference/` 域）。`nano_vllm` 负责 serving 级优化，不重复实现教材版 KV cache。

`inference/generate_core.py` 提供统一的 `sample_next_token(logits, temperature, top_k, top_p)`（greedy / temperature / top-k / top-p，softmax 数值稳定），所有 `generate*` 变体（GPT-2 / Qwen3 / 批量 / 流式 / 早停）均复用它；单元测试见 `tests/unit/test_inference.py`。

## 相关

- [模型](../40-models/index.md)
- [post-training reasoning](../20-post-training/reasoning.md)
- [测试](../50-development/testing.md)
