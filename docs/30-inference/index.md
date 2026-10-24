# 模型推断（Inference）

> 所属：[TinyLLM Docs](../README.md)

统一模型加载、文本生成、推断策略、serving、评测与 UI。

## 分页

| 页面 | 内容 |
|---|---|
| [loaders.md](loaders.md) | `models/io` 权重加载与旧路径兼容 |
| [serving.md](serving.md) | 推断原理、API 客户端、服务化 |
| [nano-vllm.md](nano-vllm.md) | 分页 KV + continuous batching |
| [evaluation.md](evaluation.md) | benchmark、judge、leaderboard |
| [reasoning-scaling.md](reasoning-scaling.md) | 推断时扩展与相关概念 |

## 代码结构

```text
inference/
├── registry.py       # 模型注册表
├── loading.py        # load_model 统一门面，调用 models/io
├── generation.py     # generate_text / generate_text_stream
├── generate_core.py  # 教材解码与实验后端
├── strategies/       # self-consistency / Best-of-N / self-refinement
├── serving/          # nano_vllm
├── api/              # Ollama / OpenAI 客户端
├── evaluation/       # benchmarks / judges / leaderboards / 统一 judge
├── apps/             # Chainlit UI
├── loaders/          # 旧权重路径兼容 shim，新代码勿用作实现位置
├── examples/
└── scripts/
```

## 公共 API

```python
from inference import list_models, load_model, generate_text

list_models()
model, cfg, extras = load_model("qwen3-0.6b")
text = generate_text(
    model,
    "Hello",
    family="qwen3",
    tokenizer_file_path=extras["tokenizer_file_path"],
)
```

`generate_text` 与 stream 均只返回**新增文本**，不包含 prompt。GPT-2/Qwen3 默认使用外部 KV cache；其他模型需显式 `use_cache=False`。`generate_core.py` 的 optimized/batched 变体是实验后端，默认链路使用统一 facade。

## 入口

```bash
bash inference/scripts/run_app.sh
python models/io/qwen3/qwen3_06B_load.py
python inference/examples/bert_fill_mask.py --text1 '今天天气[MASK]好'
```

OpenAI 客户端从 `OPENAI_API_KEY` 读取凭证，`OPENAI_BASE_URL` 可选；不再读取根目录 `config.json`。

## 相关

- [模型](../40-models/index.md)
- [reasoning 训练](../20-post-training/reasoning.md)
- [测试](../50-development/testing.md)
