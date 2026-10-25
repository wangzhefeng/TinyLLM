# 模型推断（Inference）

> 所属：[TinyLLM Docs](../README.md)

统一模型加载、文本生成、推断策略、serving、评测循环与 UI。

## 分页

| 页面 | 内容 |
|---|---|
| [loaders.md](loaders.md) | 权重 IO（`models/io`）与 HF 导出 |
| [serving.md](serving.md) | 推断原理、API 客户端、服务化 |
| [nano-vllm.md](nano-vllm.md) | 分页 KV + continuous batching |
| [evaluation.md](evaluation.md) | 评测 harness 与公共库分工 |
| [reasoning-scaling.md](reasoning-scaling.md) | 推断时扩展与相关概念 |

## 代码结构

```text
inference/
├── __init__.py       # PEP 562 懒加载导出门面
├── loading.py        # load_model 统一加载（注册表在 models/registry.py）
├── generation.py     # generate_text / generate_text_stream
├── generate_core.py  # 教材解码与实验后端
├── harness/          # 生成侧评测循环：math500.py、mmlu.py
├── strategies/       # self-consistency / Best-of-N / self-refinement / 投机解码
├── serving/nano_vllm/
├── apps/app.py       # Chainlit UI
├── examples/         # gpt2_inference / qwen3_06B / bert_fill_mask
└── scripts/          # run_app.sh / run_math500_experiments.sh / benchmark_compile.py
```

评测基础设施（客户端/验证器/judge/排行榜）在顶层公共库 `evaluation/`，见 [评测公共库](../35-evaluation/index.md)。

## 公共 API

```python
from inference import load_model, generate_text
from models.registry import list_models

list_models()
model, cfg, extras = load_model("qwen3-0.6b")
text = generate_text(
    model,
    "Hello",
    family="qwen3",
    tokenizer_file_path=extras["tokenizer_file_path"],
)
```

`generate_text` 与 stream 均只返回**新增文本**，不包含 prompt。GPT-2/Qwen3 默认使用外部 KV cache；其他模型需显式 `use_cache=False`。`generate_core.py` 的 optimized/batched 变体是实验后端，默认链路使用统一 facade。`beam_search_generate`（2026-10-24 升级）支持多 beam、逐 beam 真实生成长度的长度惩罚（`score = Σlogp / len^length_penalty`）与 EOS 冻结；beam 分叉的 cache 回滚复杂，教学实现每步整段重算（见 docstring）。

## 入口

```bash
bash inference/scripts/run_app.sh
python models/io/qwen3/qwen3_06B_load.py
python inference/examples/bert_fill_mask.py --text1 '今天天气[MASK]好'
```

OpenAI 客户端从 `OPENAI_API_KEY` 读取凭证，`OPENAI_BASE_URL` 可选（见 [评测公共库](../35-evaluation/index.md)）；不再读取根目录 `config.json`。

## 相关

- [评测公共库](../35-evaluation/index.md)
- [模型](../40-models/index.md)
- [reasoning 训练](../20-post-training/reasoning.md)
- [测试](../50-development/testing.md)
