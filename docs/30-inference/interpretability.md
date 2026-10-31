> 所属：[TinyLLM Docs](../README.md) › [Inference](index.md)

# 可解释性（教学工具）

回答"模型每一步在想什么"的最小工具集：生成时每步的 top-k 候选概率与分布熵、GPT-2 各层 attention 权重捕获。定位是教学（对照 [NamiGPT](https://github.com/aa1143/NamiGPT) ep15 与其网页演示的"每步候选词概率"），不是重型分析框架——零新依赖，渲染输出 markdown。

## 逐步生成追踪

```python
from inference.interpretability import render_trace_md, trace_generation

records, out_ids = trace_generation(
    model, token_ids, max_new_tokens=20,
    temperature=0.0, k=5, tokenizer=tokenizer,
)
print(render_trace_md(records))
```

行为契约（`tests/unit/test_interpretability.py` 锁定）：

- batch=1 教学路径，无 KV cache；采样复用 `layers/sampling.sample_next_token`；
- 贪心（temperature<=0）输出与 `generate_core.generate(use_cache=False)` 逐 token 一致；
- `eos_token_id` 语义与 `generate` 相同：完成步的 eos 不追加进序列，该步记录 `eos_stop=True`；
- 每步记录字段：

| 字段 | 含义 |
|---|---|
| `step` / `token_id` / `token` | 步序号、采样 token（`tokenizer=None` 时 `token` 为 None） |
| `entropy` | 该步全词表分布的熵（nats） |
| `topk` | 前 `k` 个候选 `[(token_id, token, prob), ...]`，概率降序 |
| `eos_stop` | 本步是否因采到 eos 结束 |

## Attention 权重捕获

`output_attentions` 是仿 `output_hidden_states` 的 **opt-in 前向契约**，当前实现于 GPT-2 栈（`MultiHeadAttention` → `TransformerBlockGPT2_124M` → `models/gpt2/gpt2_124M.py`）：

- `model(ids, output_attentions=True)` 返回 `(logits, [每层 (B, H, T, T)])`；
- 权重取 softmax 之后、dropout 之前的值；
- 与 `use_cache` / 外部 cache 互斥（抛 `ValueError`）；默认关闭时前向行为与返回类型完全不变；
- 其他模型族暂未实现（需要时按同一契约扩展各模型 forward）。

```python
from inference.interpretability import attention_weights, render_attention_md

logits, attns = attention_weights(model, token_ids)
print(render_attention_md(attns[-1], tokens=["猫", "吃", "鱼"]))
```

`render_attention_md` 取 batch/head 平均，输出 `T x T` markdown 表；因果性（上三角为 0）与行和为 1 由测试保证。

## Chainlit 教学模式

`inference/apps/app.py` gpt2 模式的设置面板新增 `show_trace` 开关：开启后回答消息追加一条 author="Trace" 的每步 top-k/熵表（走无 cache 教学路径）；qwen3 模式不受影响。

## 相关

- [Inference 总览](index.md)（`generate_core.py` 生成路径）
- [GPT-2](../40-models/gpt2.md)
- [测试](../50-development/testing.md)
