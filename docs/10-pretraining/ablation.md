> 所属：[TinyLLM Docs](../README.md) › [Pre-training](index.md)

# 组件升级消融实验

`pre_training/experiments/exp_component_ablation.py`：同一 tiny Transformer 骨架，逐项把 GPT-2 组件替换为现代化组件，同预算训练对比 next-token loss。设计对照 NamiGPT 进阶篇（GPT-2 → LLaMA 的组件替换教学）。**standalone 实验脚本，不进 CI**；组件全部复用 `layers/` 现有实现。

## 消融配置

| variant | norm | FFN | attention | 位置 |
|---|---|---|---|---|
| baseline | LayerNorm | GELU | MHA | learned PE |
| +rmsnorm | **RMSNorm** | GELU | MHA | learned PE |
| +swiglu | LayerNorm | **SwiGLU** | MHA | learned PE |
| +rope | LayerNorm | GELU | **MHA+RoPE** | **RoPE** |
| +gqa | LayerNorm | GELU | **GQA（内置 RoPE）** | **RoPE** |
| llama_style | **RMSNorm** | **SwiGLU** | **GQA** | **RoPE** |

注：仓库的 GQA 实现内置 RoPE，故 +gqa 同时切换位置编码；SwiGLU 中间维取 ~2/3×4d（Llama 约定，无偏置），各 variant 参数量略有差异（见结果表）。

## 任务与预算

- 任务：归纳头（induction）——随机段 A + 分隔符 + 重复段 A，只在重复段上算 loss，理论下界 0。注意力必须回看"第一段中同 token 的后继"，天然考察注意力机制。
- 默认预算：vocab 64、d=64、2 层 4 头、seq 64、batch 32、AdamW lr 1e-3 + cosine、400 步、seed 123。CPU <1 分钟。

```bash
python pre_training/experiments/exp_component_ablation.py            # 默认
python pre_training/experiments/exp_component_ablation.py --steps 1000 --embed-dim 128
```

结果写入 `results/ablation/{component_ablation.csv, loss_curves.csv}`（gitignored）。

## 本设置下的观察（2026-10-28，400 步）

| variant | params | final loss |
|---|---|---|
| +rope | 107,776 | 0.041 |
| +gqa | 99,584 | 0.047 |
| llama_style | 78,016 | 0.055 |
| +rmsnorm | 111,680 | 0.736 |
| baseline | 112,000 | 0.947 |
| +swiglu | 90,752 | 1.404 |

- 两个 RoPE 变体与 llama_style 都把 loss 压到 ~0.05（任务近乎学会）；learned-PE 系明显更慢。归纳任务依赖相对位置，RoPE 的相对位置结构在此占优——这是**该任务/预算下的观察**，不是普适结论。
- 参数量最少的是 llama_style（无偏置 + GQA 共享 KV 投影），现代组件组合在"单位参数收益"上占优。
- 换任务（如真实语料）、放大预算后结论可能变化；欢迎按上方命令重跑并更新本表。

## 相关

- [training](training.md)、[examples](examples.md)
- 组件实现：`layers/normalization/`、`layers/feed_forward.py`、`layers/position_encoding/RoPE.py`、`layers/attention.py`
- [论文映射](../60-reference/paper-map.md)（RoPE/GQA/SwiGLU 论文条目）
