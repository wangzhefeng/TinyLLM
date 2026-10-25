> 所属：[TinyLLM Docs](../README.md) › [Models](index.md)

# Karpathy 来源分析

> 分析对象：`~/projects/llm_proj/karpathy/` 下的两个仓库  
> 对照项目：TinyLLM（当前工作区）  
> 分析日期：2026-10-18  
> **状态**：`pre_training/extras/karpathy_proj/` 已按清理决策删除；本文保留来源分析与快照 commit，便于需要时重新引入。

## 1. 两个上游仓库的快照

| 仓库 | 当前 commit | 说明 |
|---|---|---|
| `karpathy/nanoGPT` | `3adf61e`（2025-11-12） | 已标记 deprecated，推荐转向 nanochat |
| `karpathy/nanochat` | `92d63d4`（2026-07-03） | 当前主仓库，已把 AdamW/Muon 合并进 `optim.py` |

TinyLLM 中的 karpathy 副本曾位于（现已删除）：

```text
pre_training/extras/karpathy_proj/
├── README.md
├── nanoGPT/
│   └── model.py                  # 直接复制（仅少末尾换行）
├── nanochat/
│   ├── gpt.py                    # 直接复制（旧快照）
│   ├── common.py                 # 直接复制（旧快照）
│   ├── adamw.py                  # 直接复制（旧快照，现已并入 optim.py）
│   └── muon.py                   # 直接复制（旧快照，现已并入 optim.py）
├── minGPT/                       # 来自第三个仓库 karpathy/minGPT
└── minGPT_ddp/                   # 从 minGPT 改造的 DDP 版本
```

## 2. 直接复制的内容

### 2.1 nanoGPT

| 项目文件 | 上游文件 | 状态 |
|---|---|---|
| `pre_training/extras/karpathy_proj/nanoGPT/model.py` | `karpathy/nanoGPT/model.py` | 与当前 `3adf61e` 内容一致，仅少末尾换行 |

没有复制：

- `train.py`、`sample.py`、`bench.py`、`configurator.py`
- `config/`、`data/`、`assets/`
- `scaling_laws.ipynb`、`transformer_sizing.ipynb`

因此项目里的 nanoGPT 只是**模型定义参考**，不是完整训练流程。

### 2.2 nanochat

| 项目文件 | 上游文件 | 状态 |
|---|---|---|
| `pre_training/extras/karpathy_proj/nanochat/gpt.py` | `nanochat/gpt.py` | 与 **初始 commit `3a5e0bc`（2025-10-13）** 完全一致 |
| `pre_training/extras/karpathy_proj/nanochat/common.py` | `nanochat/common.py` | 同上 |
| `pre_training/extras/karpathy_proj/nanochat/adamw.py` | `nanochat/adamw.py` | 同上；该文件在当前 nanochat 已不存在 |
| `pre_training/extras/karpathy_proj/nanochat/muon.py` | `nanochat/muon.py` | 同上；该文件在当前 nanochat 已不存在 |

这是 nanochat 的**旧快照**，与当前 `92d63d4` 有较大差异：

- 旧版是 MQA（Multi-Query Attention），当前是 GQA；
- 旧版没有 Flash Attention 3、sliding window、value embedding；
- 旧版 `Muon`/`DistAdamW` 分文件，当前合并为 `nanochat/optim.py` 的 `MuonAdamW`；
- 旧版 `common.py` 没有 `COMPUTE_DTYPE`、文件下载锁等逻辑。

没有复制：

- `tokenizer.py`、`dataloader.py`、`dataset.py`、`engine.py`
- `flash_attention.py`、`fp8.py`、`optim.py`、`checkpoint_manager.py`
- `scripts/`、`tasks/`、`tests/`、`runs/`

因此项目里的 nanochat 是**旧版模型/优化器参考**，不能直接与当前 nanochat 对照。

## 3. 改造 / 间接来源

| 项目内容 | 与 karpathy 的关系 | 判断 |
|---|---|---|
| `pre_training/extras/karpathy_proj/minGPT_ddp/model.py` | 文件头明确写 “Adapted from karpathy/minGPT” | **明确改造**，但来源是第三个仓库 `minGPT`，不是 nanoGPT/nanochat |
| `pre_training/extras/karpathy_proj/minGPT/model.py` | 与 minGPT 的 `model.py` 同源，仅加了项目头与 import 调整 | **来自 minGPT**；当前 import 路径 `karpathy_proj.miniGPT.minigpt_utils` 已失修 |
| `pre_training/extras/distributed_training/fsdp/model.py` | 注释写明 “partly inspired by the nanoGPT model” | **间接参考 nanoGPT**，不是移动 |
| `models/gpt2/gpt.py` | 使用 `@dataclass GPTConfig`、`n_layer/n_heads/embed_dim` 等 nanoGPT 风格字段 | **风格改造稿**（推断）；实现只有配置和 embedding demo，非逐行复制 |
| `utils/learning_rate.py` | `learning_rate_warmup` 的 warmup+cosine 与 nanoGPT `get_lr` 思路相同，公式略有差异 | **同源思想/独立实现**，不能确认直接复制 |
| `utils/optimizer.py` | 使用 fused AdamW，类似 nanoGPT `configure_optimizers` 的工程习惯；但没有 decay/no-decay 参数分组 | **工程模式参考**，非直接复制 |
| `pre_training/exp/exp_pretrain_gpt2_124M.py` | DDP、AMP、`torch.compile` 等工程模式与 nanoGPT `train.py` 相似；数据/损失/评估来自 LLMs-from-scratch | **工程模式参考**，主体不是 nanoGPT |

## 4. 容易误判为 karpathy 来源的内容

以下内容与 nanoGPT 风格相似，但**不是**从 `karpathy/nanoGPT` 或 `karpathy/nanochat` 移入的：

| 内容 | 实际来源 |
|---|---|
| `pre_training/extras/smolGPT/`（已删除） | `Om-Alve/smolGPT`，nanoGPT 风格的教育项目 |
| `pre_training/extras/tinyllama/`（已删除） | TinyLlama 项目；`get_lr`/`configure_optimizers` 属于 nanoGPT 同源风格 |
| `post_training/imported/minimind/`（已删除） | MiniMind 项目 |
| `post_training/finetuning/model_pretrain/` | LLMs-from-scratch 的 GPT-2 权重加载 |
| `pre_training/references/gpt2_book/` | LLMs-from-scratch ch05 |
| `models/gpt2/gpt2_124M.py`、`layers/` | LLMs-from-scratch 主线实现，不是 nanoGPT 的 `c_attn`/FlashAttention 结构 |

## 5. 结论

1. **真正直接移动的只有 5 个文件**：
   - nanoGPT：`model.py`
   - nanochat：`gpt.py`、`common.py`、`adamw.py`、`muon.py`
2. **nanochat 是旧快照**（初始 commit），不是当前 nanochat；后续若要研究 Muon/FA3/最新结构，应以 `92d63d4` 为准。
3. **minGPT/minGPT_ddp 不是这两个仓库的内容**，而是第三个仓库 `karpathy/minGPT`；`minGPT_ddp/model.py` 是明确改造。
4. **项目主线的 GPT-2 实现来自 LLMs-from-scratch**，不是 nanoGPT；不要因为都用 GPT-2 就归因到 karpathy。
5. 目前这些副本**没有被主代码 import**，只是 `pre_training/extras/` 下的参考资料。

## 6. 建议

1. 在 `pre_training/extras/karpathy_proj/README.md` 标注：
   - nanoGPT 快照：`3adf61e`
   - nanochat 快照：`3a5e0bc`（当前上游已大幅演进）
   - minGPT 属于第三来源
2. 若要继续用 nanochat 的 Muon/FA3，建议重新拉取当前 `92d63d4` 的 `optim.py`、`flash_attention.py`，而不是沿用旧 `adamw.py`/`muon.py`。
3. 修复 `minGPT/model.py` 的 `karpathy_proj.miniGPT.minigpt_utils` 旧 import（当前实际路径是 `pre_training/extras/karpathy_proj/minGPT/minigpt_utils.py`）。
4. 如果只想保留“参考实现”，可以在 README 明确“不参与主链路、不保证可运行”，避免后续误用。
