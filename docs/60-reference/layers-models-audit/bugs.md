# 代码问题清单（已逐条验证）

> 所属：[TinyLLM Docs](../../README.md) › [参考与分析](../index.md) › [layers/models 审查](index.md)

所有问题均经源码交叉验证。P0 = 运行时必挂或输出必错；P1 = 结果错误但不崩溃；P2 = 设计/结构缺陷。

> **状态（2026-10-20）**：P0 #1–#11 与 P1 #12–#17、KV cache 溢出防护均已修复；
> Phase 1 结构整改完成（目录改名、inference 归位、sys.path hack 清除、Qwen3 重复收敛、
> 硬编码 bf16、mask 语义、llama Sequential/tok_emb、#18/#19）。
> 由 `tests/unit/test_layers_regression.py`（17 项）覆盖；全量 85 项测试通过。
> 剩余：P2 #22（tokenizers 归属，暂保留）、#23（generate 变体收敛，部分完成）、
> #24/#25/#28/#29（完整接口统一与权重加载抽象，随 Phase 2 需要）。

## P0：必修 bug

| # | 位置 | 问题 |
|---|---|---|
| 1 | `layers/attention.py:262` | `attn_scores / keys.shape[-1]**-0.5`：指数为 **-0.5**，等于乘 √d 而非除 √d，softmax 饱和 |
| 2 | `layers/attention.py:875` | `compute_rope(keys, ...)` 误用 reshape 前的 3 维 `keys` 而非 `keys_new`，Qwen3-optimized 数值必错 |
| 3 | `layers/attention.py:877-879` | `if layer_idx is not None: raise "layer_idx required"` 条件写反；model 层必传 `layer_idx=i`，optimized+cache 路径 100% 崩溃 |
| 4 | `layers/inference.py:360-363` | `[:, -1]` 索引两次，(B,V) 变 (B,)，采样与 cat 全错 |
| 5 | `layers/inference.py:113,172,229` 等 | `context_length or model.pos_embed.num_embeddings`：Llama/Qwen3/Gemma3 无 `pos_embed`，不传 context_length 即 AttributeError |
| 6 | `layers/embedding/text_embedding.py:54` | `assert x == torch.arange(...)`：多元素张量断言必抛异常，该层永远不可用 |
| 7 | `layers/position_encoding/FixPE.py:33,72` | `PositionalEncoding` 定义两次互相覆盖，文件内自测代码全部不可运行 |
| 8 | `models/gpt2/gpt2_124M.py:90` | `reset_kv_cache()` 调 `blk.attn.reset_cache()`，但 block 用的 `MHAPyTorchScaledDotProduct` 无此方法；该类不支持 cache，`use_cache=True` 路径语义错误（每 token 只能看到自己） |
| 9 | `models/llama/llama2.py:38` / `llama3_8B.py:39` | `FeedForwardSiLU` 读 `cfg.d_ff`，而 llama config 只定义 `hidden_dim`，自带配置构造即 KeyError；另 `eos_token_id=50256` 是 GPT-2 的（Llama2=2、Llama3=128001） |
| 10 | `models/gemma3/Gemma3_270M.py:50` | 未做权重 tying（官方 270M `tie_word_embeddings=True`），参数虚增约 1.68 亿，加载 HF 权重时 out_head 为随机噪声 |
| 11 | tokenizers 系列 | `gemma.py:58` 把字符串赋给 `pad_token_id`；`llama2_7b_sentencepiece.py:81` `decode_pieces` 误传 int ids；`hf_gpt2_bpe.py:50` 对 property 加括号调用；`vocab_download.py` 用 `124` 而非 `124M` 必 404；`simple_bpe.py:428` `vocab_size=len(corpus)` |

## P1：结果错误但不崩溃

| # | 位置 | 问题 |
|---|---|---|
| 12 | `layers/moe.py:71` | 注释称高斯噪声，实为 `torch.rand_like` 均匀噪声 |
| 13 | `layers/moe.py:96` | `cfgs.capacity_factor = 1.0` 链式赋值强行覆盖调用方配置 |
| 14 | `layers/normailzation/rms_norm.py:62` | 基础 RMSNorm 不做 fp32 上转，与 Qwen3/Gemma3 版精度语义不一致 |
| 15 | `layers/transformer_block.py:195,231,405` 等 | Llama3/Qwen3/Gemma3 block 硬编码 `.to(torch.bfloat16)`，无视 `cfg.dtype`，fp32 训练被静默降精度 |
| 16 | `layers/tokenizers/bpe/simple_bpe.py` | encode 未按训练 rank 合并（违背 BPE 核心不变量）；decode 遇 `\n` 插入多余空格，encode/decode 不互逆 |
| 17 | `layers/attention.py:148` | KV cache `torch.zeros` 未传 dtype，bf16 模型下 cache 为 fp32 |
| 18 | `layers/attention.py:375,417,511` | `self.training = training` 覆盖 `nn.Module.training`，`model.eval()` 对这些类失效 |
| 19 | `models/qwen3/qwen3.py:76-82` | cache 模式 `current_pos` 无限增长，RoPE `cos[offset:offset+seq_len]` 越界无防护 |

## P2：设计 / 结构缺陷

| # | 位置 | 问题 |
|---|---|---|
| 20 | 目录 | `layers/normailzation/` 拼写错误（应 normalization），已被多处 import 固化；类名 `CasualAttention` 应为 Causal |
| 21 | `layers/inference.py` | 生成/解码逻辑放在 layers/，与 AGENTS.md 定义的 `inference/` 域并存且重复；tokenizers 也非"层" |
| 22 | 全部 27 个 layers 文件 | `sys.path.append(Path.cwd())` hack，依赖从仓库根启动，pytest/子目录运行即崩 |
| 23 | 重复实现 | Qwen3 四个模型文件逐字复制；`TransformerBlockQwen3` vs `_original` 仅差属性名；inference.py 6+ 个 generate 变体——P0 #1/#3/#4 均为 drift 型 bug |
| 24 | `layers/masking.py` vs `attention.py` | mask 语义分裂：True=保留 vs True=屏蔽，无组合函数 |
| 25 | 接口不统一 | forward 签名六种各异；`tok_embed` vs `tok_emb`；DotDict vs 字典索引；两个 KVCache 类接口完全不同 |
| 26 | `models/llama/llama2.py` | 用 `nn.Sequential` 堆 block，永远无法传 cache/mask，不支持增量解码 |
| 27 | `models/deepseek/mla.py:34` | 报错指向不存在的 `models/deepseek/README.md`（AGENTS.md 禁止 models/ 下放 README） |
| 28 | 包结构 | 仅 deepseek/、bert/ 有 `__init__.py`；绝对导入与 sys.path hack 混杂 |
| 29 | 权重加载 | 模型层无 from_pretrained/tie_weights；loader 散落 `inference/loaders/`，仅 GPT-2/Qwen3 能加载官方权重 |

## 验证方式

- 人工通读：`attention.py`（1092 行）、`inference.py`（715 行）、`transformer_block.py`（425 行）、`kv_cache.py` 全文。
- 交叉核对：model 层调用方（`qwen3_optimized.py:96-103` 传 `layer_idx`）、config 键（llama 无 `d_ff`）、官方架构（Gemma3 tying / Qwen3 QK-Norm / Gemma3 双 RoPE——后两者实现**正确**）。
- 两个并行子代理分别复审 layers/ 与 models/，结论一致。

## 相关

- [缺失主题清单](gaps.md)
- [落地路线](roadmap.md)
