> 所属：[TinyLLM Docs](../README.md) › [Models](index.md)

# LLM 架构

## Tokenizers

Tokens: A token is an indivisible unit of text, such as a word, subword or character,
  
and is part of a predefined vocabulary.

Tokenizer: A tokenizer $T$ divides text into tokens of an arbitrary level of granularity.

Main types of tokenizers:

- word
- subword
- character byte

Commonly-used subword-level tokenizer:

- Byte-Pair Encoding(BPE)
- Unigram

## Embeddings

Embedding: An embedding is a numerical representation of an element(e.g. token, sentence) and is characterized by a vector $x\in \mathbf{R}^{n}$.

Similarity: The cosine similarity between two tokens $t_{1}$, $t_{2}$ is quantified by:

$\text{similarity}(t_{1}, t_{2})=\frac{t_{1} \cdot t_{2}}{||t_{1}|| ||t_{2}||} =cos(\theta) \in \[-1, 1\]$

## Transformer

### Attention

#### Self-Attention

Attention: Given a query $q$, we want to know which key $k$ the query should pay "attention" to with respect to the associated value $v$.

<iframe src="images/"/>

Attention can be efficiently computed using matrices $Q$, $K$, $V$ that contain queries $q$,
  
key $k$ and values $v$ respectively, along with the dimension $d_{k}$ of keys:

$\text{attention} = \text{softmax}\Big(\frac{QK^{T}}{\sqrt{d_{k}}}\Big)V$

#### Multi-Head Attention(MHA)

Multi-Head Attention(MHA): a MHA layer performs attention computations across multiple heads,
  
then projects the result in the output space.

#### Grouped-Query Attention(GQA)

#### Multi-Query Attention(MQA)

### Transformer Block

Transformer:

Encoder:

Decoder:

Position Embeddings:

- Rotary Position Embeddings, RoPE

## Transformer 变体

- Transformer

  - 
  - [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
  - [harvardnlp/annotated-transformer](https://github.com/harvardnlp/annotated-transformer/)

- Encoder-Only: BERT(Bidirectional Encoder Representations from Transformer)
- Encoder-Decoder: T5
- Decoder-Only

  - GPT(Generative Pre-trained Transformer)
  - Llama
  - DeepSeek


### GPT

### Llama

### DeepSeek

- 独立模型目录：[deepseek.md](deepseek.md)（MLA 完整实现 + DeepSeekModel 组装，2026-10-20）
- DeepSeek-R1-Distill-Qwen 的 LoRA 微调仍在 `post_training/peft/model_peft/deepseek_lora/`。

## layers/ 新增组件（2026-10-20）

- `layers/position_encoding/ALiBi.py`：ALiBi 线性偏置位置编码（长度外推）
- `layers/position_encoding/RoPE.py`：新增 NTK-aware（`ntk_factor`）与 YaRN（`yarn_config`）外推
- `layers/attention.py`：新增独立 `SlidingWindowAttention`（SWA，含窗口 KV cache）
- `layers/kv_cache.py`：新增 `trim_kv_cache_window`（tuple cache 窗口截断）
- `layers/moe.py`：`MoEFeedForward` 新增 Switch 式负载均衡 aux loss 与 DeepSeekMoE shared expert
- `layers/lora.py`：LoRA/DoRA 通用层（`post_training/peft/layers/lora_dora.py` 为兼容 shim）
- `layers/normalization/`：目录名由 normailzation 修正为 normalization

## 计算优化

注意力机制近似：注意力计算的时间复杂度为 $\mathbb{O}(n2)$，
  
当序列长度 $n$ 增加时，计算成本也迅速上升。常见的近似方法包括：

- 稀疏注意力：注意力不在整个序列中进行，而只在更相关的 token 之间进行.
- 低秩近似：将注意力公式近似为低秩矩阵的乘积，从而显著降低计算负担。

Flash Attention：Flash Attention 是一种精确的注意力计算优化方法。它通过充分利用 GPU 硬
  
件，在快速的静态随机存取存储器（Static Random-Access Memory, SRAM）中执行矩阵操作，再
  
将结果写入较慢的高带宽内存（High Bandwidth Memory, HBM），从而实现更高效的计算。
  
注：在实际应用中，Flash Attention 可有效减少内存占用并显著加速注意力计算过程。
