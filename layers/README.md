# 模型架构

## Tokenizers

Tokens: A token is an indivisible unit of text, such as a word, subword or character, and is part of a predefined vocabulary.

Tokenizer: A tokenizer `$T$` divides text into tokens of an arbitrary level of granularity.

Main types of tokenizers:

* word
* subword
* character byte

Commonly-used subword-level tokenizer:

* Byte-Pair Encoding(BPE)
* Unigram

## Embeddings

Embedding: An embedding is a numerical representation of an element(e.g. token, sentence) and is characterized by a vector `$x\in \mathbf{R}^{n}$`.

Similarity: The cosine similarity between two tokens `$t_{1}$`, `$t_{2}$` is quantified by:

`$$\text{similarity}(t_{1}, t_{2})=\frac{t_{1} \cdot t_{2}}{||t_{1}|| ||t_{2}||} =cos(\theta) \in \[-1, 1\]$$`

## Transformer

### Attention

#### Self-Attention

Attention: Given a query `$q$`, we want to know which key `$k$` the query should pay "attention" to with respect to the associated value `$v$`.

![img](images/)

Attention can be efficiently computed using matrices `$Q$`, `$K$`, `$V$` that contain queries `$q$`, 
key `$k$` and values `$v$` respectively, along with the dimension `$d_{k}$` of keys:

`$$\text{attention} = \text{softmax}\Big(\frac{QK^{T}}{\sqrt{d_{k}}}\Big)V$$`

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

* Rotary Position Embeddings, RoPE

#### Transformer 变体

Encoder-Only: BERT(Bidirectional Encoder Representations from Transformer)

Encoder-Decoder: T5

Decoder-Only: GPT(Generative Pre-trained Transformer)

### 计算优化

注意力机制近似：注意力计算的时间复杂度为 `$\mathbb{O}(n2)$`，
当序列长度 `$n$` 增加时，计算成本也迅速上升。常见的近似方法包括：

* 稀疏注意力：注意力不在整个序列中进行，而只在更相关的 token 之间进行.
* 低秩近似：将注意力公式近似为低秩矩阵的乘积，从而显著降低计算负担。

Flash Attention：Flash Attention 是一种精确的注意力计算优化方法。它通过充分利用 GPU 硬
件，在快速的静态随机存取存储器（Static Random-Access Memory, SRAM）中执行矩阵操作，再
将结果写入较慢的高带宽内存（High Bandwidth Memory, HBM），从而实现更高效的计算。
注：在实际应用中，Flash Attention 可有效减少内存占用并显著加速注意力计算过程。


## Transformer 架构

### GPT


### Llama


### DeepSeek


## Llama

### Prompt


### Finetuning

* SFT
* LoRA

### MoE


### Knowledge diffu

### 

