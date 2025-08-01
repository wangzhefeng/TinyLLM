<details><summary>目录</summary><p>

- [Pre-training](#pre-training)
- [LLM 架构](#llm-架构)
    - [Tokenizers](#tokenizers)
    - [Embeddings](#embeddings)
    - [Transformer](#transformer)
        - [Attention](#attention)
            - [Self-Attention](#self-attention)
            - [Multi-Head Attention(MHA)](#multi-head-attentionmha)
            - [Grouped-Query Attention(GQA)](#grouped-query-attentiongqa)
            - [Multi-Query Attention(MQA)](#multi-query-attentionmqa)
        - [Transformer Block](#transformer-block)
    - [Transformer 变体](#transformer-变体)
        - [GPT](#gpt)
        - [Llama](#llama)
        - [DeepSeek](#deepseek)
    - [计算优化](#计算优化)
- [LLM 内容](#llm-内容)
    - [Tutorial](#tutorial)
        - [Pretraining](#pretraining)
            - [Transformer](#transformer-1)
            - [MoE](#moe)
        - [Finetuning](#finetuning)
            - [GRPO](#grpo)
        - [Model Evaluation](#model-evaluation)
            - [Instruction Follow Finetune](#instruction-follow-finetune)
    - [Usage](#usage)
        - [Pretraining](#pretraining-1)
        - [Finetuning](#finetuning-1)
        - [Agent](#agent)
    - [others](#others)
        - [KV Cache](#kv-cache)
</p></details><p></p>

# Pre-training

* Q&A format
* Long-context stage
* Continued pre-training
* High-quality stage
* Knowledge distillation

> * 专注于数据集过滤（而不是尽可能多地收集数据）是 LLM 训练中的一个显著趋势。
>   但前提是它必须符合一定的质量标准。

# LLM 架构

## Tokenizers




Tokens: A token is an indivisible unit of text, such as a word, subword or character, 
and is part of a predefined vocabulary.

Tokenizer: A tokenizer $T$ divides text into tokens of an arbitrary level of granularity.

Main types of tokenizers:

* word
* subword
* character byte

Commonly-used subword-level tokenizer:

* Byte-Pair Encoding(BPE)
* Unigram

## Embeddings

Embedding: An embedding is a numerical representation of an element(e.g. token, sentence) and is characterized by a vector $x\in \mathbf{R}^{n}$.

Similarity: The cosine similarity between two tokens $t_{1}$, $t_{2}$ is quantified by:

$$\text{similarity}(t_{1}, t_{2})=\frac{t_{1} \cdot t_{2}}{||t_{1}|| ||t_{2}||} =cos(\theta) \in \[-1, 1\]$$

## Transformer

### Attention

#### Self-Attention

Attention: Given a query $q$, we want to know which key $k$ the query should pay "attention" to with respect to the associated value $v$.

![img](images/)

Attention can be efficiently computed using matrices $Q$, $K$, $V$ that contain queries $q$, 
key $k$ and values $v$ respectively, along with the dimension $d_{k}$ of keys:

$$\text{attention} = \text{softmax}\Big(\frac{QK^{T}}{\sqrt{d_{k}}}\Big)V$$

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

## Transformer 变体

* Transformer
    - []()
    - [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
    - [harvardnlp/annotated-transformer](https://github.com/harvardnlp/annotated-transformer/)
* Encoder-Only: BERT(Bidirectional Encoder Representations from Transformer)
* Encoder-Decoder: T5
* Decoder-Only
    * GPT(Generative Pre-trained Transformer)
    * Llama
    * DeepSeek

### GPT

### Llama

### DeepSeek

## 计算优化

注意力机制近似：注意力计算的时间复杂度为 $\mathbb{O}(n2)$，
当序列长度 $n$ 增加时，计算成本也迅速上升。常见的近似方法包括：

* 稀疏注意力：注意力不在整个序列中进行，而只在更相关的 token 之间进行.
* 低秩近似：将注意力公式近似为低秩矩阵的乘积，从而显著降低计算负担。

Flash Attention：Flash Attention 是一种精确的注意力计算优化方法。它通过充分利用 GPU 硬
件，在快速的静态随机存取存储器（Static Random-Access Memory, SRAM）中执行矩阵操作，再
将结果写入较慢的高带宽内存（High Bandwidth Memory, HBM），从而实现更高效的计算。
注：在实际应用中，Flash Attention 可有效减少内存占用并显著加速注意力计算过程。

# LLM 内容

## Tutorial

### Pretraining

* [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch/tree/main)
* [MiniMind](https://github.com/jingyaogong/minimind)

#### Transformer

* [Paper: Transformers without Normalization](https://arxiv.org/abs/2503.10622)
* [Transformers without Normalization](https://jiachenzhu.github.io/DyT/)

#### MoE

* [Transformer 和 MoE 的差别](https://mp.weixin.qq.com/s/z5gpNkFkbR7nR4HHIHGx0g)
* [可视化图解 MoE 大模型的 7 个核心问题](https://mp.weixin.qq.com/s/-SFFB6gUp0KA4x95lCoxcg)

### Finetuning

#### GRPO

* [HuggingFace::Practical Exercise: Fine-tune a model with GRPO](https://huggingface.co/learn/nlp-course/en/chapter12/5?fw=pt)
* [Colab::Finetune LLMs with GRPO](https://colab.research.google.com/github/huggingface/notebooks/blob/main/course/en/chapter13/grpo_finetune.ipynb#scrollTo=ilrEVEdDkGgs)

### Model Evaluation

#### Instruction Follow Finetune

* [Short-answer and multiple choice benchmarks: MMLU](https://arxiv.org/abs/2009.03300)
* [Human perference comparison to other LLMs: LMSYS chatbot arena](https://arena.lmsys.org)
* [Automated conversational benchmarks: AlpacaEval](https://tatsu-lab.github.io/alpaca_eval/)

## Usage

### Pretraining

### Finetuning

### Agent

* [OpenManus 深夜开源](https://mp.weixin.qq.com/s/Z1vtpH-Wx0QPI8MFum0Wiw)
* [mannaandpoem/OpenManus](https://github.com/mannaandpoem/OpenManus)
* [Manus 开源复刻框架 OWL，测评和使用教程](https://mp.weixin.qq.com/s/lvs2y2ZnSJo5GZ7gbVkQLQ)
* [camel-ai/owl](https://github.com/camel-ai/owl)

## others

### KV Cache

* [图解 KV Cache：解锁 LLM 推理效率的关键](https://mp.weixin.qq.com/s/uWV56N-NeHA57_UeNDE67g)

