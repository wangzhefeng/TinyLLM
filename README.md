<details><summary>目录</summary><p>

- [Data Preprocessing](#data-preprocessing)
    - [数据形式](#数据形式)
    - [Filtering](#filtering)
    - [Synthetic Data](#synthetic-data)
    - [Mixing](#mixing)
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
- [LLM 推理](#llm-推理)
    - [Inference Compute Scaling](#inference-compute-scaling)
    - [Reinforcement Learning](#reinforcement-learning)
    - [Knowledge Distillation](#knowledge-distillation)
- [技巧](#技巧)
- [其他资料](#其他资料)
</p></details><p></p>

# Data Preprocessing

* Filtering, 过滤
* Synthetic Data, 合成
* Mixing, 混合

## 数据形式

* instruction-response pairs


## Filtering


## Synthetic Data

> 数据合成

* 从无到有生成指令数据--Instruction SFT
    - [Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing](https://arxiv.org/abs/2406.08464)
* 从原始语料合成指令数据--Instruction Pretraining
    - [Instruction Pre-Training: Language Models are Supervised Multitask Learners](https://arxiv.org/abs/2406.14491)
    - [Genie: Achieving Human Parity in Content-Grounded Datasets Generation](https://arxiv.org/abs/2401.14367)
* Instruction Synthesizer
    - [Mistral 7B v0.1 LLM](https://magazine.sebastianraschka.com/i/138555764/mistral-b)
    - [HotpotQA Dataset](https://arxiv.org/abs/1809.09600)

## Mixing


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

* [Byte-Pair Encoding(BPE)]()
* [Unigram]()

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

# LLM 推理

> LLM Reasoning

## Inference Compute Scaling

> 推断计算增强

推断时间计算扩展（也叫推断计算增强、测试时增强等）包含一系列在推理阶段（即用户输入提示词时）提升模型推理能力的方法，
这些方法无需对底层模型权重进行训练或修改。其核心思想是通过增加计算资源来换取性能提升，
借助思维链推理（chain-of-thought reasoning）及多种采样程序等技术，使固定参数的模型展现出更强的推理能力。

## Reinforcement Learning

> 强化学习

强化学习是一类通过最大化奖励信号来提升模型推理能力的训练方法。其奖励机制可分为两类：

* 广义奖励：如任务完成度或启发式评分
* 精准可验证奖励：如数学问题正确答案或编程任务通过率

与推断时间计算增强（inference-time compute scaling）不同，
RL 通过动态调整模型参数（weights updating）实现能力提升。
该机制使模型能够基于环境反馈，通过试错学习不断优化其推理策略。

> 注： 在开发推理模型时，
> 需明确区分此处的纯强化学习（RL）方法与常规大语言模型开发中用于偏好微调的基于人类反馈的强化学习（RLHF）（如图 2 所示）。
> 二者的核心差异在于奖励信号的来源：RLHF 通过人类对模型输出的显式评分或排序生成奖励信号，
> 直接引导模型符合人类偏好行为；纯 RL 则依赖自动化或环境驱动的奖励信号（如数学证明的正确性），
> 其优势在于客观性，但可能降低与人类主观偏好的对齐度。典型场景对比：纯 RL 训练：以数学证明任务为例，
> 系统仅根据证明步骤的正确性提供奖励；RLHF 训练：需人类评估员对不同输出进行偏好排序，
> 以优化符合人类标准（如表述清晰度、逻辑流畅性）的响应。

## Knowledge Distillation

> 知识蒸馏

模型蒸馏是指将高性能大模型习得的复杂推理模式迁移至更轻量化模型的技术。在 LLM 领域，该技术通常表现为：
使用高性能大模型生成的高质量标注指令数据集进行监督微调（Supervised Fine-Tuning, SFT）。
这种技术在 LLM 文献中常统称为知识蒸馏（Knowledge Distillation）或蒸馏（Distillation）。

与传统深度学习的区别：经典知识蒸馏中，「学生模型」需同时学习「教师模型」的输出结果和 logits，
而 LLM 的蒸馏通常仅基于输出结果进行迁移学习。

> 注：本场景采用的监督微调（SFT）技术与常规大语言模型开发中的 SFT 类似，
> 其核心差异体现在训练样本由专为推理任务开发的模型生成（而非通用 LLM）。
> 也因此，其训练样本更集中于推理任务，通常包括中间推理步骤。


# 技巧

1. 寻览 LLMs 通常不适用 dropout;
2. 现代 LLMs 在 query、key 和 value 矩阵的 `nn.Linear` 层也不使用偏置(bias)向量（与早期的 GPT 模型不同）;

# 其他资料

* [The Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison)
* [Reasoning From Scratch ch01](https://mp.weixin.qq.com/s/zQUB9ZXqtSRGJU_YWMoMEw)
