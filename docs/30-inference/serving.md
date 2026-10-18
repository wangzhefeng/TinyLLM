> 所属：[TinyLLM Docs](../README.md) › [Inference](index.md)

# LLM 推理

> LLM Inference

## 什么是 LLM 推理

LLM 推理是指使用训练好的 LLM，如 GPT-4、Llama 4 和 DeepSeek-V3，
根据用户输入（通常以自然语言提示的形式提供）生成有意义的输出。
在推理过程中，模型通过其庞大的参数集处理提示，以生成文本、代码片段、摘要和翻译等响应。

## LLM 训练与推理的区别

LM 训练和推理是模型生命周期中的两个不同阶段。

### Training

> Training: Building the model’s understanding

在构建 LLM 时，训练是初始阶段。它涉及教会模型如何识别模式并做出准确预测。
这是通过让模型接触大量数据并根据其遇到的数据调整其参数来完成的。

LLM 训练中常用的技术包括：

* Supervised learning
    - 监督学习：向模型展示输入与正确输出配对的示例。
* Reinforcement learning
    - 强化学习：允许模型通过试错学习，根据反馈或奖励进行优化。
* Self-supervised learning
    - 自监督学习：通过预测数据中缺失或损坏的部分来学习，无需显式标签。

训练计算密集，通常需要大量的 GPU 或 TPU 集群。虽然初始成本可能非常高，但基本上是一次性支出。
一旦模型达到所需精度，通常只需要定期更新或改进模型才需要重新训练。

### Inference

> Inference: Using the model in real-time

LLM 推理是指将训练好的模型应用于新数据以进行预测。与训练不同，推理是持续且实时发生的，
能够立即响应用户输入或传入数据。这是模型被积极“使用”的阶段。
训练更好、更精细调整的模型通常能提供更准确和有用的推理。

推理计算需求是持续的，并且可能会变得非常高，尤其是在用户交互和流量增长时。
每个推理请求都会消耗 GPU 等计算资源。虽然单独的每个推理步骤可能比训练小，
但随时间的累积需求可能导致显著的运营成本。

## LLM 推理原理

在推理过程中，LLM 会利用其内部的注意力机制以及对先前上下文的知识，LLM 会逐个生成文本标记，。

### Tokens 和 Tokenization

Token 是 LLMs 处理文本时使用的语言最小单位。它可以是一个词、一个词的一部分，甚至是一个字符，
具体取决于分词器(tokenizer)。每个 LLM 都有自己的分词器(tokenizer)，采用不同的分词算法(tokenization algorithms)。

分词(Tokenization)是将输入文本（如一个句子或段落）转换为 token 的过程。
分词后的输入会被转换为 ID，这些 ID 在推理过程中被传递给模型。

对于输出，LLMs 会自回归地生成新的 token。从一组初始 token 开始，
模型根据它到目前为止所看到的一切预测下一个 token。这个过程会一直重复，直到达到停止标准。

### LLM 推理的两个阶段

对于基于 Transformer 的模型如 GPT-4，整个过程分为两个阶段：prefill 和 decode。

#### Prefill

当用户发送查询时，LLM 的 tokenizer 将 Prompt 转换为一系列 tokens。在分词后，prefill 阶段开始：

1. 这些 tokens（或 tokens ID）被嵌入(Embedded)为 LLM 可以理解的数值向量。
2. 这些向量通过多个 Transformer 层，每个层都包含自注意力机制。在这里，
   为每个 token 计算 Query(Q)、Key(K)和 Value(V) 向量。这些向量决定了 tokens 如何相互关注(attend)，
   从而捕捉上下文意义(contextual meaning)。
3. 随着模型处理提示，它构建一个 KV Cache（KV 缓存）来存储每个层中每个 token 的 key 和 value 向量。
   它充当内部内存，在 decoding 期间进行更快的查找。




#### Decode


#### Collocating Prefill 和 Decode





## 模型文件格式

* GGUF

## 模型量化格式

* 4-bit(GGUF)
* 8-bit
* 16-bit(original)

## 模型运行方式

> 4-bit, 16-bit

* inference serving
* fine-tuning

## 模型运行平台

> GGUF

* Ollama
* Open WebUI
* vLLM
* llama.cpp


# 资料

* [LLM Inference in Production](https://bentoml.com/llm/)
* [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm/tree/main)
* [从零实现 vLLM (1.1）：并行词嵌入](https://mp.weixin.qq.com/s?__biz=MzU4MTgyOTk1Mw==&mid=2247484630&idx=1&sn=215458fd1c45540a379ba710d9b8fc3a&scene=21&poc_token=HC0Lh2ijWjBD1fD5j0m8-T2b7ic7nqx01zG2Uxgz)
* [从零实现 vLLM (1.2）：如何实现张量并行](https://mp.weixin.qq.com/s?__biz=MzU4MTgyOTk1Mw==&mid=2247484710&idx=1&sn=34ec37dc82d4c1c559c7d33fff593a52&chksm=fd40d108ca37581e95ce318fd1e6434da0eb4ba35372dd6ba27f5b490745a62960d4902ab68e&cur_album_id=4092515450351730699&scene=189#wechat_redirect)
