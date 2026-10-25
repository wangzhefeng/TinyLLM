> 所属：[TinyLLM Docs](../README.md) › [模型推断](index.md)

# LLM 模型推断

> LLM Inference

## 什么是模型推断

模型推断是指使用训练好的 LLM（如 GPT-4、Llama 4、DeepSeek-V3），
根据用户输入（通常以自然语言提示的形式提供）生成有意义的输出。
在推断过程中，模型通过其参数处理提示，生成文本、代码片段、摘要和翻译等响应。

## 训练与推断的区别

训练和推断是模型生命周期中的两个不同阶段。

### Training

> Training: Building the model’s understanding

训练是构建 LLM 的初始阶段，目标是让模型识别模式并做出准确预测。
训练通过让模型接触大量数据并调整参数来完成。

常用的训练技术包括：

* Supervised learning
    - 监督学习：向模型展示输入与正确输出配对的示例。
* Reinforcement learning
    - 强化学习：允许模型通过试错学习，根据反馈或奖励优化。
* Self-supervised learning
    - 自监督学习：通过预测数据中缺失或损坏的部分来学习，无需显式标签。

训练计算密集，通常需要大量 GPU 或 TPU；但基本是一次性支出。

### Inference

> Inference: Using the model in real-time

模型推断是将训练好的模型应用于新数据以产生输出。与训练不同，推断是持续、实时发生的，
能够立即响应用户输入或传入数据。这是模型被积极“使用”的阶段。
训练更充分、微调更精细的模型通常能提供更准确的推断结果。

推断计算需求持续存在，可能非常高，尤其是用户交互和流量增长时。
每个推断请求都会消耗 GPU 等资源；单次推断步骤可能比训练小，
但随时间的累积需求会带来显著运营成本。

## 模型推断原理

在推断过程中，LLM 利用内部注意力机制和上下文知识，逐个生成文本 token。

### Tokens 和 Tokenization

Token 是 LLM 处理文本时使用的最小单位，可以是一个词、词的一部分，甚至一个字符。
每个 LLM 有自己的 tokenizer 和分词算法。

分词（Tokenization）将输入文本转换为 token；分词后的输入再转换为 ID，在推断过程中传给模型。

输出阶段，LLM 自回归地生成新 token：从初始 token 开始，模型根据已看到的内容预测下一个 token，
重复直到达到停止条件。

### 模型推断的两个阶段

基于 Transformer 的模型（如 GPT-4）通常分为 prefill 和 decode 两个阶段。

#### Prefill

用户发送查询后，tokenizer 将 prompt 转换为 tokens；随后进入 prefill：

1. tokens/ID 被嵌入为模型可理解的数值向量。
2. 向量通过多个 Transformer 层；每层包含自注意力，为每个 token 计算 Q/K/V。
3. 模型在处理提示时构建 KV Cache，存储每层每个 token 的 key/value，供 decode 阶段快速查找。

#### Decode

prefill 完成后进入 decode，模型自回归地逐个生成 token：

1. 每步只输入上一步新生成的 1 个 token，而不是重新输入整个序列。
2. 注意力计算直接读取 prefill 写入的 KV Cache，只为新 token 计算 Q/K/V，
   把每步计算量从 O(n²) 降到 O(n)。
3. 对最后一个位置的 logits 应用解码策略（greedy、temperature、top-k、top-p），
   采样下一个 token，直到命中 EOS 或达到长度上限。

decode 是显存带宽密集型（memory-bound）：每步读取全部权重和 KV Cache，
但计算量很小，因此吞吐主要取决于显存带宽而非算力。

#### Collocating Prefill 和 Decode

* **混合部署（collocated）**：prefill 与 decode 跑在同一批 GPU 上，通过连续批处理
  （continuous batching）把不同请求的 prefill/decode 步骤混排在同一批次，
  实现简单、资源利用率高；但长 prompt 的 prefill 会阻塞 decode（head-of-line blocking）。
* **分离部署（disaggregated，P/D 分离）**：prefill 和 decode 部署在独立 GPU 池，
  prefill 完成后把 KV Cache 传给 decode 节点。两阶段可独立扩缩容，
  延迟更稳定，代价是 KV 传输开销与系统复杂度。

教学级实现见 [nano_vllm](nano-vllm.md)（分页 KV + 连续批处理）。

## 模型文件格式

* GGUF

## 模型量化格式

* 4-bit（GGUF）
* 8-bit
* 16-bit（original）

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

## 聊天界面（Chainlit）

`inference/apps/app.py` + `inference/scripts/run_app.sh` 提供本地聊天 UI，
用法 `bash inference/scripts/run_app.sh [orig|own] [gpt2|qwen3]`：

- **qwen3 模式**：`inference.loading.load_model` 加载（默认 key
  `qwen3-0.6b`，`QWEN3_MODEL_KEY` 可换），Qwen3 tokenizer，
  `format_qwen3_chat(..., add_thinking=...)` 组 prompt；
- **thinking 开关**：Chainlit ChatSettings Switch，开启时模型生成
  `<think>...</think>` 再回答；
- **流式输出**：走 `inference/generation.generate_text_stream`（KV cache），
  `<think>` 段与正式回答分离渲染（thinking 消息独立 author）；
- gpt2 模式行为不变（`GPT2_WEIGHTS=orig|own` 选择权重来源，无独立 shim 文件）；
- chainlit 为可选依赖（`pip install chainlit`）。

## 推断吞吐对比脚本

`inference/scripts/benchmark_compile.py`：eager vs `torch.compile` 的
tokens/s 对比（`--model` / `--max-new-tokens` / `--repeat`）。设备感知：
仅 CUDA 跑 compile 对比（Inductor），MPS/CPU 打印说明后只跑 eager；
每轮用 `generate_stats` 报告 Time / tokens/sec，末尾汇总加速比。

# 资料

* [LLM Inference in Production](https://bentoml.com/llm/)
* [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm/tree/main)
* [从零实现 vLLM (1.1)：并行词嵌入](https://mp.weixin.qq.com/s?__biz=MzU4MTgyOTk1Mw==&mid=2247484630&idx=1&sn=215458fd1c45540a379ba710d9b8fc3a&scene=21&poc_token=HC0Lh2ijWjBD1fD5j0m8-T2b7ic7nqx01zG2Uxgz)
* [从零实现 vLLM (1.2)：如何实现张量并行](https://mp.weixin.qq.com/s?__biz=MzU4MTgyOTk1Mw==&mid=2247484710&idx=1&sn=34ec37dc82d4c1c559c7d33fff593a52&chksm=fd40d108ca37581e95ce318fd1e6434da0eb4ba35372dd6ba27f5b490745a62960d4902ab68e&cur_album_id=4092515450351730699&scene=189#wechat_redirect)
