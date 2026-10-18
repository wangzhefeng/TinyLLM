> 所属：[TinyLLM Docs](../../README.md) › [Post-training](../index.md) › [Fine-tuning](index.md)

# 微调数据

## 常用数据

* 现有的开源数据集，[Spider](https://huggingface.co/datasets/xlangai/spider)
* 使用 LLM 来创建合成数据集，[Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)
* 使用人工来创建数据集，如 [Dolly](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
* 使用上述方法的组合，如：
    - [Orca](https://huggingface.co/datasets/Open-Orca/OpenOrca)
    - [Orca: Progressive Learning from Complex Explanation Traces of GPT-4](https://arxiv.org/abs/2306.**02707**)

数据集合：

* [LLM Datasets](https://github.com/mlabonne/llm-datasets)
* [大语言模型高质量数据集汇总](https://github.com/ninehills/blog/issues/129)
* [Hugging Face datasets](https://huggingface.co/datasets)

## 数据格式

* 对话格式

> conversational

```json
{"messages": [{"role": "system", "content": "You are..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"messages": [{"role": "system", "content": "You are..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"messages": [{"role": "system", "content": "You are..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

* 指令格式

> instruction

```json
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
```

## 数据预处理

* [Distlabel](https://distilabel.argilla.io/latest/)

## 文本分类数据（text_clf）

# 数据处理

1. 数据下载

```bash
$ bash data_load.py
```

2. 数据处理

```bash
$ bash data_process.py
```

3. 创建 data loaders

> 测试

```bash
$ bash data_loader.py
```
