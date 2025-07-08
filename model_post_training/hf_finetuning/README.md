

# 数据

## 常用数据


* 现有的开源数据集，Spider
* 使用 LLM 来创建合成数据集，Alpaca
* 使用人工来创建数据集，如 Dolly
* 使用上述方法的组合，如 Orca

## 数据格式

* 对话格式

```json
{"messages": [{"role": "system", "content": "You are..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"messages": [{"role": "system", "content": "You are..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"messages": [{"role": "system", "content": "You are..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

* 指令格式

```json
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
```
