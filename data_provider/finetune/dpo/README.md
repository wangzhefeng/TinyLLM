
# Preference Finetuning 简介

偏好微调（Preference Finetuning） 旨在使指令微调后的 LLM 更加符合人类偏好。

生成偏好微调数据集有多种方法：

1. 使用指令微调 LLM 生成多个响应，并由人工根据偏好标准进行排序。
2. 使用指令微调 LLM 生成多个响应，并由 LLM 根据设定的偏好标准进行排序。
3. 使用 LLM 基于特定偏好标准直接生成偏好（Preferred）和非偏好（Dispreferred）响应。

指令数据集格式如下：

* 输入（Input）

```json
[
    {
        "instruction": "What is the state capital of California?",
        "input": "",
        "output": "The state capital of California is Sacramento.",
    },
    {
        "instruction": "Provide a synonym for 'fast'.",
        "input": "",
        "output": "A synonym for 'fast' is 'quick'.",
    },
    {
        "instruction": "What is the capital of Greece?",
        "input": "",
        "output": "The capital of Greece is Athens.",

    },
    ...
]
```

生成的数据集格式如下，其中 **较礼貌的响应** 被标记为 `'chosen'`（偏好响应），
**较不礼貌的响应** 被标记为 `'rejected'`（非偏好响应）。

输出（Output）

```json
[
    {
        "instruction": "What is the state capital of California?",
        "input": "",
        "output": "The state capital of California is Sacramento.",
        "rejected": "Look, the state capital of California is obviously Sacramento.",
        "chosen": "The state capital of California is Sacramento."
    },
    {
        "instruction": "Provide a synonym for 'fast'.",
        "input": "",
        "output": "A synonym for 'fast' is 'quick'.",
        "chosen": "A suitable alternative to 'fast' would be 'quick'.",
        "rejected": "A synonym for 'fast' is 'quick'."
    },
    {
        "instruction": "What is the capital of Greece?",
        "input": "",
        "output": "The capital of Greece is Athens.",
        "chosen": "I'd be happy to help! The capital of Greece is indeed Athens.",
        "rejected": "The capital of Greece is Athens."
    },
    ...
]
```

# DPO: Direct Preference Optimization

* [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
* [Tutorial](https://github.com/MLNLP-World/LLMs-from-scratch-CN/blob/main/ch07/04_preference-tuning-with-dpo/dpo-from-scratch.ipynb)

直接偏好优化（DPO）是一种替代强化学习（RLHF）的方法，可用于微调大语言模型（LLM）。
DPO 可用于微调（对齐）模型，使其生成的响应更符合用户期望和指令。

DPO 训练的目标是让模型更倾向于生成 'chosen' 风格的响应，而 避免 'rejected' 风格。
