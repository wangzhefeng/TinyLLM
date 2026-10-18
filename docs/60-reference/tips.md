> 所属：[TinyLLM Docs](../README.md) › [Reference](index.md)

# 技巧与资料

1. 寻览 LLMs 通常不适用 dropout;
2. 现代 LLMs 在 query、key 和 value 矩阵的 `nn.Linear` 层也不使用偏置(bias)向量（与早期的 GPT 模型不同）;

# 其他资料

- [The Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison)
- [Reasoning From Scratch ch01](https://mp.weixin.qq.com/s/zQUB9ZXqtSRGJU_YWMoMEw)
