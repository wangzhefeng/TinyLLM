> 所属：[TinyLLM Docs](../README.md) › [Post-training](index.md)


# 知识蒸馏

# 教程

* [GitHub: dkozlov/awesome-knowledge-distillation](https://github.com/dkozlov/awesome-knowledge-distillation)
* [PyTorch: Knowledge Distillation Tutorial](https://pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html)

# 文献

* [Distilling the Knowledge in a Neural Network](https://arxiv.org/pdf/1503.02531)
* [A Survey on Knowledge Distillation of Large Language Models](https://arxiv.org/abs/2402.13116)
* [知识蒸馏研究综述](http://cjc.ict.ac.cn/online/onlinepaper/HZH315.pdf)
* [知识蒸馏全面综述]
    - [article](https://mp.weixin.qq.com/s/K3vHMW7QdPNLxnlpR0ij3w)
    - [GitHub](https://github.com/IPL-sharif/KD_Survey)

# 两条实现线（2026-10-21）

| 文件 | 类型 | 说明 |
|---|---|---|
| `post_training/reasoning/training/distillation.py` | logit 蒸馏 | Hinton 2015：KL(softmax(teacher/T) ‖ softmax(student/T))·T² + 可选硬标签 CE |
| `post_training/reasoning/training/distill_reasoning.py` | 推理轨迹蒸馏 | reasoning-from-scratch ch08：教师采样 → verifier 过滤 → 学生 SFT（只对 completion 计损） |
