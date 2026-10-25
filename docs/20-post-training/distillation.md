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

# ch08 蒸馏工程链路（2026-10-24）

| 步骤 | 文件 | 说明 |
|---|---|---|
| 教师数据生成 | `post_training/reasoning/data_gen.py` | MATH-500 子集 → 教师生成 → verifier 过滤 → verified JSONL |
| 批量训练原子 | `post_training/reasoning/training/distill_reasoning.py` | `distill_sft_epoch`：右填充批量 completion-only CE |
| 训练入口 | `post_training/reasoning/run.py` | JSONL → 学生批量 SFT → checkpoint + metrics CSV |

参考上游：rasbt/reasoning-from-scratch `ch08/02_generate_distillation_data`、`ch08/04_train_with_distillation`。

## 数据生成

```bash
python -m post_training.reasoning.data_gen --backend local       # 教师模型本地采样
python -m post_training.reasoning.data_gen --backend ollama --model qwen3:30b-thinking
python -m post_training.reasoning.data_gen --backend openrouter  # 需 OPENROUTER_API_KEY
```

- 题目默认取 MATH-500 训练子集（`--train-size` 按 `--seed` 抽样，与 RLVR 入口同口径）。
- local 后端复用 `generate_reasoning_traces`（`--num-samples` 条/题 + verifier 过滤）。
- 记录字段：`{"problem", "prompt", "completion", "expected", "verified"}`；只落盘 verified，函数返回全部。
- 输出默认 `dataset/finetuning/math_distill/train.jsonl`；`average_field_lengths` 统计各字段 token 长度。

## 批量训练

- `distill_sft_epoch(dataset, student, optimizer, batch_size=4, pad_id=None, max_grad_norm=1.0)`；
  dataset 为 `(prompt_ids, completion_ids)` 列表，右填充组 batch，CE 只对 completion token
  （`reasoning_sft_loss` 的向量化版本），返回 epoch 平均 loss。
- 右填充 + target 掩码保证 pad 不影响 loss（见 `tests/unit/test_distillation.py` 的无 pad 比对）。

## 训练入口

```bash
python post_training/run.py --task distill_reasoning \
  --model qwen3-0.6b --data dataset/finetuning/math_distill/train.jsonl \
  --epochs 2 --batch-size 4 --lr 5e-5 --device cpu
```

产物：`--out`（默认 `results/checkpoints/distill/`，每 `--save-every` epoch 存一次）与
`--logs`（默认 `logs/distill/distill_metrics.csv`）。数据文件缺失时报错并提示先跑 data_gen。

## 相关

- 推断侧 HF 导出：[loaders](../30-inference/loaders.md)
- 数据路径：[datasets](../50-development/datasets.md)
- RLVR（ch06/ch07）：[rlhf](rlhf.md)
