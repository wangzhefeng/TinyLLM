> 所属：[TinyLLM Docs](../README.md) › [Models](index.md)

<details><summary>目录</summary><p>

- [BERT 学习](#bert-学习)
  - [模型原理](#模型原理)
  - [模型结果](#模型结果)
  - [训练过程](#训练过程)
  - [应用场景](#应用场景)
- [问题](#问题)
  - [问题 1: BERT 与微调的关系](#问题-1-bert-与微调的关系)
  - [问题 2: BERT 与 Embedding 的关系](#问题-2-bert-与-embedding-的关系)
- [参考](#参考)
</p></details><p></p>

# BERT（独立研究与学习目录）

本目录专门用于 BERT 的学习、从零实现与下游实验，与 GPT/Llama/Qwen 等 decoder-only 模型分开维护。

## 目录

BERT 代码已按三域架构分布，模型本体只保留在 `models/bert/`：

```text
run_bert.py                              # 预训练入口（根目录）：训练循环 / 验证 / 早停 / checkpoint

models/bert/
└── model.py                             # 从零实现 BERT：Embedding / Encoder / Pooler / MLM+NSP

pre_training/data_provider/bert/         # 数据层（原根级 data_provider/bert/）
├── data_loader.py                       # BertDataset（input_ids / segment_ids / mask / label）
└── data_processor.py                    # 文本清洗、正负样本构造、词表构建

pre_training/experiments/bert/           # 预训练实验：corpus 加载 / WordPiece 训练 / 编码
post_training/sft/bert_sentiment/        # 下游情感分类（微调从零 BERT 的 checkpoint）
inference/examples/bert_fill_mask.py     # 推断示例：fill-mask（MLM）与 NSP 判定

tests/unit/
├── test_bert.py                         # 模型前向 / loss 反传
└── test_bert_training.py                # run_bert 训练流水线、checkpoint 保存/加载
```

## 从零 BERT 的结构

- `BertEmbedding`：token + position + segment 三种嵌入相加，LayerNorm + Dropout。
- `MultiHeadAttention`：按 hidden 维切分 head，`scores / sqrt(head_dim)`，padding mask（从 input_ids 推导），输出投影。
- `BertEncoder`：标准 Post-LN 结构，`LayerNorm(x + sublayer(x))`；FFN 激活为 GELU。
- `BertPooler`：取 `[CLS]` 表示。
- `MaskedLMHead`：transform（Dense+GELU+LayerNorm）+ decoder，decoder 与词嵌入**权重共享**。
- `Model`：MLM 头（`cls_mask`）+ NSP 头（`cls_next`），训练时返回 `loss = loss_mask + loss_next`。
- 参数初始化：BERT 标准 `N(0, 0.02)`（`initializer_range`）。权重共享后必须显式初始化，否则 nn.Embedding 默认的 N(0,1) 会让 MLM logits 方差过大、初始 loss 远高于 ln(vocab_size)。

## 快速验证

```python
from models.bert.model import Model

config = {
    "vocab_size": 30522,
    "hidden_size": 128,
    "feed_num": 512,
    "head_num": 4,
    "layer_num": 2,
}
model = Model(config)
```

`forward(batch_idx, batch_seg_idx, batch_mask_val=None, batch_label=None)`：

- 传入 `batch_mask_val` 与 `batch_label` 时返回 loss；
- 否则返回 `(MLM 预测, NSP 预测)`。

## 预训练入口（run_bert.py）

数据：`dataset/finetuning/THUCNews/`（`task2.csv` 句子对 + `index_2_word.txt` 词表，由 `data_processor.py` 生成）。

```bash
# 快速验证（2 千条样本、小模型、CPU 可跑）
python run_bert.py --max_samples 2000 --train_epochs 2 --hidden_size 64 --feed_num 128

# 完整训练（默认 hidden_size=128 / 2 层 / 4 头 / batch=32 / 10 epochs）
python run_bert.py
```

- 流程：加载词表与 `task2.csv` → 末尾 `--dev_size` 条作验证集 → AdamW + 线性 warmup/衰减 → 训练（MLM+NSP 联合 loss）→ 每 epoch 验证输出 `dev loss / acc_mask / acc_next`。
- 早停按 dev loss 保存最优 checkpoint 到 `--checkpoints`（默认 `./checkpoints/bert/`，gitignored），结束时另存 `bert_pretrain_final.pt`（含 config，可用于恢复模型）。
- `--resume <ckpt>` 可加载已有权重继续训练。

## 说明

- `post_training/sft/bert_sentiment/` 是下游情感分类微调：复用 `run_bert.py` 的 checkpoint 作 backbone 加分类头，数据为 `dataset/finetuning/sentiment/`（中文评论二分类）。支持三种对照模式：`--init_mode pretrained`（默认，验证预训练价值）、`--init_mode scratch`（随机初始化对照）、`--freeze_backbone`（冻结 backbone 只训分类头）。入口：`python post_training/sft/bert_sentiment/run.py --help`。
- `pre_training/experiments/bert/` 是语料加载 / WordPiece 训练 / 编码的实验脚本，导入已修复（`tokenizers`/`transformers`），`main()` 中的路径仍需按本机调整。
- 推断：`inference/examples/bert_fill_mask.py` 加载 `run_bert.py` 的 checkpoint 做 fill-mask 与 NSP，输入无需手动 padding（模型内部从 input_ids 推导 padding mask）。
- v2 修正记录（2025-10-18，对照标准 BERT）：多头注意力由「按序列维切分」改为按 hidden 维切分并补 scaling/padding mask/输出投影；`AddNorm` 改为标准残差+LayerNorm；FFN 由 ReLU 改为 GELU；MLM 头补 transform 并与词嵌入权重共享；增加 `N(0, 0.02)` 参数初始化。

---

# BERT 学习

BERT 模型（Bidirectional Encoder Representations from Transformers）是一种在自然语言处理（NLP）领域具有深远影响的模型。

BERT 的出现是基于深度学习在 NLP 领域的快速发展。在 BERT 之前，已经有许多预训练语言模型，如 ELMo 和 GPT，
它们展示了预训练模型在 NLP 任务中的强大性能。然而，这些模型通常基于单向的上下文信息，即只考虑文本中的前向或后向信息，
这限制了它们对文本的全局理解。BERT 旨在通过引入双向上下文信息来解决这一问题，从而更准确地表示文本中的语义信息。

## 模型原理

BERT 是一种基于 Transformer 架构的双向语言表示模型。它通过对海量文本数据的预训练，学习得到丰富的上下文表示，
然后在下游任务上进行微调，实现极高的性能。BERT 被开发为一个深层双向模型，
它能够从输入序列的最初层一直到输出层有效地捕捉到目标单词周围的全方位上下文信息。

## 模型结果

BERT 的结构基于 Transformer 编码器（Encoder）部分，完整的 BERT 模型包括一个词嵌入层、多层的 Transformer Encoder，以及用于特定任务的输出层（映射层）。

* Input：BERT使用三种嵌入向量的组合作为输入表示，包括 Token Embedding（词嵌入）、Segment Embedding（片段嵌入）和 Position Embedding（位置嵌入）。
  这三种嵌入相加，BERT 就能为每个输入词生成一个综合表示，这个表示同时蕴含了词本身的含义、它在句子中的角色（例如，属于哪个句子）以及它在句子中的位置信息。
* Transformer Encoder：BERT 的核心部分是由多个堆叠的 Transformer 编码器组成。
  每个编码器层都包含多头自注意力机制（Multi-Head Self-Attention）和前馈神经网络（Feed-Forward Neural Network），
  并通过层归一化和残差连接进行稳定和优化。

## 训练过程

BERT 的训练过程包括预训练和微调两个阶段。

* 预训练：在预训练阶段，BERT 使用大量的无标签语料进行训练，
  通过 **Masked Language Model(MLM)** 和 **Next Sentence Prediction(NSP)** 这两个任务来学习语言的表示。
  MLM 任务要求模型预测被 mask 掉的 token，而 NSP 任务则要求模型判断两个句子是否连续。
* 微调：在微调阶段，BERT 可以针对特定的任务进行训练，以适应不同的应用场景。通常，
  只需要在 Transformer 的输出上添加一个与任务相关的输出层，并进行少量的训练即可。

## 应用场景

BERT 模型广泛应用于各种自然语言处理任务，如文本分类、情感分析、命名实体识别、翻译和问答系统等。
由于 BERT 模型采用了 Transformer 架构和自注意力机制，可以有效地捕捉文本中的语义信息，因此在许多任务中表现出了出色的性能。

# 问题

## 问题 1: BERT 与微调的关系

## 问题 2: BERT 与 Embedding 的关系

# 实验
- [bert-experiments.md](bert-experiments.md)：v1→v2 修正对比、三模式微调对照、DAPT 结论。

# 参考

* [datawhalechina/hand-bert](https://github.com/datawhalechina/hand-bert)
* [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
* [李沐 Video](https://www.bilibili.com/video/BV1PL411M7eQ/?spm_id_from=333.337.search-card.all.click&vd_source=a7a5ff2f9d1f5a8f5e83d4a2d2ed8fb0)
