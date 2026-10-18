# BERT 实验记录

> 所属：[TinyLLM Docs](../README.md) › [Models](index.md) › [BERT](bert.md)

本文记录从零 BERT 的关键对照实验，数据均为本机（CPU/MPS）单次运行结果，绝对数值仅供定性参考。

## 实验 1：模型实现修正（v1 → v2）

`models/bert/model.py` 对照标准 BERT 的修正（多头注意力切分维度 + scaling + padding mask、Post-LN、GELU、MLM 头 transform + 权重共享、N(0,0.02) 初始化）。

同配置对比（10k 句子对、10 epochs、hidden 64、batch 32）：

| 指标 | v1 修正前 | v2 修正后 |
|---|---|---|
| dev loss（最终） | 6.81 | **6.66** |
| acc_mask（最终） | 7.25% | **7.66%** |
| acc_next | 10 轮始终 50.0%（没学到） | 50% → **54.5%** |

要点：v1 没有 padding mask，[CLS] 注意力被约 110 个 [PAD] 淹没，NSP 永远学不出来；v2 补上 mask 后 NSP 脱离随机基线。

## 实验 2：预训练 → 微调三模式对照

下游任务：sentiment 中文评论二分类（train 16883 / valid 2111 / test 2111）。
backbone：THUCNews 20k 句子对预训练 10 epochs（hidden 128，1.06M 参数，acc_mask 19.3%）。

| 模式 | 8k 样本 / 3 epochs | 500 样本 / 5 epochs |
|---|---|---|
| pretrained（全量微调） | test 86.03% | test 59.92% |
| scratch（随机初始化） | test **86.69%** | test **66.51%** |
| pretrained + frozen（只训分类头） | test 55.76% | test 49.22% ≈ 随机 |

结论：此规模下预训练无收益。任务信号高度词汇化（好/差/喜欢），1M 参数的 char 级小模型直接从任务数据学就够了；frozen 结果说明 NSP 训练出的 [CLS] 表示对情感没有线性可分性。

## 实验 3：DAPT（领域自适应预训练）

流程：THUCNews checkpoint → 用 sentiment train 文本构造 34k NSP 句子对继续 MLM 5 epochs（acc_mask 19.3% → 28.5%）→ 重跑实验 2。

| 模式 | 8k / 3ep（DAPT 前→后） | 500 / 5ep（DAPT 前→后） |
|---|---|---|
| pretrained | 86.03% → 86.22% | 59.92% → 60.59% |
| frozen | 55.76% → 54.19% | 49.22% → 51.26% |
| scratch（不变） | 86.69% | **66.51%** |

附加实验（500 样本、lr 5e-5）：pretrained 53.20% vs scratch 49.83%——只有双方都被过小学习率「削弱」时预训练才显现优势。

结论：DAPT 方向正确（领域内 acc_mask +9pp）但收益被规模吃掉。文献中 DAPT 的增益建立在 BERT-base（110M 参数、亿级词预训练）之上；1M 参数 + 几万条语料的表示太弱，迁移价值有限。**迁移学习的收益是规模依赖的。**

## 待完成项

学习闭环（数据处理 → 预训练 → 推理 → 微调）已完备；相对完整 BERT 仍缺（按优先级）：

| 优先级 | 待办 | 说明 |
|---|---|---|
| P0 | 扩大 backbone 复验迁移结论 | hidden 256 + 4 层 + 全量 THUCNews（126 万句对）预训练，复验实验 2/3（预训练/DAPT 收益是规模依赖的） |
| P0 | 接入 WordPiece 分词对照 char 级 | `pre_training/experiments/bert/tokenizer.py` 已有 WordPiece 训练脚本，未接入主链路；BERT 原生分词方式 |
| P1 | 高效注意力 | 手写 softmax(QK^T)V → `F.scaled_dot_product_attention`（FlashAttention 路径） |
| P1 | 预训练评估指标 | 补困惑度（ppl）等标准指标，替代/补充 acc_mask |
| P1 | BERT 配置入 `config/` | 当前为 argparse 拼装，GPT-2/Llama/Qwen3 均有配置目录 |
| P2 | 更多下游任务形态 | NER（序列标注）、抽取式 QA、句向量/检索，当前仅二分类 |
| P2 | 分布式训练 | GPT-2 有 DP/DDP 脚本（`pre_training/scripts/`），BERT 未接入 |
| P2 | 模型导出/部署 | ONNX 导出、batched inference、服务化均未覆盖 |

## 复现命令

```bash
# 预训练 backbone
python run_bert.py --max_samples 20000 --train_epochs 10
# DAPT 数据 + 继续预训练
python pre_training/experiments/bert/dapt_data.py
python run_bert.py --data_dir ./dataset/finetuning/sentiment --task_file task2_sentiment.csv \
  --vocab_file <abs_path>/dataset/finetuning/THUCNews/index_2_word.txt \
  --resume ./checkpoints/bert/bert_pretrain_best.pt --train_epochs 5 --checkpoints ./checkpoints/bert_dapt/
# 三模式微调对照
python post_training/sft/bert_sentiment/run.py --init_mode pretrained --max_samples 8000 --train_epochs 3
python post_training/sft/bert_sentiment/run.py --init_mode scratch   --max_samples 8000 --train_epochs 3
python post_training/sft/bert_sentiment/run.py --init_mode pretrained --freeze_backbone --max_samples 8000 --train_epochs 3
```
