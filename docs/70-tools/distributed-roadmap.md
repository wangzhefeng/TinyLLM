# 分布式模块后续路线图（09+）

> 所属：[TinyLLM Docs](../README.md) › [Tools](index.md) › [分布式训练工具](distributed.md)

第一期（00–08）已覆盖缺失清单中的 TP、PP、组合并行、手写 ZeRO、成熟框架对照
（见 [分布式训练工具](distributed.md)）。本页记录剩余两个空白方向
——SP/CP（序列/上下文并行）与 MoE/EP（专家并行）——的扩展计划与优先级，
以及穿插进行的非新范式工作。

## 第一优先：09-sp 序列并行（Megatron-SP）

05-tp 的自然延伸，概念门槛最低，可完全沿用"手写 + 等价验证"范式在 CPU+gloo 闭环。

- **内容**：TP 中 LayerNorm/Dropout 沿 hidden 维是重复的，SP 把它们沿**序列维**切到
  各 rank，与 TP 组合后激活显存才完整除以 TP 度数。通信原语是 05-tp f/g 对偶的
  另两个组合：前向 all-gather ↔ 反向 reduce-scatter。
- **实现要点**：复用 `05-tp/parallel_linear.py` 的 autograd 原语骨架，补
  `_GatherFromSequenceRegion` / `_ReduceScatterToSequenceRegion`；扩展 `tp_model.py`
  为 TP+SP 版，注释讲清 SP 区域（LayerNorm/残差）与 TP 区域（Linear）的边界与通信衔接。
- **验证**：照 05-tp 的 `verify_equivalence.py` 范式与单进程基线对账。

## 第二优先：10-cp 上下文并行（Ring Attention）

长上下文训练的标配。与 SP 解决不同问题：SP 省激活，CP 让序列长度超过单卡容量。

- **内容**：简化版 Ring Attention——序列沿长度维切分，K/V 沿 ring 逐跳 send/recv，
  在线 softmax（flash-attention 式分块累计）保证数学等价。只支持因果 mask 的教学简化版。
- **实现要点**：用 `_common` 的 toy Transformer + gloo `batch_isend_irecv`
  （06-pp 已趟过 gloo 点对点通信的坑）。**先做最小 send/recv 复现实验确认 gloo
  点对点模式可用，再投入完整实现**（教训来自 06-pp 的 1F1B gloo 崩溃）。
- **验证**：CP=2 与单进程完整 attention 输出对账。
- 备选：09/10 可合并为一个 `09-long-context/` 单元，共同回答"序列变长怎么办"。

## 第三优先：11-moe-ep 专家并行

概念独立（不依赖 09/10），与 `models/` 里的 deepseek/qwen3 MoE 模型直接呼应。

- **内容**：toy MoE 层（top-k router + N 个专家 FFN）+ EP——专家切到各 rank，token
  经 all-to-all dispatch/combine 路由到对应专家再收回。文档中建立
  "教学实现 ↔ 项目模型"的映射。
- **关键前置实验**：先验证 gloo 对 `all_to_all_single` 的支持；不可用时退化为
  "手写 all-gather + 本地选择"的语义等价实现，注释说明生产环境用 NCCL all-to-all。
- **验证**：EP=2 与单进程 MoE 前向/训练对账，同现有范式。

## 穿插进行（非新范式）

1. **GPU 实测补验**（建议放在任何 09+ 之前）：`01-dp`、`04-fsdp`、
   `07-compose/fsdp_tp.py` 目前只有"优雅报错"被测过，训练路径未真跑。找多卡机器
   跑 `04-fsdp/run.sh`、`run_dcp_api.sh` 与 07 的 CUDA 版，把结果回填
   [distributed.md](distributed.md) 的验证状态列。
2. **性能对比页**：用 `tools/profiling/` 对各单元做标准化吞吐/显存测量
   （`_common/metrics.py` 已留 `peak_memory_mb`/`tokens_per_sec` 接口），结果沉淀为
   distributed.md 的对比表——让"每个范式省什么、代价是什么"从定性注释升级为定量数据。

## 不建议现在做

- **3D 组合（TP+PP+DP）**：07-compose 的 2D 已讲透通信组划分的核心概念，3D 是工程
  组合而非新概念，投入产出比低，指路 torchtitan（见
  [框架对照](distributed-frameworks.md)）。
- **activation checkpointing（激活重计算）**：单卡技术而非分布式模式，主题不纯；
  更适合作为 `utils/` 或训练主链路的独立特性。

## 新单元规范（沿用第一期）

- 复用 `_common/`；torchrun 入口；`--verify` 等价验证且必须实跑 PASS；
- 冒烟测试纳入 `tests/unit/test_tools_distributed.py`（`TINYLLM_SLOW_TESTS=1`）；
- 文档单页 ≤150 行，登记本目录索引与 [repo-map](../00-overview/repo-map.md)。

## 相关

- [分布式训练工具总览](distributed.md)
- [成熟框架对照与上游导读](distributed-frameworks.md)
