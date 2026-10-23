# 分布式训练工具

> 所属：[TinyLLM Docs](../README.md) › [Tools](index.md)

`tools/distributed/` 是分布式并行的**教学参考库**，不接入训练主链路
（主链路 DDP 由 `utils/ddp_utils.py` 承担）。单元按学习动线 `00 → 08` 组织：
单卡 → DP → DDP → ZeRO（手写）→ FSDP2（工业形态）→ TP → PP → 组合 → 成熟框架。

核心教学方法是**数值等价性验证**：03/05/06/07 都提供与单进程基线的对照且全部 PASS，
用实验证据说明“并行只改分布、不改语义”。

## 单元总览

| 单元 | 学什么 | 运行（仓库根目录） | 验证状态 | 需要 CUDA |
|---|---|---|---|---|
| `00-single-gpu/` | 单卡训练基线（等价验证的参照物） | `.venv/bin/python tools/distributed/00-single-gpu/main.py` | CPU 可跑（~1.4s） | 否 |
| `01-dp/` | `nn.DataParallel` 单机多卡 | `.venv/bin/python tools/distributed/01-dp/main.py` | CUDA-only，无卡优雅报错 | 是 |
| `02-ddp/` | DDP：`multigpu_spawn.py` / `multigpu_torchrun.py`（含 snapshot 续训）/ `multinode.py` / `nn_style/`（作者模块化版，走 `utils.ddp_utils`） | `bash tools/distributed/02-ddp/nn_style/ddp_script.sh` | CPU + gloo 可跑 | 否 |
| `03-zero/` | 手写 ZeRO stage 1/2/3（`zero1/2/3.py` + `ShardedAdam`），stage 0 为 DDP 等价基线 | `bash tools/distributed/03-zero/run_all.sh`（底层为 `train.py --stage {0,1,2,3}`） | 与单进程逐位一致，max\|Δ\|=0.0 | 否 |
| `04-fsdp/` | FSDP2 `fully_shard` + DCP checkpoint 双 API（`--dcp-api` / `--mixed-precision` / `--explicit-prefetch`） | `bash tools/distributed/04-fsdp/run.sh` | CUDA-only，入口有清晰报错 | 是 |
| `05-tp/` | 手写 Megatron 式 TP：Column/RowParallelLinear + f/g autograd 原语（支持 process_group 子组） | `bash tools/distributed/05-tp/run.sh` | 前向/训练 diff 2.4e-07，PASS | 否 |
| `06-pp/` | `torch.distributed.pipelining`：`--schedule {gpipe,1f1b}`、`--verify` | `bash tools/distributed/06-pp/run.sh --verify` | GPipe CPU 等价 PASS；1F1B 见“已知限制” | 否（1F1B 需要） |
| `07-compose/` | 组合并行：手写 TP×DP 二维网格（`--tp-size` / `--dp-size`）+ `fsdp_tp.py`（torchtitan 风格 FSDP2+TP） | `bash tools/distributed/07-compose/run.sh` | 4 进程 2×2 网格等价 PASS | 否（`fsdp_tp.py` 需要） |
| `08-frameworks/` | accelerate / DeepSpeed 对照手写版 | 见 [框架对照](distributed-frameworks.md) | accelerate 已实测；deepspeed 为守卫式示例 | 否 |

shell 脚本内部会自动 cd 到仓库根，与调用时的 cwd 无关；细节看各脚本头部注释。

## 等价性验证

- 统一方法：同一玩具数据集/模型（`_common/`）、同一 seed，单进程基线 vs 并行实现，
  逐步对照 loss 序列与最终参数。
- 独立验证入口：`03-zero/verify_equivalence.py`、`05-tp/verify_equivalence.py`、
  `07-compose/verify_equivalence.py`；`06-pp` 用 `--verify` 开关。
- 建议每个单元先跑验证再看实现。

## 共享约定

- `_common/`：统一玩具数据集、ToyMLP 与 toy Transformer、torchrun 启动辅助、指标输出。
- checkpoint 统一写入 `results/checkpoints/distributed/<unit>/`。
- 测试：`tests/unit/test_tools_distributed.py`（默认快速层 ~4s；`TINYLLM_SLOW_TESTS=1`
  启用 torchrun 双进程冒烟层 ~15s），见 [测试](../50-development/testing.md)。

## 已知限制

- `06-pp` 的 1F1B：torch 2.5.1 + gloo 下 `Schedule1F1B` 框架级崩溃，入口防御性报错；需 CUDA + NCCL。
- `04-fsdp` 与 `07-compose/fsdp_tp.py`：FSDP2 依赖 CUDA，torch 2.5.1 CPU 下 `fully_shard` 硬报错。
- `01-dp`：DataParallel 仅支持单机多卡 CUDA。
- accelerate：CPU 多进程要用 torchrun + `ACCELERATE_USE_CPU=True`；
  `accelerate launch --cpu` 不会拉起多进程（见框架对照页）。

## 相关

- [成熟框架对照与上游导读](distributed-frameworks.md)
- [后续路线图（09+：SP/CP、MoE/EP）](distributed-roadmap.md)
- [预训练训练循环](../10-pretraining/training.md)
- [LLMs-from-scratch 附录 A](../60-reference/llms-from-scratch/appendices.md)
- [Tools 索引](index.md)
