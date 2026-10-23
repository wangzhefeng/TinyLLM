# 分布式成熟框架对照与上游导读

> 所属：[TinyLLM Docs](../README.md) › [Tools](index.md) › [分布式训练工具](distributed.md)

本页说明 `tools/distributed/08-frameworks/`，并给出手写实现对应的上游参考导读
（上游代码不入库，只留链接）。定位：学完手写单元后，对照“工业界如何封装同一件事”。

## 08-frameworks/

### accelerate_example/（HF Accelerate）

- 对照单元：`02-ddp/`。训练循环几乎不变，`Accelerator` 接管进程组初始化、
  模型/优化器包装与梯度同步。
- 运行（CPU 已实测，torchrun 拉起 2 进程）：

```bash
ACCELERATE_USE_CPU=True torchrun --standalone --nproc_per_node=2 \
    tools/distributed/08-frameworks/accelerate_example/main.py
```

- 坑：`accelerate launch --cpu` 走 simple_launcher，只会拉起单个进程
  （CPU 多进程需要 MPI）；因此 CPU 场景改用 torchrun，`Accelerator`
  从 `RANK`/`WORLD_SIZE` 环境变量自动接管（MULTI_CPU + gloo）。

### deepspeed_zero/（DeepSpeed ZeRO）

- 对照单元：`03-zero/`（手写 ZeRO stage 1/2/3）。
- 守卫式示例：未安装 deepspeed 时优雅退出并给出安装指引，未在仓库环境中实测。
- 配置见 `tools/distributed/08-frameworks/deepspeed_zero/ds_config.json`。

## 手写实现的上游导读

- **Megatron-LM**：TP/PP 的原始出处。`05-tp/parallel_linear.py` 的
  Column/RowParallelLinear 与 f/g autograd 原语对应其 `megatron/core/tensor_parallel/`；
  GPipe/1F1B 调度见 `megatron/core/pipeline_parallel/`。
  上游：<https://github.com/NVIDIA/Megatron-LM>
- **torchtitan**：FSDP2 + TP 组合的官方参考实现。`07-compose/fsdp_tp.py` 的
  二维网格（`fully_shard` + TP 子组）按 torchtitan 风格组织。
  上游：<https://github.com/pytorch/torchtitan>

## 相关

- [分布式训练工具总览](distributed.md)
- [外部参考项目](../50-development/reference-projects.md)
