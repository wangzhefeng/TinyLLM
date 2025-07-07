# DeepSpeed 使用

## DeepSpeed Model

DeepSpeed 模型训练是通过 DeepSpeed 引擎完成的。
该引擎可以包装任何类型为 torch.nn.module 的任意模型，
并具有一套最小的 API 用于模型训练和检查点保存。

初始化 DeepSpeed 引擎：

```python
import deepspeed

model_engine, optimizer, _, _ = deepspeed_initialize(
    args = cmd_args,
    model = model,
    model_parameters = params
)
```

> 如果你已经设置了分布式环境，
> 你需要将：`torch.distributed.init_process_group()` 替换为：`deepspeed.init_distributed()`。
>
> 但如果不需要在 `deepspeed.initialize()` 之后设置分布式环境，则不必使用此函数，
> 因为 DeepSpeed 将在其 `initialize` 自动初始化分布式环境。无论如何，
> 如果已经设置好了 `torch.distributed.init_process_group`，需要将其移除。

### Training

一旦 DeepSpeed 引擎被初始化，它就可以使用三个简单的 API 来训练模型：前向传播（可调用对象）、
反向传播（ backward ）和权重更新（ step ）。

```python
for step, batch in enumerate(data_loader):
    # forward
    loss = model_engine(batch)
    # backward
    model_engine.backward(loss)
    # weight update
    model_engine.step()
```

在底层，DeepSpeed 自动执行分布式数据并行训练所需的操作，以混合精度和预定义的学习率调度器进行：

* Gradient Averaging，梯度平均
    - 在分布式数据并行训练中，`backward` 确保在训练一个 `train_batch_size` 后，
      梯度在数据并行过程中被平均。
* Loss Scaling，损失缩放
    - 在 FP16/混合精度训练中，DeepSpeed 引擎自动处理损失缩放，以避免梯度中的精度损失。
* Learning Rate Scheduler，学习率调度器
    - 当使用 DeepSpeed 的学习率调度器（在 `ds_config.json` 文件中指定）时，
      DeepSpeed 在每个训练步骤（当 `model_engine.step()` 执行时）调用调度器的 step() 方法。
    - 当不使用 DeepSpeed 的学习率调度器时：
        - 如果调度器需要在每个训练步骤执行，
          那么用户可以在初始化 DeepSpeed 引擎时将调度器传递给 `deepspeed.initialize`，
          并让 DeepSpeed 管理其更新或保存/恢复。
        - 如果计划表需要在其他任何间隔执行（例如训练轮次），则用户不应在初始化时将计划表传递给 DeepSpeed，
          而必须显式管理它。

### Model Checkpointing

保存和加载训练状态是通过 DeepSpeed 中的 `save_checkpoint` 和 `load_checkpoint` API 处理的，
该 API 需要两个参数来唯一标识一个检查点：

* `ckpt_dir`：检查点将保存的目录
* `ckpt_di`：一个唯一标识目录中检查点的标识符。在以下代码片段中，我们使用损失值作为检查点标识符。

DeepSpeed 可以自动保存和恢复模型、优化器和学习率调度器的状态，同时将这些细节隐藏起来。
然而，用户可能希望保存特定于给定模型训练的额外数据。为了支持这些项目，
`save_checkpoint` 接受客户端状态字典 `client_sd` 进行保存。
这些项目可以作为返回参数从 `load_checkpoint` 中检索。
在下述示例中，`step` 值被存储为 `client_sd` 的一部分。

```python
# load checkpoint
_, client_sd = model_engine.load_checkpoint(args.load_dir, args.ckpt_id)
step = client_sd['step']

# advance data loader to ckpt step
dataloader_to_step(data_loader, step + 1)

for step, batch in enumerate(data_loader):
    # forward() method
    loss = model_engine(batch)

    # runs backpropagation
    model_engine.backward(loss)

    # weight update
    model_engine.step()

    # save checkpoint
    if step % args.save_interval:
        client_sd['step'] = step
        ckpt_id = loss.item()
        model_engine.save_checkpoint(args.save_dir, ckpt_id, client_sd = client_sd)
```

## DeepSpeed 配置

DeepSpeed 的功能可以通过一个 config JSON 文件来启用、禁用或配置，
该文件应指定为 `args.deepspeed_config`。下面是一个示例配置文件。

```json
// ds_config.json
{
    "train_batch_size": 8,
    "gradient_accumulation_steps": 1,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 0.00015
        }
    },
    "fp16": {
        "enabled": true
    },
    "zero_optimization": true
}
```

## 启动 DeepSpeed 训练

DeepSpeed 安装入口 `deepspeed` 来启动分布式训练。我们通过以下假设来展示 DeepSpeed 的一个使用示例：

1. 已经将 DeepSpeed 集成到你的模型中
2. `client_entry.py` 是你的模型的入口脚本
3. `client args` 是 `argparse` 命令行参数
4. `ds_config.json` 是 DeepSpeed 的配置文件

## 资源配置

### 资源配置-多节点

#### hostfile

DeepSpeed 使用与 OpenMPI 和 Horovod 兼容的 `hostfile` 配置多节点计算资源。
`hostfile` 是一个主机名（或 SSH 别名）列表，这些是可以通过无密码 SSH 访问的机器，
以及槽位数量(slot counts)，这些指定了系统上可用的 GPU 数量。

例如：下面的 `hostfile` 指定了名为 `worker-1` 和 `worker-2` 的两台机器，每台机器都有四个 GPU 用于训练。

```
worker-1 slots=4
worker-2 slots=4
```

Hostfiles 通过 `--hostfile` 命令行选项指定。如果没有指定 `hostfile`，
DeepSpeed 会搜索 `/job/hostfile`。如果没有指定或找到 `hostfile`，
DeepSpeed 会查询本地机器上的 GPU 数量，以发现可用的本地插槽数量。

以下命令将在 `myhostfile` 中指定的所有可用节点和 GPU 上启动一个 PyTorch 训练作业：

```bash
$ deepspeed --hostfile=myhostfile \
    <client_entry.py> <client args> \
    --deepspeed --deepspeed_config ds_config.json
```

#### num_nodes 和 num_gpus

或者，DeepSpeed 允许你将模型的分布式训练限制在可用的节点和 GPU 的子集上。
这一功能通过两个命令行参数 `--num_nodes` 和 `--num_gpus` 启用。
例如，可以使用以下命令将分布式训练限制在仅使用两个节点：

```bash
$ deepspeed --num_nodes=2 --num_gpus 8 \
    <client_entry.py> <client args> \
    --deepspeed --deepspeed_config ds_config.json
```

#### include 和 exclude

可以使用 `--include` 和 `--exclude` 标志来包含或排除特定资源。
例如，要在节点 `worker-2` 上使用除 GPU `0` 以外的所有可用资源，
并在 `worker-3` 上使用 GPU `0` 和 GPU `1`：

```bash
$ deepspeed --exclude="worker-2:0@worker-3:0,1" \
    <client_entry.py> <client args> \
    --deepspeed --deepspeed_config ds_config.json
```

同样地，你也可以在 `worker-2` 上仅使用 GPU `0` 和 `1`：

```bash
$ deepspeed --include="worker-2:0,1" \
    <client_entry.py> <client args> \
    --deepspeed --deepspeed_config ds_config.json 
```

#### 不使用无密码 SSH 启动

DeepSpeed 现在支持在不使用无密码 SSH 的情况下启动训练作业。
这种模式在 Kubernetes 等云环境中特别有用，这些环境允许灵活的容器编排，
而使用无密码 SSH 设置 leader-worker 架构会增加不必要的复杂性。

要使用此模式，您需要在所有节点上分别运行 DeepSpeed 命令。命令应按以下结构运行：

```bash
deepspeed --hostfile=myhostfile \
    --no_ssh \
    --node_rank=<n> \
    --master_addr=<addr> --master_port=<port> \
    <client_entry.py> <client args> \
    --deepspeed --deepspeed_config ds_config.json
```

每个节点必须使用唯一的 `node_rank` 启动，并且所有节点都需要提供领导节点的地址和端口（`rank 0`）。
这种模式使启动器表现得类似于 PyTorch 文档中描述的 `torchrun` 启动器。

#### 多节点环境变量

在跨多个节点进行训练时，我们发现支持传播用户定义的环境变量很有用。
默认情况下，DeepSpeed 会传播所有已设置的 NCCL 和 PYTHON 相关的环境变量。
如果您想传播额外的变量，可以在名为 `.deepspeed_env` 的点文件中指定它们，
该文件包含用换行符分隔的 `VAR=VAL` 条目列表。
DeepSpeed 启动器将检查您正在执行的本地路径以及您的家目录（`~/`）。
如果您想覆盖此文件的默认名称或路径并用自己的名称指定，可以使用环境变量 `DS_ENV_FILE`。
这主要适用于您启动多个作业，而所有作业都需要不同的变量的情况。

作为一个具体的例子，某些集群需要在训练之前设置特殊的 `NCCL` 变量。
用户只需将这些变量添加到其主目录中的一个 `.deepspeed_env` 文件中，该文件看起来像这样：

```bash
NCCL_IB_DISABLE=1
NCCL_SOCKET_IFNAME=eth0
```

DeepSpeed 将确保在训练作业中，在每个节点上启动每个进程时都设置这些环境变量。

### 资源配置-单节点

如果我们只在一个节点上运行（该节点有一个或多个 GPU），DeepSpeed 就不需要像上面描述的那样使用主机文件。
如果未检测到或未传入主机文件，DeepSpeed 将查询本地机器上的 GPU 数量，以发现可用的插槽数量。 
`--include` 和 `--exclude` 参数按正常方式工作，但用户应将 `localhost` 指定为主机名。

此外，`CUDA_VISIBLE_DEVICES` 可以与 `deepspeed` 一起使用，以控制在一个节点上应使用哪些设备。
因此，以下任一方式都可以用于仅在当前节点的设备 `0` 和 `1` 上启动：

```bash
$ deepspeed --include localhost:0,1 ...
```

```bash
$ CUDA_VISIBLE_DEVICES=0,1 deepspeed ...
```


# 参考

* [deepspeed.ai](https://www.deepspeed.ai/)
* [deepspeedai/DeepSpeed](https://github.com/deepspeedai/DeepSpeed)
* [deepspeed 知乎](https://www.zhihu.com/people/deepspeed/posts)
