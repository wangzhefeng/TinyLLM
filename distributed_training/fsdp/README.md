# FSDP2 原理

在 DistributedDataParallel (DDP)训练中，每个 rank 拥有一个模型副本并处理一批数据，
最后使用 all-reduce 来跨 rank 同步梯度。

与 DDP 相比，FSDP 通过分片模型参数、梯度和优化器状态来减少 GPU 内存占用。
这使得训练无法在单个 GPU 上运行的模型成为可能。如下图所示，

![img](images/fsdp_workflow.png)

* 在正向和反向传播计算之外，参数(parameters)是完全分片的(fully sharded)
* 在正向和反向传播之前，分片参数(sharded parameters)会全部聚合(all-gathered)为未分片参数(unsharded parameters)
* 在反向传播过程中，局部的未分片梯度(unsharded gradients)会通过 reduce-scatter 聚合为分片梯度(sharded gradients)
* 优化器(optimizer)使用分片梯度(sharded parameters)更新分片参数(sharded gradients)，
  从而产生分片优化器状态(sharded optimizer stats)

FSDP 可以被视为将 DDP 的全归约(all-reduce)操作分解为：归约散播(reduce-scater)和全收集(all-gather)操作：

![img](images/fsdp_sharding.png)

# FSDP2 使用

## Model Initialization

* **在子模块上应用 `fully_shard`**：与 DDP 不同，我们不仅应该在根模型上应用 fully_shard，还应该在子模块上应用。

```bash
$ torchrun --nproc_per_node 2 train.py
```

```python
from torch.distributed.fsdp import fullly_shard, FSDPModule

# model
model = Transformer()

# 首先对每一层应用 fully_shard，然后对根模型应用
for layer in model.layers:
    fullly_shard(layer)
fully_shard(layer)

assert isinstance(model, Transformer)
assert isinstance(model, FSDPModule)
print(model)
```

* `model.parameters() as DTensor`：`fully_shard` 在不同 rank 之间分片参数，
  并将 `model.parameters()` 从普通的 `torch.Tensor` 转换为 `DTensor` 来表示分片参数。
  FSDP2 默认在 `dim-0` 上分片，因此 DTensor 的放置是 `Shard(dim=0)`。
  假设我们有 N 个 rank，并且分片前参数有 N 行。分片后，每个 rank 将拥有参数的 1 行。
  我们可以使用 `param.to_local()` 检查分片参数。

```python
from torch.distributed.tensor import DTensor

for param in model.parameters():
    assert isinstance(param, DTensor)
    assert param.placements == (Shard(0),)
    # inspect sharded parameters with param.to_local()

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
```

## Forward/Backward with Prefetching

## Mixed Precision

## Gradient Clipping and Optimizer with State Dicts with DTensor APIs


## State Dict with DCP APIs


# CODES

* [pytorch/examples/distributed/FSDP2/](https://github.com/pytorch/examples/tree/main/distributed/FSDP2)
