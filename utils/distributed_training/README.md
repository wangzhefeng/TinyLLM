<details><summary>目录</summary><p>

- [大模型分布式并行训练](#大模型分布式并行训练)
    - [流水线并行](#流水线并行)
    - [数据并行](#数据并行)
        - [数据并行(DP)](#数据并行dp)
        - [分布式数据并行(DDP)](#分布式数据并行ddp)
    - [张量并行](#张量并行)
    - [数据并行+张量并行](#数据并行张量并行)
</p></details><p></p>


# 大模型分布式并行训练

## 流水线并行



## 数据并行

### 数据并行(DP)

建议使用 `torch.nn.parallel.DistributedDataParallel` 而不是 `torch.nn.DataParallel` 进行多 GPU 训练，即使只有一个节点。

`DistributedDataParallel` 和 `DataParallel` 的区别是：`DistributedDataParallel` 在多进程(multiprocessing)中为每个 GPU 创建一个进程，而 `DataParallel` 使用多线程(multithreading)。通过使用多进程，每个 GPU 都有自己的专用进程，这避免了 Python 解释器 GIL 带来的性能开销。


* [DataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.DataParallel.html)
* [使用 `nn.parallel.DistributedDataParallel` 而不是 `multiprocessing` 或 `nn.DataParallel`](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-nn-ddp-instead)

### 分布式数据并行(DDP)

* [Distributed Data Parallel](https://docs.pytorch.org/docs/stable/notes/ddp.html#ddp)


## 张量并行



## 数据并行+张量并行



