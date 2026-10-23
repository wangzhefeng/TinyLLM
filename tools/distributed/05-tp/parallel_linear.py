# -*- coding: utf-8 -*-

# ***************************************************
# * File        : parallel_linear.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-10-24
# * Version     : 1.0.102402
# * Description : 05 单元：手写 Megatron 式张量并行（TP）原语。
# *               - ColumnParallelLinear：按输出维度切分权重，
# *                 前向 = 本地 matmul，无通信；可选 all-gather 输出。
# *               - RowParallelLinear：按输入维度切分权重，
# *                 前向 = 本地 matmul + all-reduce 求和。
# *               通信全部用纯 torch.distributed 集合通信实现
# *               （不依赖 DTensor，教学上每一步通信都显式可见）；
# *               world_size=1 时退化为普通 nn.Linear。
# *               process_group 参数（默认 None = 全局进程组）让通信
# *               限定在指定子组内：07-compose 用它把 TP 通信约束到
# *               节点内 TP 子组，与 DP 维度正交组合。默认行为不变，
# *               05-tp 全部用法不受影响。
# * Link        : https://arxiv.org/abs/1909.08053 (Megatron-LM)
# * Requirement : torch >= 2.5；gloo（CPU）或 nccl（CUDA）后端
# ***************************************************

__all__ = [
    "ColumnParallelLinear",
    "RowParallelLinear",
]

# python libraries
import math

import torch
import torch.distributed as dist
import torch.nn as nn


def _world_size(group=None) -> int:
    """进程组 world size（group=None 为全局组；未初始化时为 1，退化为单进程）。"""
    return dist.get_world_size(group) if dist.is_available() and dist.is_initialized() else 1


def _rank(group=None) -> int:
    """进程组内 rank（group=None 为全局组；未初始化时为 0）。"""
    return dist.get_rank(group) if dist.is_available() and dist.is_initialized() else 0


# ---------------------------------------------------------------------------
# 通信原语（Megatron 论文中的 f / g 算子）
#
# Megatron-LM 的核心洞察：前向与反向的通信算子互为对偶——
#   * 前向恒等（identity）  <->  反向 all-reduce
#   * 前向 all-reduce       <->  反向恒等（identity）
# 直觉：Y = f(X) 若前向对 X 不做任何事，则每个 rank 只算出了 dL/dX 的一部分
# （来自本地权重分片的那部分），反向时必须把各 rank 的部分梯度 all-reduce
# 求和才能得到完整 dL/dX；反之，若前向已经 all-reduce 把各分片结果加总
# （每个 rank 都拿到完整的 Y），则 dL/d（本地分片输出）就等于完整的 dL/dY，
# 反向无需任何通信。
# ---------------------------------------------------------------------------


class _CopyToParallelRegion(torch.autograd.Function):
    """
    f 算子：前向恒等，反向 all-reduce。

    用在「输入被所有 rank 复制共享」的并行层入口（Column 并行的输入侧）：
    前向时各 rank 拿到同一份 X，各自与本地权重分片做 matmul；
    反向时 dL/dX = sum_r(dL/dX 的第 r 个分片贡献)，必须 all-reduce。
    group 指定通信发生的进程组（None = 全局组）。
    """

    @staticmethod
    def forward(ctx, x, group=None):
        ctx.group = group
        return x

    @staticmethod
    def backward(ctx, grad_output):
        if _world_size(ctx.group) > 1:
            # 对偶点 1：前向是 identity，反向补上 all-reduce。
            # grad_output 是 autograd 新产生的张量，原地 all-reduce 安全。
            dist.all_reduce(grad_output, group=ctx.group)
        return grad_output, None


class _ReduceFromParallelRegion(torch.autograd.Function):
    """
    g 算子：前向 all-reduce，反向恒等。

    用在「各 rank 只算出部分和」的并行层出口（Row 并行的输出侧）：
    前向把各 rank 的局部 matmul 结果 all-reduce 成完整输出；
    反向时 dL/d（本地部分和） = 完整的 dL/dY，直接透传即可。
    group 指定通信发生的进程组（None = 全局组）。
    """

    @staticmethod
    def forward(ctx, x, group=None):
        if _world_size(group) > 1:
            # 对偶点 2：前向 all-reduce 求和，反向下面的 backward 就是 identity。
            dist.all_reduce(x, group=group)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class _GatherFromParallelRegion(torch.autograd.Function):
    """
    all-gather 算子：前向沿最后一维拼接各 rank 的分片，反向取本 rank 切片。

    与 f/g 同源的对偶：前向 all-gather（每个 rank 都得到完整张量），
    反向只需把 dL/dY 中属于本 rank 的那一段切出来——恰好等价于
    「对本 rank 分片而言，上游梯度就是它对应切片的梯度」。
    group 指定通信发生的进程组（None = 全局组）。
    """

    @staticmethod
    def forward(ctx, x, group=None):
        world_size = _world_size(group)
        if world_size == 1:
            return x
        ctx.rank = _rank(group)
        ctx.world_size = world_size
        # gloo 的 all_gather 要求各 rank 张量形状一致（分片等长，已断言保证）
        tensor_list = [torch.empty_like(x) for _ in range(world_size)]
        dist.all_gather(tensor_list, x.contiguous(), group=group)
        return torch.cat(tensor_list, dim=-1)

    @staticmethod
    def backward(ctx, grad_output):
        if getattr(ctx, "world_size", 1) == 1:
            return grad_output, None
        # 反向 = split：只保留本 rank 负责的那一段
        chunk = grad_output.shape[-1] // ctx.world_size
        return grad_output.narrow(-1, ctx.rank * chunk, chunk).contiguous(), None


class ColumnParallelLinear(nn.Module):
    """
    列并行 Linear：Y = X @ A，把权重 A 按「输出维度」切到各 rank。

    完整权重 A = [A_0, A_1, ..., A_{p-1}]（按输出行切块），rank r 只持有
    A_r（形状 out_features/p x in_features），前向：

        Y_r = X @ A_r      （本地 matmul，无通信；X 在所有 rank 上相同）

    输出 Y_r 是完整 Y 沿最后一维的 1/p 切片。若后续算子（如 GELU、按 head
    分组的注意力）能直接在切片上计算，就不需要任何通信——这是 Megatron
    MLP/Attention 零额外通信的关键。gather_output=True 时再 all-gather
    拼回完整 Y（用于需要完整输出的场合，代价是一次 all-gather）。

    反向：X 的梯度被 f 算子（_CopyToParallelRegion）自动 all-reduce，
    权重分片的梯度由本地 autograd 正常算出。

    process_group：通信与切分所依据的进程组（None = 全局组）。
    组合并行（如 07-compose 的 TP×DP）时传入 TP 子组，即可让
    all-reduce / all-gather 只发生在该子组内。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = False,
        process_group=None,
    ):
        super().__init__()
        world_size = _world_size(process_group)
        rank = _rank(process_group)
        assert out_features % world_size == 0, (
            f"ColumnParallelLinear: out_features={out_features} 不能被 "
            f"world_size={world_size} 整除（每个 rank 需要等长的输出分片）"
        )
        self.in_features = in_features
        self.out_features = out_features
        self.out_features_per_rank = out_features // world_size
        self.gather_output = gather_output
        self.process_group = process_group
        self.world_size = world_size
        self.rank = rank

        # 本 rank 只持有输出维度的一个切片：(out/p, in)
        self.weight = nn.Parameter(torch.empty(self.out_features_per_rank, in_features))
        self.bias = (
            nn.Parameter(torch.empty(self.out_features_per_rank)) if bias else None
        )
        self.reset_parameters()

    def reset_parameters(self):
        # 与 nn.Linear 默认初始化保持一致的量级（按完整 in_features 归一）
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def load_full_weight(self, weight: torch.Tensor, bias: torch.Tensor | None = None):
        """从完整 (out, in) 权重中切出本 rank 的分片（用于确定性初始化/等价验证）。"""
        assert weight.shape == (self.out_features, self.in_features)
        start = self.rank * self.out_features_per_rank
        with torch.no_grad():
            self.weight.copy_(weight[start : start + self.out_features_per_rank, :])
            if self.bias is not None:
                assert bias is not None and bias.shape == (self.out_features,)
                self.bias.copy_(bias[start : start + self.out_features_per_rank])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # f 算子：前向恒等；反向时这里会把各 rank 对 X 的部分梯度 all-reduce
        x = _CopyToParallelRegion.apply(x, self.process_group)
        # 本地 matmul：Y_r = X @ A_r^T（+ b_r），全程无通信
        out = torch.nn.functional.linear(x, self.weight, self.bias)
        if self.gather_output:
            # 可选：all-gather 拼回完整输出（反向自动变为取本地切片）
            out = _GatherFromParallelRegion.apply(out, self.process_group)
        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"out_per_rank={self.out_features_per_rank}, "
            f"bias={self.bias is not None}, gather_output={self.gather_output}, "
            f"rank={self.rank}/{self.world_size}"
        )


class RowParallelLinear(nn.Module):
    """
    行并行 Linear：Y = X @ B，把权重 B 按「输入维度」切到各 rank。

    完整权重 B = [B_0; B_1; ...; B_{p-1}]（按输入列切块），同时输入 X 也
    已经按同一维度被上游切好（典型情形：上游是一个 ColumnParallelLinear，
    其输出天然就是沿最后一维的切片）。rank r 只持有 B_r
    （形状 out_features x in_features/p），前向：

        Y = sum_r(X_r @ B_r) = all_reduce( X_r @ B_r )

    即本地 matmul 之后做一次 all-reduce 求和——这就是 Row 并行唯一的
    通信点。bias 不加进局部结果（否则会被加 p 次），而是在 all-reduce
    之后由每个 rank 各自加一份完整 bias。

    反向：all-reduce 的反向是恒等（g 算子），各 rank 拿到的 dL/dY 直接
    用于本地权重分片的梯度计算；输入侧不需要 f 算子，因为 X_r 本身就是
    分片（它的梯度只需回传给上游的 Column 层，由上游的 f 算子汇总）。

    process_group：通信与切分所依据的进程组（None = 全局组），
    组合并行时传入 TP 子组。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        input_is_parallel: bool = True,
        process_group=None,
    ):
        super().__init__()
        world_size = _world_size(process_group)
        rank = _rank(process_group)
        assert in_features % world_size == 0, (
            f"RowParallelLinear: in_features={in_features} 不能被 "
            f"world_size={world_size} 整除（每个 rank 需要等长的输入分片）"
        )
        self.in_features = in_features
        self.out_features = out_features
        self.in_features_per_rank = in_features // world_size
        # input_is_parallel=True 表示上游已经把输入沿最后一维切好（标准接法）；
        # False 表示输入是完整张量，本层先切出本 rank 的一段再计算。
        self.input_is_parallel = input_is_parallel
        self.process_group = process_group
        self.world_size = world_size
        self.rank = rank

        # 本 rank 只持有输入维度的一个切片：(out, in/p)
        self.weight = nn.Parameter(torch.empty(out_features, self.in_features_per_rank))
        # bias 是完整的 (out,)，all-reduce 之后再加（见 forward 注释）
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def load_full_weight(self, weight: torch.Tensor, bias: torch.Tensor | None = None):
        """从完整 (out, in) 权重中切出本 rank 的分片（沿输入维度/列切）。"""
        assert weight.shape == (self.out_features, self.in_features)
        start = self.rank * self.in_features_per_rank
        with torch.no_grad():
            self.weight.copy_(weight[:, start : start + self.in_features_per_rank])
            if self.bias is not None:
                assert bias is not None and bias.shape == (self.out_features,)
                self.bias.copy_(bias)  # bias 完整复制，见 forward 注释

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.input_is_parallel:
            # 输入是完整张量：先切出本 rank 负责的一段
            start = self.rank * self.in_features_per_rank
            x = x.narrow(-1, start, self.in_features_per_rank)
        # 本地部分和：Y_r = X_r @ B_r^T（不加 bias，否则 all-reduce 后会被加 p 次）
        out = torch.nn.functional.linear(x, self.weight)
        # g 算子：前向 all-reduce 把各 rank 的部分和加成完整 Y；反向为恒等
        out = _ReduceFromParallelRegion.apply(out, self.process_group)
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"in_per_rank={self.in_features_per_rank}, "
            f"bias={self.bias is not None}, "
            f"input_is_parallel={self.input_is_parallel}, "
            f"rank={self.rank}/{self.world_size}"
        )




# 测试代码 main 函数
def main():
    """
    单进程自检（world_size=1 时两种并行层都退化为普通 Linear）：
    构造完整 Linear 与并行层，喂同样输入，检查输出与梯度一致。
    """
    torch.manual_seed(0)
    batch, in_f, out_f = 4, 8, 16

    full = nn.Linear(in_f, out_f, bias=True)
    x = torch.randn(batch, in_f)

    # Column 并行 + gather：应完全还原完整 Linear
    col = ColumnParallelLinear(in_f, out_f, bias=True, gather_output=True)
    col.load_full_weight(full.weight, full.bias)
    assert torch.allclose(col(x), full(x), atol=1e-6)

    # Column -> Row 串联（经典 Megatron 两层结构）：应还原 W2 @ W1
    full2 = nn.Linear(out_f, in_f, bias=True)
    col2 = ColumnParallelLinear(in_f, out_f, bias=True, gather_output=False)
    row2 = RowParallelLinear(out_f, in_f, bias=True)
    col2.load_full_weight(full.weight, full.bias)
    row2.load_full_weight(full2.weight, full2.bias)
    ref = full2(full(x))
    got = row2(col2(x))
    assert torch.allclose(got, ref, atol=1e-6)

    # 反向：并行结构的梯度拼起来应等于完整模型的梯度
    ref.sum().backward()
    got.sum().backward()
    assert torch.allclose(col2.weight.grad, full.weight.grad, atol=1e-6)
    assert torch.allclose(row2.weight.grad, full2.weight.grad, atol=1e-6)
    print(f"single-process fallback OK (world_size={_world_size()})")


if __name__ == "__main__":
    main()
