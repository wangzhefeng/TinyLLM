# 05-tp：Megatron 式张量并行（TP）训练示例（从仓库根目录运行，路径与 cwd 无关）
# CPU + gloo 即可运行；NUM_TRAINERS 为每节点进程数（即 TP 的 world_size）。
# dim=16 / n_heads=4 / FFN hidden=64 均可被 2 整除；改进程数需保证整除。

NUM_TRAINERS=2

# 1) 数值等价验证（硬性自检：TP vs 完整模型，前向 logits 与 loss 序列逐步一致）
.venv/bin/python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_TRAINERS \
    tools/distributed/05-tp/verify_equivalence.py

# 2) TP 训练示例（loss 下降 + 参数量切分对比表）
.venv/bin/python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_TRAINERS \
    tools/distributed/05-tp/main.py
