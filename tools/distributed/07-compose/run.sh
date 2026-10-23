# 07-compose：组合并行（TP x DP）示例（从仓库根目录运行，路径与 cwd 无关）
# CPU + gloo 即可运行。NUM_TRAINERS=4 组织成 2x2 网格（TP 组内 2 进程做
# 张量并行，DP 组间 2 组做数据并行）；dim=16 / n_heads=4 / FFN hidden=64
# 均可被 tp_size=2 整除，改 --tp-size 需保证整除且 tp*dp == world_size。

NUM_TRAINERS=4

# 1) 数值等价验证（硬性自检：2TPx2DP 组合 vs 单进程参考，
#    前向 logits、loss 序列、训练后复制参数逐一断言）
.venv/bin/python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_TRAINERS \
    tools/distributed/07-compose/verify_equivalence.py --tp-size 2 --dp-size 2

# 2) TP x DP 组合训练示例（网格拓扑 + 每 rank 参数量 + loss 下降）
.venv/bin/python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_TRAINERS \
    tools/distributed/07-compose/main.py --tp-size 2 --dp-size 2

# 3) 2 进程退化解（可选）：1TPx2DP（纯数据并行）或 2TPx1DP（纯张量并行）
# .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 \
#     tools/distributed/07-compose/verify_equivalence.py --tp-size 1 --dp-size 2
# .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 \
#     tools/distributed/07-compose/verify_equivalence.py --tp-size 2 --dp-size 1

# 4) 工业形态 FSDP2 + TP（CUDA-only，需要 GPU 机器，本机无 GPU 会明确报错）：
# .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 \
#     tools/distributed/07-compose/fsdp_tp.py --tp-size 2 --dp-size 2
