# 08-frameworks/accelerate_example：HF Accelerate 最小示例（从仓库根目录运行）
# 对照 02-ddp：训练代码见 main.py 注释，框架接管进程组/包装/梯度同步。
NUM_PROCESSES=2

# ---- CPU（本机无 GPU，已实测）----
# accelerate 1.5.x 的 `accelerate launch --cpu` 走 simple_launcher，只会拉起
# 单个进程（多进程 CPU 需要 MPI/--mpirun_hostfile）。因此 CPU 场景用 torchrun
# 拉起 N 个进程（与 02-ddp 相同），Accelerator 从 RANK/WORLD_SIZE 环境变量
# 自动接管（DistributedType.MULTI_CPU + gloo）；ACCELERATE_USE_CPU 固定走 CPU。
ACCELERATE_USE_CPU=True ACCELERATE_MIXED_PRECISION=no \
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_PROCESSES \
    tools/distributed/08-frameworks/accelerate_example/main.py

# ---- GPU 机器（未在本机验证）----
# 有 CUDA 时 accelerate launch 会走 multi_gpu_launcher（内部同样是 torchrun）：
# accelerate launch --num_processes $NUM_PROCESSES --mixed_precision no \
#     tools/distributed/08-frameworks/accelerate_example/main.py
