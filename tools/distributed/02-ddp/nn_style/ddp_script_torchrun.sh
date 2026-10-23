# 02-ddp/nn_style：torchrun 启动（单机，推荐）
# NUM_TRAINERS 为每节点进程数；多卡机器可设为 $(nvidia-smi -L | wc -l)。
# 如需限制可见 GPU，在 torchrun 同一行前置：CUDA_VISIBLE_DEVICES=0,1 torchrun ...
NUM_TRAINERS=2

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_TRAINERS \
    tools/distributed/02-ddp/nn_style/main_torchrun.py
