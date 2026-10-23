# 04-fsdp：FSDP2 + 显式前向/反向预取（参数名统一为 --explicit-prefetch）
NUM_TRAINERS=2

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_TRAINERS \
    tools/distributed/04-fsdp/train.py \
    --explicit-prefetch
