# 04-fsdp：FSDP2 + 混合精度（bfloat16 参数 / float32 梯度规约）。
# 注意：04-fsdp 为 CUDA-only（fully_shard 依赖 CUDA stream），无 GPU 机器会在入口明确报错。
NUM_TRAINERS=2

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_TRAINERS \
    tools/distributed/04-fsdp/train.py \
    --mixed-precision
