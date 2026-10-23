# 04-fsdp：FSDP2 + dcp_api checkpoint（torch.distributed.checkpoint.state_dict）
NUM_TRAINERS=2

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_TRAINERS \
    tools/distributed/04-fsdp/train.py \
    --dcp-api
