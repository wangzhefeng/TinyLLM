
torchrun --nproc_per_node=1 run_generate.py \
    --ckpt-path checkpoints/{checkpoint_number}/ \
    --tp 1
    --pp 1
