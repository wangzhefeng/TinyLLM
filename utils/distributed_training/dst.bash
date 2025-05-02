CUDA_VISIBLE_DEVICES=0

python -m torch.distributed.launch \
    --nproc_per_node=2 \
    distributed_sampler_training.py
