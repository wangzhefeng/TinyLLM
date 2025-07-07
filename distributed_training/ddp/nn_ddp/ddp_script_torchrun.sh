CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_TRAINERS=8

# gpu 2
torchrun --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_TRAINERS \
    main.py


# gpu all(8)
# torchrun \
#     --standalone \
#     --nnodes=1 \
#     --nproc_per_node=$(nvidia-smi -L | wc -l) \
#     main.py
