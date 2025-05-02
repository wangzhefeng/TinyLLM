CUDA_VISIBLE_DEVICES=0,1

# method 1
python DDP-script.py

# method 2
torchrun --nproc_per_node=2 DDP-script-torchrun.py
