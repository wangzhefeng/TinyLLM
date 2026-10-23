# 02-ddp/nn_style：mp.spawn 启动（单机）
# 进程数由脚本自动决定（CUDA 可用时取 GPU 数，否则取 2 用于 CPU 冒烟），也可用 --nprocs 显式指定。
# 如需限制可见 GPU：CUDA_VISIBLE_DEVICES=0,1 python -u tools/distributed/02-ddp/nn_style/main.py
python -u tools/distributed/02-ddp/nn_style/main.py
