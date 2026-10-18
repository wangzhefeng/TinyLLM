
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=8 run_train.py
    --config-file scripts/nanotron_tiny_llama/config_tiny_llama.yaml
