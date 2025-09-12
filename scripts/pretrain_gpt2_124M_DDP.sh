#!/bin/bash

# *********************************************
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2021.01.01
# * Version     : 1.0.0
# * Description : description
# * Link        : link
# **********************************************

# ------------------------------------------------------------
# nproc_per_node: 控制启动多少个进程，每个进程负责一张卡, nproc_per_node 应 ≤ CUDA_VISIBLE_DEVICES 中的设备数量
# CUDA_VISIBLE_DEVICES: CUDA_VISIBLE_DEVICES 是一个底层系统环境变量，用于限定当前进程可以使用的 GPU 物理设备，
#                       并允许自定义设备编号映射。在 PyTorch 分布式训练中，它和 LOCAL_RANK + --nproc_per_node 配合，
#                       实现精确的 GPU 资源隔离与分配。
# LOCAL_RANK: 当前进程在本节点内的编号（0 ~ nproc_per_node-1），用于映射到 CUDA_VISIBLE_DEVICES 中的设备, 
#             每个进程使用 LOCAL_RANK 作为索引访问 CUDA_VISIBLE_DEVICES 列表中的设备
# ------------------------------------------------------------

# 设置使用的 GPU 物理编号（例如只用第 2、3、6、7 张卡）
export CUDA_VISIBLE_DEVICES=0,1,2,3
# 设置日志名称
export LOG_NAME=gpt2_124M_DDP

# tokenizer 模型名称
tk_model_name=gpt2_124M
# LM 模型名称
lm_model_name=gpt2_124M

# 启动 4 个进程，每个进程绑定一个 GPU
torchrun \
    --nproc_per_node=4 \
    --master_addr=localhost \
    --master_port=12355 \
    ./exp/exp_pretrain_gpt2_124M.py \
    --task_name tiny_gpt2_124M_pretrain_DDP \
    --des 'Tiny GPT2-124M Pretrain using DDP' \
    --is_train 1 \
    --is_test 1 \
    --is_inference 0 \
    --data_source local \
    --url "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt" \
    --data_path ./dataset/pretrain/gpt \
    --data_file the-verdict.txt \
    --data_name the-verdict \
    --train_ratio 0.95 \
    --batch_size 2 \
    --num_workers 0 \
    --tokenizer_model $tk_model_name \
    --vocab_size 50257 \
    --model_name $lm_model_name \
    --context_length 1024 \
    --embed_dim 768 \
    --n_heads 12 \
    --n_layers 12 \
    --dropout 0.1 \
    --qkv_bias \
    --dtype float32 \
    --use_amp 0 \
    --learning_rate 5e-4 \
    --initial_lr 3e-5 \
    --min_lr 1e-6 \
    --weight_decay 0.1 \
    --lradj type1 \
    --seed 42 \
    --itrs 1 \
    --train_epochs 30 \
    --patience 14 \
    --checkpoints ./saved_results/pretrained_models/ \
    --test_results ./saved_results/test_results/ \
    --max_new_tokens 50 \
    --use_gpu 1 \
    --gpu_type cuda \
    --use_dp 0 \
    --use_ddp 1
