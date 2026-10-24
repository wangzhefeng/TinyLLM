#!/bin/bash

# *********************************************
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2021.01.01
# * Version     : 1.0.0
# * Description : description
# * Link        : link
# **********************************************

export CUDA_VISIBLE_DEVICES=0,1,2,3
export LOG_NAME=gpt2_124M_DP

tk_model_name=tiktoken_gpt2_bpe
lm_model_name=gpt2_124M

# DP 为单进程多卡，直接调根入口（exp_pretrain_gpt2_124M.py 没有 __main__ 参数解析）
python -u run_gpt2_124M.py \
    --task_name gpt2_124M_pretrain_DP \
    --des 'GPT2-124M Pretrain using DP' \
    --is_train 1 \
    --is_test 1 \
    --is_inference 0 \
    --data_source local \
    --data_url "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt" \
    --data_path ./dataset/pretraining \
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
    --dtype bfloat16 \
    --use_amp 0 \
    --learning_rate 5e-4 \
    --initial_lr 3e-5 \
    --min_lr 1e-6 \
    --weight_decay 0.1 \
    --lradj type1 \
    --lr_schedule warmup_cosine \
    --warmup_ratio 0.2 \
    --grad_clip 1.0 \
    --grad_accum_steps 1 \
    --pad_vocab_to_multiple_of 64 \
    --save_every_steps 0 \
    --resume "" \
    --seed 42 \
    --itrs 1 \
    --train_epochs 30 \
    --patience 14 \
    --checkpoints ./results/pretrained_models/ \
    --test_results ./results/test_results/ \
    --max_new_tokens 50 \
    --use_gpu 1 \
    --gpu_type cuda \
    --use_dp 1 \
    --use_ddp 0
