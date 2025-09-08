export CUDA_VISIBLE_DEVICES=0
export LOG_NAME=gpt2_124M

tk_model_name=gpt2_124M
lm_model_name=gpt2_124M

torchrun --nproc_per_node=4 ./exp/exp_pretrain_gpt2_124M_DDP.py \
    --task_name tiny_gpt2_124M_pretrain \
    --des 'Tiny GPT2-124M Pretrain' \
    --is_train 1 \
    --is_test 0 \
    --is_inference 0 \
    --data_path ./dataset/pretrain/gpt \
    --data_file the-verdict.txt \
    --data_name the-verdict \
    --tokenizer_model $tk_model_name \
    --model_name $lm_model_name \
    --train_ratio 0.95 \
    --vocab_size 50257 \
    --context_length 1024 \
    --embed_dim 768 \
    --n_heads 12 \
    --n_layers 12 \
    --dropout 0.1 \
    --qkv_bias \
    --dtype float32 \
    --use_amp 0 \
    --max_new_tokens 50 \
    --seed 42 \
    --itrs 1 \
    --train_epochs 30 \
    --batch_size 2 \
    --learning_rate 5e-4 \
    --initial_lr 3e-5 \
    --min_lr 1e-6 \
    --weight_decay 0.1 \
    --lradj type1 \
    --patience 14 \
    --checkpoints ./saved_results/pretrained_models/ \
    --test_results ./saved_results/test_results/ \
    --num_workers 0 \
    --use_gpu 1 \
    --gpu_type cuda \
    --use_multi_gpu 0 \
    --devices 0,1,2,3,4,5,6,7
