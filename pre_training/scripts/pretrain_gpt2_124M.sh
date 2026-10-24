export CUDA_VISIBLE_DEVICES=0
export LOG_NAME=gpt2_124M

tk_model_name=tiktoken_gpt2_bpe
lm_model_name=gpt2_124M

python -u run_gpt2_124M.py \
    --task_name gpt2_124M_pretrain \
    --des 'GPT2-124M Pretrain' \
    --is_train 1 \
    --is_test 1 \
    --is_inference 0 \
    --data_source local \
    --data_url "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt" \
    --data_path ./dataset/pretraining \
    --data_file the-verdict.txt \
    --data_name the-verdict \
    --train_ratio 0.95 \
    --batch_size 16 \
    --num_workers 0 \
    --tokenizer_model $tk_model_name \
    --vocab_size 50257 \
    --model_name $lm_model_name \
    --context_length 256 \
    --embed_dim 768 \
    --d_ff 3072 \
    --n_heads 24 \
    --n_layers 24 \
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
    --train_epochs 100 \
    --patience 14 \
    --checkpoints ./results/pretrained_models/ \
    --test_results ./results/test_results/ \
    --max_new_tokens 50 \
    --use_cache 0 \
    --use_gpu 1 \
    --gpu_type cuda \
    --use_dp 0 \
    --use_ddp 0
