export CUDA_VISIBLE_DEVICES=0
export LOG_NAME=gpt2_124M

tk_model_name=tiktoken_gpt2_bpe
lm_model_name=gpt2_124M

python -u ./exp/exp_pretrain_gpt2_124M.py \
    --task_name gpt2_124M_pretrain \
    --des 'GPT2-124M Pretrain' \
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
    --dtype bfloat16 \
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
    --use_multi_gpu 0 \
    --use_ddp 0 \
    --devices 0,1,2,3,4,5,6,7
