export CUDA_VISIBLE_DEVICES=0
export LOG_NAME=gpt

model_name=gpt

python -u ./model_pre_training/run_gpt.py \
    --task_name tiny_gpt_pretrain \
    --des 'Tiny GPT Pretrain' \
    --is_train 1 \
    --is_test 0 \
    --is_inference 0 \
    --data_path ./dataset/pretrain/gpt \
    --data_file the-verdict.txt \
    --data_name the-verdict \
    --model_name $model_name \
    --context_length 256 \
    --vocab_size 50257 \
    --emb_dim 768 \
    --n_heads 12 \
    --n_layers 12 \
    --dropout 0.1 \
    --qkv_bias 0 \
    --dtype float32 \
    --max_new_tokens 50 \
    --tokenizer_model gpt2 \
    --seed 42 \
    --iters 1 \
    --train_epochs 30 \
    --batch_size 2 \
    --train_ratio 0.95 \
    --learning_rate 5e-4 \
    --initial_lr 3e-5 \
    --min_lr 1e-6 \
    --weight_decay 0.1 \
    --lradj type1 \
    --patience 14 \
    --checkpoints ./saved_results/pretrained_models/ \
    --test_results ./saved_results/test_results/ \
    --use_amp 0 \
    --num_workers 0 \
    --use_gpu 1 \
    --gpu_type cuda \
    --use_multi_gpu 0 \
    --devices 0,1,2,3,4,5,6,7
